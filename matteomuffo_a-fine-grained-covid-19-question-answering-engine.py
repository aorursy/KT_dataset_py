!pip install semanticscholar sentence_transformers
import base64
import csv
import glob
import json
import numpy as np
import pandas as pd
import re
import semanticscholar as sch
import torch
from IPython.display import display, Latex, HTML
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
root_path = "/kaggle/input/CORD-19-research-challenge"
metadata_path = "{}/metadata.csv".format(root_path)
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
all_json = glob.glob("{}/**/*.json".format(root_path), recursive=True)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except:
                self.abstract.append("No abstract available")
            for entry in content["body_text"]:
                self.body_text.append(entry['text'])
            self.abstract = '. '.join(self.abstract)
            self.body_text = '. '.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'publish_time': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    try:
        authors = meta_data['authors'].values[0].split(';')
        dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if Null value
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information
    dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
    # add the publishing data
    dict_['publish_time'].append(meta_data['publish_time'].values[0])
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'publish_time'])
df_covid.drop_duplicates(['title'], inplace=True)
df_covid.dropna(subset=['body_text'], inplace=True)
covid_terms =['covid', 'coronavirus disease 19', 'sars cov 2', '2019 ncov', '2019ncov', '2019 n cov', '2019n cov',
              'ncov 2019', 'n cov 2019', 'coronavirus 2019', 'wuhan pneumonia', 'wuhan virus', 'wuhan coronavirus',
              'coronavirus 2', 'covid-19', 'SARS-CoV-2', '2019-nCov']
covid_terms = [elem.lower() for elem in covid_terms]
covid_terms = re.compile('|'.join(covid_terms))
def checkYear(date):
    return int(date[0:4])

def checkCovid(row, covid_terms):
    return bool(covid_terms.search(row['body_text'].lower())) and checkYear(row['publish_time']) > 2019
df_covid['is_covid'] = df_covid.apply(checkCovid, axis=1, covid_terms=covid_terms)
df_covid_only = df_covid[df_covid['is_covid']==True]
df_covid_only = df_covid_only.reset_index(drop=True)
len(df_covid_only)
def preprocessing(text):
    # remove mail
    text = re.sub(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', 'MAIL', text)
    # remove doi
    text = re.sub(r'https\:\/\/doi\.org[^\s]+', 'DOI', text)
    # remove https
    text = re.sub(r'(\()?\s?http(s)?\:\/\/[^\)]+(\))?', '\g<1>LINK\g<3>', text)
    # remove single characters repeated at least 3 times for spacing error (e.g. s u m m a r y)
    text = re.sub(r'(\w\s+){3,}', ' ', text)
    # replace tags (e.g. [3] [4] [5]) with whitespace
    text = re.sub(r'(\[\d+\]\,?\s?){3,}(\.|\,)?', ' \g<2>', text)
    # replace tags (e.g. [3, 4, 5]) with whitespace
    text = re.sub(r'\[[\d\,\s]+\]', ' ',text)
     # replace tags (e.g. (NUM1) repeated at least 3 times with whitespace
    text = re.sub(r'(\(\d+\)\s){3,}', ' ',text)
    # replace '1.3' with '1,3' (we need it for split later)
    text = re.sub(r'(\d+)\.(\d+)', '\g<1>,\g<2>', text)
    # remove all full stops as abbreviations (e.g. i.e. cit. and so on)
    text = re.sub(r'\.(\s)?([^A-Z\s])', ' \g<1>\g<2>', text)
    # correctly spacing the tokens
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    # return lowercase text
    return text.lower()
df_covid_only['preproc_body_text'] = df_covid_only['body_text'].apply(preprocessing)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
def checkAnyStop(token_list, token_stops):
    return any([stop in token_list for stop in token_stops])

def firstFullStopIdx(token_list, token_stops):
    """
    Returns the index of first full-stop token appearing.  
    """
    idxs = []
    for stop in token_stops:
        if stop in token_list:
            idxs.append(token_list.index(stop))
    minIdx = min(idxs) if idxs else None
    return minIdx

puncts = ['!', '.', '?', ';']
puncts_tokens = [tokenizer.tokenize(x)[0] for x in puncts]

def splitTokens(tokens, punct_tokens, split_length):
    """
    To avoid splitting a sentence and lose the semantic meaning of it, a paper is splitted 
    into chunks in such a way that each chunk ends with a full-stop token (['.' ';' '?' or '!']) 
    """
    splitted_tokens = []
    while len(tokens) > 0:
        if len(tokens) < split_length or not checkAnyStop(tokens, punct_tokens):
            splitted_tokens.append(tokens)
            break
        # to not have too long parapraphs, the nearest fullstop is searched both in the previous 
        # and the next strings.
        prev_stop_idx = firstFullStopIdx(tokens[:split_length][::-1], puncts_tokens)
        next_stop_idx = firstFullStopIdx(tokens[split_length:], puncts_tokens)
        if pd.isna(next_stop_idx):
            splitted_tokens.append(tokens[:split_length - prev_stop_idx])
            tokens = tokens[split_length - prev_stop_idx:]
        elif pd.isna(prev_stop_idx):
            splitted_tokens.append(tokens[:split_length + next_stop_idx + 1])
            tokens = tokens[split_length + next_stop_idx + 1:] 
        elif prev_stop_idx < next_stop_idx:
            splitted_tokens.append(tokens[:split_length - prev_stop_idx])
            tokens = tokens[split_length - prev_stop_idx:]
        else:
            splitted_tokens.append(tokens[:split_length + next_stop_idx + 1])
            tokens = tokens[split_length + next_stop_idx + 1:] 
    return splitted_tokens

def splitParagraph(text, split_length=90):
    tokens = tokenizer.tokenize(text)
    splitted_tokens = splitTokens(tokens, puncts_tokens, split_length)
    return [tokenizer.convert_tokens_to_string(x) for x in splitted_tokens]
df_covid_only['body_text_parags'] = df_covid_only['preproc_body_text'].apply(splitParagraph)
df_covid_only.head()
text = df_covid_only['body_text_parags'].to_frame()
body_texts = text.stack().tolist()
encoding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
covid_encoded = []
for body in tqdm(body_texts):
    covid_encoded.append(encoding_model.encode(body, show_progress_bar=False))
def computeMaxCosine(encoded_query, encodings):
    cosines = cosine_similarity(encoded_query[0].reshape(1, -1), encodings)
    return float(np.ndarray.max(cosines, axis=1))
def extractPapersIndexes(query, num_papers=5):
    encoded_query = encoding_model.encode([query.replace('?', '')], show_progress_bar=False)
    cosines_max = []
    for idx in range(len(covid_encoded)):
        paper = np.array(covid_encoded[idx])
        result = computeMaxCosine(encoded_query, paper)
        cosines_max.append(result)
        
    indexes_max_papers = np.array(cosines_max).argsort()[-num_papers:][::-1]
    return indexes_max_papers
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD)
qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD)

qa_model = qa_model.to(torch_device)
qa_model.eval()
def answerQuestion(question, paper):
    """
    This funtion provides the best answer found by the Q&A model, the chunk containing it
    among all chunks of the input paper and the score obtained by the answer
    """
    inputs = [qa_tokenizer.encode_plus(
        question, paragraph, add_special_tokens=True, return_tensors="pt") for paragraph in paper]
    answers = []
    confidence_scores = []
    for n, Input in enumerate(inputs):
        input_ids = Input['input_ids'].to(torch_device)
        token_type_ids = Input['token_type_ids'].to(torch_device)
        if len(input_ids[0]) > 510:
            input_ids = input_ids[:, :510]
            token_type_ids =token_type_ids[:, :510]
        text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[0])
        start_scores, end_scores = qa_model(input_ids, token_type_ids=token_type_ids)
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        # if the start token of the answer is contained in the question, the start token is moved to
        # the first one of the chunk 
        check = text_tokens.index("[SEP]")
        if int(answer_start) <= check:
            answer_start = check+1
        answer = qa_tokenizer.convert_tokens_to_string(text_tokens[answer_start:(answer_end+1)])
        answer = answer.replace('[SEP]', '')
        confidence = start_scores[0][answer_start] + end_scores[0][answer_end]
        if answer.startswith('. ') or answer.startswith(', '):
            answer = answer[2:]
        answers.append(answer)
        confidence_scores.append(float(confidence))
    
    maxIdx = np.argmax(confidence_scores)
    confidence = confidence_scores[maxIdx]
    best_answer = answers[maxIdx]
    best_paragraph = paper[maxIdx]

    return best_answer, confidence, best_paragraph
def findStartEndIndexSubstring(context, answer):   
    """
    Search of the answer inside the paragraph. It returns the start and end index.
    """
    search_re = re.search(re.escape(answer.lower()), context.lower())
    if search_re:
        return search_re.start(), search_re.end()
    else:
        return 0, len(context)
def postprocessing(text):
    # capitalize the text
    text = text.capitalize()
    # '2 , 3' -> '2,3'
    text = re.sub(r'(\d) \, (\d)', "\g<1>,\g<2>", text)
    # full stop
    text = re.sub(r' (\.|\!|\?) (\w)', lambda pat: pat.groups()[0]+" "+pat.groups()[1].upper(), text)
    # full stop at the end
    text = re.sub(r' (\.|\!|\?)$', "\g<1>", text)
    # comma
    text = re.sub(r' (\,|\;|\:) (\w)', "\g<1> \g<2>", text)
    # - | \ / @ and _ (e.g. 2 - 3  -> 2-3)
    text = re.sub(r'(\w) (\-|\||\/|\\|\@\_) (\w)', "\g<1>\g<2>\g<3>", text)
    # parenthesis
    text = re.sub(r'(\(|\[|\{) ([^\(\)]+) (\)|\]|\})', "\g<1>\g<2>\g<3>", text)
    # "" e.g. " word "  -> "word"
    text = re.sub(r'(\") ([^\"]+) (\")', "\g<1>\g<2>\g<3>", text)
    # apostrophe  e.g. l ' a  -> l'a
    text = re.sub(r'(\w) (\') (\w)', "\g<1>\g<2>\g<3>", text)
    # '3 %' ->  '3%'
    text = re.sub(r'(\d) \%', "\g<1>%", text)
    # '# word'  -> '#word'
    text = re.sub(r'\# (\w)', "#\g<1>", text)
    # https and doi
    text = re.sub(r'(https|doi) : ', "\g<1>:", text)
    return text
keywords_list = []
idx_design_map = []
with open("/kaggle/input/loe-keywords/loe_keywords.tsv") as infile:
    tsvreader = csv.reader(infile, delimiter="\t")
    for idx, line in enumerate(tsvreader):
        if idx == 0 or idx == 1:
            continue
        keywords_list.append(line[2].split(", "))
        idx_design_map.append(line[0])

keywords_list = [list(filter(None, elem)) for elem in keywords_list]
minimum_occurrencies = [2, 2, 2, 1, 4, 3, 3, 1, 1]
def getSingleLOEScore(paper_idx, loe_idx):
    keywords_analysed = re.compile("|".join(keywords_list[loe_idx]))
    return len(re.findall(keywords_analysed, df_covid_only['body_text'][paper_idx]))
def getPaperLOE(paper_idx):
    title_search = re.compile("systematic review|meta-analysis")
    if title_search.search(df_covid_only['title'][paper_idx]):
        return "systematic review and meta-analysis"
    scores = [getSingleLOEScore(paper_idx, idx) for idx in range(len(idx_design_map))]
    sorted_indexes = np.argsort(scores)[::-1]
    for sorted_index in sorted_indexes:
        if scores[sorted_index] >= minimum_occurrencies[sorted_index]:
            return idx_design_map[sorted_index]
    return "Information not available"
scimago_jr = pd.read_csv('/kaggle/input/scimagojournalcountryrank/scimagojr 2018.csv', sep=';')
scimago_jr.drop_duplicates(['Title'], inplace=True)
scimago_jr = scimago_jr.reset_index(drop=True)
def getAPIInformations(paper_id):
    paper_api = sch.paper(paper_id)
    if paper_api:
        return paper_api['influentialCitationCount'], paper_api['venue']
    else:
        return "Information not available", None

def getSingleContext(context, start, end):
    before_answer = context[:start]
    answer = "***" + context[start:end] + "***"
    after_answer = context[end:]
    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer
    context_answer = """<div class="single_answer">{}</div>""".format(postprocessing(content))
    return context_answer

def getAllContexts(question, indexes_papers):
    answers_list = []
    for paper_index in indexes_papers:
        answer, conf, paragraph = answerQuestion(question, df_covid_only['body_text_parags'][paper_index])
        if answer:
            author = df_covid_only['authors'][paper_index] if not pd.isna(df_covid_only['authors'][paper_index]) else "not available"
            journal = df_covid_only['journal'][paper_index] if not pd.isna(df_covid_only['journal'][paper_index]) else "not available"
            title = df_covid_only['title'][paper_index] if not pd.isna(df_covid_only['title'][paper_index]) else "not available"
            publish_time = df_covid_only['publish_time'][paper_index] if not pd.isna(df_covid_only['publish_time'][paper_index]) else "not available"
            start, end = findStartEndIndexSubstring(paragraph, answer)
            answer_parag = getSingleContext(paragraph, start, end)
            paper_citations_count, journal_api = getAPIInformations(df_covid_only['paper_id'][paper_index])
            journal = journal_api if journal_api else journal
            journal_row = scimago_jr[scimago_jr['Title'].apply(lambda x: x.lower()) == journal.lower()]
            journal_score = journal_row['SJR'].item() if not journal_row.empty else "Information not available"
            loe = getPaperLOE(paper_index)
            paper_answer = { 
                "answer": answer_parag, 
                "title": title,
                "journal": journal,
                "author": author,
                "publish_time": publish_time,
                "journal_score": journal_score,
                "paper_citations_count": paper_citations_count,
                "level_of_evidence": loe
            }
            answers_list.append(paper_answer)  
    return answers_list
def layoutStyle():
    style = """
        .single_answer {
            border-left: 3px solid red;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        .answer{
            color: red;
        }    
    """
    return "<style>" + style + "</style>"
def getAnswerDiv(i, answ_id, answer, button=True):
    div = """<div id="{answ_id}" class="tab-pane fade">"""
    if i is 0:
        div = """<div id="{answ_id}" class="tab-pane fade in active">"""
    div += """
        <h2>Answer</h2>
        <p>{answer}</p>
        <h3>Title</h3>
        <p>{title}</p>
        <h3>Author(s)</h3>
        <p>{author}</p>
        <h3>Journal</h3>
        <p>{journal}</p>
        <h3>Publication date</h3>
        <p>{publish_time}</p>
        """
    if button:
        div += getAnswerAddInfo("ad_{}".format(answ_id), answer)
    div += """</div>"""
    return div.format(
        answ_id=answ_id, 
        answer=answer['answer'], 
        title=answer['title'],
        author=answer['author'], 
        journal=answer['journal'],
        publish_time=answer['publish_time']
    )

def getAnswerAddInfo(answ_id, answer):
    div = """
        <button type="button" class="btn-warning" data-toggle="collapse" data-target="#{answ_id}">Additional Info</button>
        <div id="{answ_id}" class="collapse">
        <h4>Scimago Journal Score</h4>
        <p>{journal_score}</p>
        <h4>Paper citations</h4>
        <p>{paper_citations_count}</p>
        <h4>Level Of Evidence</h4>
        <p>{level_of_evidence}</p>
        </div>
        """
    return div.format(
        answ_id=answ_id, 
        journal_score=answer['journal_score'], 
        paper_citations_count=answer['paper_citations_count'],
        level_of_evidence=answer['level_of_evidence'])

def getAnswerLi(i, answer_id):
    div = """<li><a data-toggle="tab" href="#{ansid}">Ans {i}</a></li>"""
    if i is 0:
        div = """<li class="active"><a data-toggle="tab" href="#{ansid}">Ans {i}</a></li>"""
    return div.format(i=i+1, ansid=answer_id)

def getQuestionDiv(question, answers, topic_id, question_id, button=True):
    div = """
    <div>
      <button type="button" class="btn btn-info" data-toggle="collapse" data-target="#{question_id}">{question}</button>
      <div id="{question_id}" class="collapse">
        <nav class="navbar navbar-light" style="background-color: #E3F2FD; width: 400px">
            <div>
                <ul class="nav navbar-nav nav-tabs">""".format(
        question_id="{}_{}".format(topic_id, question_id), question=question)
    for i, answer in enumerate(answers):
        div += getAnswerLi(i, "{}_{}_{}".format(topic_id, question_id, i+1))
    div += """</ul>
        </div>
    </nav>
    <div class="tab-content">"""
    for i, answer in enumerate(answers):
        div += getAnswerDiv(
            i, "{}_{}_{}".format(topic_id, question_id, i+1), answer)
    return div + """</div> <br> </div> </div>"""

def getTaskDiv(task, questions):
    """
    :param task:
    :param questions: dict
    {question: [{"answer": "", "title": "", "journal": "", "authors": "", "journal_score": "", 
     "paper_citations_count": "", "level_of_evidence": ""}]
    :return:
    """
    topic_id = task.replace(" ", "").replace("?", "")
    quest_id = 0
    questions_div = ""
    queries = questions['queries']
    for question, answers in queries.items():
        questions_div += getQuestionDiv(question, answers, topic_id, quest_id)
        quest_id += 1
    topic_header = """
       <div>
         <button type="button" class="btn" data-toggle="collapse" data-target="#{id1}" style="font-size:20px">&#8226{task}: {task_name}</button>
         <div id="{id1}" class="collapse">
         {body}
         </div>
       </div>""".format(id1=topic_id, task=task, body=questions_div, task_name=questions['task_name'])
    return topic_header

def getHtmlCode(tasks, style):
    header = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
          <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
          {style}
        </head>
        <body> """.format(style=style)
    body = ""
    for task, questions in tasks.items():
        body += getTaskDiv(task, questions)
    html_code = header + body
    html_code += "</body>"
    return html_code
def getFullDictQueries(data, num_papers=5):
    final_dict = {}
    for task in data.keys():
        print(f"Processing: {task}")
        task_questions = data[task]['questions']
        dict_task_quest ={}
        dict_queries = {}
        for idx, query in enumerate(task_questions):
            print(f"Getting answers from query: {query}")
            indexes_papers = extractPapersIndexes(query, num_papers=num_papers)
            dict_queries[query] = getAllContexts(query, indexes_papers)
        dict_task_quest['queries'] = dict_queries
        dict_task_quest['task_name'] = data[task]['area']
        final_dict[task] = dict_task_quest
    return final_dict
data = {"Task1":
         {'area':'Transmission, incubation, and environmental stability',
          'questions': ["Are movement control strategies effective?",
                        "Are there diagnostics to improve clinical processes?",
                        "Does the environment affect transmission?",
                        "How long does the virus persist on surfaces?",
                        "How long individuals are contagious?",
                        "Is personal protective equipment effective ?",
                        "What is known about immunity?",
                        "What is the natural history of the virus?",
                        "What is the range of the incubation period in humans?"]},
        "Task2":
          {'area':"COVID-19 risk factors",
           'questions': ["Are co-infections risk factors?",
                        "Are male gender individuals more at risk for COVID-19?",
                        "Are pulmunary diseases risk factors?",
                        "Are there any public health mitigation measures considered effective?",
                        "Do we consider chronic kidney disease a risk factor for COVID-19?",
                        "Do we consider chronic respiratory diseases risk factors for COVID-19?",
                        "Do we consider drinking a potential risk factor for COVID-19?",
                        "Do we consider respiratory system diseases a risk factor for COVID-19?",
                        "How does chronic liver disease increases the risk for COVID-19?",
                        "How does obesity increases the risk for COVID-19?",
                        "How does overweight increases the risk for COVID-19?",
                        "Is cancer a risk factor for COVID-19?",
                        "Is cardio-cerebrovascular disease a risk factor for COVID-19?",
                        "Is cerebrovascular disease a risk factor for COVID-19?",
                        "Is individual's age considered a potential risk factor?",
                        "Is smoking a risk factor ?",
                        "What do we know about risk factors related to COPD?",
                        "What do we know about risk factors related to Diabetes?",
                        "What do we know about risk factors related to heart diseases?",
                        "What do we know about risk factors related to hypertension?",
                        "What is the basic reproductive number?",
                        "What is the serial interval?",
                        "What is the severity of the disease?",
                        "Which are high-risk patient groups?",
                        "Which are the environmental risk factors?"]},
        "Task3":
          {'area':"Vaccines, therapeutics, interventions, and clinical studies",
           'questions': ["Are there any drugs proven to be effective in treating COVID-19 patients?",
                        "What is the best method to combat the hypercoagulable state seen in COVID-19?",
                        "What is the efficacy of novel therapeutics being tested currently?"]},
        
        "Task5":
          {'area':"Non-pharmaceutical interventions",
           'questions': []},
        
        "Task6":
          {'area':"Ethical and social science considerations",
           'questions': []},
        
        "Task7":
          {'area':"Medical care",
           'questions': []},
        
        "Task4":
          {'area':"Diagnostics and surveillance",
           'questions': ["Are there diagnosis techniques based on antibodies?",
                        "Are there diagnosis techniques based on nucleic-acid tech?",
                        "Are there new advances in diagnosing SARS-COV-2?",
                        "Are there point-of-care tests being developed?",
                        "Are there rapid bed-side tests?",
                        "How does viral load relate to disease presentations?",
                        "How does viral load relate to likelihood of a positive diagnostic test?",
                        "Is there any policy or protocol for screening and testing?",
                        "What do we know about diagnostics and coronavirus?"]},
        
        "Task9":
          {'area':"Information sharing and inter-sectoral collaboration",
           'questions': []},

        "Task5":
          {'area':"How geography affects virality",
           'questions': ["Are there geographic variations in the mortality rate of COVID-19?",
                        "Are there geographic variations in the rate of COVID-19 spread?",
                        "Is there any evidence to suggest geographic based virus mutations?"]},
        
        "Task6":
          {'area':"Relevant factors",
           'questions': ["Are inter/inner travel restrictions effective?",
                        "Are multifactorial strategies effective to prevent secondary transmission?",
                        "How does temperature and humidity affect the transmission of 2019-nCoV?",
                        "Is case isolation effective?",
                        "Is community contact reduction effective?",
                        "Is personal protective equipment effective?",
                        "Is school distancing effective?",
                        "Is the transmission seasonal?",
                        "Is workplace distancing effective?",
                        "Significant changes in transmissibility in changing seasons?"]},
        
        "Task7":
          {'area':"Models and open questions",
           'questions': ["Are there changes in COVID-19 as the virus evolves?",
                        "Are there studies about phenotypic change?",
                        "Are there studies to monitor potential adaptations?",
                        "What do models for transmission predict?",
                        "What is known about mutations of the virus?",
                        "What is the human immune response to COVID-19?",
                        "What regional genetic variations (mutations) exist?"]},
        
        "Task8":
          {'area':"Patient descriptions",
           'questions': ["Can asymptomatic transmission occur during incubation?",
                        "How many pediatric patients were asymptomatic?",
                        "Is COVID-19 associated with cardiomyopathy and cardiac arrest?",
                        "What do we know about disease models?",
                        "What is the incubation period across different age groups?",
                        "What is the Incubation Period of the Virus?",
                        "What is the length of viral shedding after illness onset?",
                        "What is the longest duration of viral shedding?",
                        "What is the median viral shedding duration?",
                        "What is the natural history of the virus from an infected person?",
                        "Which is the proportion of patients who were asymptomatic?"]},
        
        "Task9":
          {'area':"Population studies",
           'questions': ["how to communicate with health care workers?",
                        "how to interact with high-risk elderly people?",
                        "what is the best management of patients who are underhoused or otherwise lower socioeconomic status?",
                        "best modes of communicating with target high-risk populations?",
                        "What are recommendations for combating and overcoming resource failures?",
                        "What are ways to create hospital infrastructure to prevent nosocomial outbreaks and protect uninfected patients?"]},
        
        "Task10":
          {'area':"Material studies",
           'questions': ["how about adhesion to hydrophilic or phobic surfaces?",
                        "how about decontamination based on physical science?",
                        "How does the virus persist on different materials?",
                        "Is there susceptibility to environmental cleaning agents?",
                        "What do we know about viral shedding in blood?",
                        "What do we know about viral shedding in stool?",
                        "What do we know about viral shedding in urine?",
                        "What do we know about viral shedding nasopharynx?"]},
        "Task11":
          {'area':"Miscellaneous",
           'questions': ["Is there more than one strain in circulation?",
                        "Are there methods to control the spread in communities?",
                        "Which efforts have been made to identify the underlying drivers of fear?",
                        "Are there oral medications that might potentially work?",
                        "Which are the best ways of communicating with target high-risk populations?"]},
        "Task12":
         {'area':'CMS',
          'questions': ["Is CT scan a reliable tool to detect the presence of covid-19 infection?",
                        "Which serological rapid test is shown to be the most reliable (in terms of specificity and sensitivity) to detect the presence of covid-19 infection?",
                        "Which is the average duration of the incubation period of covid-19 virus?",
                        "Does the duration of the incubation period of covid-19 virus depend on individual characteristics (such as age, gender, comorbidities, etc.)?",
                        "Does the viral load affect the severity of symptoms from covid-19?",
                        "Is there scientific evidence that flu vaccine prevents the infection from covid-19?",
                        "Is there scientific evidence that some blood types are more prone to be infected by covid-19?",
                        "Are tracking apps an effective tool to prevent the spread of covid-19?",
                        "Is there scientific evidence that warm weather reduces the spread of covid-19?",
                        "Is there scientific evidence that conjunctivitis is a symptom of covid-19?",
                        "What is the false positive and false negative rate in Diasorin serological rapid test?",
                        "Are the antibodies IgM an effective measure to detect the presence of covid-19?",
                        "What is the average persistence of IgM antibodies in the blood, for individuals infected by covid-19?",
                        "Is there scientific evidence that some ethnic groups are more affected by covid-19?",
                        "Which comorbidities are responsible for more severe clinical conditions caused by covid-19?",
                        "How many days, on average, does the intensive care treatment last, for individuals infected by covid-19?"]}

       }
full_code = getFullDictQueries(data)
display(HTML(getHtmlCode(full_code, layoutStyle())))
from bs4 import BeautifulSoup
cnt = 1
df_list = []
for task_key in full_code.keys():
    results_table = pd.DataFrame(columns=['Question', 'Title', 'Authors', 'Answer', 'Journal', 'Publication date', 'Journal score', 'Paper citations count',
                                     'Level of evidence'])
    for question_key in full_code[task_key]['queries'].keys():
        for idx in range(len(full_code[task_key]['queries'][question_key])):
            row = [question_key,
                   full_code[task_key]['queries'][question_key][idx]['title'],
                   full_code[task_key]['queries'][question_key][idx]['author'],
                   BeautifulSoup(full_code[task_key]['queries'][question_key][idx]['answer']).text,
                   full_code[task_key]['queries'][question_key][idx]['journal'],
                   full_code[task_key]['queries'][question_key][idx]['publish_time'],
                   full_code[task_key]['queries'][question_key][idx]['journal_score'],
                   full_code[task_key]['queries'][question_key][idx]['paper_citations_count'],
                   full_code[task_key]['queries'][question_key][idx]['level_of_evidence']
                  ]
            results_table.loc[cnt] = row
            cnt += 1
    df_list.append(results_table)
def create_download_link(df, title, filename):  
    csv = df.to_csv(sep="\t")
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df_list[0], "Download task 1 table", "task_1.tsv")
create_download_link(df_list[1], "Download task 2 table", "task_2.tsv")
create_download_link(df_list[2], "Download task 3 table", "task_3.tsv")
create_download_link(df_list[3], "Download task 4 table", "task_4.tsv")
create_download_link(df_list[4], "Download task 5 table", "task_5.tsv")
create_download_link(df_list[5], "Download task 6 table", "task_6.tsv")
create_download_link(df_list[6], "Download task 7 table", "task_7.tsv")
create_download_link(df_list[7], "Download task 8 table", "task_8.tsv")
create_download_link(df_list[8], "Download task 9 table", "task_9.tsv")
create_download_link(df_list[9], "Download task 10 table", "task_10.tsv")
create_download_link(df_list[10], "Download task 11 table", "task_11.tsv")
create_download_link(df_list[11], "Download task 12 table", "task_12.tsv")
