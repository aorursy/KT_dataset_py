import pandas as pd
import numpy as np
import json
import re
import os
import io
from zipfile import ZipFile
from urllib.request import urlopen, urlretrieve
def downloadFile(url, target):
    target_dir = target.rsplit("/",1)[0] 
    if not os.path.exists(target_dir) and target!=target_dir:
        os.makedirs(target_dir)
    if not os.path.isfile(target):
        urlretrieve(url, target)

        
def downloadZip(url, target):
    if os.path.isdir(target) == False:
        with urlopen(url) as zipresp:
            with ZipFile(io.BytesIO(zipresp.read())) as zfile:
                zfile.extractall(target)

                
def get_all_fulltext_files(path_data):
    filepaths = []
    for dirpath, dirnames, filenames in os.walk(path_data):
        for filename in filenames:
            if filename.endswith(".json"):
                filepaths.append(dirpath+"/"+filename)
    return filepaths
# Specify where the dataset is located or should be stored:
path_data = '../input/CORD-19-research-challenge'

# List all fulltext files:
fulltext_files = get_all_fulltext_files(path_data)
questions = ['Ethical and moral issues during a pandemic',
             'Ethical principles applied during a pandemic',
             'Sustained education in ethics',
             'Team at WHO to connect global networks of social sciences',
             'Impact and conseqeuences of school closures on society during a pandemic',
             'Physical and psychological health of doctors and nurses during Covid-2019',
             'Needs of staff and hospitals during Covid-2019',
             'Reasons and drivers of anxiety']
import nltk
stopwords = set(nltk.corpus.stopwords.words('english')) 
def tokenize(sentence):
    
    global stopwords
       
    # Ensure that 'WHO' is not interpreted as 'who'
    sentence.replace('WHO', 'worldhealthorganization')
    sentence.replace('World Health Organization', 'worldhealthorganization')
    
    # Split current sentence into words:
    words = re.split(r'\W+', sentence)
            
    # Transform all words into tokens
    tokens = []
    for word in words:
        word = word.strip() # removes any whitespaces before or after word
        word = word.lower() # removes any uppercases
        if re.search("^[0-9\.\-\s\/]+$", word)==None and len(word)>0 and word not in stopwords:                      
            tokens.append(word)
            
    return tokens


def tokenize_multiple(paragraph=None, arr=None):
    
    # Split input into single sentences:
    if paragraph==None: paragraph = '; '.join(arr)
    sentences = re.split('[;.?:]\s', paragraph) 
        
    # Loop over sententences and tokenize them
    tokenized_sentences, tokenized_sentences_id = [], []
    for sentence_id, sentence in enumerate(sentences):  
        tokenized_sentence = tokenize(sentence)
        if len(tokenized_sentence) >= 1:
            tokenized_sentences.append(tokenized_sentence)
            tokenized_sentences_id.append(sentence_id)

    return tokenized_sentences, tokenized_sentences_id
# Tokenize titles and abstracts from meta data file:
meta_data              = pd.read_csv(path_data+"/metadata.csv")
titles                 = meta_data['title'].dropna()
abstracts              = meta_data['abstract'].dropna().str.replace('Abstract ','')
tokenized_titles, _    = tokenize_multiple(arr=titles)
tokenized_abstracts, _ = tokenize_multiple(arr=abstracts)
print(len(titles), 'titles and', len(abstracts), 'abstracts were tokenized successfully.\n-----------')

# Tokenize all fulltext files
tokenized_fulltexts = []
print('Start tokenizing', len(fulltext_files), 'fulltexts.')
for i, filepath in enumerate(fulltext_files):
    file                  = json.load(open(filepath))
    body_text             = pd.DataFrame(file["body_text"])
    if 'text' in body_text.columns:
        paragraphs            = pd.DataFrame(file["body_text"])['text']
        fulltext              = ' '.join(paragraphs)
        tokenized_fulltext, _ = tokenize_multiple(paragraph=fulltext)
        tokenized_fulltexts += tokenized_fulltext
    if (i+1)%1000==0 or (i+1)==len(fulltext_files): print(i+1, 'fulltext files were tokenized successfully.')
                
# Merge all tokenizations together
tokenized_sentences = tokenized_titles + tokenized_abstracts + tokenized_fulltexts
import logging
import gensim 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model_w2v = gensim.models.Word2Vec(tokenized_sentences, size=200, window=10, min_count=10, workers=8, iter=20)
model_w2v.wv.most_similar(positive='ethical')
model_w2v.save('../output/models/word2vec.model')
import gensim 
from scipy import spatial
import math
model_w2v = gensim.models.Word2Vec.load('../output/models/word2vec.model')
def embedd(tokenized_sentence, model):
    embedded_words = np.array([model.wv[word] for word in tokenized_sentence if word in model.wv])
    if len(embedded_words) != 0:
        return np.mean(embedded_words, axis=0)
    else:
        return None;
def embedd_multiple(arr, model):
    arr_embedded = []
    for row in arr:
        row_embedded = embedd(row, model)
        arr_embedded.append(row_embedded)
    return arr_embedded
def computeSimilarity(model, threshold):
    
    global fulltext_files
    global questions
    
    # Create directory
    if not os.path.exists('../output/results'):
        os.makedirs('../output/results')
    
    # Embedd questions
    tokenized_questions, _ = tokenize_multiple(arr=questions)
    embedded_questions     = embedd_multiple(arr=tokenized_questions, model=model)
    
    # Define vector to store similarities (Structure: question_id - file_id - sentence_id - similarity)
    similarities = np.zeros((0,4))

    # Loop over files
    print('Start evaluating', len(fulltext_files), 'fulltext files.')
    for file_id, file_path in enumerate(fulltext_files):

        # Load and tokenize file
        file       = json.load(open(file_path))
        body_text  = pd.DataFrame(file["body_text"])
        if 'text' in body_text.columns:
            paragraphs = body_text['text']
            fulltext   = ' '.join(paragraphs)
            tokenized_sentences, sentence_ids = tokenize_multiple(paragraph=fulltext)

            # Embedd all tokenized sentences and compute their similarity with each of the questions
            for tokenized_sentence, sentence_id in zip(tokenized_sentences, sentence_ids): 
                embedded_sentence = embedd(tokenized_sentence, model)
                if embedded_sentence is not None:
                    for question_id, embedded_question in enumerate(embedded_questions):
                        # Exclude tokens that appear in question and sentence (prevent sentences to just repeat question)
                        tokenized_question = tokenized_questions[question_id]
                        shared_tokens = np.intersect1d(tokenized_sentence, tokenized_question)
                        for token in shared_tokens:
                            embedded_sentence -= model.wv[token]*(1/len(tokenized_sentence))              
                        # Compute similarities
                        brevity_penalty = 1 if len(tokenized_sentence)>len(tokenized_question) else math.exp(1 - len(tokenized_question)/len(tokenized_sentence))
                        similarity = brevity_penalty*abs(1 - spatial.distance.cosine(embedded_question, embedded_sentence))
                        if (similarity>threshold):
                                similarities = np.concatenate((similarities, [[question_id, file_id, sentence_id, similarity]]))

        # Progress bar
        if (file_id+1)%500==0: 
            print(file_id+1, 'files embedded successfully.')
            np.save('../output/results/res_all.npy', similarities)
            
    return similarities
results = computeSimilarity(model=model_w2v, threshold=0.7)
# Store all results
np.save('../output/results/res_all.npy', results)

# Store only top k results for each question:
k = 50
results_reduced = np.zeros((0,4))
for question_id in range(len(questions)):
    res_question_all = results[results[:,0]==question_id]
    res_question_topK = res_question_all[res_question_all[:,3].argsort()][::-1][0:k]
    results_reduced = np.concatenate((results_reduced, res_question_topK))
results_reduced
np.save('../output/results/res_reduced.npy', results_reduced)
from IPython.core.display import display, HTML
def output(path, threshold=0, max_rows=100):
    
    global questions
    global fulltext_files
    
    # Load results:
    results = np.load(path, allow_pickle=True)
    
    # Filter only results above threshold
    results = results[results[:,3]>threshold]
    
    for question_id, question in enumerate(questions):
    
        # Print html title and prepare table
        display(HTML('<h3>'+question+'</h3>'))
        df_table = pd.DataFrame(columns = ['<p align="left">Excerpt','Relevancy'])

        # Filter and rank results to this question
        results_subset  = results[results[:,0]==question_id]
        results_ranked  = results_subset[results_subset[:,3].argsort()][::-1][0:max_rows]
        
        for result in results_ranked:

            # Retrieve parameters
            file_id     = int(result[1])
            sentence_id = int(result[2])
            similarity  = result[3]

            # Load original sentence
            filepath    = fulltext_files[file_id]
            file        = json.load(open(filepath))     
            paragraphs  = pd.DataFrame(file["body_text"])['text']
            fulltext    = ' '.join(paragraphs)
            sentence    = re.split('[;.?:]\s', fulltext)[sentence_id]           

            # Output results
            df_table.loc[len(df_table)] = ['<p align="left">'+sentence+'</p>', np.round(similarity,2)]

        # Print table
        pd.set_option('display.max_colwidth', -1)
        df_table = HTML(df_table.to_html(escape=False,index=False))
        display(df_table)
        
output('../output/results/res_reduced.npy', max_rows=10)
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
# GLOBAL VARIABLES

question_id = None
answer_id = None
path_res_all = None
path_res_relevant = None
res_all = None
res_relevant = None
meta_data = None
        
        
    
# WIDGETS

# Widget to select question
w_header       = widgets.HTML(value='<h3>Please select a research question and review the excerpts manually:</h3>')
w_questions    = widgets.Dropdown(options=questions, value=questions[0], layout = widgets.Layout(width='100%'))

# Buttons for navigation
w_btn_del      = widgets.Button(description='Delete entry', button_style='danger', icon='trash')
w_btn_add      = widgets.Button(description='Mark as relevant', button_style='success', icon='check')
w_btn_prev     = widgets.Button(description='< Previous')
w_btn_next     = widgets.Button(description='Next >')
w_buttons_nav  = widgets.HBox([w_btn_prev, widgets.HBox([w_btn_del, w_btn_add]), w_btn_next], layout=widgets.Layout(display='flex', justify_content='space-between'))

# Widgets to display an answer to the selected questions
w_progress     = widgets.HTML(value='1')
w_excerpt      = widgets.Textarea(description='Excerpt:', value='', layout = widgets.Layout(width='100%'))
w_paper_id     = widgets.HTML(description='Paper ID:', value='')
w_paper_title  = widgets.HTML(description='Title:', value='')
w_accuracy     = widgets.HTML(description='Accuracy:', value='')
w_date         = widgets.HTML(description='Date:', value='')
w_authors      = widgets.HTML(description='Authors:', value='')
w_paragraph    = widgets.HTML(description='Paragraph:', value='')
w_fulltext     = widgets.HTML(description='Fulltext:', value='')

# Button to edit excert
w_btn_save     = widgets.Button(description='Save as relevant', button_style='success', icon='check')
w_btn_undo     = widgets.Button(description='Undo', button_style='', icon='undo')
w_buttons_edit = widgets.HBox([w_btn_undo, w_btn_save], layout=widgets.Layout(display='flex', justify_content='flex-end'))

# Buttons to show more details
w_btn_show_paragraph = widgets.Button(description='Display paragraph')
w_btn_show_fulltext  = widgets.Button(description='Display fulltext')
w_buttons_more       = widgets.HBox([w_btn_show_paragraph, w_btn_show_fulltext], layout=widgets.Layout(display='flex', justify_content='flex-end'))

# Container that wraps everything together
w_body      = widgets.VBox([w_buttons_nav, w_progress, w_paper_title, w_date, w_authors, w_excerpt, w_buttons_edit, w_paragraph, w_buttons_more,  w_fulltext])

# Widget to show warning message:
w_warning   = widgets.HTML(value="<center>You have reviewed all answers. Please choose a different question or move on to the next section to retrieve/visualize the results.</center>")



# FUNCTION TO DISPLAY EXCERPTS

def load_excerpt(show_fulltext=False, show_paragraph=False):
    
    global question_id
    global answer_id
    global res_all
    global meta_data
    
    # Retrieve list of all answers to that question
    answers = res_all[res_all[:,0]==question_id]
    if len(answers) != 0:
        
        # Load file that contains current answer
        row              = answers[answer_id]
        file_id          = int(row[1])
        sentence_id      = int(row[2])
        similarity_score = row[3]
        filepath         = fulltext_files[file_id]
        file             = json.load(open(filepath))

        # Load meta data and relevant fields
        paper_id         = file["paper_id"]
        paragraphs       = pd.DataFrame(file["body_text"])['text']
        fulltext         = ' '.join(paragraphs)
        sentence         = re.split('[;.?:]\s', fulltext)[sentence_id]
        meta_data_rows   = meta_data.loc[meta_data['sha']==paper_id]
        
        # Display answer
        w_progress.value = '<center><font color="grey">'+str(answer_id+1)+'/'+str(len(answers))+'</font></center>'
        w_body.layout.display = 'flex'
        w_excerpt.value  = sentence
        w_paper_id.value = paper_id
        w_accuracy.value = str(np.round(similarity_score*100,1))+"%"
        if len(meta_data_rows) == 1:
            w_date.value = str(meta_data_rows.iloc[0]['publish_time'])
            w_authors.value = str(meta_data_rows.iloc[0]['authors'])
            w_paper_title.value = str(meta_data_rows.iloc[0]['title'])
        for paragraph in paragraphs:
            if sentence in paragraph:
                w_paragraph.value = paragraph.replace(sentence, '<font color="blue"><b>'+sentence+'</b></font>')  
                w_fulltext.value = '<br><br>'.join(paragraphs).replace(sentence, '<font color="blue"><b>'+sentence+'</b></font>')
                break;
            w_fulltext.value = ' '.join(paragraphs).replace(sentence, '<font color="blue"><b>'+sentence+'</b></font>')
            w_paragraph.value = ''

        # Hide widgets
        w_buttons_edit.layout.display = 'none'
        w_paragraph.layout.display = 'none'
        w_fulltext.layout.display = 'none'
        w_warning.layout.display = 'none'

        # Enable buttons
        if w_paragraph.value != '':
            w_btn_show_paragraph.layout.display = 'inline'
        w_buttons_more.layout.display = 'flex'
    
    else: 
        w_body.layout.display = 'none'
        w_warning.layout.display = 'flex'

        

# EVENTS

# Event: Change question
def change_question(evt):
    if evt['type'] == 'change' and evt['name'] == 'value':
        global question_id
        global answer_id
        question = evt['new']
        question_id = questions.index(question)
        answer_id = 0
        load_excerpt()
w_questions.observe(change_question)

# Event: Show previous excerpt
def prev(evt): 
    global answer_id
    answer_id = max(answer_id-1, 0)
    load_excerpt()
w_btn_prev.on_click(prev)
    
# Event: Show next excerpt
def next(evt): 
    global answer_id
    answers   = res_all[res_all[:,0]==question_id]
    answer_id = min(answer_id+1, len(answers)-1)
    load_excerpt()
w_btn_next.on_click(next)

# Event: Mark an excerpt as relevant
def delete(evt):
    global res_all
    global question_id
    global answer_id
    answers = res_all[res_all[:,0]==question_id]
    row     = answers[answer_id]
    for result_id, result in enumerate(res_all):
        if np.array_equal(result, row):
            res_all = np.delete(res_all, result_id, axis=0)
    answer_id -= 1
    np.save(path_res_all, res_all)
    next(evt)
w_btn_del.on_click(delete)

# Event: Mark an excerpt as relevant
def save(evt):
    global res_relevant
    global question_id
    question    = questions[question_id]
    excerpt     = w_excerpt.value
    authors     = w_authors.value
    year        = w_date.value[:4]
    paper_id    = w_paper_id.value
    paper_title = w_paper_title.value
    res_relevant = np.concatenate((res_relevant, [[question, excerpt, paper_id, paper_title, authors, year]]), axis=0)
    np.save(path_res_relevant, res_relevant)
    delete(evt)
w_btn_add.on_click(save)
w_btn_save.on_click(save)

# Event: Display fulltext
def show_fulltext(evt):
    w_fulltext.layout.display = 'flex'
    w_buttons_more.layout.display = 'none'
w_btn_show_fulltext.on_click(show_fulltext)

# Event: Display paragraph
def show_paragraph(evt):
    w_paragraph.layout.display = 'flex'
    w_btn_show_paragraph.layout.display = 'none'
w_btn_show_paragraph.on_click(show_paragraph)

# Event: Show buttons for editing the excerpt
def print_evt(evt):
    if evt['type'] == 'change' and evt['name'] == 'value':
        w_buttons_edit.layout.display = 'flex'
w_excerpt.observe(print_evt)

# Event: Revert the excerpt to what it was before
def print_evt(evt):
    load_excerpt()
w_btn_undo.on_click(print_evt)



# INITIALIZE UI

def initializeUI(p_res_all, p_res_relevant='', continueReview=False):
    
    # Load results
    global res_all
    global path_res_all
    path_res_all = p_res_all
    res_all = np.load(path_res_all)
    res_all = res_all[res_all[:,3].argsort()][::-1]  # Sorts results by importance

    # Load previously reviewed results (Structure: question, excerpt, paper_id, paper_title, author, year)
    global res_relevant
    global path_res_relevant
    path_res_relevant = p_res_relevant
    res_relevant = np.load(path_res_relevant) if continueReview else np.zeros((0,6))
    
    # Load meta data
    global path_data
    global meta_data
    meta_data      = pd.read_csv(path_data+"/metadata.csv")
    
    # Load first excerpt
    global question_id
    global answer_id
    question_id    = 0
    answer_id      = 0
    load_excerpt()
    
    display(w_header, w_questions, w_warning, w_body)
initializeUI(p_res_all="../output/results/res_reduced.npy", p_res_relevant="../output/results/res_relevant.npy", continueReview=False)
from IPython.core.display import display, HTML
# Load relevant results:
res_relevant = np.load("../output/results/res_relevant.npy")

def display_results(results):
    """Displays extract, authors, publication year and name of the articles for each 
    research question given in results."""
    global questions
    
    for question in questions:
    
        # Print html title:
        display(HTML('<h2>'+question+'</h2>'))
        # Find all rows correspionding to question:
        rows = res_relevant[res_relevant[:,0] == question]
        for row in rows:
            question, excerpt, paper_id, title, authors, year = row
            display(HTML('<h4>'+excerpt+'</h4>'+'<p><font color="grey"><i>~ '+authors+' ('+year+'). '+title+'</i></font></p><br>'))
        display(HTML('<br>'))
        
# Display results
display_results(res_relevant)