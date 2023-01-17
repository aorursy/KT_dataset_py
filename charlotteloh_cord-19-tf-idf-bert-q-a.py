from IPython.core.display import display, HTML

import glob

for filename in glob.glob('/kaggle/input/cord-answers1/*.txt'):

    f = open(filename,"r")

    ans = f.read()

    display(HTML(ans))
import numpy as np

import pandas as pd 

import glob

import json

import math



root_dir = '/kaggle/input/CORD-19-research-challenge' 

df = pd.read_csv(f'{root_dir}/metadata.csv') # Reading the metadata of the data set

sha_abstract = df[['sha', 'abstract']] # Filtering out rows with both `SHA` and `Abstract`

sha_abstract = sha_abstract.dropna().reset_index()[['sha', 'abstract']] # Separating `SHA` and `Abstract` columns into separate lists



corpus = list(sha_abstract.values[:,1])

sha = list(sha_abstract.values[:,0])
fs = open('/kaggle/input/stopwords/stopwordlist.txt','r') # Source of stopwordlist: https://gist.github.com/sebleier/554280

stopWordsList = fs.read().split(" ")
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



query = ['coronavirus coronaviruses cov covid']

qC = query+corpus

featuresize = 512 # It was found that increasing the featuresize did not change the number of abstracts found

vectorizer = TfidfVectorizer(stopWordsList,max_features=featuresize)

X = vectorizer.fit_transform(qC).toarray()

#print(vectorizer.get_feature_names())



covidCS = []

covid_SHA = []

ID = []

for i in range(len(corpus)):

    cossim = cosine_similarity(X[0].reshape(1,-1),X[i+1].reshape(1,-1))

    if cossim > 0.: 

        covidCS.append(cossim)

        covid_SHA.append(sha[i])

        ID.append(i)

print("{} out of {} abstracts contains your query for featuresize = {}".format(len(covidCS),len(corpus),featuresize))

# Each covid_SHA may have multiple SHAs. 

# Create a list of SHAs, with each element containing a list of SHAs pointing to the same article

full_sha = []

for sha in covid_SHA:

    newsha = sha.split('; ')

    full_sha.append(newsha)

    

doc_paths = glob.glob(f'{root_dir}/*/*/*/*.json')

root_dir = '/kaggle/input/CORD-19-research-challenge'

df = pd.read_csv(f'{root_dir}/metadata.csv')

sub_df = df[['sha', 'title', 'authors', 'url']]

def get_text(full_sha):

    # Initialise the full_text array of articles

    full_text=[]

    for full_sha_num in full_sha: # for each list of SHA pointing to the same article

        sha = full_sha_num[0]

        document_path = [path for path in doc_paths if sha in path] # we find the document path for the first SHA in the list as all SHA points to the same article

        with open(document_path[0]) as f:

            file = json.load(f)

            article_series = sub_df[sub_df['sha'].str.contains(sha, na=False)]

            sha = article_series.values[0][0]

            title = article_series.values[0][1]

            authors = article_series.values[0][2]

            url = article_series.values[0][3]

            for text_part in file['body_text']:

                text = text_part['text']

                # remove citations from each paragraph

                for citation in text_part['cite_spans']:

                    text = text.replace(citation['text'], "")

                full_text.append([sha, title, authors, url, text])

    return full_text

full_text = get_text(full_sha)

full_text_df = pd.DataFrame(full_text, columns=['sha','title','authors', 'url', 'text'])

pd.set_option('display.max_colwidth', 50)

full_text_df.head()

print("No. of paragraphs: ",len(full_text_df))
import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer



model = BertForQuestionAnswering.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')

model.eval(); # set model in eval mode
# Formatting functions to be used later



def highlight_paragraph(paragraph, token_start, token_end,num_seg_a):

    para =""

    para = para + " ".join(paragraph[num_seg_a:token_start])

    para = para + "[start]"  + " ".join(paragraph[token_start:token_end+1])+ "[end]"

    para = para + " ".join(paragraph[token_end+1:])

    return para



def split_paragraph(paragraph, qns):

    input_ids = tokenizer.encode(paragraph)

    #print(len(input_ids))

    qns_ids = tokenizer.encode(qns)

    #print(len(qns_ids))

    total_ids = qns_ids + input_ids

    tokens = tokenizer.convert_ids_to_tokens(total_ids)

    assert len(tokens) == len(qns_ids) + len(input_ids)

    num_sections = math.ceil(len(tokens)/512) 

    sections = []

    while len(tokens) > 512: 

        one_section = tokens[:512]

        one_sect_id = total_ids[:512]

        found_end = False

        i = 0

        last_index = len(one_section)

        while not found_end:

            i += 1

            if (one_section[last_index-i-1][-1] == "." and one_section[last_index-i][0].isupper()):

                found_end = True               

                sent_end = i

                sections.append((one_section[:last_index-sent_end],one_sect_id[:last_index-sent_end]))

                tokens = tokens[:len(qns_ids)+1]+tokens[last_index-sent_end:]

                total_ids = total_ids[:len(qns_ids)+1]+tokens[last_index-sent_end:]

    sections.append((tokens,total_ids))

    return sections
def askQuestion(taskno, paragraphs, titles, authors, urls):

    task = taskno[0] # element 0 is the task topic where tf-idf is performed as first filter

    question = taskno[1] # element 1 is the specific question

    tP = [task]+paragraphs

    vectorizer = TfidfVectorizer(stopWordsList,max_features=featuresize) 

    Y = vectorizer.fit_transform(tP).toarray() # vectorize task topic and every paragraph

    taskCS = []

    

    counter_threshold = 0

    # use cosine similarity to find top k paragraphs relevant to task topic

    for i in range(len(paragraphs)):

        cossim = cosine_similarity(Y[0].reshape(1,-1),Y[i+1].reshape(1,-1))

        taskCS.append(cossim[0][0])

        

    ## Find top k similar paragraphs

    k = 20



    ranked = np.argsort(taskCS)[::-1][:k]

    ranked_CS = [taskCS[i] for i in ranked]

    ranked_para = [paragraphs[i] for i in ranked]

    ranked_url = [urls[i] for i in ranked]

    ranked_title = [titles[i] for i in ranked]

    ranked_author = [authors[i] for i in ranked]

    

    sectionlist = []

    answerlist=[]

    urllist=[]

    authorlist=[]

    titlelist=[]

    

    for i in range(k):

        answer_text = ranked_para[i]

        sections = split_paragraph(answer_text,question)

        

        for s in range(len(sections)):

            urllist.append(ranked_url[i])

            authorlist.append(ranked_author[i])

            titlelist.append(ranked_title[i])

            

            tokens, input_ids = sections[s]



            sep_index = input_ids.index(tokenizer.sep_token_id) # Search the input_ids for the first instance of the `[SEP]` token.

            num_seg_a = sep_index + 1 # The number of segment A tokens includes the [SEP] token istelf

            num_seg_b = len(input_ids) - num_seg_a # The remainder are segment B.   

            segment_ids = [0]*num_seg_a + [1]*num_seg_b # Construct the list of 0s and 1s.

            assert len(segment_ids) == len(input_ids) # There should be a segment_id for every input token.

    

            start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.

                                     token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text



            answer_start = torch.argmax(start_scores) # Find the tokens with the highest `start` and `end` scores.

            answer_end = torch.argmax(end_scores)   

            answer = ' '.join(tokens[answer_start:answer_end+1]) # Combine the tokens in the answer and print it out.



            para = highlight_paragraph(tokens,answer_start,answer_end,num_seg_a)

            para = para.replace(' ##', '')

            para = para.replace('[CLS]', '')

            para = para.replace('[SEP]', '')

            max_length = 10 # we limit answer lengths to < 10 words

            if (answer_start != answer_end & answer_end - answer_start < max_length):  

                answerlist.append(answer)

                sectionlist.append(para)

                

            else:

                sectionlist.append("Answer not found")

                answerlist.append("Answer not found")

    

    new_df = pd.DataFrame(list(zip(answerlist,sectionlist,titlelist,authorlist,urllist)),columns=['Answers','Evidence','Title','Authors','URL'])

    

    new_df = new_df[~new_df['Answers'].str.match("Answer not found")]

    new_df = new_df.iloc[:,1:]

    return new_df
paragraphs = list(full_text_df.values[:,4])

SHAp = list(full_text_df.values[:,0])

titles = list(full_text_df.values[:,1])

authors = list(full_text_df.values[:,2])

urls = list(full_text_df.values[:,3])



featuresize = 1024



task1 = ['incubation', "How long is the incubation period?"]

task2 = ['movement strategies',"Effectiveness of movement control strategies"]

task3 = ['transmission asymptomatic',"Prevalence of asymptomatic shedding and transmission"]

task4 = ['transmission', 'mode of transmission']
#for taskno in [task1,task2,task3]:

taskno = task1

answers_df = askQuestion(taskno, paragraphs, titles, authors, urls)



html_df = answers_df.to_html()

html_df = html_df.replace('[start]', '<span style="color: blue;"> ')

html_df = html_df.replace('[end]', ' </span>')

html_df = "<h2>Question: "+taskno[1]+"</h2>"+html_df

html_df = "<h2>Topics: "+taskno[0]+"</h2>"+html_df

display(HTML(html_df))
filtered = answers_df.loc[[7,9,11],['Evidence','Title','Authors', 'URL']]

html_df = filtered.to_html()

html_df = html_df.replace('[start]', '<span style="color: blue;"> ')

html_df = html_df.replace('[end]', ' </span>')

html_df = "<h2>Question: "+taskno[1]+"</h2>"+html_df

html_df = "<h2>Topics: "+taskno[0]+"</h2>"+html_df

display(HTML(html_df))



# Save answers as txt file

# with open("incubation_answers.txt", "w+") as f:

#     f.write(html_df)
for filename in glob.glob('/kaggle/input/cord-answers1/*.txt'):

    f = open(filename,"r")

    ans = f.read()

    display(HTML(ans))