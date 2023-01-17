!pip install sentence-transformers
#importing the required libraries
import glob
import json
import logging
import os
import prettytable
import pickle
import re
import shutil
import tqdm
import textwrap
import warnings
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import plotly.offline as py
import scattertext as st

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.bleu_score import sentence_bleu

import torch
import tensorflow as tf
import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer
from sentence_transformers import models, SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from transformers import BertForQuestionAnswering, BertTokenizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from absl import logging
from IPython.core.display import display, HTML
from PIL import Image
from tqdm.notebook import tqdm
from ipywidgets import *
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
warnings.simplefilter('ignore')

print("Packages imported successfully")
from IPython.core.display import display, HTML 
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns', 500)
DATA_PATH = '/kaggle/input/datatask6'
OUTPUT_PATH = '/kaggle/working/'
MODEL_PATH = '/kaggle/input/models/scibert-nli/'
# Specify the Kaggle Username and Key to use the Kaggle Api

# os.environ['KAGGLE_USERNAME'] = '*************'
# os.environ['KAGGLE_KEY'] = '****************'
# from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# api.authenticate()

# api.dataset_download_files(dataset="allen-institute-for-ai/CORD-19-research-challenge", path=DATA_PATH, unzip=True)
# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# The raw code for this IPython notebook is by default hidden for easier reading.
# To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# Helper class to extract data and convert it to df
class PaperReader:
    
    # initializing
    def __init__(self, root_path):
        self.root_path = root_path
        self.filenames = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
        print(str(len(self.filenames))+' files were found')
    
    # load files
    def load_files(self):
        raw_files = []

        for filename in tqdm(self.filenames):
            file = json.load(open(filename, 'rb'))
            raw_files.append(file)

        return raw_files
    
    # extract values from keys
    def extract_value(self, key, dictionary):
        
        for k, v in dictionary.items():
            
            if k == key:
                yield v
                break
            elif isinstance(v, dict):
                for result in self.extract_value(key, v):
                    yield result
            elif isinstance(v, list):
                
                for d in v:
                    if type(d) == dict:
                        for result in self.extract_value(key, d):
                            yield result  

    # Function to generate Clean DF
    def generate_clean_df(self):
        
        raw_files = self.load_files()
        cleaned_files = []

        for content in tqdm(raw_files):
            
            # extract paper_id ( sha)
            paper_id = list(self.extract_value('paper_id', content))
            paper_id = paper_id[0]

            if 'metadata' in content.keys():
                # extract title
                title = list(self.extract_value('metadata', content)) 
                title = title[0]['title']
#             else:
#                 title = np.nan

            # extract abstract
            if 'abstract' in content.keys():
                abstract = list(self.extract_value('abstract', content)) 
                abstract = ' \n\n\n '.join([element['text'] for element in abstract[0] if len(element['text']) > 200 ])
#            else:
#                abstract = np.nan

            # extract body
            if 'body_text' in content.keys():
                body = list(self.extract_value('body_text', content)) 
                body = ' \n\n\n '.join([element['section'] + ' \n ' + element['text'] for element in body[0] if len(element['text']) > 200])
#             else:
#                 body = np.nan

            # extract bib_entries
            if 'bib_entries' in content.keys():
                bib_entries = list(self.extract_value('bib_entries', content)) 
                bib = []
                for edx, el in enumerate(bib_entries[0]):
                    #index = bib_entries[0][el]['ref_id']
                    bib_title = bib_entries[0][el]['title']
                    bib.append(bib_title)

                bib = ' \n\n'.join(bib)
#             else:
#                 bib = np.nan

            # extract red entries
            if 'ref_entries' in content.keys():
                ref_entry = list(self.extract_value("ref_entries",  content)) 
                ref_ent = []
                for rdx, re in enumerate(ref_entry[0]): 
                    #print(re)
                    #print(ref_entry[0][re])
                    ref_ent.append(ref_entry[0][re]['type'] + ' \n ' + ref_entry[0][re]['text'] )
                ref_entry = ' \n\n\n '.join(ref_ent)
#             else:
#                 ref_entry = np.nan

            # extract back matter
            if 'back_matter' in content.keys():
                back_matter = list(self.extract_value("back_matter", content)) 
                back_matter = ' \n\n\n '.join([element['section'] + ' \n ' + element['text'] for element in back_matter[0]])
#             else:
#                 back_matter = np.nan

            # create a dataframe of extracted json file
            file_dataframe = pd.DataFrame.from_dict({   'paper_id': [paper_id], 
                                            'title': [title],
                                            'abstract': [abstract],
                                            'text' : [body],
                                            'bib_entries': [bib],
                                            'ref_entries': [ref_entry],
                                            'back_matter': [back_matter]
                                        })

            # append file_dataframe to file_paths_df list
            cleaned_files.append(file_dataframe)
        
        # concat all the dataframes present in file_paths_df list         
        data = pd.concat(cleaned_files)
        
        # reset index of data dataframe
        data.reset_index(inplace = True, drop = True)
        
        return data
#PR = PaperReader(INPUT_PATH)
# generate Dataframe
#papers_df = PR.generate_clean_df()
#papers_df.head()
# save extracted data
# papers_df.to_excel(DATA_PATH + 'extracted_data.xlsx', index = False)
# helper function to locate the json files
def find_file(filename):
    result = []
    for root, dir, files in os.walk(DATA_PATH):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result[0]
task_questions = ["Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.",
            "Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.",
            "Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.",
            "Methods to control the spread in communities, barriers to compliance and how these vary among different populations",
            "Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.",
            "Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.",
            "Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).",
            "Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."]
questions = ["How to scale up Non Pharmaceutical Interventions for COVID-19 ?",
            "What are the methods to compare Non Pharmaceutical Interventions for COVID-19 ?",
            "What are the effect of travel bans for COVID-19 ?",
            "What are the methods to control the community spread for COVID-19 ?",
            "What are the Non Pharmaceutical Interventions models to predict costs and benefits for COVID-19 ?",
            "What are the required policy changes to enable the compliance of individuals with limited resources for COVID-19 ?",
            "Why people fail to comply with public health advice for COVID-19 ?",
            "What is the economic impact of pandemic ?"]
question = questions[2]
print(f"This is the question you are looking at: ")
print(f"  ==>   {question}")
papers_df = pd.read_excel(DATA_PATH + '/extracted_data.xlsx')
papers_df.columns
papers_df.head()
papers_df.isnull().sum()
papers_df['Embedding_Col'] = papers_df['abstract'].copy(deep= True)
papers_df.head()
# creating mask where "abstract" is missing
abs_mask = papers_df['Embedding_Col'].isna()
#abs_mask
# copy title to Embdding_Col where abstract is missing
papers_df['Embedding_Col'][abs_mask] = papers_df['title'][abs_mask].copy(deep = True)
# create mask for not null values in Embdding_Col column 
abs_sec_mask = papers_df['Embedding_Col'].notna()
abs_sec_mask.value_counts()
working_Data = papers_df[abs_sec_mask].copy(deep = True)
working_Data.isnull().sum()
# Model Training part

# if not os.path.isdir(MODEL_PATH+'SciBERT-NLI Pretrained/):
#     os.mkdir(MODEL_PATH+'SciBERT-NLI Pretrained/)
# else:
#     pass


# tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")

# model = AutoModel.from_pretrained("gsarti/scibert-nli")

# model.save_pretrained(MODEL_PATH+'SciBERT-NLI Pretrained/)

# tokenizer.save_pretrained(MODEL_PATH+'SciBERT-NLI Pretrained/)
# word_embedding_model = models.BERT(MODEL_PATH+'SciBERT-NLI Pretrained/)

# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),

#                        pooling_mode_mean_tokens=True,

#                        pooling_mode_cls_token=False,

#                        pooling_mode_max_tokens=False)

# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# shutil.rmtree(MODEL_PATH+'SciBERT-NLI Pretrained/)

# model.save(MODEL_PATH+'SciBERT-NLI Pretrained/)
# load the scibert-nli model
sciBert_model = SentenceTransformer(MODEL_PATH)
# remove duplicate values 
working_Data.drop_duplicates(['Embedding_Col'], keep='first', inplace=True)
metadata_df = pd.read_csv(DATA_PATH + '/metadata.csv')
metadata_df.rename(columns={'sha': 'paper_id'}, inplace=True)
merged_data = pd.merge(working_Data, 
                       metadata_df[['paper_id','cord_uid', 'source_x', 'publish_time', 'url']], 
                      on = 'paper_id', 
                       how='left')
merged_data.head()
str_mask = merged_data['Embedding_Col'].str.contains('2019-nCoV')
merged_data['Embedding_Col'][str_mask]  = merged_data['Embedding_Col'][str_mask].str.replace('2019-nCoV', 'covid-19')
merged_data.drop_duplicates(['Embedding_Col'], keep='first', inplace=True)
# merged_data.to_excel(DATA_PATH + '/merged_data.xlsx', index = False)
# merged_data = pd.read_excel(DATA_PATH + '/merged_data.xlsx')
merged_data['Embedding_Col'] = merged_data['Embedding_Col'].str.strip()
corpus = [re.sub(' \n\n\n ','',x) for x in merged_data['Embedding_Col'].to_list()]
# create embedding for the text in Embdding_Col column 
# fulldfembedding = sciBert_model.encode(corpus, show_progress_bar=True)
# with open(DATA_PATH + '/Abstract_Title_embd.pkl', 'wb') as emb:
#     pickle.dump(fulldfembedding, emb)
# load pickled file of embeddings
with open(DATA_PATH + '/Abstract_Title_embd.pkl', 'rb') as emb:
    fulldfembedding = pickle.load(emb)
def ask_question(query, model, corpus, corpus_embed, top_k=20):
    """
    Adapted from https://www.kaggle.com/dattaraj/risks-of-covid-19-ai-driven-q-a
    """
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            results.append([count + 1, idx, corpus[idx], round(1 - distance, 4)])
    return results


def show_answers(results):
    table = prettytable.PrettyTable(
        ['Rank', 'S.No.', 'Embedding Column', 'Score']
    )
    for res in results:
        rank = res[0]
        sno = res[1]
        text = res[2]
        text = textwrap.fill(text, width=75)
        text = text + '\n\n'
        score = res[3]
        table.add_row([
            rank,
            sno,
            text,
            score
        ])
    print('\n')
    print(str(table))
    print('\n')
# get the results from the question 3 -> "What are the effect of travel bans for COVID-19 ?"
question = questions[2]
results = ask_question(question, sciBert_model, corpus, fulldfembedding)
#results
#question
#@supress_stdout
def hash_results(results, merged_data):
    hashed_paras = []
    context = []
    para_from_paperid = {}
    para_from_papertitle = {}
    para_from_paperurl = {}

    for result in results:
        index = result[1]
        paper_title = merged_data['title'].iloc[index]
        #print(f"result: {paper_title}")
        paper_sha = merged_data['paper_id'].iloc[index]
        paper_cord_uid = merged_data['cord_uid'].iloc[index]
        paper_url = merged_data['url'].iloc[index]
        #print(f"type of paper_sha is: {type(paper_sha)} paper_sha is: {paper_sha}")
        body = merged_data['text'].iloc[index]
        if isinstance(body, str):
            for para in body.split(' \n\n\n '):
                mystring = para.replace('\n', '')

                context.append(mystring)
                #mystring = 'b' + "'" +mystring
                # Assumes the default UTF-8
                hash_object = hashlib.sha256(mystring.encode("utf-8"))
                hex_dig = hash_object.hexdigest()
                hashed_paras.append(hex_dig)
                para_from_paperid[hex_dig] = merged_data['paper_id'].iloc[1306]
                para_from_papertitle[hex_dig] = paper_title
                para_from_paperurl[hex_dig] = paper_url
    
    return hashed_paras,context, para_from_paperid,para_from_papertitle, para_from_paperurl    
hashed_paras,context, para_from_paperid,para_from_papertitle, para_from_paperurl = hash_results(results, merged_data)
show_answers(results)
context_list = [x for x in context if len(x) > 150]
def find_top_para_bert(context, question):
    # create embeddings for paragraphs 
    para_embeddings = sciBert_model.encode(context, show_progress_bar=True)
    # create embeddings for question 
    query_embed = sciBert_model.encode(question, show_progress_bar=True)
    for query, query_emb in zip(question, query_embed):
        distances = cdist([query_emb], para_embeddings, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:10]):
            results.append([count+1, idx, context[idx], round(1 - distance, 4) ])
    return results
    
top_para_bert = find_top_para_bert(context_list, question)
show_answers(top_para_bert)
def find_top_para_CVect(context, question):
    para_list = context
    #tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    count_vec_total_text = count_vectorizer.fit_transform(para_list)
    count_vec_question = count_vectorizer.transform([question])
    distance_array = cosine_similarity(count_vec_question, count_vec_total_text)
    distance_array = distance_array[0]
    
    top_n = 10
    top_passages = []
    for index in distance_array.argsort()[::-1][:top_n]:
        #print(index, ' ', para_list[index])
        #print('-' * 120)
        top_passages.append((index, para_list[index]))
    return top_passages
top_para_CVec = find_top_para_CVect(context_list, question)
for index, para in top_para_CVec:
    print(index, para)
    print('-' * 120)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
squad_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
def squad_bert_results(top_para_bert, top_para_CVec):
    cv_indexes = [i[0] for i in top_para_CVec ]
    scibert_indexes = [x[1] for x in top_para_bert]
    indices  = set(cv_indexes + scibert_indexes )
    countVec_para_list = [x[1].replace(' \n ', ' ') for x in top_para_CVec if x[0] in indices ]
    sciBert_para_list = [x[2].replace(' \n ', ' ') for x in top_para_bert if x[1] in indices]
    all_paras = countVec_para_list + sciBert_para_list
    predicted_answer_list = []
    predicted_passage_list = []
    results = {}
    for padx, passage in enumerate(all_paras):
        inputs = tokenizer.encode_plus(question, passage, add_special_tokens=True, return_tensors="pt",max_length=512)
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer_start_scores, answer_end_scores = squad_model(**inputs)

        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

        Flag = False
        hash_object = hashlib.sha256(passage.encode("utf-8"))
        hex_dig = hash_object.hexdigest()
        results[padx] = [question, answer,passage, para_from_paperid[hex_dig], para_from_papertitle[hex_dig], para_from_paperurl[hex_dig] ]
        if len(answer) >10 and '[CLS]' not in answer and '[SEP]' not in answer :
            predicted_answer_list.append([padx, answer])
            Flag = True
            hash_object = hashlib.sha256(passage.encode("utf-8"))
            hex_dig = hash_object.hexdigest()
            results[padx] = [Flag, question, answer,passage, para_from_paperid[hex_dig], para_from_papertitle[hex_dig], para_from_paperurl[hex_dig] ]
            predicted_passage_list.append(passage)
            """
            print(f"==============================================================")
            print(f"QUESTION: {question}")
            print(f"--------------------------------------------------------------")
            print(f"ANSWER: {answer}\n")
            print(f"this is from paragraph: ")
            print(passage)
            print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(f"paper cord_uid: {para_from_paperid[hex_dig]}")
            print(f"paper title: {para_from_papertitle[hex_dig]}")
            print(f"paper url: {para_from_paperurl[hex_dig]}")
            print(f"==============================================================")  
            """
    return results, predicted_answer_list
squad_results, predicted_answer_list = squad_bert_results(top_para_bert, top_para_CVec)
predicted_answer_list
def bleu_post_process(predicted_answer_list):
    predicted_answer = predicted_answer_list
    
    bleu_matrix = np.zeros((len(predicted_answer),len(predicted_answer)))
    for idx, i in enumerate(predicted_answer):
        for jdx,j in enumerate(predicted_answer):
            if idx != jdx:
                reference = [word_tokenize(predicted_answer[idx][1])]
                candidate = word_tokenize(predicted_answer[jdx][1])
                bleu_matrix[idx,jdx] = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))   
            elif idx == jdx:
                bleu_matrix[idx,jdx] = 1
    twoGram=pd.DataFrame(bleu_matrix)
    elements =[]

    elements2remove = []

    for row in range(0,len(twoGram)):
        elements.append([row,twoGram.iloc[row][(twoGram.iloc[row] >=0.001) & (twoGram.iloc[row] != 1)].index])

    for pdx, p in enumerate(elements):
        if len(p[1]) >0 :
            for item in p[1]:
                if item > pdx:
                    elements2remove.append(item)
                    
    remove_indices = list(set(elements2remove))
    
    Processed_answers = [[j,i] for j, i in enumerate(predicted_answer) if j not in remove_indices]
    
    return twoGram, Processed_answers

twoGram, Processed_answers =bleu_post_process(predicted_answer_list)

cm = sns.light_palette("blue", as_cmap=True)
#twoGram=pd.DataFrame(bleu_matrix)
print('\n')
print("2-Gram BLEU Score Matrix")
twoGram_Style=twoGram.style.background_gradient(cmap=cm)
display(twoGram_Style)
for pdx, p in enumerate(Processed_answers):
    print(f"==============================================================")
    print(f"QUESTION: {squad_results[p[1][0]][1]}")
    print(f"--------------------------------------------------------------")
    print(f"ANSWER: {squad_results[p[1][0]][2]}\n")
    print(f"this is from paragraph: ")
    print(squad_results[p[1][0]][3])
    print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(f"paper cord_uid: {squad_results[p[1][0]][4]}")
    print(f"paper title: {squad_results[p[1][0]][5]}")
    print(f"paper url: {squad_results[p[1][0]][6]}")
    print(f"==============================================================")  
    

solutions = {}
for qdx, question in enumerate(questions):
    results = ask_question(question, sciBert_model, corpus, fulldfembedding)
    hashed_paras,context, para_from_paperid,para_from_papertitle, para_from_paperurl = hash_results(results, merged_data)
    top_para_bert = find_top_para_bert(context, question)
    top_para_CVec = find_top_para_CVect(context, question)
    squad_results, predicted_answer_list = squad_bert_results(top_para_bert, top_para_CVec)
    twoGram, Processed_answers =bleu_post_process(predicted_answer_list)
    df_list = []
    for pdx, p in enumerate(Processed_answers):
        df = pd.DataFrame({'QUESTION': [squad_results[p[1][0]][1]],    
                            'paper cord_uid': [squad_results[p[1][0]][4]],
                            'paper title': [squad_results[p[1][0]][5]],
                            'paper url': [squad_results[p[1][0]][6]],
                              'Short Answer ': [squad_results[p[1][0]][2]],
                                'Relevant Paragraph' :[squad_results[p[1][0]][3]]})
        df_list.append(df)
    frame = pd.concat(df_list)
    frame.reset_index(inplace=True)
    frame.drop(['index'], inplace = True, axis = 1)
    solutions[qdx+1]=frame 
final_list = [solutions[keys] for keys in solutions.keys()]
final_df = pd.concat(final_list)
final_df.reset_index(inplace=True)
final_df.drop(['index'], inplace = True, axis = 1)
final_df.head()
#final_df.head().style.set_properties(subset=['Relevant Paragraph'], **{'width': '300px'})
!jupyter nbextension enable --py widgetsnbextension
import ipywidgets as widgets
#from ipywidgets import interactive
from IPython.display import display
from IPython.html.widgets import interactive 
items = ['All']+sorted(final_df['QUESTION'].unique().tolist())
 
def view(Question=''):
    if Question=='All': display(final_df)
    display(final_df[final_df['QUESTION']==Question])
    
def question_ask(question, sciBert_model, merged_data, fulldfembedding):
    corpus = [re.sub(' \n\n\n ','',x) for x in merged_data['Embedding_Col'].to_list()]
    results = ask_question(question, sciBert_model, corpus, fulldfembedding)
    hashed_paras,context, para_from_paperid,para_from_papertitle, para_from_paperurl = hash_results(results, merged_data)
    top_para_bert = find_top_para_bert(context, question)
    top_para_CVec = find_top_para_CVect(context, question)
    squad_results, predicted_answer_list = squad_bert_results(top_para_bert, top_para_CVec)
    twoGram, Processed_answers =bleu_post_process(predicted_answer_list)
    df_list = []
    for pdx, p in enumerate(Processed_answers):
        df = pd.DataFrame({'QUESTION': [squad_results[p[1][0]][1]],    
                            'paper cord_uid': [squad_results[p[1][0]][4]],
                            'paper title': [squad_results[p[1][0]][5]],
                            'paper url': [squad_results[p[1][0]][6]],
                              'Short Answer ': [squad_results[p[1][0]][2]],
                                'Relevant Paragraph' :[squad_results[p[1][0]][3]]})
        df_list.append(df)
    frame = pd.concat(df_list)
    frame.reset_index(inplace=True)
    frame.drop(['index'], inplace = True, axis = 1)
    return frame
 
w = widgets.Dropdown(options=items)
q = widgets.Text(value='question')
#interactive(view, Question=w)
#create tabs
tab_nest = widgets.Tab()
# tab_nest.children = [tab_visualise]
tab_nest.set_title(0, 'Results')
#tab_nest.set_title(1, 'Ask Question')


#interact function in isolation
f = interactive(view, Question=w);
#f1 = interactive(question_ask, sciBert_model=sciBert_model, merged_data=merged_data, fulldfembedding=fulldfembedding, question=q);
tab_nest.children = [VBox(children = f.children) ]
display(tab_nest)
#question = input('Enter Question: ')
#question_ask(question, sciBert_model, merged_data, fulldfembedding)
#@title Load the Universal Sentence Encoder's TF Hub module

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
    return model(input)
def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=0.8)
    g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
    message_embeddings_ = embed(messages_)
    plot_similarity([textwrap.fill(x)[:100] for x in messages_], message_embeddings_, 90)
# showing only first 100 leters in the labels
l = [x[1][1] for x in Processed_answers]

#np.inner(embed(l), embed(l))
run_and_plot(l)
final_df.head()
## Excel Results reading to Pandas
#****### The following code lets the reader view the most relevant answers in the already run results. The excel file is organized in tabs 6.1, 6.2, 6.3... 6.8, corresponding to each sub-task. The result for each sub-task can be passed by the corresponding sheet name to the code below.
### Note the display is limited up to top 5 results, a complete set of paper results is captured in the attached excel
#References: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
# import pandas as pd

# df = pd.read_excel (r'Task6 Results and Analysis_final.xlsx', sheet_name='6.3')
# pd.set_option('display.max_colwidth', None)
# df.head()
predicted_list = [y[1] for y in predicted_answer_list]
#predicted_list
def gen_wordcloud(passage_list, ericsson_style=False):
    '''
    i/p: a list of passages: list[str]
    o/p: None, plots the picture
    '''
    all_passages = ' '.join(predicted_list)
    
    stopwords = set(STOPWORDS)
    
    if ericsson_style:
        mask = np.array(Image.open('/kaggle/input/ericssonlogo/ericsson-logo-clipart.jpg'))

        wordcloud = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=1000, mask=mask).generate(all_passages)

        # create coloring from image
        image_colors = ImageColorGenerator(mask)
        plt.figure(figsize=[18,10])
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.axis("off")
        plt.show()

    else:
        wordcloud = WordCloud().generate(all_passages)

        wordcloud = WordCloud(max_font_size=40).generate(all_passages)
        plt.figure(figsize = (15, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
# run this function for a paragraph list of each question
gen_wordcloud(predicted_list)
#os.listdir('/kaggle/input/ericssonlogo/')
# gen_wordcloud(predicted_list, ericsson_style=True)