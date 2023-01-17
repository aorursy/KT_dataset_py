!curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz

!mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
!update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
!update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
!pip install pyserini==0.8.1.0
from pyserini.search import pysearch
COVID_INDEX = '../input/luceneindexcovidparagraph20200410/lucene-index-covid-paragraph-2020-04-10'
searcher = pysearch.SimpleSearcher(COVID_INDEX)
def get_articles(query):
    hits = searcher.search(query)
    #print(len(hits))
    # Prints the first 10 hits
    return hits
query = 'range of incubation periods for COVID-19'
hits = get_articles(query)
for i in range(0, 10):
    #print some relevant fields
    print(f'{i+1} {hits[i].docid} {hits[i].score} {hits[i].lucene_document.get("title")} {hits[i].lucene_document.get("doi")}')
hits[0].contents.split('\n')
import json
def get_para_results(query):
    hits = searcher.search(query,10) 
    temp = {} # to store the doi of the articles being returned so we know if the article is repeated
    i = 0
    output = []
    while i<len(hits) and i<10:
        outJson = {}
        outJson['rank'] = i+1
        # check if the current article has a paragraph returned or not ('has_full_text' in the dataset)
        if '.' in hits[i].docid:
            doc_id = hits[i].docid.split('.')[0]
            para_id = hits[i].docid.split('.')[1]
            doi = hits[i].lucene_document.get('doi')
            paragraph = {}
            paragraph['score'] = hits[i].score
            paragraph['text'] = hits[i].contents.split('\n')[-1] # get the last element, since the contents are sorted as [title, abstract, paragraph]
            paragraph['id'] = para_id
            # check if the doi (same article) has not appeared before in the list
            if doi not in temp:
                outJson['abstract'] = hits[i].lucene_document.get('abstract') # include abstract if new article
                article_data = json.loads(searcher.doc(doc_id).lucene_document().get('raw')) # get all the relevant data from the dataset 
                if 'body_text' in article_data:
                    outJson['body_text'] = article_data['body_text'] # include 'body_text' in case needed later
                temp[doi] = i
            outJson['paragraphs'] = []
            outJson['paragraphs'].append(paragraph)
        else:
            # no paragraph present, which means article does not have full text available
            outJson['abstract'] = hits[i].lucene_document.get('abstract')
            outJson['score'] = hits[i].score
        outJson['title'] = hits[i].lucene_document.get('title')
        outJson['sha'] = hits[i].lucene_document.get('sha')
        outJson['doi'] = hits[i].lucene_document.get('doi')
        output.append(outJson)
        i+=1
    return output
query = 'range of incubation periods for COVID-19'
i = 1
for item in get_para_results(query):
    if i>10:
        break
    print(item)
    i+=1
def information_retrieval(file_name, topk = 10):

    with open(file_name) as f:
        json_file = json.load(f)
    subtasks = json_file["sub_task"]
    
    all_results = []
    data_for_qa = []
    for item in subtasks:
        questions = item["questions"]
        for query in questions:
            result_item = {"question" : query}
            retri_result = get_para_results(query)
            result_item["data"] = retri_result

            qa_item = {"question": query}
            context = []
            titles = []
            doi = []
            count = 1
            for item in retri_result:
                if count>topk:
                    break
                if 'abstract' in item and len(item['abstract']) > 0:
                    context.append(item['abstract'])
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1
                if 'paragraphs' in item:
                    context.append(item['paragraphs'][0]['text'])   
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1

            qa_item["data"] = {"answer": "", "context": context, "doi": doi, "titles": titles}

            all_results.append(result_item)
            data_for_qa.append(qa_item)

    return data_for_qa

def parse_ir_results(query, retri_result, topk = 10):
    all_results = []
    data_for_qa = []
    qa_item = {"question": query}
    result_item = {"question" : query}
    result_item["data"] = retri_result
    context = []
    titles = []
    doi = []
    count = 1
    for item in retri_result:
        if count>topk:
            break
        if 'abstract' in item and len(item['abstract']) > 0:
            context.append(item['abstract'])
            doi.append(item["doi"])
            titles.append(item["title"])
            count+=1
        if 'paragraphs' in item:
            context.append(item['paragraphs'][0]['text'])   
            doi.append(item["doi"])
            titles.append(item["title"])
            count+=1
    qa_item["data"] = {"answer": "", "context": context, "doi": doi, "titles": titles}

    all_results.append(result_item)
    data_for_qa.append(qa_item)    

    return all_results, data_for_qa

    
def information_retrieval_query(query):

    retri_result = get_para_results(query)
    all_results, data_for_qa = parse_ir_results(query, retri_result ,topk = 20)
    
    return all_results, data_for_qa
### 3.1 install the prerequisite
import os
import sys
import json

!pip uninstall tensorflow -y
!pip uninstall tensorflow-gpu -y
!pip install tensorflow==1.13.1
!pip install caireCovid==0.1.8
import tensorflow as tf
import caireCovid
from caireCovid import QaModule
from caireCovid.qa_utils import stop_words
import math
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
### 3.2 Check all version
print(tf.__version__)
# QA System
class QA_System():
    def _init_(self):
        # Load the QA models. Please refer to [Github](https://github.com/yana-xuyan/caireCovid) for details.
        self.model = QaModule(['mrqa', 'biobert'], ["/kaggle/input/pretrained-qa-models/mrqa/1564469515", "/kaggle/input/pretrained-qa-models/biobert/1585470591"], \
                              "/kaggle/input/xlnetlargecased/xlnet_cased_L-24_H-1024_A-16/spiece.model", "/kaggle/input/pretrained-qa-models/bert_config.json", \
                              "/kaggle/input/bert-base-cased/vocab.txt")
    def getAnswer(self, query):
        _, data_for_qa = information_retrieval_query(query)
        answers =  self.model.getAnswers(data_for_qa)
        return answers
    def getAnswers(self, filename):
        _, data_for_qa = information_retrieval(query)
        answers = self.model.getAnswers(data_for_qa)
        return answers
    def makeFormatAnswers(self, answers):
        format_answers = []
        for i in range(len(answers[0]['data']['answer'])):
                format_answer = {}
                format_answer['question'] = answers[0]['question']
                format_answer['answer'] = answers[0]['data']['answer'][i]
                format_answer['context'] = answers[0]['data']['context'][i]
                format_answer['doi'] = answers[0]['data']['doi'][i]
                format_answer['title'] = answers[0]['data']['title'][i]
                format_answer["confidence"] = answers[0]['data']['confidence'][i]
                format_answer["raw"] = answers[0]['data']['raw'][i]
                format_answers.append(format_answer)
        return format_answers

def get_QA_answer_api(query):
    url = "http://eez114.ece.ust.hk:5000/query_qa"
    payload = "{\n\t\"text\": \""+query+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    response = response.json()
    return response
import argparse
import sys
import pandas as pd
import csv
import requests
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize # use sentence tokenize
from IPython.core.display import display, HTML
from nltk import word_tokenize, pos_tag, sent_tokenize
from caireCovid.qa_utils import stop_words
stop_words.append('including')

def rankAnswers(answers):
    for item in answers:
        query = item["question"]
        context = item['context']
        # make new query with only n. and adj.
        tokens = word_tokenize(query.lower())
        tokens = [word for word in tokens if word not in stop_words]
        tagged = pos_tag(tokens)
        query_token = [tag[0] for tag in tagged if 'NN' in tag[1] or 'JJ' in tag[1] or 'VB' in tag[1]]

        text = context.lower()
        count = 0
        text_words = word_tokenize(text)
        for word in text_words:
            if word in query_token:
                count += 1
            
        match_number = 0
        for word in query_token:
            if word == 'covid-19':
                continue
            if word in text_words:
                match_number += 1
        matching_score = count / (1 + math.exp(-len(text_words)+50))/ 5 + match_number*10
        item['matching_score'] = matching_score
        item['rerank_score'] = matching_score + 0.5 * item['confidence']
    
    # sort QA results
    answers.sort(key=lambda k: k["rerank_score"], reverse=True)
#     print([item['rerank_score'] for item in answers])
    return answers

def highlight_qaresult(qaresult):
    if qaresult == []:
        print('API broken')
        return 1
    ## tokenize query
    query = qaresult[0]['question']
    query_tokens = word_tokenize(query.lower())
    query_tokens = [word for word in query_tokens if word not in stop_words]
    tagged = pos_tag(query_tokens)
    query_tokens = [tag[0] for tag in tagged if 'NN' in tag[1] or 'JJ' in tag[1] or 'VB' in tag[1]]

    ## highlihgt answer
    for i in range(len(qaresult)):
        context_1 = "<style type='text/css'>mark { background-color:yellow; color:black; } </style>"
        golden = qaresult[i]['answer']
        context = qaresult[i]['context']
        context_sents = sent_tokenize(context)
        golden_sents = sent_tokenize(golden)
        for sent in context_sents:
            if sent not in golden:
                context_1 += sent
            else:
                context_1 += "<mark>"
                for word in sent.split():
                    word_tokens = word_tokenize(word)
                    if len(word_tokens) > 1:
                        for j in word_tokens:
                            if j.lower() in query_tokens:
                                context_1 = context_1 + "<b>" + j + "</b>"
                            else:
                                context_1 = context_1 + j
                        context_1 = context_1 + " "
                    else:
                        for j in word_tokens:
                            if j.lower() in query_tokens:
                                context_1 = context_1 + "<b>" + j + " </b>"
                            else:
                                context_1 = context_1 + j + " "
                context_1 += " </mark>"
        qaresult[i]['context'] = context_1
    return qaresult

def display_QA(result):
    result = highlight_qaresult(result)
    pdata = []
    count = 0
    for i in range(len(result)):
        count += 1
        line = []
        context_1 = "<div> "
        context = result[i]['context']
        context_1 = context_1 + context
        context_1 += " </div>"
        line.append(context_1)
        context_2 = '<a href= "https://doi.org/'
        context_2 += result[i]['doi']
        context_2 += '">'
        context_2 += result[i]['title']
        context_2 += '</a>'
        line.append(context_2)
        pdata.append(line)
        if count > 5:
            break
    df = pd.DataFrame(pdata, columns = ['QA results', 'title'])
    df = df.style.set_properties(**{'text-align': 'left','mark-color': 'red'})
    display(df)
!pip install easydict
!pip install covidSumm==0.1.4
!pip install fairseq
import covidSumm
import requests
import json
import os
import argparse
from covidSumm.abstractive_utils import get_ir_result, result_to_json, get_qa_result
from covidSumm.abstractive_model import abstractive_summary_model
from covidSumm.abstractive_config import set_config
from covidSumm.abstractive_bart_model import *
def get_summary_list(article_list, abstractive_model):
    summary_list = []
    for i in range(len(article_list)):
        article = article_list[i]
        summary_results = abstractive_model.generate_summary(article)
        result = ""
        for item in summary_results:
            result += item.replace('\n', ' ')
        summary_list.append(result)
    return summary_list

def get_answer_summary(query, abstractive_model):
    paragraphs_list = get_qa_result(query, topk = 3)
    answer_summary_list = abstractive_model.generate_summary(paragraphs_list)
    answer_summary = ""
    for item in answer_summary_list:
        answer_summary += item.replace('\n', ' ')
    answer_summary_json = {}
    answer_summary_json['summary'] = answer_summary
    answer_summary_json['question'] = query
    return answer_summary_json

def get_article_summary(query, abstractive_summary_model):
    article_list, meta_info_list = get_ir_result(query, topk = 10)  
    summary_list = get_summary_list(article_list, abstractive_summary_model)
    summary_list_json = []
    
    for i in range(len(summary_list)):
        json_summary = {}
        json_summary = result_to_json(meta_info_list[i], summary_list[i])
        summary_list_json.append(json_summary)

    return summary_list_json

def get_bart_answer_summary_from_qa(query, qa_result, bart_model):
    # we select top3
    paragraphs_list = []
    topk = 3

    for i in range(topk):
        if 'context' in qa_result[i].keys():
            one_line = {}
            one_line['src'] = qa_result[i]['context']
            one_line['tgt'] = ""
            paragraphs_list.append(one_line)
    
    answer_summary_list = bart_model.bart_generate_summary(paragraphs_list)
    answer_summary_result = ""
    for item in answer_summary_list:
        answer_summary_result += item.replace('\n', ' ')
    
    answer_summary_json = {}
    answer_summary_json['summary'] = answer_summary_result
    answer_summary_json['question'] = query
    return answer_summary_json
args = set_config()
args['model_path'] = '/kaggle/input/carieabssummmodel/'
summary_model_1 = abstractive_summary_model(config = args)
model_path = "/kaggle/input/bartsumm/bart.large.cnn"
summary_model_2 = Bart_model(model_path)
from IPython.core.display import display, HTML
import pandas as pd

def display_summary(ans_summary_json, model_type):
    question = ans_summary_json['question']
    text = ans_summary_json['summary']
    question_HTML = '<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'
    display(HTML(question_HTML))

    execSum_HTML = '<div style="font-family: Times New Roman; font-size: 18px; margin-bottom:1pt"><b>' + model_type + ' Abstractive Summary:</b>: '+text+'</div>'
    display(HTML(execSum_HTML))

def display_article_summary(result, query):
    question_HTML = '<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:28px"><b>Query</b>: '+query+'</div>'
    pdata = []
    abstract = ""
    summary = ""
    for i in range(len(result)):
        if 'abstract' in result[i].keys():
            line = []
            context_2 = '<a href= "https://doi.org/'
            context_2 += result[i]['doi']
            context_2 += ' target="_blank">'
            context_2 += result[i]['title']
            context_2 += '</a>'
            line.append(context_2)
            
            abstract = "<div> " 
            abstract += result[i]['abstract']
            abstract += " </div>"
            line.append(abstract)
            summary = "<div> " + result[i]['summary'] + " </div>"
            line.append(summary)


            pdata.append(line)
    display(HTML(question_HTML))
    df = pd.DataFrame(pdata, columns = ['Title','Abstract','Summary'])
    HTML(df.to_html(render_links=True, escape=False))
#     display(HTML(df.to_html(render_links=True, escape=False)))
    df = df.style.set_properties(**{'text-align': 'left'})
    display(df)

query = "How incubation period for COVID-19 varies across age?"
def run_example(query):
    # Given one query, we retrieve the relevant paragraphs and feed the (paragraph, query) pairs into the QA system 
    qa_result = get_QA_answer_api(query)
    # Answer Reranking
    qa_result = rankAnswers(qa_result)
    
    # Input "summary_model_2" is the BART summarization model.
    # Function "get_bart_answer_summary" is loaded from covidSumm.abstractive_bart_model
    # Given one query, we take top-3 reranked paragraphs from the QA module and summarize them into one paragraph
    answer_summary_2 = get_bart_answer_summary_from_qa(query, qa_result, summary_model_2)
    display_summary(answer_summary_2, 'BART')
    display_QA(qa_result)
run_example(query)
query = "What is the range of incubation periods for COVID-19 in humans?"
run_example(query)
query = "How incubation period for COVID-19 varies across age?"
run_example(query)
query = "How incubation period for COVID-19 varies across health status?"
run_example(query)
query = "How is prevalence of asymptomatic shedding and transmission?"
run_example(query)
query = "How is prevalence of asymptomatic shedding and transmission for children?"
run_example(query)
query = "How Seasonality affects COVID-19 transmission?"
run_example(query)
query = "What do we know about Physical science of the COVID-19, including charge distribution, adhesion to hydrophilic / phobic surfaces, environmental survival and viral shedding?"
run_example(query)
query = "What do we know about persistence and stability of COVID-19 on different substrates and sources, including nasal discharge, sputum, urine, fecal matter, blood?"
run_example(query)
query = "What do we know about persistence of COVID-19 on surfaces of different materials including copper, stainless steel, plastic?"
run_example(query)
query = "What do we know about Natural history of COVID-19 and shedding of it from an infected person?"
run_example(query)
query = "What do we know about implementation of diagnostics and products to improve clinical processes for COVID-19?"
run_example(query)
query = "what do we know about COVID-19 models, including animal models for infection, disease and transmission?"
run_example(query)
query = "What are tools and studies to monitor phenotypic change and potential adaptation of COVID-19?"
run_example(query)
query = "What do we know about immune response and immunity for COVID-19?"
run_example(query)
query = "How movement control strategies help to prevent COVID-19 secondary transmission in health care and community settings?"
run_example(query)
query = "How personal protective equipment (PPE) helps to reduce risk of COVID-19 transmission in health care and community settings?"
run_example(query)
query = "What is role of the environment in COVID-19 transmission?"
run_example(query)
query = "Is smoking a risk factor for COVID-19?"
run_example(query)
query = "Is pre-existing pulmonary disease a risk factor for COVID-19?"
run_example(query)
query = "Does co-morbidities make COVID-19 more transmissible or virulent?"
run_example(query)
query = "Are Neonates more potential for COVID-19?"
run_example(query)
query = "What are Socio-economic factors and economic impact of COVID-19?"
run_example(query)
query = "What are behavioral factors for COVID-19?"
run_example(query)
query = "What are the differences of Socio-economic and behavioral factors for COVID-19?"
run_example(query)
query = "What is the basic reproductive number of COVID-19?"
run_example(query)
query = "What is the incubation period of COVID-19?"
run_example(query)
query = "What is the serial interval  of COVID-19?"
run_example(query)
query = "What are the modes of transmission of COVID-19?"
run_example(query)
query = "What are the environmental factors of COVID-19?"
run_example(query)
query = "What do we know about susceptibility of populations for COVID-19?"
run_example(query)
query = "What are public health mitigation measures effective for controlling COVID-19?"
run_example(query)