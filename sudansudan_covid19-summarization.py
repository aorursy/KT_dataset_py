!pip uninstall covidSumm --y
!pip install easydict
!pip install -i https://test.pypi.org/simple/ covidSumm==0.1.3
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
args = set_config()
args['model_path'] = '/kaggle/input/carieabssummmodel/'
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
query = 'What is the range of incubation periods for COVID-19 in humans'
args = set_config()
args['model_path'] = '/kaggle/input/carieabssummmodel/'
summary_model_1 = abstractive_summary_model(config = args)
model_path = "/kaggle/input/bartsumm/bart.large.cnn"
summary_model_2 = Bart_model(model_path)
answer_summary_1 = get_answer_summary(query, summary_model_1)
display_summary(answer_summary_1, 'UniLM')
answer_summary_2 = get_bart_answer_summary(query, summary_model_2)
display_summary(answer_summary_2, 'BART')
article_summary_1 = get_article_summary(query, summary_model_1)
display_article_summary(article_summary_1, query)
article_summary_2 = get_bart_article_summary(query, summary_model_2)
display_article_summary(article_summary_2, query)

from covidSumm.abstractive_api import *
answer_summary_1 = abstractive_api_uni_para(query)
answer_summary_1
from covidSumm.abstractive_utils import *
test_answer = abstractive_api(query, 'unilm_para')
test_answer
test_answer = abstractive_api(query, 'bart_article')
test_answer