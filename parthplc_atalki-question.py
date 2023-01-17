# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from nltk import tokenize
!pip install git+https://github.com/boudinfl/pke.git

!python -m nltk.downloader stopwords

!python -m nltk.downloader universal_tagset

!python -m spacy download en # download the english model
import time

import torch

from transformers import T5ForConditionalGeneration,T5Tokenizer





def set_seed(seed):

  torch.manual_seed(seed)

  if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



set_seed(42)



model = T5ForConditionalGeneration.from_pretrained('../input/questiongenerationmultiple/result')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
def greedy_decoding (inp_ids,attn_mask,model,tokenizer):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=80)

    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return Question.strip().capitalize()
def topkp_decoding (inp_ids,attn_mask,model,tokenizer):

  topkp_output = model.generate(input_ids=inp_ids,

                                 attention_mask=attn_mask,

                                 max_length=80,

                               do_sample=True,

                               top_k=100,

                               top_p=0.95,

                               num_return_sequences=3,

                                no_repeat_ngram_size=2,

                                early_stopping=True

                               )

  Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]

  return list(Question.strip().capitalize() for Question in Questions)

def beam_search_decoding (inp_ids,attn_mask,model,tokenizer):

      beam_output = model.generate(input_ids=inp_ids,

                                     attention_mask=attn_mask,

                                     max_length=80,

                                   num_beams=100,

                                   num_return_sequences=3,

                                   no_repeat_ngram_size=2,

                                   early_stopping=True

                                   )

      Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in

                   beam_output]

      return  Questions
def t5_answer(text,answer,model,tokenizer):

    con = "context:%s answer:%s</s>" %(text,answer)

    encoding = tokenizer.encode_plus(con, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = beam_search_decoding(input_ids, attention_masks,model,tokenizer)

    return str(output)

from pke.unsupervised import TopicRank

def extract_keyword(text):

    extractor = TopicRank()



    # load the content of the document, here in CoreNLP XML format

    # the input language is set to English (used for the stoplist)

    # normalization is set to stemming (computed with Porter's stemming algorithm)

    extractor.load_document(input=text,

                            language="en")



    # select the keyphrase candidates, for TopicRank the longest sequences of 

    # nouns and adjectives

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})



    # weight the candidates using a random walk. The threshold parameter sets the

    # minimum similarity for clustering, and the method parameter defines the 

    # linkage method

    extractor.candidate_weighting(threshold=0.95,

                                  method='average')

    keyphrases = []



    # print the n-highest (10) scored candidates

    for (keyphrase, score) in extractor.get_n_best(n=2):

        keyphrases.append(keyphrase)

    return keyphrases
import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase





test = "Tim's linkedin profile is https://www.linkedin.com/in/"

print(decontracted(test))
text = "Tim's linkedin profile is https://www.linkedin.com/in/"

print(text.lower())

extract_keyword(decontracted(text))
def span(text,keyword):

    text = text.lower()

    text = decontracted(text)

    l = text.split()

    keyword = list(keyword.split())

    keyword = keyword[0]

    k = l.index(keyword)

    span = keyword

    

    if (len(l)>4):

        if k == 0 or k==1:

            span = keyword+" "+l[k+1]+" "+l[k+2]+" "+l[k+3]

        elif k == (len(l)-1) or k==(len(l)-2):

            span =l[k-3]+" "+l[k-2]+" "+l[k-1]+" "+keyword

        else:

            span = l[k-2]+" "+l[k-1]+" "+keyword+" "+l[k+1]+" "+l[k+2]

    return span

    
span(text,extract_keyword(text)[1])
def para_to_sent(para):

    keyphrases = tokenize.sent_tokenize(para)

    return keyphrases
import ast
def final_question_keywords(text):

    token_sent = para_to_sent(text)

    keywords = []

    i = 0

    for x in token_sent:

        k = span(x,extract_keyword(x)[0])

        keywords.append(k)

    return keywords
text = '''Tim is a skilled Software Testing professional. Tim's experience includes test team management, client and vendor 

management, automation test engineering, test infrastructure creation and maintenance. 

Tim is an expert in building testing teams from scratch. Tim puts together processes, strategies, tools and frameworks, 

skilled resources for efficient and seamless delivery. 

Expertise in designing and developing test strategy, test plan, test cases and generating test reports and defect reports.

Extensive experience in coordinating testing effort, responsible for test deliverables,

status reporting to management, issue escalations etc. Have Extensive experience in automating and 

testing enterprise web applications using Selenium 1.0 and Selenium 2.0 (WebDriver) tool using both Microsoft and JAVA technology stack.

Tim is very good at JAVA, C# and python language programming. Tim has strong experience in C# programming using MS Visual Studio and

other MS technology stack in particular MSTest unit testing framework. Tim has very good experience in setting up Continuous Integration

environment using Team Foundation Server, creating Build definitions and management etc. Domain experience includes Superannuation, 

Telco, Insurance, Banking, Finance and healthcare domains. Tim's linkedin profile is https://www.linkedin.com/in/timothy-r-alex-ai/.

Tim's contact phone number is 0470139767. Tim's email address is timothyrajan@gmail.com. 

Tim has good exposure to gitlab and github hosting platforms.'''
keywords = final_question_keywords(text)
print(keywords)
import ast 

def final_questions(text,keywords,model,tokenizer):

    token_sent = para_to_sent(text)

    i = 0

    for x in keywords:

        quest = t5_answer(text,keywords[i],model,tokenizer)

        res = ast.literal_eval(quest) 

        longest_string = max(res, key=len)

        quest = longest_string.lstrip('question:')

        i = i+1

        print(quest)        
final_questions(text,keywords,model,tokenizer)
def extract_multiple_keyword(text):

    extractor = TopicRank()



    # load the content of the document, here in CoreNLP XML format

    # the input language is set to English (used for the stoplist)

    # normalization is set to stemming (computed with Porter's stemming algorithm)

    extractor.load_document(input=text,

                            language="en")



    # select the keyphrase candidates, for TopicRank the longest sequences of 

    # nouns and adjectives

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})



    # weight the candidates using a random walk. The threshold parameter sets the

    # minimum similarity for clustering, and the method parameter defines the 

    # linkage method

    extractor.candidate_weighting(threshold=0.95,

                                  method='average')

    keyphrases = []



    # print the n-highest (10) scored candidates

    for (keyphrase, score) in extractor.get_n_best(n=50):

        keyphrases.append(keyphrase)

    return keyphrases
extract_multiple_keyword(text)
def generaten_multiple_questions(text):

    keywords = extract_multiple_keyword(text)

    i=0

    for x in keywords:

        quest = t5_answer(text,x,model,tokenizer)

        res = ast.literal_eval(quest) 

        longest_string = max(res, key=len)

        quest = longest_string.lstrip('question:')

        i = i+1

        print(quest)        
generaten_multiple_questions(text)