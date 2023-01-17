import os.path
from pathlib import Path

import subprocess
version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
print(version)
%%capture
!pip install pyserini==0.8.1.0
!pip install transformers
!pip install nltk
import json
if(not('11.0.2' in str(version))):
    print('jdk upgrade required')
    !curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz

    !mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz
    !update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1
    !update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java
else:
    print('jdk level is Ok ')


import json
import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.system("ls /usr/lib/jvm")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"
!ls '/usr/lib/jvm'
from IPython.core.display import display, HTML
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
import numpy as np
import string
import torch
import numpy
from tqdm import tqdm
#%tensorflow_version 1.x
!pip install tensorflow==1.15.2
import tensorflow
print(tensorflow.__version__)
%%capture

!wget https://www.dropbox.com/s/j55t617yhvmegy8/lucene-index-covid-2020-04-10.tar.gz
!tar xvfz lucene-index-covid-2020-04-10.tar.gz
!wget https://www.dropbox.com/s/szakwmvco88hp3m/synonyms.csv?dl=0
!mv synonyms.csv?dl=0 synonyms.csv
!du -h lucene-index-covid-2020-04-10
from transformers import *
#let us try different BERT models, so far BERT model had better performance

#dtokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
#dmodel = AutoModelForQuestionAnswering.from_pretrained('allenai/scibert_scivocab_cased')
#dtokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.0_pubmed_pmc', do_lower_case=False)
#dmodel = AutoModelForQuestionAnswering.from_pretrained('monologg/biobert_v1.0_pubmed_pmc')
#dtokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#dmodel = AutoModelForQuestionAnswering.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

dtokenizer= BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
dmodel=BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
scoredic={}

			
# retrun synonyms for term 
def getsynonym(term):
    

  df = pd.read_csv('../input/synonymscsv/synonyms.csv')

  mylist=[]
  for col in df.columns:
    for index, rows in df.iterrows():
      if rows[col]==term:
        df2=rows[:]
        df2=df2.dropna()
        mylist = df2.values.tolist()
        break;
  return mylist
#return a string composed of the query and adding more synonynms.
def expandquery(query):
  
  searchquery=""
  querylist =query.split(" ")
  listofwords=[]
  for term in querylist:
    synonymlist = getsynonym(term)
    if not synonymlist == []:
      listofwords=listofwords+synonymlist
    else:
      searchquery=searchquery+" "+term
  myset = set(listofwords)
  mylist =list(myset)
  searchquery2=" ".join(str(item) for item in mylist)
  searchquery = searchquery+" "+searchquery2
  
  return searchquery
import unicodedata

def normalize_caseless(text):
    return unicodedata.normalize("NFKD", text.casefold())

def caseless_equal(left, right):
    return normalize_caseless(left) == normalize_caseless(right)
#return a string composed of the query after removing stop words
def removeCovidStopwords(query):
  stop_wordsCovid =set(['what','how',"which","where","virus","viral","viruses","infection","disease","patients","study",",","?"])
  stop_words=set(stopwords.words("english"))
  searchquery=""
  word_tokens = word_tokenize(query)
  print(type(stop_wordsCovid))
  filtered_sentence = [w for w in word_tokens if ((not w in stop_words)and(not w in stop_wordsCovid))]
  searchquery=" ".join(str(item) for item in filtered_sentence)
  return searchquery
# return keywords to be used with pyserini
def extractquerysearch(query):
  searchquery=""
  searchquery = normalize_caseless(query)
  searchquery = removeCovidStopwords(searchquery)
  searchquery=expandquery(searchquery)

  return searchquery
# Clean some extra text in paper abstract for a better presentation of results
def cleantext(paragraph):
  if paragraph.startswith('abstract')or paragraph.startswith('Abstract')or paragraph.startswith('ABSTRACT'):
    paragraph =paragraph[8:]
  
  return paragraph
query='What is known about covid-19 transmission, incubation, and environmental stability?'
searchquery=extractquerysearch(query)
print("keywords extracted are:",searchquery)
from pyserini.search import pysearch

searcher = pysearch.SimpleSearcher('lucene-index-covid-2020-04-10/')
hits = searcher.search(searchquery)

display(HTML('<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:12px"><b>Query</b>: '+query+'</div>'))


# Prints the first 10 hits
for i in range(0, 10):
  score=hits[i].score
  scoredic.update({hits[i].lucene_document.get("title") :score })
  display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + 
               F'{i+1} {hits[i].docid} ({hits[i].score:1.2f}) -- ' +
               F'{hits[i].lucene_document.get("authors")} et al. --' + 
               F'<a href="https://doi.org/{hits[i].lucene_document.get("doi")}">{hits[i].lucene_document.get("doi")}</a>.'+
               '<br>' +'<b> Paper Title: </b> '+
               F'{hits[i].lucene_document.get("title")}. '
               
               + '</div>'))
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
fig, ax = plt.subplots()

titles = list(scoredic.keys())
y_pos = np.arange(len(titles))
scores = list(scoredic.values())
error = np.random.rand(len(titles))

ax.barh(y_pos, scores, xerr=error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(titles)
ax.invert_yaxis()  
ax.set_xlabel('Scores')
ax.set_title(query)

plt.show()

def answer_question(question, answer_text,dtokenizer,dmodel):
    
    answer = "No highlight detected"
    if not question or not answer_text:
      print("Empty question or Empty abstract")
      return answer
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = dtokenizer.encode(question, answer_text,max_length=512)
    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(dtokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1
    
    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a
    
    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    
    start_scores, end_scores = dmodel(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = dtokenizer.convert_ids_to_tokens(input_ids)
    
    # Start with the first token.
    answer = tokens[answer_start]
    #if bert didn't get the tokens right, then the function retrun and highlight the keywords instead
    if answer==dtokenizer.cls_token:
      answer = "No highlight detected"
      return answer
    # if the first token is [sep] then skip and move forward  
    if answer==dtokenizer.sep_token:
      answer=""

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            if tokens[i-1]=='(' or tokens[i-1]  == '-':
              answer += tokens[i]
            elif tokens[i] == ')' or tokens[i]  == '-':
              answer += tokens[i]
            else:
              answer += ' ' + tokens[i]

    if answer==dtokenizer.sep_token:
      answer='No highlight detected'
    return answer
def highlightanswer(str,paragraph):
  str_start=""
  str_end=""
  flag='none'
  paragraph=normalize_caseless(paragraph)
  str=normalize_caseless(str)
  try:
    indx = paragraph.index(str)
  except:
    return str_start, str, str_end,flag

  if indx==-1:
    return str_start, str, str_end,flag
  str_start=paragraph[0:indx]
  str_end=paragraph[indx+len(str):]
  flag='done'
  return str_start, str, str_end, flag
def highlight_keywords(answer_text):

  abstractwords= word_tokenize(answer_text)
  searchquery_tokenized=word_tokenize(searchquery)
  abstractpara=""

  for wrd in abstractwords:
    if wrd in searchquery_tokenized:
      abstractpara = abstractpara+" "+"<font color='red'>"+wrd+"</font>"
    else:
      abstractpara = abstractpara+" "+wrd
  
  return abstractpara
def display_marker_result():
  display(HTML('<div style="font-family: Times New Roman; font-size: 20px; padding-bottom:12px; background:#e3e3e3"><b>Query</b>: '+query+'</div>'))
  # Prints the first 10 hits
  for i in range(0, 10):
    abstract=cleantext(hits[i].lucene_document.get("abstract"))
    answer =answer_question(query,abstract,dtokenizer,dmodel)
    strstart, highlighted, strend, myflag= highlightanswer(answer,abstract)
    if answer=='No highlight detected':
      display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + '<b>'+
               F'{i+1}'+') Score: </b>'+ F'{hits[i].score:1.2f}' +'-- <b>Authors: </b>'+
               F'{hits[i].lucene_document.get("authors")} et al. ' +'-- <b>DOI: </b>'+
               F'<a href="https://doi.org/{hits[i].lucene_document.get("doi")}">{hits[i].lucene_document.get("doi")}</a>.'+
               '<br> <b>Paper Title: </b>'+ F'{hits[i].lucene_document.get("title")}. ' +
               '<br> <b>Abstract: </b><br>'+
               F'{highlight_keywords(abstract)}'
               +'<font color="red">'+
               '<br><br><b>High Lights: </b> highlighting detected keywords </font><br>'+
                '</div> --------------------------------------------------------------------------------------------------------------------------------------' ))
    elif myflag=='none':
      display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + '<b>'+
               F'{i+1}'+') Score: </b>'+ F'{hits[i].score:1.2f}' +'-- <b>Authors: </b>'+
               F'{hits[i].lucene_document.get("authors")} et al. ' +'-- <b>DOI: </b>'+
               F'<a href="https://doi.org/{hits[i].lucene_document.get("doi")}">{hits[i].lucene_document.get("doi")}</a>.'+
               '<br> <b>Paper Title: </b>'+ F'{hits[i].lucene_document.get("title")}. ' +
               '<br> <b>Abstract: </b><br>'+
               F'{abstract}'
               +'<font color="red">'+
               '<br><br><b>High Lights: </b>'+F'{highlighted} '+'</font><br>'+
                '</div> --------------------------------------------------------------------------------------------------------------------------------------' ))
    else:
        display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:10px">' + '<b>'+
               F'{i+1}'+') Score: </b>'+ F'{hits[i].score:1.2f}' +'-- <b>Authors: </b>'+
               F'{hits[i].lucene_document.get("authors")} et al. ' +'-- <b>DOI: </b>'+
               F'<a href="https://doi.org/{hits[i].lucene_document.get("doi")}">{hits[i].lucene_document.get("doi")}</a>.'+
               '<br> <b>Paper Title: </b>'+ F'{hits[i].lucene_document.get("title")}. ' +
               '<br> <b>Abstract: </b><br>'+
               F'{strstart} ' +'<font color="red">'+F'{highlighted} '+'</font>'+F'{strend}'
              
               +'<font color="red">'+
               '<br><br><b>High Lights: </b>'+F'{highlighted} '+'</font><br>'+
               '</div> ---------------------------------------------------------------------------------------------------------------------------------------' ))

display_marker_result()
query ='what are the effectiveness of drugs being developed and tried to treat COVID-19 patients?'
searchquery=extractquerysearch(query)
print("keywords extracted is: ",searchquery)
hits = searcher.search(searchquery)
display_marker_result()
query="What do we know about COVID-19 risk factors?"
searchquery=extractquerysearch(query)
print("keywords extracted is: ",searchquery)
hits = searcher.search(searchquery)
display_marker_result()