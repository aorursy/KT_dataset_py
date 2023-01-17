!pip install -U sentence-transformers
!pip install openpyxl
!pip install googletrans

from openpyxl.workbook import Workbook
import pandas as pd
import os

import scipy
from sentence_transformers import SentenceTransformer

import re, math
from collections import Counter
import pandas as pd
from googletrans import Translator
answer=open("../input/answer.txt",encoding="utf8")
ans=[]
for Line_in_File in answer:
    ans.append(Line_in_File)
keyword=open("../input/keyword.txt",encoding="utf8")
kwd=[]
for Line_in_File in keyword:
    kwd.append(Line_in_File)


print(ans,kwd)

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import re
kwd_dic={}
for i in range(len(kwd)):
  synonyms = []
  wd=re.sub("\n","",str(kwd[i]))
  for syn in wordnet.synsets(str(wd)):
    for l in syn.lemmas():
      synonyms.append(l.name())
  kwd_dic[str(wd)]=synonyms
print(kwd_dic)


nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
snt=word_tokenize(str(ans[0]))

print(snt)

import itertools
words=[]
for i in range(len(snt)):
  try:
    k=0
    #print(snt[i])
    #print(len(kwd_dic))
    for key,value in kwd_dic.items():
      #print(key)
      if snt[i]==key:
        words.append(value)
        print(len(value))
        k=1
        break
    if k==0:
      raise Exception("I know python!")
  except Exception as e:
    #print(e)
    words.append([snt[i]])
words

sentences=list(itertools.product(*words))
sentences[0:10]

print(len(sentences))
sentences1=[]
for i in range(len(sentences)):
  sntns=' '.join(sentences[i])
  sentences1.append(sntns)
!pip install --upgrade language_tool_python
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')
text = 'living giving'
matches = tool.check(text)
len(matches)
sentences1[0:10]
with open('student.txt', 'w') as f:
    for item in sentences1:
        f.write("%s\n" % item)
import nltk
from googletrans import Translator
nltk.download('wordnet')
from nltk.corpus import wordnet
import re
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize 
import itertools
#!pip install --upgrade language_tool_python
import language_tool_python
#!pip install -U sentence-transformers
#!pip install openpyxl
#!pip install googletrans
from openpyxl.workbook import Workbook
import pandas as pd
import os

import scipy
from sentence_transformers import SentenceTransformer

import re, math
from collections import Counter
correct_ans=[]
wrong_ans=[]
def bert_distance(ans,keyword,student_answer,threshold):
  count=0
  global correct_ans
  global wrong_ans
  i=0
  j=0
  embedder = SentenceTransformer('bert-base-nli-mean-tokens')
  corpus_embeddings = embedder.encode(student_answer)
  queries = [ans]
  query_embeddings = embedder.encode(queries)
  for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
    distances=list(distances)
    print(distances)
    for distance in range(len(distances)):
      if distances[distance]<threshold:
        correct_ans.append(distance)
      else:
        wrong_ans.append(distance)
  #df.to_excel("answer_check.xlsx",index=False)
  #return [correct_ans,wrong_ans]

def answer_checking(ans_loc,student,keyword_loc,choice):
  global correct_ans
  global wrong_ans
  if choice==1:
    answer=open(ans_loc,encoding="utf8")
    ans=[]
    for Line_in_File in answer:
        ans.append(Line_in_File)
    keyword=open(keyword_loc,encoding="utf8")
    keywrd=[]
    for Line_in_File in keyword:
        keywrd.append(re.sub("\n","",str(Line_in_File)))
    stnt=open(student,encoding="utf8")
    student_answer=[]
    for Line_in_File in stnt:
        student_answer.append(re.sub("\n","",str(Line_in_File)))
    ans[0]=str(ans[0]).lower()
    for k in range(len(keywrd)):
      keywrd[k]=keywrd[k].lower()
    for k in range(len(student_answer)):
      student_answer[k]=student_answer[k].lower()
    
    df=bert_distance(str(ans[0]),keywrd,student_answer[0:100],0.0299)
    answer=open(ans_loc,encoding="utf8")
    ans=[]
    for Line_in_File in answer:
        ans.append(Line_in_File)
    keyword=open(keyword_loc,encoding="utf8")
    kwd=[]
    for Line_in_File in keyword:
        kwd.append(Line_in_File)
    nltk.download('wordnet')

    kwd_dic={}
    for i in range(len(kwd)):
      synonyms = []
      wd=re.sub("\n","",str(kwd[i]))
      for syn in wordnet.synsets(str(wd)):
        for l in syn.lemmas():
          synonyms.append(l.name())
      kwd_dic[str(wd)]=synonyms
    #print(kwd_dic)

    nltk.download('punkt')

    snt=word_tokenize(str(ans[0]))
    words=[]
    for i in range(len(snt)):
      try:
        k=0
        #print(snt[i])
        #print(len(kwd_dic))
        for key,value in kwd_dic.items():
          #print(key)
          if snt[i]==key:
            words.append(value)
            print(len(value))
            k=1
            break
        if k==0:
          raise Exception("I know python!")
      except Exception as e:
        #print(e)
        words.append([snt[i]])

    sentences=list(itertools.product(*words))
    print(len(sentences))
    sentences1=[]
    tool = language_tool_python.LanguageTool('en-US')
    for i in range(len(sentences[0:100])):
      sntns=' '.join(sentences[i])
      matches = tool.check(sntns)
      if len(matches)==0:
        sentences1.append(sntns)
    for k in range(len(keywrd)):
      keywrd[k]=keywrd[k].lower()
    for k in range(len(student_answer)):
      student_answer[k]=student_answer[k].lower()
    for sents in range(len(sentences1)):
      sentences1[sents]=sentences1[sents].lower()
      bert_distance(sentences1[sents],keywrd,student_answer[0:100],0.02)
    df=pd.DataFrame()
    df['Student_ans']=student_answer[0:100]
    df['ans']=[0 for x in range(len(student_answer[0:100]))]
    for chk in range(len(correct_ans)):
      df['ans'][correct_ans[chk]]=1
    df.to_excel("answers.xlsx",index=False)
  else:
    ans=str(ans_loc).lower()
    for k in range(len(keyword_loc)):
      keyword_loc[k]=keyword_loc[k].lower()
    student=student.lower()
    df=bert_distance(ans,keyword_loc,[student],0.0299)
    if len(correct_ans)>0:
      print('Correct answer')
      correct_ans=[]
    else:
      print('Wrong answer')
#if you want to predict for complete answer file use 1 for choice and give location of files 
#if you want to predict for single student answer user 0 for choice and give answers as parameter
answer_checking(ans_loc='this is wrong answer',student="it's wrong answer",keyword_loc=['wrong','correct'],choice=0)

len(correct_ans)

