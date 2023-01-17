# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import tensorflow
import re
import time 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


    
lines=open('../input/movie-conversations/movie_lines.txt', encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('../input/movie-conversations/movie_conversations.txt', encoding='utf-8',errors='ignore').read().split('\n')

id2line={}

for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]]=_line[4]
        

conversation_id=[]

for con in conversations :
    _conversation=con.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_id.append(_conversation.split(','))
quentions=[]
answers=[]

for conv in conversation_id:
    for i in range(len(conv)-1):
        quentions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
print(quentions)
print(answers)
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

#clean clean_quentions clean_quentions 
clean_quentions=[]
for q in quentions:
    clean_quentions.append(clean_text(q))
        

clean_answser=[]
for a in answers:
    text=clean_text(a)
    clean_answser.append(text)

     
word2count={}

for q in clean_quentions:
    for word in q.split():
        if word not in word2count:
            word2count[word]=1
        else:
             word2count[word]+=1
             
for a in  clean_answser :
    for word in a.split():
        if word not in word2count:
            word2count[word]=1
        else:
             word2count[word]+=1
number_word=0
int_quentionword={}
th=30

for word, count in word2count.items():
    if(count>=th):
        int_quentionword[word]=number_word
        number_word+=1
        
number_word=0
int_answerword={}
th=30

for word, count in word2count.items():
    if(count>=th):
        int_answerword[word]=number_word
        number_word+=1
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    int_quentionword[token] = len(int_quentionword) + 1
for token in tokens:
    int_answerword[token] = len(int_answerword) + 1
    
int_answerword={w_i : w for w,w_i in int_answerword.items() }

#Add [EOS] for answers
for i in range(len(clean_answser)):
    clean_answser[i]+=' <EOS>'
    
q_2_int=[]

for q in clean_quentions:
 ints=[]
 for word in q.split():
     if word not in int_quentionword:
         ints.append(int_quentionword['<OUT>'])
     else:
         ints.append(int_quentionword[word])
 q_2_int.append(ints)
         
as_2_int=[]

for a in clean_answser:
 ints=[]
 for word in a.split():
     if word not in int_answerword:
         ints.append(int_answerword['<OUT>'])
     else:
         ints.append(int_answerword[word])
 as_2_int.append(ints)
 
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(q_2_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(q_2_int[i[0]])
            sorted_clean_answers.append(as_2_int[i[0]])