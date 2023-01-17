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
questions=pd.read_csv('/kaggle/input/pythonquestions/Questions.csv',encoding='latin1')
questions.head()
questions.columns
questions.loc[67]['Title']
questions.loc[67]['Body']
from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
  print("Message: {}".format(messages[i]))
  print("Embedding size: {}".format(len(message_embedding)))
  message_embedding_snippet = ", ".join(
      (str(x) for x in message_embedding[:3]))
  print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
message_embeddings.shape
questions.shape
title=list(questions['Title'].values)
title[45:78]
embed1=embed(title[:100000])
embed2=embed(title[100000:200000])
embed3=embed(title[200000:300000])
embed4=embed(title[300000:400000])
embed5=embed(title[400000:500000])
embed6=embed(title[500000:])
embeds=[np.array(embed1),np.array(embed2),np.array(embed3),np.array(embed4),np.array(embed5),np.array(embed6)]
names=['embed1','embed2','embed3','embed4','embed5','embed6']
import pickle
for i in range(6):
    file=open(names[i],'wb')
    pickle.dump(embeds[i],file)
    file.close()
for i in embeds:
    print(i.shape)
#search_results
inputs=input('enter query :')
vector=embed([inputs])
vector=np.array(vector)
for i in range(len(embeds)):
    s=np.dot(vector,embeds[i].T)
    norm_a=np.linalg.norm(embeds[i],axis=1)
    norm_a=norm_a*np.linalg.norm(vector)
    s=np.reshape(s,-1)
    norm=s/norm_a
    if i==0:
        m=list(s)
    else:
        m.extend(list(s))

        
m=np.array(m)

ind = np.argpartition(m, -10)[-10:]
x=questions.loc[ind[0]]['Body']
len(x)
x
def preprocess(x):
    m=re.finditer('\n',x)
    mp = [match.start() for match in m]
    clean=[]
    for i in range(len(mp)):
        if i==0:
            clean.append(x[:mp[i]])
            continue
        clean.append(x[mp[i-1]:mp[i]])
    clean_ans=[]
    for i in clean:
        i=re.sub('\n', '', i)
        i=re.sub('<.*?>', '', i)
        if i=='':
            continue
        clean_ans.append(i)
    return clean_ans     
        
for i in ind:
    print('Title : '+questions.loc[i]['Title'])
    x=questions.loc[i]['Body']
    x=preprocess(x)
    for j in x:
        print(j)
    break
    
