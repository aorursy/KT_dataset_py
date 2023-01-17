# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
train = pd.read_csv("drive/app/toxic_train.csv")
test=pd.read_csv("drive/app/toxic_test.csv")
print(train.head())
#creat regular expression for tokenization
newp='["#$%&\'()*+-/:;<=>@[\]^_`{|}~]'

import re
from nltk.tokenize import RegexpTokenizer
toker = RegexpTokenizer('[a-z]+'+newp+'[a-z]+|[a-z]+\d+[a-z]+|[a-z]+')

train['comment_text']=train['comment_text'].map(str.lower)
test['comment_text']=test['comment_text'].map(str.lower)
train['token']=train['comment_text'].map(toker.tokenize)
test['token']=test['comment_text'].map(toker.tokenize)

print(train.head())
#convert feature("repeat") to 1 from 0
train["repeat"]=0
test["repeat"]=0

def addrepeat(df: pd.DataFrame):
  print('start')
  import collections
  for i,row in df.iterrows():
    c=collections.Counter(row['token'])
    c2=collections.Counter()
    for i2, value in c.items():
      value=(value//3)*3
      d={}
      d[value]=1
      c2.update(d)

    for k, v2 in c2.items():
      if k >= 5 and(k*v2/len(row['token'])) >=0.5:
        df.loc[i,'repeat']=1
  
  print(df[df['repeat']==1].head())

addrepeat(train)
addrepeat(test)
import nltk
nltk.download('words')
nltk.download('names')
#set for all words common in toxic comment
s_alltier={'suck', 'shut', 'faggot', 'motherfucker', 'dickhead', 'smash', 'cocksucker', 'fucking',
        'nigger', 'fuck', 'cock', 'bastard', 'moron', 'bitch', 'nigga', 'dumb', 'nipple', 'slut' ,'dog',
        'jew', 'piss', 'fucksex', 'boobs', 'fistfuck', 'stupid', 'balls', 'anal', 'fag', 'bush', 
         'trash', 'crap', 'idiot', 'pussy', 'mexicans', 'vagina', 'hitler', 'gay', 'bullshit', 'wanker',
         'murder', 'homosexuality', 'die', 'racist', 'basterd', 'motherfucking', 'bark', 'flithy', 
         'ass', 'cunt', 'fat', 'shit', 'freak', 'brain', 'bash', 'wtf', 'filthy', 'suffer', 'blood', 
         'bleed', 'deserve', 'ban', 'mad', 'harassement', 'abuse', 'suicide', 'sexuality', 'hack', 
           'safe', 'cut', 'rape', 'swear', 'knife', 'shoot', 'splatter', 'hook', 'sack', 'dick', 'penis', 'damn', 
         'life', 'pit', 'necrophiliac', 'legal', 'destroy', 'corpse', 'kill', 'gut', 'cancer', 'horrible', 
           'gun', 'nut', 'dirty', 'hit', 'punch', 'violence', 'stab'}


#set for english dictionary 
from nltk.corpus import words
word_list=words.words()
word_list.append('fuck')
s=set()
s.update(word_list)
s.update(s_alltier)

#set for name dictionary
from nltk.corpus import names
name_list=names.words()
for a in range(0,len(name_list),1):
  name_list[a]=name_list[a].lower()
s.update(name_list)

# specific set of words for toxic comment
s_test={'idiot', 'tail', 'fucking', 'dick', 'sucker', 'faggot', 'pisser', 'bastard', 'retard', 'vagina', 'queer', 'dumb', 'suck', 'freak', 'cock', 'suicide', 'fucker', 'hitler', 'murderer', 'nigger', 'wank', 'vaginal', 'homosexuality', 'ball', 'fuck', 'cocksucker', 'barky', 'motherfucker', 'nigga'}
