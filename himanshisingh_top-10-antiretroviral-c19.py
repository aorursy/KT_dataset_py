# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! pip install distance 
import pandas as pd

import numpy as np

import json

import os.path

import re

import distance

import matplotlib.pyplot as plt
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv")

metadata=metadata[metadata["sha"]==metadata["sha"]] 

metadata=metadata[metadata["has_full_text"]]
def path(shaid):

    for dirname, _, files in os.walk('/kaggle/input'):

        if shaid+'.json' in files:

            return os.path.join(dirname,shaid+".json")
metadata["Path"]=metadata.apply(lambda x: path(x["sha"]),axis=1) #this takes a while unfotunately

metadata=metadata[metadata["Path"]==metadata["Path"]]
STRING='vir '

Texts={} 

valid_id=[]

for shaid,file in zip(metadata["sha"],metadata["Path"]):

    with open(file, 'r') as f:

        doc=json.load(f)

    MainText=''

    for item in doc["body_text"]:

        MainText=MainText+(re.sub('[^a-zA-Z0-9]', ' ', item["text"].lower()))

    if STRING in MainText:

        Texts[shaid]=MainText

        valid_id.append(shaid)
metadata=metadata[metadata["sha"].isin(valid_id)]
MIN_LENGTH=5 

drugs=[]

for shaid in valid_id:

    iterator=re.finditer(STRING,Texts[shaid])

    for m in iterator:

        drugs.append(Texts[shaid][Texts[shaid].rfind(' ',0, m.end()-2):m.end()])

drugs=[i for i in drugs if len(i)>MIN_LENGTH]

drugs_set=list(set(drugs))

drugs_set=sorted(drugs_set)
THRESH=2 

incorrects=[]

corrects=[]

from itertools import combinations

for str1,str2 in combinations(drugs_set,2):

    if (distance.levenshtein(str1, str2)<THRESH) and (drugs.count(str1)>10 or drugs.count(str2)>10):

            if drugs.count(str1)>drugs.count(str2):

                incorrect=str2

                correct=str1

            else:

                incorrect=str1

                correct=str2

            if incorrect not in incorrects:

                incorrects.append(incorrect)

                corrects.append(correct)

for item in incorrects:

    drugs_set.remove(item)
for shaid in valid_id:

    for inc in range(0,len(incorrects)):

        re.sub(incorrects[inc],corrects[inc], Texts[shaid])
MAXPLOT=10 

cs=[]

for item in drugs_set:

    cs.append(drugs.count(item))

cs=np.array(cs)

plt.figure(figsize=(10,5))

plt.bar(np.array(drugs_set)[(-cs).argsort()[:MAXPLOT]], cs[(-cs).argsort()[:MAXPLOT]], width = 0.5, color = ['green'])

plt.xticks(rotation=90,fontsize=18)

plt.yticks(fontsize=10)

plt.ylabel("occurance", fontsize=10)

plt.show()