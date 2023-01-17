! pip install distance #Make sure you have Internet(ON) in the Kaggle notebook settings
import pandas as pd

import numpy as np

import json

import os.path

import re

import distance

import matplotlib.pyplot as plt
metadata=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

metadata=metadata[metadata["sha"]==metadata["sha"]] #filters out entries having sha=NaN

metadata=metadata[metadata["has_full_text"]] #filters out entries for which the full text is not available
def path(shaid): #returns path of .json file

    for dirname, _, files in os.walk('/kaggle/input'):

        if shaid+'.json' in files:

            return os.path.join(dirname,shaid+".json")
metadata["Path"]=metadata.apply(lambda x: path(x["sha"]),axis=1) #this takes a while unfotunately

metadata=metadata[metadata["Path"]==metadata["Path"]]

metadata.shape
STRING='vir ' #note the space at the end

KEY='corona' #filter for keywords such as corona, covid, sars, mers, etc.
Texts={} #dictionary: {id: "Text"}; adapted from cristian's notebook (https://www.kaggle.com/crprpr/vaccine-data-filter)

Abs_and_concl_w_punct={}

valid_id=[]

for shaid,file in zip(metadata["sha"],metadata["Path"]):

    with open(file, 'r') as f:

        doc=json.load(f)

    MainText=''

    A_C_w_p=''

    for item in doc["body_text"]:

        MainText=MainText+(re.sub('[^a-zA-Z0-9]', ' ', item["text"].lower()))

        if (item["section"]=="Discussion") or (item["section"]=="Abstract") or (item["section"]=="Conclusion"):

            A_C_w_p=A_C_w_p+item["text"].lower()

    if (STRING in MainText) and (KEY in MainText):

        Texts[shaid]=MainText

        Abs_and_concl_w_punct[shaid]=A_C_w_p

        valid_id.append(shaid)
metadata=metadata[metadata["sha"].isin(valid_id)] #filter only articles that contain names of antivirals
MIN_LENGTH=6 #most likely names of antivirals are longer than 4 letters + 2 spaces; shorter words are probably acronyms 

drugs=[]

for shaid in valid_id:

    iterator=re.finditer(STRING,Texts[shaid])

    for m in iterator:

        drugs.append(Texts[shaid][Texts[shaid].rfind(' ',0, m.end()-2):m.end()])

drugs=[i for i in drugs if len(i)>MIN_LENGTH]

drugs_set=list(set(drugs))

count=[]

for d in drugs_set:

    count.append(-drugs.count(d))

drugs_set=list(np.array(drugs_set)[np.array(count).argsort()])  # thanks Julian for the suggestion
len(drugs_set)
THRESH=2 #Threshold for the Levenshtein distance

incorrects=[]

corrects=[]

from itertools import combinations

for str1,str2 in combinations(drugs_set,2):

    if (distance.levenshtein(str1, str2)<THRESH) and (drugs.count(str1)>10 or drugs.count(str2)>10):

            if drugs.count(str1)>=drugs.count(str2):

                incorrect=str2

                correct=str1

            else:

                incorrect=str1

                correct=str2

            print(str1, "(",drugs.count(str1),")", "and", str2, "(",drugs.count(str2),")", "look very similar.")

            if incorrect not in incorrects:

                print("I will substitute", incorrect, "with", correct,".")

                incorrects.append(incorrect)

                corrects.append(correct)
for item in incorrects:

    drugs_set.remove(item)
len(drugs_set)
for shaid in valid_id:

    for inc in range(0,len(incorrects)):

        Texts[shaid]=re.sub(incorrects[inc],corrects[inc], Texts[shaid])
antivirals=pd.DataFrame(drugs_set,columns=["Drug"])



def count1(drug,druglist):

    return druglist.count(drug)



def count2(drug):

    n=0

    for shaid in valid_id:

        iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])

        for m in iterator:

            n+=1 

    return n

        

antivirals['Count_text'] = antivirals.apply(lambda x: count1(x["Drug"],drugs),axis=1) #counts occurences in the whole text

antivirals['Count_abs_concl'] = antivirals.apply(lambda x: count2(x["Drug"]),axis=1) #counts occurences in abstract + conclusions
MAXPLOT=20 #plot the MAXPLOT most mentioned antivirals

plt.figure(figsize=(20,5))

plt.bar(antivirals["Drug"][(-antivirals["Count_text"].to_numpy()).argsort()[:MAXPLOT]], antivirals["Count_text"][(-antivirals["Count_text"].to_numpy()).argsort()[:MAXPLOT]])

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=12)

plt.ylabel("Counts",fontsize=15)

plt.show()
from textblob import TextBlob

import nltk

histo=[]

nltk.download('punkt')
def Sentiment(drug): #looks for the drug name in the abstract or conclusions and measures the sentiment

    s=0

    smax=-1

    for shaid in valid_id:

            iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])

            for m in iterator:

                beg_comma=Abs_and_concl_w_punct[shaid].rfind(',',0, m.start())+1

                end_comma=Abs_and_concl_w_punct[shaid].find(',',m.end()-1,len(Texts[shaid]))

                beg_dot=Abs_and_concl_w_punct[shaid].rfind('.',0, m.start())+1

                end_dot=Abs_and_concl_w_punct[shaid].find('.',m.end()-1,len(Texts[shaid]))

                beg=max(beg_comma,beg_dot)

                end=min(end_comma,end_dot)

                blob = TextBlob(Abs_and_concl_w_punct[shaid][beg:end])

                s+=blob.sentiment.polarity

    return s



THRESH=0.3

def Sentence(drug): #records positive senctences, with the doi for reference

    nice_sentence=[]

    for shaid in valid_id:

            iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])

            for m in iterator:

                beg_comma=Abs_and_concl_w_punct[shaid].rfind(',',0, m.start())+1

                end_comma=Abs_and_concl_w_punct[shaid].find(',',m.end()-1,len(Texts[shaid]))

                beg_dot=Abs_and_concl_w_punct[shaid].rfind('.',0, m.start())+1

                end_dot=Abs_and_concl_w_punct[shaid].find('.',m.end()-1,len(Texts[shaid]))

                beg=max(beg_comma,beg_dot)

                end=min(end_comma,end_dot)

                blob = TextBlob(Abs_and_concl_w_punct[shaid][beg:end])

                if blob.sentiment.polarity > THRESH:

                    for doi in metadata[metadata["sha"]==shaid]["doi"]:

                        nice_sentence.append(str(blob)+" [ "+doi+" ]")

    if len(nice_sentence)==0:

        nice_sentence="Nothing found"

    return nice_sentence
antivirals['Sentiment'] = antivirals.apply(lambda x: Sentiment(x["Drug"]),axis=1)

antivirals["Nice_sentence"] = antivirals.apply(lambda x: Sentence(x["Drug"]),axis=1)

#antivirals['Sentiment_norm']=antivirals["Sentiment"]/antivirals["Count_abs_concl"]
MAXPLOT=20

plt.figure(figsize=(20,5))

plt.bar(antivirals["Drug"][(-antivirals["Sentiment"].to_numpy()).argsort()[:MAXPLOT]], antivirals["Sentiment"][(-antivirals["Sentiment"].to_numpy()).argsort()[:MAXPLOT]])

plt.xticks(rotation=90)

plt.xticks(rotation=90,fontsize=12)

plt.yticks(fontsize=12)

plt.ylabel("Total Sentiment",fontsize=15)

plt.show()
for sentences in antivirals[antivirals["Drug"]==" oseltamivir "]["Nice_sentence"]:

    for s in sentences:

        print(s)
for sentences in antivirals[antivirals["Drug"]==" favipiravir "]["Nice_sentence"]:

    for s in sentences:

        print(s)