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
def get_year_month_day(x,k):

    s = str(x)

    s = s.split('-')

    try:

        return int(s[k])

    except:

        return 0
metadata["year"]=metadata.apply(lambda x: get_year_month_day(x["publish_time"],0),axis=1)

metadata["month"]=metadata.apply(lambda x: get_year_month_day(x["publish_time"],1),axis=1)

metadata["day"]=metadata.apply(lambda x: get_year_month_day(x["publish_time"],2),axis=1)

# metadata = metadata[metadata["year"]>2019]
metadata["Path"]=metadata.apply(lambda x: path(x["sha"]),axis=1) #this takes a while unfotunately

metadata=metadata[metadata["Path"]==metadata["Path"]]

metadata.shape
STRING='vir ' #note the space at the end
Texts={} #dictionary: {id: "Text"}; adapted from cristian's notebook (https://www.kaggle.com/crprpr/vaccine-data-filter)

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
metadata_old = metadata.copy()
metadata  = metadata_old.copy()

sha_selected=[]

metadata['selected']=False

for a in Texts.keys():

    t = Texts[a]

    s = t.split(' ')

    # f = s.find("table")

    # key_words = [ 'ratio','recovered','death','covid-19','covid19'  ]

    key_words = [ 'covid-19','covid19', 'covid 19'  ]

    for k in key_words:

        indexes = [index for index in range(len(s)) if s[index] == k]

        if len(indexes) > 0:

            idx = metadata[metadata['sha']==a].index  

            metadata.at[idx,'selected'] = True 

            

            

    key_words = [ ['result','in', 'death'] ]

    for k in key_words:

        indexes = [index for index in range(len(s)) if s[index] == k[0]]

        if len(indexes) > 0:

            for i in indexes:

                if s[i+1]==k[1] and s[i+2] == k[2]:

                    idx = metadata[metadata['sha']==a].index  

                    metadata.at[idx,'selected'] = True 

             



x = metadata[metadata['selected']==True]

x.shape

metadata_old = metadata.copy()

#metadata = x
metadata=metadata[metadata["sha"].isin(valid_id)] #filter only articles that contain names of antivirals



metadata.shape
MIN_LENGTH=6 #most likely names of antivirals are longer than 4 letters + 2 spaces; shorter words are probably acronyms 

drugs=[]

drugs_data = {}

for shaid in valid_id:

    iterator=re.finditer(STRING,Texts[shaid])

    for m in iterator:

        d = Texts[shaid][Texts[shaid].rfind(' ',0, m.end()-2):m.end()]

        drugs.append(d)        

        meta = metadata[metadata['sha']==shaid]

        

        info = [int(meta.year),int(meta.month),int(meta.day),shaid,meta.publish_time]

        try:

            drugs_data[d].append(info)

        except:

            drugs_data[d] = []

            drugs_data[d].append(info)

            

drugs=[i for i in drugs if len(i)>MIN_LENGTH]

drugs_set=list(set(drugs))

drugs_set=sorted(drugs_set)
THRESH=2 #Threshold for the Levenshtein distance

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

            print(str1, "(",drugs.count(str1),")", "and", str2, "(",drugs.count(str2),")", "look very similar. I will substitute", incorrect, "with", correct)

            if incorrect not in incorrects:

                incorrects.append(incorrect)

                corrects.append(correct)

for item in incorrects:

    drugs_set.remove(item)
len(drugs_set)
for shaid in valid_id:

    for inc in range(0,len(incorrects)):

        re.sub(incorrects[inc],corrects[inc], Texts[shaid])
MAXPLOT=20 #plot the MAXPLOT most mentioned antivirals

cs=[]

for item in drugs_set:

    cs.append(drugs.count(item))

cs=np.array(cs)

plt.figure(figsize=(20,5))

plt.bar(np.array(drugs_set)[(-cs).argsort()[:MAXPLOT]], cs[(-cs).argsort()[:MAXPLOT]])

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.ylabel("Counts", fontsize=15)

plt.show()
def plot_drug(a,sx):

    dd = drugs_data[a]



    df = pd.DataFrame.from_dict(dd)

    df.columns=["year", "month", "day", "sha","publish_time"]

    x = (df["year"]).astype(str)+"-"+(df["month"]).astype(str)+"-"+(df["day"]).astype(str)

    try:

        df['date']=pd.to_datetime(x)



        k = df.groupby(['sha','date']).size().reset_index(name='Size')

         

        kk = k[['date','Size']] 

        #plt.scatter(kk['date'], kk['Size'] ,s=250,label=a)

        # plt.plot(kk['date'], kk['Size'],'o--',linewidth=.1, markersize=10+sx ,label=a)

        plt.plot(kk['date'], [sx+1]*len(kk['date']),'o--',linewidth=.1, markersize=33 ,label=a)

    except:        

        pass

 
MAXPLOT=20 #plot the MAXPLOT most mentioned antivirals

cs = []

for item in drugs_set:

    cs.append(drugs.count(item))

cs = np.array(cs)



a = np.array(drugs_set)[(-cs).argsort()[:MAXPLOT]]



plt.figure(figsize=(50,15))



cnt=0

for a_itr in a:

    cnt+=1

    plot_drug(a_itr,(MAXPLOT-cnt))

plt.xticks(rotation=90,fontsize=25)

plt.yticks(fontsize=25)

plt.legend()

plt.legend(loc=1, prop={'size': 25})

plt.grid(True)

plt.show()

metadata[metadata['sha']==shaid]
Texts[shaid]