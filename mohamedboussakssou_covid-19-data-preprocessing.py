import numpy as np 

import pandas as pd 



import os
import json

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image

import re

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")
dirs = ['/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/',

        '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/',

        '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/',

        '/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/'

       ]





filenames=[]

docs =[]

for d in dirs:

    for file in os.listdir(d):

        filename = d +file

        j = json.load(open(filename, 'rb'))

        

        paper_id =j['paper_id']

        

        title = j['metadata']['title']

        authors = j['metadata']['authors']

        list_authors =[]

        for author in authors:

            if(len(author['middle'])==0):

                middle =""

            else :

                middle = author['middle'][0]

            _authors =author['first']+ " "+ middle +" "+ author['last']

            list_authors.append(_authors)

            

        try :

            abstract =  j['abstract'][0]['text']

        except :

            abstract =" "

        

        full_text =""

        for text in  j['body_text']:

            full_text += text['text']

        

        docs.append([paper_id,title,list_authors,abstract,full_text])



df = pd.DataFrame(docs,columns=['paper_id','title','list_authors','abstract','full_text'])

df.to_csv('/kaggle/working/data.csv')
df.head()
incubation = df[df['full_text'].str.contains('incubation')]

incubation.head()
all_incubation_paragraph=[]

for text in incubation['full_text'].values:

    for paragraph in text.split('. '):

        if 'incubation' in paragraph:

            all_incubation_paragraph.append(paragraph)

            

    

len(all_incubation_paragraph)

days_incubation=[]

for t in all_incubation_paragraph:

    day=re.findall(r"\d{1,2} day", t)

    if (len(day)==1):

        days_incubation.append (day[0].split(" "))

        

days_incubation_1=[]

for d in days_incubation:

    days_incubation_1.append(float(d[0]))

len(days_incubation_1)
plt.xlabel("days incubation")

plt.hist(days_incubation_1)
np.mean(days_incubation_1)
transmission = df[df['full_text'].str.contains('transmission')]



all_transmission_paragraph=[]

for text in transmission['full_text'].values:

    for paragraph in text.split('. '):

        if 'transmission' in paragraph:

            all_transmission_paragraph.append(paragraph)

            

print(len(all_transmission_paragraph))

feet_transmission=[]

for t in all_transmission_paragraph:

    feet = re.findall(r"\d{1,2} feet", t)

    if len(feet)==1:

        feet_transmission.append(feet)

feet_transmission_1=[]

for d in feet_transmission:

    feet_transmission_1.append(float(d[0].split(' ')[0]))

len(feet_transmission_1)
np.mean(feet_transmission_1)
plt.hist(feet_transmission_1)