from IPython.display import Image

Image("../input/taskrelatedpapers/Task_Related_Papers_unsorted.png")
#Importing all the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

from datetime import date

import itertools

import re

import string

import nltk

import os

import json
#Converting the data from kaggle into a dataframe

count=0

df=pd.DataFrame()



dir="/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/"

for file in os.listdir(dir):

    if file.endswith(".json"):

        file= os.path.join(dir,file)

        data = json.loads(open(file).read())

        df.loc[count,'subset_type']="common"

        try:

             df.loc[count,'paper_id']=data['paper_id']

        except:

             df.loc[count,'paper_id']=None

        try:

             df.loc[count,'title']=data['metadata']['title']

        except:

             df.loc[count,'title']=None

        try:

             df.loc[count,'abstract']=data['abstract'][0]['text']

        except:

             df.loc[count,'abstract']=None

        try:

             df.loc[count,'text']=data['body_text'][0]['text']

        except:

             df.loc[count,'text']=None

    count=count+1

    

dir="/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/"

for file in os.listdir(dir):

    if file.endswith(".json"):

        file= os.path.join(dir,file)

        data = json.loads(open(file).read())

        df.loc[count,'subset_type']="non-common"

        try:

             df.loc[count,'paper_id']=data['paper_id']

        except:

             df.loc[count,'paper_id']=None

        try:

             df.loc[count,'title']=data['metadata']['title']

        except:

             df.loc[count,'title']=None

        try:

             df.loc[count,'abstract']=data['abstract'][0]['text']

        except:

             df.loc[count,'abstract']=None

        try:

             df.loc[count,'text']=data['body_text'][0]['text']

        except:

             df.loc[count,'text']=None

    count=count+1
df
#Shape of the dataframe

df.shape
##Lemmatize words based on parts of speech

from nltk.corpus import wordnet



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



lemmatizer = nltk.stem.WordNetLemmatizer()

df['lemma'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(y,get_wordnet_pos(y)) for y in x.split()])
##Convert vector to string

df['lemma'] = df['lemma'].apply(lambda x:' '.join(map(str, x)))

df['lemma'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.%-]', value=r'')
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(analyzer='word', stop_words = 'english')

for i in range(0,len(df)):

    value = [df['lemma'][i]]

    try:

        x = v.fit_transform(value)#.values.astype('str'))

    except ValueError:

        continue

    #words = v.get_feature_names()

    feature_array = np.array(v.get_feature_names())

    tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:25]

    words=",".join(top_n)

    df.loc[i,'keywords']=words
# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt



stopwords=['disease','process','development']

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white',

                stopwords=stopwords,

                min_font_size = 10).generate(str(df['keywords']))



# %% [code]

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()

#Import csv with task and task details

data = pd.read_csv("../input/task-data/task_data.csv",encoding='cp1252')

data['Task Details'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.-]', value=r'')
##Lemmatize words based on parts of speech

from nltk.corpus import wordnet



def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)



lemmatizer = nltk.stem.WordNetLemmatizer()

data['lemma'] = data['Task Details'].apply(lambda x: [lemmatizer.lemmatize(y,get_wordnet_pos(y)) for y in x.split()])
data
##Convert vector to string

data['lemma'] = data['lemma'].apply(lambda x: ' '.join(map(str, x)))

temp=pd.DataFrame(data['lemma'],columns=['lemma'])

temp=temp.rename(index={0:12014,1:12015,2:12016,3:12017,4:12018,5:12019,6:12020,7:12021,8:12022,9:12023,10:12024,11:12025})

df_consolidated=pd.DataFrame(data=df['abstract'],columns=['abstract'])

for i in range(0,len(df)):

    if df_consolidated.loc[i,'abstract']==None:

        df_consolidated.loc[i,'abstract']=df.loc[i,'text']

df_consolidated=df_consolidated.rename(columns={'abstract':'lemma'})

df_consolidated['lemma'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.-]', value=r'')

df_consolidated=pd.concat([df_consolidated,temp])
df_consolidated
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(analyzer='word', min_df = 0.04, stop_words = 'english')

x = v.fit_transform(df_consolidated.lemma.values.astype('str'))

df2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
from matplotlib.ticker import StrMethodFormatter

from numpy import mean

x=mean(df2)

x=x.sort_values(ascending=False)

x=x.drop(['diseases','viruses','virus','disease','viral','health','cell'])

x=x[:50]

ax = x.plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)



  # Despine

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)



# Switch off ticks

ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")



# Draw vertical axis lines

vals = ax.get_xticks()

for tick in vals:

    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)



# Set x-axis label

ax.set_xlabel("TF-IDF score", labelpad=20, weight='bold', size=12)



# Set y-axis label

ax.set_ylabel("Word List", labelpad=20, weight='bold', size=12)



# Format y-axis label

ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))   

from sklearn.metrics.pairwise import cosine_similarity

new_similarity=cosine_similarity(df2)
#Narrowing results only to task  similarity

new_similarity=new_similarity[12014:,:12014]
##Loop through each row to find the top 30 similar papers

final_new=[]

for  i in range(0,len(new_similarity)):

    b=[]

    a=new_similarity[i]

    if len(a)==0 or np.count_nonzero(a)==0:

            continue

    for j in range(0,31):

        maxi=max(a)

        x=[]

        s=[]

        for l,k in enumerate(a):

            if k==maxi:

                x.append(l)

                s.append(l+j)

        a=np.delete(a,x)

        b.append(s)

    if i in b[0]:

        b[0].remove(i)

    final_new.append(b)



##Eliminate anything  excess of 30

add=[]

for i in final_new:

    x=[]

    for j in i:

        for k in j:

            if len(x)>=30:

                del x[30:]

                break

            x.append(k)

    add.append(x)

top = pd.DataFrame(add)



top['Top Related Papers']='['+top[0].astype(str)+','+top[1].astype(str)+','+top[2].astype(str)+','+top[3].astype(str)+','+top[4].astype(str)+','+top[5].astype(str)+','+top[6].astype(str)+','+top[7].astype(str)+','+top[8].astype(str)+','+top[9].astype(str)+','+top[10].astype(str)+','+top[11].astype(str)+','+top[12].astype(str)+','+top[13].astype(str)+','+top[14].astype(str)+','+top[15].astype(str)+','+top[16].astype(str)+','+top[17].astype(str)+','+top[18].astype(str)+','+top[19].astype(str)+','+top[20].astype(str)+','+top[21].astype(str)+','+top[22].astype(str)+','+top[23].astype(str)+','+top[24].astype(str)+','+top[25].astype(str)+','+top[26].astype(str)+','+top[27].astype(str)+','+top[28].astype(str)+','+top[29].astype(str)+']'



top=top.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],axis=1)



data_final=pd.concat([data,top],axis=1)

key=[]

for i in add:

    k=[]

    for  j in i:

        k.append([df.loc[j,'keywords']])

    #print(k)

    key.append(str(k))

#print(key)

data_final=pd.concat([data_final,pd.DataFrame(key,columns=['Keywords'])],axis=1)
from sklearn.feature_extraction.text import TfidfVectorizer

c=0

for i in add:

    text=""

    for j in i:

        text=text+" "+df.loc[j,'text']

    inp=[text]

    v = TfidfVectorizer(analyzer='word', stop_words = 'english')

    x = v.fit_transform(inp)#.values.astype('str'))

    #p=v.get_feature_names()

    feature_array = np.array(v.get_feature_names())

    tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]

    top_n = feature_array[tfidf_sorting][:25]

    #print(p)

    data_final.loc[c,'Overall Keywords']=str(top_n)

    #print(top_n)

    c=c+1
indiv=data_final.loc[1,'Keywords']

stopwords=['viral','factor','human','virus','disease','different','health','study','infect','infection','cause','case']

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords=stopwords,

                min_font_size = 10).generate(str(indiv))



# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()

indiv=data_final.loc[2,'Keywords']

stopwords=['include','factor','human','virus','disease','different','health','study','infect','infection','cause','important','case']

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords=stopwords,

                min_font_size = 10).generate(str(indiv))



# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()

indiv=data_final.loc[8,'Keywords']

stopwords=['cell','diseases','case','information','human','virus','disease','different','health','study','infect','infection','cause','important']

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords=stopwords,

                min_font_size = 10).generate(str(indiv))



# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()

indiv=data_final.loc[0,'Keywords']

stopwords=['viral','detect','case','human','virus','disease','different','health','study','infect','infection','cause','important']

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords=stopwords,

                min_font_size = 10).generate(str(indiv))



# %% [code]

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()

##Loop through each row to find the similar papers

final_new_limitless=[]

for  i in range(0,len(new_similarity)):

    a=new_similarity[i]

    b=np.argsort(a)

    for i in range(0,len(b)):

        if a[i]>=0.5:           

            break

    b=b[i:]

    final_new_limitless.append(b)
count=0

df['Related_to_Task_No']=''

for i in final_new_limitless:

    for j in i:

        df.loc[j,'Related_to_Task_No']= str(df.loc[j,'Related_to_Task_No']) + "," + str(count)  

    count=count+1



print(df)

data_final.to_csv('Task_output_submission.csv')

df.to_csv('Article_output_submission.csv')
