## This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk # natural language processing

import re # regular expression

from bs4 import BeautifulSoup #scraping HTML

from nltk.corpus import stopwords

import seaborn as sns # visualization

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from string import punctuation

from nltk.collocations import BigramCollocationFinder







# nltk workspace



stop = set(stopwords.words('english'))

lemmatizer = nltk.stem.WordNetLemmatizer()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Define function

def strip_punctuation(s):

    # input str, output str, strip out punctuations

    return ''.join(c for c in s if c not in punctuation)



def remove_html(s):

    #input str, output str, remove html from content

    soup = BeautifulSoup(s,'html.parser')

    content = soup.get_text()

    return content



def text_transform(dataframe):

    # input data frame, process title and content. 

    dataframe['title'] = dataframe['title'].apply(lambda x: strip_punctuation(str.lower(x)))

    dataframe['content'] = dataframe['content'].apply(lambda x: strip_punctuation(str.lower(remove_html(x).replace("\n"," "))))



def load_data(name):

    utl = "../input/"+name+'.csv'

    files = pd.read_csv(utl)

    text_transform(files)

    files['category'] = name

    return files



def merge_data(list_of_files):

    list_of_dataframe = [""]*len(list_of_files)

    for i in range(0,len(list_of_files)):

        list_of_dataframe[i] = load_data(list_of_files[i])

    data = pd.concat(list_of_dataframe,axis = 0, ignore_index =True)

    return data
def list_to_str(lists):

    # input list, output string.

    strs = ""

    for content in lists:

        strs+=content

    return strs



def to_plain_text(dataframe):

    # input dataframe, output str. transform the text column in dataframe to str

    text=list_to_str(dataframe['all_text'].apply(lambda x: x.replace('\n',' ')).tolist())

    return text



def to_nltk_text(dataframe):

    #input dataframe, output nltk text object. 

    text = to_plain_text(dataframe)

    token = nltk.word_tokenize(text)

    lemmatizer = nltk.stem.WordNetLemmatizer()

    lemmas = [lemmatizer.lemmatize(t) for t in token]

    return nltk.Text(lemmas)
def hasNumbers(inputString):

     return any(char.isdigit() for char in inputString)





def freqDist(text,include_bigram = True):

    #input str, output dictionary of word and count pair. Calculate (absolute) term frequencies.

    #Incorporate bigrams into the model

    text_bigr = []

    if include_bigram == True:

        text_bigr = list(nltk.bigrams(all_text))

    freqDist = {}

    for data in [text,text_bigr]:

        for word in data:

            if word in freqDist:

                freqDist[word] +=1

            else:

                freqDist[word] = 1

    return freqDist



def relativeFreq(subset,alls,sort=True,adjusted=0):

    #input subset and alls are dictionaries from freqDist function. subset is a subset of text from 

    #the specific topic we are interested in studying whereas alls the totality of text data. we have

    #at disposal.if sort equals to True, output will be sorted based on relative frequencies. Adjusted 

    #is a manual adjustment to terms that have an overall low volume.

    result = [" "]*len(subset)

    result_dict = {}

    modifier = 1

    for i, key in enumerate(subset.keys()):

        if alls[key] > adjusted and hasNumbers(key) == False:

            modifier = 1

        else:

            modifier = 0

        tf = float(subset[key])/alls[key]

        result[i]=(key,tf*modifier)

        result_dict[key] = tf*modifier

    if sort == True:

        result.sort(key=lambda tup: tup[1], reverse = True)

    return [result,result_dict]
#Taking biology as an example

list_of_files = ['biology','cooking','crypto','diy','robotics','travel','test']

data = merge_data(list_of_files)

data['all_text'] = data['title'] + " " + data['content']

all_text = to_nltk_text(data)

Fdist_all = freqDist(all_text)
#Define Function tag Explained

def tagExplained(s,all_text,Fdist_all):

    #s, string, category of interest.

    interest = to_nltk_text(data[data['category'] == s])

    Fdist_interest = freqDist(interest)

    relative_Freq_dict = relativeFreq(Fdist_interest,Fdist_all)[1]

    tags=data[data['category']==s]['tags'].apply(lambda x:nltk.word_tokenize(x)).tolist()

    tags = [x for record in tags for x in record]

    tags=[(lemmatizer.lemmatize(x.split('-')[0]),lemmatizer.lemmatize(x.split('-')[1])) if "-" in x else lemmatizer.lemmatize(x) for x in tags]

    relative_score = [relative_Freq_dict[x] if x in relative_Freq_dict else -1.0 for x in tags ]

    per_of_tag_explained = sum(1 if x != -1.0 else 0 for x in relative_score)/float(len(relative_score))

    return relative_score,per_of_tag_explained



   

    
#f,axes = plt.subplots(1,1,figsize=(10,10),sharex=False,sharey=False)

tag_explained = [0.0]*len(list_of_files[:-1])

score = [""]*len(list_of_files[:-1])

for i,category in enumerate(list_of_files[:-1]):

    score[i],tag_explained[i] = tagExplained(category,all_text,Fdist_all)
for i in range(0,len(list_of_files[:-1])):

    sns.distplot(score[i],label = list_of_files[i],hist=False)
#Tag prediction for test dataset

test = data[data['category'] == 'test']

vocabulary_text = to_nltk_text(test)

Fdist_test = freqDist(vocabulary_text)

relative_freq, relative_freq_dict = relativeFreq(Fdist_test,Fdist_all)
test['all_text_lemma'] = test['all_text'].apply(lambda x: [lemmatizer.lemmatize(token) for token in nltk.word_tokenize(x.replace('\n',' '))])

test['all_text_lemma2'] = test['all_text_lemma'].apply(lambda x: x+list(nltk.bigrams(x)))
import heapq

Top_N = 5

#top_Per = sum([sum(x)/len(x) for x in score])/len(score)

top_Per =0.9

print(top_Per)

def pickTheBest(l):

    result = {}

    for lemma in l:

        if lemma in relative_freq_dict:

            result[relative_freq_dict[lemma]] = lemma

    return result

test['relative_freq'] = test['all_text_lemma2'].apply(pickTheBest)



def tags(dic):

    result = heapq.nlargest(Top_N,list(dic.keys()))

    result = [dic[x] for x in result if x>=top_Per]

    result = "".join([x[0]+"-"+x[1]+" "if type(x) == tuple else x+" " for x in result])

    return result

            

    
test['tags'] = test['relative_freq'].apply(tags)
test[['all_text','tags']][0:100]
test[['id','tags']].to_csv('submission.csv',index=False)
# Visualization: How many of the tags are included in the category vocabulary?

#sns.barplot(x=list_of_files[:-1],y=tag_explained,color=sns.light_palette((210, 90, 60), input="husl"))
##all_text = to_nltk_text(data)

#biology = to_nltk_text(data[data['category'] == 'biology'])
#Fdist_all = freqDist(all_text)

#Fdist_biology = freqDist(biology)

#relative_Freq,relative_Freq_dict = relativeFreq(Fdist_biology,Fdist_all)
#

#tags=data[data['category']=='biology']['tags'].apply(lambda x: [lemmatizer.lemmatize(t) for t in nltk.word_tokenize(x)]).tolist()

#tags=[tag for record in tags for tag in record]

#freqTag = freqDist(tags,include_bigram = False)
##freqtag_df = pd.DataFrame.from_dict(freqTag,orient='index')\

                         #.reset_index()\

                         #.rename(columns={'index':'tags',0:"freq"})\

                         #.sort_values('freq',axis=0,ascending=False)
##freqtag_df['tag_revised'] = freqtag_df['tags'].apply(lambda x: (x.split('-')[0],x.split('-')[1]) if "-" in x else x)
##freqtag_df[0:100]
##freqtag_df['freq_p'] = freqtag_df['freq'].apply(lambda x: float(x)/freqtag_df.freq.sum())

#freqtag_df['relative_score'] = freqtag_df['tag_revised'].apply(lambda x: relative_Freq_dict[x] if x in relative_Freq_dict else -1.0)
#investigation = freqtag_df[freqtag_df['relative_score']==-1.0]

#print(len(investigation))

#investigation[0:50]

# The majority of unidentifiable seems to come from compounded word - maybe we should also consider bigram  