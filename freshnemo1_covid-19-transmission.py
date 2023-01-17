!pip install -U scikit-learn

!pip install pandas

!pip install wordcloud

label_name_list = ["Transmission", "Non-transmission"]

import csv

#spacy was used to preprocess the text, we add title and abstract together

litcovid_pd = pd.read_csv('/kaggle/input/litcovid/litcovid.tsv',index_col=None, sep='\t', quoting=csv.QUOTE_NONE)

print(litcovid_pd.columns)

text_list = litcovid_pd["text"]

label_list = litcovid_pd["label"]

from collections import Counter

print(Counter(label_list))

print(text_list[:2])

print(label_list[:2])

from sklearn.pipeline import Pipeline

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm

import random

seed = 1000

c = list(zip(text_list, label_list))

random.Random(seed).shuffle(c)

text_list, label_list = zip(*c)

split_length = int(len(text_list)*0.8)

train_text, test_text = text_list[:split_length], text_list[split_length:]

train_label, test_label = label_list[:split_length], label_list[split_length:]

clf_svm = svm.SVC(kernel='linear',C=2000.0, verbose=False, probability=True) #linear, rbf

sgd = Pipeline([

        ("tfidf", TfidfVectorizer(stop_words=None)),

        ("clf", clf_svm)])

sgd.fit(train_text, train_label)

predicted = sgd.predict(test_text)

print(predicted[:10])

print(metrics.classification_report(test_label, predicted,digits=4))

train_pd = litcovid_pd[litcovid_pd["label"]=="Transmission"]

print(train_pd.shape)

words = []

import nltk

from nltk.corpus import stopwords

stopwords_set = set(stopwords.words('english'))

stopwords_set.update(["covid-19","sars","coronavirus","cov-2","2019","2019-ncov","virus","2020"])

for text in list(train_pd["text"]):

    words.extend([word for word in text.split(" ") if len(word) >= 2 and word not in stopwords_set])

from collections import Counter

from wordcloud import WordCloud

words_counter = Counter(words)

wordcloud = WordCloud(width=900,height=500,background_color='white',normalize_plurals=False).generate_from_frequencies(words_counter)

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")


!pip install pandas

!pip install tqdm

!pip install spacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
import pandas as pd

import os 

import nltk

from nltk.tokenize import sent_tokenize,word_tokenize

from tqdm import tqdm

import spacy

from spacy.matcher import PhraseMatcher

import en_core_sci_sm

import holoviews as hv

from holoviews import opts, dim

hv.extension('bokeh')

hv.output(size=200)
def build_raw_data (filepath):

    allfile=pd.read_csv(filepath)

    #get alldoucment

    allfile['abstract']=allfile.abstract.astype(str)

    #get allsentence

    allsent=[]

    allid=[]

    for i in tqdm(range(len(allfile))):

        temp=allfile.abstract.iloc[i]

        temp=sent_tokenize(temp)

        for j in range(len(temp)):

            allsent.append(temp[j])

            allid.append(allfile.pid.iloc[i])

            

    allsent=pd.DataFrame(allsent,columns=['sent'])

    allsent['pid']=allid

    

    return allfile, allsent
allfile, allsent=build_raw_data ('/kaggle/input/kaggle-snorkel/NIH_ab.csv')
nlp = en_core_sci_sm.load()


with open('/kaggle/input/kaggle-snorkel/keylist2.txt', "r") as f:

    alist =f.read().splitlines()

    for line in alist:

        keylist=line.split(',')
valuelist = []

with open('/kaggle/input/kaggle-snorkel/valuelist2.txt', "r") as f:

    alist =f.read().splitlines()

    for line in alist:

        valuelist.append(line.split(','))

def matcher(nlp,keylist,valuelist):

    matcher = PhraseMatcher(nlp.vocab)

    for i in range(len(keylist)):

        valuelist1=[x.lower() for x in valuelist[i]]

        pattern=[nlp(text) for text in valuelist1]

        matcher.add(keylist[i], None, *pattern)

    return matcher
def reportfu(dataframe,matcher,nlp):

    allrule=[]

    allid=[]

    for i in tqdm(range(len(dataframe.abstract.values))):

        pid=dataframe.pid.values[i]

        doc=dataframe.abstract.values[i]

        doc=nlp(str(doc).lower())

        matches = matcher(doc)

        for match_id, start, end in matches:

            rule_id = nlp.vocab.strings[match_id]

            allrule.append(rule_id)

            allid.append(pid)

    returnframe=pd.DataFrame(allid,columns=['pid'])

    returnframe['allrule']=allrule

    returnframe['count']=[1]*len(returnframe)

    return returnframe
matchers=matcher(nlp, keylist,valuelist)


allmatchnu=reportfu(allfile,matchers,nlp)
allmatchnu=allmatchnu.drop_duplicates()

keys=list(allmatchnu.allrule.drop_duplicates().values)

values=list(range(len(keys)))

newdict=dict()

for i in range(len(keys)):

    newdict[keys[i]]=values[i]

allmatchnu['allrulec']=allmatchnu['allrule']

allmatchnu1=pd.DataFrame()

for i in keys:

    tempid=allmatchnu[allmatchnu.allrule==i].pid.values

    tempframe=allmatchnu[['allrule','allrulec','pid']]

    tempframe=tempframe[tempframe.pid.isin(tempid)]

    tempframe=tempframe.groupby(['allrulec']).count().reset_index()

    tempframe['allrule']=[i]*len(tempframe)

    allmatchnu1=pd.concat([allmatchnu1,tempframe])

allmatchnu1['allrule']=allmatchnu1.allrule.replace(newdict)

allmatchnu1['allrulec']=allmatchnu1.allrulec.replace(newdict)

allmatchnu1=allmatchnu1.rename(columns={'allrule':'target','allrulec':'source','pid':'value'})

allmatchnu1=allmatchnu1[allmatchnu1.target!=allmatchnu1.source]
newdict1=pd.DataFrame.from_dict(newdict,orient='index').reset_index()

newdict1=newdict1.rename(columns={'index':'Title',0:'index'})

newdict11 = hv.Dataset(pd.DataFrame(newdict1['Title']), 'index')

chord = hv.Chord((allmatchnu1, newdict11)).select(value=(5, None))

chord.opts(

    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 

               labels='Title', node_color=dim('index').str()))
from IPython.display import Image



Image('/kaggle/input/clamp-image/image.png')