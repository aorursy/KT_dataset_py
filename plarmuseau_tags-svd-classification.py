# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import seaborn as sns

import pandas as pd

import itertools 

import csv

import collections

import matplotlib.pyplot as plt



sns.set_context("paper")

%matplotlib inline



RES_DIR = "../input/"
# Load train data (skips the content column)

def load_train_data():

    categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']

    train_data = []

    for cat in categories:

        data = pd.read_csv("{}{}.csv".format(RES_DIR, cat), usecols=['id', 'title', 'tags'])

        data['category'] = cat

        train_data.append(data)

    

    return pd.concat(train_data)
train_data = load_train_data()

train_data.head()
# Distribution of questions by category

ax = train_data['category'].value_counts().plot(kind='bar')

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=30)

plt.show()
# Summary about tags

tag_lists = [t.strip().split() for t in train_data['tags'].values]

tag_lists2 = [t.strip().split() for t in train_data['title'].values]

all_tags = list(itertools.chain(*tag_lists,*tag_lists2))

tag_list_size = np.array([len(x) for x in tag_lists])

print("""The corpus is composed by {} questions. Overall {} tags have been used, of which {} unique ones. 

Average number of tags per question {:.2f} (min={}, max={}, std={:.2f})""".format(

    len(train_data),

    len(all_tags), len(set(all_tags)),

    tag_list_size.mean(), 

    min(tag_list_size), max(tag_list_size),

    tag_list_size.std()))
# Distribution of number of tags per question

sns.distplot(tag_list_size, kde=False)

sns.plt.show()
# Utility function to return top occuring tags in the passed df

def get_top_tags(df, n=None):

    itag_lists = [t.strip().split() for t in df['tags'].values]

    itag_lists2 = [t.strip().split() for t in df['title'].values]

    tags = list(itertools.chain(*itag_lists,*itag_lists2))

    top_tags = collections.Counter(list(tags)).most_common(n)

    tags, count = zip(*top_tags)

    return tags, count

# Utility function to return top occuring tags in the passed df
# Created DataFrame indexed on tags

tags_df = pd.DataFrame(index=set(itertools.chain(*tag_lists,*tag_lists2)))

# For each category create a column and update the flag to tag count

for i, (name, group) in enumerate(train_data.groupby('category')):

    tags_df[name] = 0

    tmp_index, count = get_top_tags(group)

    tmp = pd.Series(count, index=tmp_index)

    tags_df[name].update(tmp)

# Number of categories for which a tag appeared at least 1 time

tags_df['categories_appears'] = tags_df.apply(lambda x: x.astype(bool).sum(), axis=1)

tags_df['categories_appears'].value_counts()
#import the test data

test = pd.read_csv("../input/test.csv")

test.head()
# viewing the table of tags

A=tags_df

A
#Solving the question with a Singular Value Decomposition, 

#this is the core function
from numpy.linalg import inv

U,s,V=np.linalg.svd(A,full_matrices=False)

# reconstruct

S=np.diag(s)



iS=inv(S)

US=np.dot(U,iS)

US

# A fill up with US matrix

US_df=pd.DataFrame(data=US, index=tags_df.index, columns=tags_df.columns)

# with this simple math i know all the relations between all the tags and the documents

# 
#learn how to use dataframes...  and yes the algorithm knows extreme tourism antarctica has something to do with travel...

df1=US_df['extreme-tourism':'extreme-tourism':]

df2=US_df['antarctica':'antarctica':]

frames = [df1,df2]

Qtemp=pd.concat(frames).sum()

np.dot(Qtemp,V)/np.dot(np.abs(Qtemp),np.abs(V))
columns = ['biology','cooking','crypto','diy','robotics','travel']

#,'categories_appears']

data = {'biology': [0],'cooking': [0],'crypto': [0],'diy': [0],'robotics': [0],'travel': [0],'categories_appears': [0]}

newDF = pd.DataFrame(data, columns=columns,index = ['blanco'])

#print(newDF)

for xya in range(0,8926):

    temptxt = test['title'][xya] + test['content'][xya]

    tempspl = temptxt.strip().split() 

    Qtemp=newDF

    for sword in tempspl:

        if sword in US_df.index:

            #print(US_df.loc[sword:sword,:])

            Qtemp=Qtemp.append(US_df.loc[sword:sword,:])

            #print(Qtemp)

    simila=np.dot(Qtemp.sum(),V)/np.dot(np.abs(Qtemp.sum()),np.abs(V))

    tempprnt=''

    for xyb in range(0,5):

        if simila[xyb]>0.89 or simila[xyb]==np.amax(simila[0:5]):

            tempprnt+=columns[xyb]+' '

    

    print(test['id'][xya],',',tempprnt)



    

    

 