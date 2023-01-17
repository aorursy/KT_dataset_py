from sklearn import *

import sklearn

import pandas as pd

import numpy as np

import sys

train = pd.read_csv('../input/training_variants')

test = pd.read_csv('../input/test_variants')

train_text = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])



print('train',train.shape)

print('test',test.shape)

total_unique_gene=len(train.Gene.unique())+len(test.Gene.unique())

total_unique_variation=len(train.Variation.unique())+len(test.Variation.unique())

print("*"*100)



print('Test data has ',100*(test.shape[0]-train.shape[0])/train.shape[0],'% more data')

print("*"*100)



print('Count of unique train Genes',len(train.Gene.unique()))

print('Count of unique train Variation',len(train.Variation.unique()))

print("*"*100)



print('Count of unique test Genes',len(test.Gene.unique()))

print('Count of unique test Variation',len(test.Variation.unique()))



print("*"*100)

print("Number of unique genes in the train and test dataset" ,total_unique_gene)

print("Number of unique variation in the train and test dataset" ,total_unique_variation)

print("*"*100)



print('Percentage of genes that are in the train set but not in test set and viceversa',

      len(set(train.Gene.unique()).symmetric_difference(set(test.Gene.unique())))*100/total_unique_gene)

print('Percentage of variation that are in the train set but not in test set and viceversa',

      len(set(train.Variation.unique()).symmetric_difference(set(test.Variation.unique())))*100/total_unique_variation)



print("*"*100)



print('Intersection of train and test Genes',len(set(train.Gene.unique()).intersection(set(test.Gene.unique())))*100/total_unique_gene)



print('Intersection of train and test Variation',len(set(train.Variation.unique()).intersection(set(test.Variation.unique())))*100/total_unique_variation)

print('Text values with null values in training set',train_text.loc[train_text.Text=='null','Text'].shape[0])

print('Text values with null values in testing  set',test_text.loc[test_text.Text=='null','Text'].shape[0])

train.isnull().sum()
train.groupby('Gene').ID.count().sort_values(ascending=False).head()
test.groupby('Gene').ID.count().sort_values(ascending=False).head()
train.groupby('Variation').ID.count().sort_values(ascending=False).head()
test.groupby('Variation').ID.count().sort_values(ascending=False).head()
%matplotlib inline

df=(train.Gene.value_counts(normalize=True)*100)

df[df>1].plot(kind='barh',title='% of GENES in training set')
df=(test.Gene.value_counts(normalize=True)*100)

df[df>=0.5].plot(kind='barh',title='% of GENES in test set')
num_common_genes=set(train.Gene).intersection(set(test.Gene))



print( 'Common genes in train and test {0}'.format(len(num_common_genes)))

train.Class.value_counts(normalize=True).plot(kind='bar',title='Class distribution in train set')
import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = (20,5)



train['Gene_Length']=train.Gene.apply(lambda x:len(x))

pd.crosstab(train.Class,train.Gene_Length).plot(kind='bar',title='Class distribution versus Gene length')
train_top=train.Gene.value_counts(normalize=True)*100

top_genes=train_top[train_top>1].index

uq=train.Class.unique()

train_top_genes=train[train.Gene.isin(top_genes)]

topGenes=train_top_genes.Gene.unique()



test_top_genes=test[test.Gene.isin(top_genes)]

print(train_top_genes.Gene.unique())

print(test_top_genes.Gene.unique())
#For every class find out what is the most predominant genes

plt.rcParams["figure.figsize"] = (20,10)

import pylab

row=0

col=0

fig, axes = plt.subplots(nrows=3, ncols=3)

#pylab.gca().get_ylabel().set_fontsize(10)

for i in range(0,len(uq)):

    ax1=axes[row,col]

    train_top_genes[train_top_genes.Class==uq[i]].Gene.value_counts().reindex(topGenes).plot(kind='bar',ax=ax1,title=uq[i])

    plt.subplots_adjust(top=1.3)

    ax1.set_yticks(np.arange(0,105, 5))



    col=col+1

    if(col==3):

        col=0

        row=row+1

        

        
plt.rcParams["figure.figsize"] = (10,10)

fig, axes = plt.subplots(nrows=11, ncols=2)



uq_genes=train_top_genes.Gene.unique()

row=0

col=0

for i in range(0,len(uq_genes)):

    ax1=axes[row,col]



    train_top_genes[train_top_genes.Gene==uq_genes[i]].Class.value_counts().reindex(uq).plot(kind='bar',ax=ax1,title=uq_genes[i])

    col=col+1

    if(col==2):

        col=0

        row=row+1

    plt.subplots_adjust(top=2.5)



        



from sklearn import preprocessing 

train_top_genes['sod']='train'

test_top_genes['sod']='test'



df1=train_top_genes.loc[:,['ID','Gene','Variation','sod']].append(test_top_genes)

le = preprocessing.LabelEncoder()

var=le.fit_transform(df1.Variation)

le = preprocessing.LabelEncoder()

var=le.fit_transform(df1.Variation)

df1['Int_var']=var

df1.head()
train_top_genes=df1[df1.sod=='train'].sort_values(by='Gene')

test_top_genes=df1[df1.sod=='test'].sort_values(by='Gene')

df=train_top_genes.groupby(['Gene','Int_var'],as_index=False).ID.count()

df.head()
import seaborn as sns

sns.set(color_codes=True)

plt.rcParams["figure.figsize"] = (20,5)



sns.stripplot(x="Gene", y="Int_var", data=train_top_genes)

sns.plt.suptitle('Train title versus variation ')

sns.stripplot(x="Gene", y="Int_var", data=test_top_genes)

sns.plt.suptitle('Test title versus variation ')



train=train.merge(train_text)

test=test.merge(test_text)

print(train.shape)

print(test.shape)

train.Text.head()
train.iloc[0].Text
train['length_str']=train.Text.apply(lambda x:len(x))

test['length_str']=test.Text.apply(lambda x:len(x))

train['word_count_str']=train.Text.apply(lambda x:len(x.split()))

test['word_count_str']=test.Text.apply(lambda x:len(x.split()))

train.head()


sns.distplot(test.word_count_str,label='test')

sns.distplot(train.word_count_str,label='train')

plt.legend(loc='upper right')
plt.rcParams["figure.figsize"] = (10,5)



train.groupby('Class').word_count_str.median().plot(kind='bar',title='Word_Count by class')
#For every class find out what is the most predominant genes

plt.rcParams["figure.figsize"] = (20,10)

import pylab

row=0

col=0

fig, axes = plt.subplots(nrows=3, ncols=3)



#pylab.gca().get_ylabel().set_fontsize(10)

for i in range(0,len(uq)):

    ax1=axes[row,col]

    ax1.set_ylim(0,0.000175)

    ax1.set_title('Class '+str(i), fontsize =16)

    sns.kdeplot(train.loc[train.Class==uq[i],'word_count_str'],ax=ax1,shade=True, color="r")



    plt.subplots_adjust(top=1.3)



    col=col+1

    if(col==3):

        col=0

        row=row+1

        

        
test.word_count_str.describe()
train.word_count_str.describe()


#Packages Imports

import matplotlib.pyplot as plt

import nltk

import numpy as np

import pandas as pd

import re

from collections import Counter

from nltk.stem import WordNetLemmatizer

from os import path

from PIL import Image

from wordcloud import WordCloud, ImageColorGenerator



from nltk.tokenize import sent_tokenize



from nltk import word_tokenize

from nltk.corpus import stopwords

stop = stopwords.words('english')





train['txt']=train.Text.apply(lambda x:' '.join(i for i in x.lower().split() if i not in stop))



df=train.groupby('Class').txt.sum()

df.head()
%matplotlib inline

plt.rcParams["figure.figsize"] = (10,10)

import pylab



#pylab.gca().get_ylabel().set_fontsize(10)

for i in range(0,len(uq)):

    wordcloud =WordCloud(background_color='black',  max_font_size=50, max_words=100).generate(df.loc[uq[i]])

    plt.imshow(wordcloud)

    plt.title('Class '+str(i))

    plt.axis("off")

    plt.show()





    
plt.rcParams["figure.figsize"] = (20,5)

import pylab

row=0

col=0

fig, axes = plt.subplots(nrows=3, ncols=3)



#pylab.gca().get_ylabel().set_fontsize(10)

for i in range(0,len(uq)):

    ax1=axes[row,col]

    ax1.set_ylim(0,70000)

    ax1.set_title('Class '+str(i), fontsize =16)





    pd.Series(pd.Series(df.loc[uq[i]]).str.split()[0]).value_counts().sort_values(ascending=False)[:50].plot(kind='barh',ax=ax1)

    plt.subplots_adjust(top=3)



    col=col+1

    if(col==3):

        col=0

        row=row+1