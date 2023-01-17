#%% [markdown]

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from wordcloud import WordCloud as wc

from nltk.corpus import stopwords

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

from sklearn.ensemble import RandomForestClassifier

import sklearn

import string

import scipy

import numpy

import nltk

import json

import sys

import csv

import os

nltk.download('averaged_perceptron_tagger')

nltk.download("stopwords")

# # Version of the different libraries

print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))



from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.corpus import stopwords

data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."



print(word_tokenize(data))

print(sent_tokenize(data))



from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize,word_tokenize



words=["game","gaming","gamed","games"]

ps=PorterStemmer()



for word in words:

    print(ps.stem(word))



from nltk.tokenize import PunktSentenceTokenizer



sentences=nltk.sent_tokenize(data)

for set in sentences:

    print(nltk.pos_tag(nltk.word_tokenize(set)))



# # How to make the use of the sns i am not able to get it in poproperly

sns.set(style='white',context='notebook',palette="deep")



# # EDA

# ## I will be going to write the diffrent exploratoion technique which can be used to explore the dataset

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print('shape of the train',train.shape)

print('shape of the test',test.shape)



train.size # finding the size of the training set

type(train) # tells us about the object type

train.describe() #describe use us about the data

train.sample(5)



# # Data Cleaning

# # for finding that there is any kind of the null element is present or not(sum of the null values)

train.isnull().sum()

# # but if we have the null values used it for finding the result in the dataset

print('Before Dropping the items',train.shape)

train=train.dropna()

print('After droping',train.shape)

# # for finding the unique items for the target with command below:

# # getting all the unique from the dataset

train_target=train['target'].values

np.unique(train_target)



train.head(5)



train.tail(5)



train.describe()



# Data preprocessing refers to the transformations applied to our data before feeding it to the algorithm.



# Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis. there are plenty of steps for data preprocessing and we just listed some of them in general(Not just for Quora) :



#     removing Target column (id)

#     Sampling (without replacement)

#     Making part of iris unbalanced and balancing (with undersampling and SMOTE)

#     Introducing missing values and treating them (replacing by average values)

#     Noise filtering

#     Data discretization

#     Normalization and standardization

#     PCA analysis

#     Feature selection (filter, embedded, wrapper)

#     Etc.



# now we will be going to perfrom some queries on the dataset



train.where(train['target']==1).count()





train[train['target']>1]



train.where(train['target']==1).head(5)

# Imbalanced dataset is relevant primarily in the context of supervised machine learning involving two or more classes.

# Imbalance means that the number of data points available for different the classes is different: If there are two classes, then balanced data would mean 50% points for each of the class. For most machine learning techniques, little imbalance is not a problem. So, if there are 60% points for one class and 40% for the other class, it should not cause any significant performance degradation. Only when the class imbalance is high, e.g. 90% points for one class and 10% for the other, standard optimization criteria or performance measures may not be as effective and would need modification.

# Now  we will be going to  explore the exploreing  question

question=train['question_text']

i=0

for q in question[:5]:

        i=i+1

        print("Question came from the Quora Data_set=="+q)



train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))

# # Some Feature Engineering 

print(train.columns)

train.head()



# # Count Plot

ax=sns.countplot(x='target',hue='target',data=train,linewidth=5,edgecolor=sns.color_palette("dark",3))

plt.title('Is data set imbalance')

plt.show()



ax=train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True)

ax.set_title('target')

ax.set_ylabel('')

plt.show()



train.hist(figsize=(15,20))

plt.figure()



# #  Creating the histogram which can be used to make the 

# # Making the violin plot



sns.violinplot(data=train,x='target',y='num_words')





# # Making the kde plot

sns.FacetGrid(train,hue="target",size=5).map(sns.kdeplot,"num_words").add_legend()

plt.show()





# # Box Plot

train['num_words'].loc[train['num_words']>60]=60

axes=sns.boxplot(x='target',y='num_words',data=train)

axes.set_xlabel('Target',fontsize=12)

axes.set_title("No of words in each class",fontsize=15)

plt.show()








