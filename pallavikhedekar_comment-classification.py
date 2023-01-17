# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#basics

import pandas as pd 

import numpy as np



#misc

import gc

import time

import warnings

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing the dataset

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

train.shape
train.tail(5)
test.tail(5)
train['comment_text'][0]
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)

train.describe()
len(train),len(test)
print("Check for missing values in Train dataset")

null_check=train.isnull().sum()

print(null_check)

print("Check for missing values in Test dataset")

null_check=test.isnull().sum()

print(null_check)

print("filling NA with \"unknown\"")

train["comment_text"].fillna("unknown", inplace=True)

test["comment_text"].fillna("unknown", inplace=True)
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

# import matplotlib.gridspec as gridspec 

import seaborn as sns

# from wordcloud import WordCloud ,STOPWORDS

# from PIL import Image

# import matplotlib_venn as venn



x=train.iloc[:,2:].sum()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)

#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
x=train.iloc[:,2:].sum()

#marking comments without any tags as "clean"

rowsums=train.iloc[:,2:].sum(axis=1)

x=rowsums.value_counts()



#plot

plt.figure(figsize=(8,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[2])

plt.title("Multiple tags per comment")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of tags ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
temp_df=train.iloc[:,2:-1]

# filter temp by removing clean comments

# temp_df=temp_df[~train.clean]



corr=temp_df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)

main_col="toxic"

corr_mats=[]

for other_col in temp_df.columns[1:]:

    confusion_matrix = pd.crosstab(temp_df[main_col], temp_df[other_col])

    corr_mats.append(confusion_matrix)

out = pd.concat(corr_mats,axis=1,keys=temp_df.columns[1:])



def highlight_min(s):

    '''   highlight the maximum in a Series yellow.    '''

    is_max = s == s.min()

    return ['background-color: yellow' if v else '' for v in is_max]



#cell highlighting

out = out.style.apply(highlight_min,axis=0)

out
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
train['comment_text'][:20]
import re

def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"don't", "donot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = re.sub('\::+', ' ', text)

    text = text.strip(' ')

    return text
train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
test['comment_text'] = test['comment_text'].map(lambda com : clean_text(com))
train.columns
X_train = train.comment_text

X_test = test.comment_text



print(X_train.shape , X_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=5000,stop_words='english')

vect
X_dtm_train = vect.fit_transform(X_train)

X_dtm_test = vect.fit_transform(X_test)

X_dtm_train
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=10.0)



# create submission file

submission_binary = pd.read_csv('../input/sample_submission.csv')



for label in cols_target:

    print('... Processing {}'.format(label))

    y = train[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm_train, y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm_train)

    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))

    # compute the predicted probabilities for X_test_dtm

    test_y_prob = logreg.predict_proba(X_dtm_test)[:,1]

    submission_binary[label] = test_y_prob
submission_binary.head()
submission_binary.to_csv('submission_binary.csv',index=False)