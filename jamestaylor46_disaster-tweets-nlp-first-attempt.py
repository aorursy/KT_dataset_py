import os
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import string

import re

!pip install pyspellchecker

from spellchecker import SpellChecker



from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression





train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# Printing the head of the DataFrame to get an overview of what it looks like

print(train_df.head())
print(train_df.info())

print(test_df.info())
sns.set_style('whitegrid')

sns.countplot(x='target', data=train_df)



train_df['target'].value_counts(normalize='True')
train_df['word_count']=train_df['text'].str.split().map(lambda x: len(x))

train_df['char_count']=train_df['text'].str.len()



grid = sns.FacetGrid(train_df, col='target')



grid.map(plt.hist, 'word_count')



print('The average word count for Non-Disaster tweets is {}'.format(train_df[train_df['target']==0]['word_count'].mean()))

print('The average word count for Disaster tweets is {}'.format(train_df[train_df['target']==1]['word_count'].mean()))
grid = sns.FacetGrid(train_df, col='target')



grid.map(plt.hist, 'char_count')



print('The average character count for Non-Disaster tweets is {}'.format(train_df[train_df['target']==0]['char_count'].mean()))

print('The average character count for Disaster tweets is {}'.format(train_df[train_df['target']==1]['char_count'].mean()))
# Removing Punctuation

def remove_punctuation(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



train_df['text']=train_df['text'].apply(lambda x : remove_punctuation(x))

test_df['text']=test_df['text'].apply(lambda x : remove_punctuation(x))





# Removing HTML tags

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



train_df['text']=train_df['text'].apply(lambda x : remove_html(x))

test_df['text']=test_df['text'].apply(lambda x : remove_html(x))





# Removing URLs

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



train_df['text']=train_df['text'].apply(lambda x : remove_URL(x))

test_df['text']=test_df['text'].apply(lambda x : remove_URL(x))





# Correct Spelling

spell = SpellChecker()



def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)





#train_df['text']=train_df['text'].apply(lambda x : correct_spellings(x))

#test_df['text']=test_df['text'].apply(lambda x : correct_spellings(x))
X_train, X_test, y_train, y_test = train_test_split(train_df['text'], train_df['target'], random_state=1)
pl = Pipeline([

        ('vec', CountVectorizer()),

        ('clf', LogisticRegression())

    ])
pl.fit(X_train, y_train)



accuracy = pl.score(X_test, y_test)



print(accuracy)
# Finally we'll input our predictions into the sample submission and submit to Kaggle for final scoring



submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



submission["target"] = pl.predict(test_df['text'])



submission.to_csv("submission.csv", index=False)