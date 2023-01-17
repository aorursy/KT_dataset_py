# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
FILEPATH_TRAIN = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
df = pd.read_csv(FILEPATH_TRAIN, encoding = "ISO-8859-1")
df.sample(2)
df_toxic = df.drop(['id', 'comment_text'], axis=1)

counts = []

categories = list(df_toxic.columns.values)



for i in categories:

    counts.append((i, df_toxic[i].sum()))

df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])



df_stats
%matplotlib inline

import re

import matplotlib

import matplotlib.pyplot as plt
df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))

plt.title("Number of comments per category")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('category', fontsize=12)
import seaborn as sns
rowsums = df.iloc[:,2:].sum(axis=1)

x=rowsums.value_counts()

#plot

plt.figure(figsize=(8,5))

ax = sns.barplot(x.index, x.values)

plt.title("Multiple categories per comment")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('# of categories', fontsize=12)
print('Percentage of comments that are not labelled:')

print(len(df[(df['toxic']==0) & (df['severe_toxic']==0) & (df['obscene']==0) & (df['threat']== 0) & (df['insult']==0) & (df['identity_hate']==0)]) / len(df))
lens = df.comment_text.str.len()

lens.hist(bins = np.arange(0,5000,50))
print('Number of missing comments in comment text:')

df['comment_text'].isnull().sum()
df['comment_text'][0]
def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))

df['comment_text'][0]
from sklearn.model_selection import train_test_split
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train, test = train_test_split(df, random_state=42, test_size=0.33, shuffle=True)



X_train = train.comment_text

X_test = test.comment_text



print(X_train.shape)

print(X_test.shape)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
# Define a pipeline combining a text feature extractor with multi lable classifier

NB_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', OneVsRestClassifier(MultinomialNB(

                    fit_prior=True, class_prior=None))),

            ])



for category in categories:

    print('... Processing {}'.format(category))

    # train the model using X_dtm & y

    NB_pipeline.fit(X_train, train[category])

    # compute the testing accuracy

    prediction = NB_pipeline.predict(X_test)

    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))