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
import os

import csv

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

#printmd('**bold**')
data_path="/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv"
data_raw = pd.read_csv(data_path)
data_raw.head()
print("Number of rows in data =",data_raw.shape[0])

print("Number of columns in data =",data_raw.shape[1])

print("\n")

printmd("**Sample data:**")

data_raw.head()


missing_values_check = data_raw.isnull().sum()

print(missing_values_check)



# Comments with no label are considered to be clean comments.

# Creating seperate column in dataframe to identify clean comments.



# We use axis=1 to count row-wise and axis=0 to count column wise



rowSums = data_raw.iloc[:,2:].sum(axis=1)

clean_comments_count = (rowSums==0).sum(axis=0)



print("Total number of comments = ",len(data_raw))

print("Number of clean comments = ",clean_comments_count)

print("Number of comments with labels =",(len(data_raw)-clean_comments_count))
categories = list(data_raw.columns.values)

categories = categories[2:]

print(categories)
# Calculating number of comments in each category



counts = []

for category in categories:

    counts.append((category, data_raw[category].sum()))

df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])

df_stats
sns.set(font_scale = 2)

plt.figure(figsize=(15,8))



ax= sns.barplot(categories, data_raw.iloc[:,2:].sum().values)



plt.title("Comments in each category", fontsize=24)

plt.ylabel('Number of comments', fontsize=18)

plt.xlabel('Comment Type ', fontsize=18)



#adding the text labels

rects = ax.patches

labels = data_raw.iloc[:,2:].sum().values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)



plt.show()
rowSums = data_raw.iloc[:,2:].sum(axis=1)

multiLabel_counts = rowSums.value_counts()

multiLabel_counts = multiLabel_counts.iloc[1:]



sns.set(font_scale = 2)

plt.figure(figsize=(15,8))



ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)



plt.title("Comments having multiple labels ")

plt.ylabel('Number of comments', fontsize=18)

plt.xlabel('Number of labels', fontsize=18)



#adding the text labels

rects = ax.patches

labels = multiLabel_counts.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
from wordcloud import WordCloud,STOPWORDS



plt.figure(figsize=(40,25))



# toxic

subset = data_raw[data_raw.toxic==1]

text = subset.comment_text.values

cloud_toxic = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 1)

plt.axis('off')

plt.title("Toxic",fontsize=40)

plt.imshow(cloud_toxic)





# severe_toxic

subset = data_raw[data_raw.severe_toxic==1]

text = subset.comment_text.values

cloud_severe_toxic = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 2)

plt.axis('off')

plt.title("Severe Toxic",fontsize=40)

plt.imshow(cloud_severe_toxic)





# obscene

subset = data_raw[data_raw.obscene==1]

text = subset.comment_text.values

cloud_obscene = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 3)

plt.axis('off')

plt.title("Obscene",fontsize=40)

plt.imshow(cloud_obscene)





# threat

subset = data_raw[data_raw.threat==1]

text = subset.comment_text.values

cloud_threat = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 4)

plt.axis('off')

plt.title("Threat",fontsize=40)

plt.imshow(cloud_threat)





# insult

subset = data_raw[data_raw.insult==1]

text = subset.comment_text.values

cloud_insult = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 5)

plt.axis('off')

plt.title("Insult",fontsize=40)

plt.imshow(cloud_insult)





# identity_hate

subset = data_raw[data_raw.identity_hate==1]

text = subset.comment_text.values

cloud_identity_hate = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          collocations=False,

                          width=2500,

                          height=1800

                         ).generate(" ".join(text))



plt.subplot(2, 3, 6)

plt.axis('off')

plt.title("Identity Hate",fontsize=40)

plt.imshow(cloud_identity_hate)



plt.show()

data = data_raw

data = data_raw.loc[np.random.choice(data_raw.index, size=2000)]

data.shape
data.shape
import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
def cleanHtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', str(sentence))

    return cleantext





def cleanPunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    cleaned = cleaned.strip()

    cleaned = cleaned.replace("\n"," ")

    return cleaned





def keepAlpha(sentence):

    alpha_sent = ""

    for word in sentence.split():

        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)

        alpha_sent += alpha_word

        alpha_sent += " "

    alpha_sent = alpha_sent.strip()

    return alpha_sent
data['comment_text'] = data['comment_text'].str.lower()

data['comment_text'] = data['comment_text'].apply(cleanHtml)

data['comment_text'] = data['comment_text'].apply(cleanPunc)

data['comment_text'] = data['comment_text'].apply(keepAlpha)

data.head()
stop_words = set(stopwords.words('english'))

stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])

re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):

    global re_stop_words

    return re_stop_words.sub(" ", sentence)



data['comment_text'] = data['comment_text'].apply(removeStopWords)

data.head()


stemmer = SnowballStemmer("english")

def stemming(sentence):

    stemSentence = ""

    for word in sentence.split():

        stem = stemmer.stem(word)

        stemSentence += stem

        stemSentence += " "

    stemSentence = stemSentence.strip()

    return stemSentence



data['comment_text'] = data['comment_text'].apply(stemming)

data.head()
from sklearn.model_selection import train_test_split



train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)



print(train.shape)

print(test.shape)


train_text = train['comment_text']

test_text = test['comment_text']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')

vectorizer.fit(train_text)

vectorizer.fit(test_text)


x_train = vectorizer.transform(train_text)

y_train = train.drop(labels = ['id','comment_text'], axis=1)



x_test = vectorizer.transform(test_text)

y_test = test.drop(labels = ['id','comment_text'], axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.multiclass import OneVsRestClassifier


%%time



# Using pipeline for applying logistic regression and one vs rest classifier

LogReg_pipeline = Pipeline([

                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),

            ])



for category in categories:

    printmd('**Processing {} comments...**'.format(category))

    

    # Training logistic regression model on train data

    LogReg_pipeline.fit(x_train, train[category])

    

    # calculating test accuracy

    prediction = LogReg_pipeline.predict(x_test)

    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

    print("\n")
%%time



# using binary relevance

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB



# initialize binary relevance multi-label classifier

# with a gaussian naive bayes base classifier

classifier = BinaryRelevance(GaussianNB())



# train

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)



# accuracy

print("Accuracy = ",accuracy_score(y_test,predictions))

print("\n")
# using classifier chains

from skmultilearn.problem_transform import ClassifierChain

from sklearn.linear_model import LogisticRegression
%%time



# initialize classifier chains multi-label classifier

classifier = ClassifierChain(LogisticRegression())



# Training logistic regression model on train data

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)



# accuracy

print("Accuracy = ",accuracy_score(y_test,predictions))

print("\n")
# using Label Powerset

from skmultilearn.problem_transform import LabelPowerset


%%time



# initialize label powerset multi-label classifier

classifier = LabelPowerset(LogisticRegression())



# train

classifier.fit(x_train, y_train)



# predict

predictions = classifier.predict(x_test)



# accuracy

print("Accuracy = ",accuracy_score(y_test,predictions))

print("\n")
# http://scikit.ml/api/api/skmultilearn.adapt.html#skmultilearn.adapt.MLkNN



from skmultilearn.adapt import MLkNN

from scipy.sparse import csr_matrix, lil_matrix
%%time



classifier_new = MLkNN(k=10)



# Note that this classifier can throw up errors when handling sparse matrices.



x_train = lil_matrix(x_train).toarray()

y_train = lil_matrix(y_train).toarray()

x_test = lil_matrix(x_test).toarray()



# train

classifier_new.fit(x_train, y_train)



# predict

predictions_new = classifier_new.predict(x_test)



# accuracy

print("Accuracy = ",accuracy_score(y_test,predictions_new))

print("\n")
 