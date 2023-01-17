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
data_path = "../input/janatahack-independence-day-2020-ml-hackathon/train.csv"
test_data_path = "../input/janatahack-independence-day-2020-ml-hackathon/test.csv"
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

rowSums = data_raw.iloc[:,3:].sum(axis=1)
clean_comments_count = (rowSums==0).sum(axis=0)

print("Total number of comments = ",len(data_raw))
print("Number of clean comments = ",clean_comments_count)
print("Number of comments with labels =",(len(data_raw)-clean_comments_count))
categories = list(data_raw.columns.values)
categories = categories[3:]
print(categories)
counts = []
for category in categories:
    counts.append((category, data_raw[category].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number of comments'])
df_stats
df_stats.category.tolist()
import textwrap

sns.set(font_scale = 2)
plt.figure(figsize=(15,8))

ax= sns.barplot(categories, data_raw.iloc[:,3:].sum().values)

plt.title("Comments in each category", fontsize=24)
plt.ylabel('Number of comments', fontsize=18)
plt.xlabel('Comment Type ', fontsize=18)

#adding the text labels
rects = ax.patches
labels = data_raw.iloc[:,3:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

labels = df_stats.category.tolist()
# labels.sort()
labels=[textwrap.fill(text,12) for text in labels]
pos = np.arange(len(labels)) 
plt.xticks(pos, labels)

plt.show()
data_raw.iloc[:,3:]
rowSums = data_raw.iloc[:,3:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[:]

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
data_raw.head()
# Combine the Title and Abstract data
data_raw['TEXT'] = data_raw['TITLE'].map(str) + data_raw['ABSTRACT'].map(str)
from wordcloud import WordCloud,STOPWORDS

plt.figure(figsize=(40,25))

# Computer Science
subset = data_raw[data_raw['Computer Science']==1]
text = subset.TEXT.values
cloud_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("Computer Science",fontsize=40)
plt.imshow(cloud_toxic)


# Physics
subset = data_raw[data_raw.Physics==1]
text = subset.TEXT.values
cloud_severe_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 2)
plt.axis('off')
plt.title("Physics",fontsize=40)
plt.imshow(cloud_severe_toxic)


# Mathematics
subset = data_raw[data_raw.Mathematics==1]
text = subset.TEXT.values
cloud_obscene = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 3)
plt.axis('off')
plt.title("Mathematics",fontsize=40)
plt.imshow(cloud_obscene)


# Statistics		
subset = data_raw[data_raw.Statistics==1]
text = subset.TEXT.values
cloud_threat = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 4)
plt.axis('off')
plt.title("Statistics",fontsize=40)
plt.imshow(cloud_threat)


# Quantitative Biology
subset = data_raw[data_raw['Quantitative Biology']==1]
text = subset.TEXT.values
cloud_insult = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 5)
plt.axis('off')
plt.title("Quantitative Biology",fontsize=40)
plt.imshow(cloud_insult)


# Quantitative Finance
subset = data_raw[data_raw['Quantitative Finance']==1]
text = subset.TEXT.values
cloud_identity_hate = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

plt.subplot(2, 3, 6)
plt.axis('off')
plt.title("Quantitative Finance",fontsize=40)
plt.imshow(cloud_identity_hate)

plt.show()
data_raw.head()
test = pd.read_csv(test_data_path)
# Combine the Title and Abstract data
test['TEXT'] = test['TITLE'].map(str) + test['ABSTRACT'].map(str)
print(test.shape)
print("\n")
test.head()
data_raw['split']= 1
test['split']=0
data = pd.concat([data_raw,test])
data.info()

# data = data_raw
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
data['TEXT'] = data['TEXT'].str.lower()
data['TEXT'] = data['TEXT'].apply(cleanHtml)
data['TEXT'] = data['TEXT'].apply(cleanPunc)
data['TEXT'] = data['TEXT'].apply(keepAlpha)
data.head()
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data['text'] = data['TEXT'].apply(removeStopWords)
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

data['text'] = data['text'].apply(stemming)
data.head()
test_data = data[data['split']==0]
test_data.shape
train_df = data[data['split']==1]
train_df.columns
train_df['Computer Science']=train_df['Computer Science'].astype(int)
train_df['Physics']=train_df['Physics'].astype(int)
train_df['Mathematics']=train_df['Mathematics'].astype(int)
train_df['Statistics']=train_df['Statistics'].astype(int)
train_df['Quantitative Biology']=train_df['Quantitative Biology'].astype(int)
train_df['Quantitative Finance']=train_df['Quantitative Finance'].astype(int)
train_df.head()
train_df.shape
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_df, random_state=42, test_size=0.30, shuffle=True)

print(train.shape)
print(test.shape)
train_text = train['text']
test_text = test['text']  #test data from the train as validation 
test_df = test_data['text'] # test data to predict 
train_text.head()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)
vectorizer.fit(test_df)
x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['ID','TITLE','ABSTRACT','TEXT','text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['ID','TITLE','ABSTRACT','TEXT','text'], axis=1)

test_to_predict  = vectorizer.transform(test_df)
print(x_train.shape)
print("="*20)
print(x_test.shape)
print("="*20)
print(test_to_predict.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
%%time

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])


predictions = []
accuracy_list = []
for category in categories:
    printmd('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    accuracy = accuracy_score(test[category], prediction)
    predictions.append(prediction)
    accuracy_list.append(accuracy)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n")
predictions
sample = pd.read_csv('../input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
sample.shape
%%time

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])


predictions = []
# accuracy_list = []
for category in categories:
    printmd('**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(test_to_predict)
#     accuracy = accuracy_score(test[category], prediction)
    predictions.append(prediction)
#     accuracy_list.append(accuracy)
#     print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("Complete....")
    print("\n")
predictions
sample['Computer Science']=predictions[0]
sample['Physics']=predictions[1]
sample['Mathematics']=predictions[2]
sample['Statistics']=predictions[3]
sample['Quantitative Biology']=predictions[4]
sample['Quantitative Finance']=predictions[5]
sample.head()
sample.to_csv("submission.csv",index=False)