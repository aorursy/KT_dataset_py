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
import pandas as pd





df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')

df.head()
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

df = df.rename(columns={"v1":"labels", "v2":"text"})

df.head()
df.describe()
df.labels.value_counts()
df.labels.value_counts().plot.bar()
# Replacing spam with 1 and ham with 0

df['spam']=df['labels']

for i,j in df.iterrows():

    # i is index

    # j is (labels, text)

    if j['labels']=='ham':

        j['spam'] = 0

    else:

        j['spam']=1
df.head()
import string

print(string.punctuation)
from nltk.corpus import stopwords

print(stopwords.words('english')[10:15])
def punctuation_stopwords_removal(sms):

    # filters charecter-by-charecter : ['h', 'e', 'e', 'l', 'o', 'o', ' ', 'm', 'y', ' ', 'n', 'a', 'm', 'e', ' ', 'i', 's', ' ', 'p', 'u', 'r', 'v', 'a']

    remove_punctuation = [ch for ch in sms if ch not in string.punctuation]

    # convert them back to sentences and split into words

    remove_punctuation = "".join(remove_punctuation).split()

    filtered_sms = [word.lower() for word in remove_punctuation if word.lower() not in stopwords.words('english')]

    return filtered_sms
print(punctuation_stopwords_removal("Hello we need to send this report by EOD.!!! yours sincerely, Purva"))
print(df.head())
from collections import Counter



data_ham = df[df['spam']==0].copy()

data_spam = df[df['spam']==1].copy()

print(data_ham[:2])

print(data_spam[:2])
data_ham.loc[:, 'text'] = data_ham['text'].apply(punctuation_stopwords_removal)

print(data_ham[:1])
words_data_ham = data_ham['text'].tolist()
words_data_ham[:3]
data_spam.loc[:, 'text']=data_spam['text'].apply(punctuation_stopwords_removal)

print(data_spam[:1])

#words_data_spam = data_spam['text'].tolist()
words_data_spam = data_spam['text'].tolist()

print(words_data_spam[:2])
ham_list = []

for sublist in words_data_ham:

    for word in sublist:

        ham_list.append(word)



spam_list = []

for sublist in words_data_spam:

    for word in sublist:

        spam_list.append(word)
ham_count = Counter(ham_list)

spam_count = Counter(spam_list)



ham_top_30_words = pd.DataFrame(ham_count.most_common(30), columns=['word', 'count'])

spam_top_30_words = pd.DataFrame(spam_count.most_common(30), columns=['word', 'count'])
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='word', y='count', 

            data=ham_top_30_words, ax=ax)

plt.title("Top 30 Ham words")

plt.xticks(rotation='vertical');


fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='word', y='count', 

            data=spam_top_30_words, ax=ax)

plt.title("Top 30 Spam words")

plt.xticks(rotation='vertical');
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=punctuation_stopwords_removal).fit(df['text'])
len(bow_transformer.vocabulary_)
sample_spam = df['text'][8]

bow_sample_spam = bow_transformer.transform([sample_spam])

print(sample_spam)

print(bow_sample_spam)
print('Printing bag of words for sample 1')

row, cols = bow_sample_spam.nonzero()

for col in cols:

    print(bow_transformer.get_feature_names()[col])
import numpy as np

print(np.shape(bow_sample_spam))
sample_ham = df['text'][4]

bow_sample_ham = bow_transformer.transform([sample_ham])

print(sample_ham)

print(bow_sample_ham)

rows, cols = bow_sample_ham.nonzero()

print('Printing ')

for col in cols:

    print(bow_transformer.get_feature_names()[col])
from sklearn.feature_extraction.text import TfidfTransformer



# bag of words in vectorized format

bow_data = bow_transformer.transform(df['text'])

print(bow_data[:1])

tfidf_transformer = TfidfTransformer().fit(bow_data)
tfidf_sample_ham = tfidf_transformer.transform(bow_sample_ham)

print('Sample HAM : ')

print(tfidf_sample_ham)



tfidf_sample_spam = tfidf_transformer.transform(bow_sample_spam)

print('Sample SPAM : ')

print(tfidf_sample_spam)
final_data_tfidf = tfidf_transformer.transform(bow_data)

print(final_data_tfidf)

print(np.shape(final_data_tfidf))
from sklearn.model_selection import train_test_split



data_tfidf_train, data_tfidf_test, label_train, label_test = train_test_split(final_data_tfidf, df["spam"], test_size=0.3, random_state=5)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import seaborn as sns



def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    #fig, ax = plt.subplots(figsize=(4,4))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  

                cmap="Blues", square=True, cbar=False)

    #  

    plt.ylabel('true label')

    plt.xlabel('predicted label')
data_tfidf_train = data_tfidf_train.A

data_tfidf_test = data_tfidf_test.A
print(data_tfidf_train.dtype)

print(label_train.dtype)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



spam_detect_model_MNB = MultinomialNB()

spam_detect_model_MNB.fit(data_tfidf_train, np.asarray(label_train, dtype="float64"))

pred_test_MNB = spam_detect_model_MNB.predict(data_tfidf_test)

acc_MNB = accuracy_score(np.asarray(label_test, dtype="float64"), pred_test_MNB)

print(acc_MNB)
from sklearn.metrics import roc_curve, auc



fpr, tpr, thr = roc_curve(np.asarray(label_test, dtype="float64"), spam_detect_model_MNB.predict_proba(data_tfidf_test)[:,1])

plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc_knn4 = auc(fpr, tpr) * 100

plt.legend(["AUC {0:.3f}".format(auc_knn4)]);
def plot_confusion_matrix(y_true, y_pred):

    mtx = confusion_matrix(y_true, y_pred)

    #fig, ax = plt.subplots(figsize=(4,4))

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  

                cmap="Blues", square=True, cbar=False)

    #  

    plt.ylabel('true label')

    plt.xlabel('predicted label')
plot_confusion_matrix(np.asarray(label_test, dtype="float64"), pred_test_MNB)