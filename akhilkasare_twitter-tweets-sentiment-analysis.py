# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

import string

%matplotlib inline 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding ="ISO-8859-1" ,

                 names=["target", "ids", "date", "flag", "user", "text"])
dataset.head()
dataset.info()
dataset.describe(include = 'O')
# Printing the length of the dataset

print("Dataset length : {}".format(len(dataset)))
print("Dataset shape : {}".format(dataset.shape))
#Checking for null values

dataset.isnull().any()
dataset.isnull().sum()
# Checking the target values

dataset.target.value_counts()
g = sns.countplot(dataset['target'], data=dataset)

g.set_xticklabels(["Negative", "Positive"], rotation=0)
dataset.user.value_counts()
import re
def remove_noise(text):

    # Dealing with Punctuation

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
# Applying the remove_noise function on the dataset



dataset['text'] = dataset['text'].apply(lambda x : remove_noise(x))
dataset.head()
dataset['text'] = dataset['text'].apply(lambda x : x.lower())
dataset.head()
dataset['text'][12314]
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



stop = stopwords.words('english')
# Removing stopwords



def remove_stopwords(text):

    text = [item for item in text.split() if item not in stop]

    return ' '.join(text)



dataset['cleaned_data'] = dataset['text'].apply(remove_stopwords)
dataset.head()
from nltk.stem.porter import PorterStemmer



stemmer = PorterStemmer()



def stemming(text):

    text = [stemmer.stem(word) for word in text.split()]

    return ' '.join(text)



dataset['stemed_text'] = dataset['cleaned_data'].apply(stemming)



dataset.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt



fig, (ax1) = plt.subplots(1, figsize=[7, 7])

wordcloud = WordCloud( background_color='white', width=600, height=600).generate(" ".join(dataset['stemed_text']))



ax1.imshow(wordcloud)

ax1.axis('off')

ax1.set_title('Frequent Words',fontsize=16);
from sklearn.feature_extraction.text import TfidfVectorizer





tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=500000)



x = tfidf.fit_transform(dataset['stemed_text'])
tfidf.get_feature_names()[:20]
tfidf.get_params()
print(x[0].todense())
y = dataset['target']
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    See full source and example: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
from sklearn.model_selection import train_test_split

import itertools

import numpy as np

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, classification_report



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
from sklearn.naive_bayes import MultinomialNB



multinb = MultinomialNB(alpha=1.9)

multinb.fit(x_train, y_train)



multi_predict = multinb.predict(x_test)

multinb_accuracy_score = accuracy_score(y_test, multi_predict)

print("The Accuracy score for MultinomialNB is : {}".format(multinb_accuracy_score))



multinb_conf_mat = confusion_matrix(y_test, multi_predict)

plot_confusion_matrix(multinb_conf_mat, classes = ['FAKE', 'REAL'])
%%time



multinb_classifier = MultinomialNB(alpha=0.1)



previous_score = 0



# We are taking values from 0 to 1 with an increament of 0.1 



for alpha in np.arange(0,2,0.1):

    sub_classifier = MultinomialNB(alpha=alpha)

    sub_classifier.fit(x_train, y_train)

    y_pred = sub_classifier.predict(x_test)

    score = accuracy_score(y_test, y_pred)

    

    if score> previous_score:

        classifier = sub_classifier

        print("Alpha is : {} & Accuracy is : {}".format(alpha, score))
from sklearn.naive_bayes import BernoulliNB



bernoullinb = BernoulliNB(alpha=2)

bernoullinb.fit(x_train, y_train)



bernoulli_pred = bernoullinb.predict(x_test)



bernoulli_acc_score = accuracy_score(y_test, bernoulli_pred)

print("The Accuracy score for BernoulliNB is {} : ".format(bernoulli_acc_score))



print("=======================================================================================")



bernoullinb_conf_mat = confusion_matrix(y_test, bernoulli_pred)

plot_confusion_matrix(bernoullinb_conf_mat, classes = ['FAKE', 'REAL'])





bernoullinb_class_report = classification_report(y_test, bernoulli_pred)

print(bernoullinb_class_report)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

log_pred = log_reg.predict(x_test)



log_reg_conf_mat = confusion_matrix(y_test, log_pred)

plot_confusion_matrix(log_reg_conf_mat, classes = ['FAKE', 'REAL'])





log_reg_class_report = classification_report(y_test, log_pred)

print(log_reg_class_report)
import pickle



file = open('vectoriser-ngram-(1,2).pickle','wb')

pickle.dump(tfidf, file)

file.close()



file = open('Sentiment-LR.pickle','wb')

pickle.dump(log_reg, file)

file.close()



file = open('Sentiment-BNB.pickle','wb')

pickle.dump(bernoullinb, file)

file.close()