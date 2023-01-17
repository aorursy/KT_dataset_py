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
# Esentials

import pandas as pd

import numpy as np

import os



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style("darkgrid")



# Ignore useless warnings

import warnings

warnings.filterwarnings("ignore")



#Limiting floats output to 2 decimal points

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) 



# Text Analysis

from collections import Counter

import nltk

from nltk.corpus import stopwords

import string



# Modelling Library

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.pipeline import Pipeline





#print(os.getcwd())
sms = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin1')[['v1', 'v2']]

sms.columns = ['Label','Message']

print('Dataset Dimension:', sms.shape)

sms.head()
# dataset grouped as per Label

print(sns.countplot(data=sms, x='Label'))

plt.title('Spam/ham Count')
count = pd.value_counts(sms["Label"], sort= True)

count.plot(kind='pie', figsize=(15,5), autopct='%1.0f%%')

plt.title('Spam/ham Distribution')

plt.ylabel('')
# Length of the messages are calculated and plotted

sms['Length'] = sms.Message.apply(len)

sms.hist(column='Length',by='Label',bins=50, figsize=(15,6))
sms.groupby('Label').describe()
# longest message in the dataset

sms[sms.Length == 910].Message.iloc[0]
# Example of spam message

sms[sms.Length == 157].Message.iloc[0]
def process_text(text):

    '''

    What will be covered:

    1. Remove punctuation

    2. Remove stopwords

    3. Return list of clean text words

    '''

    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', '16' ,'im', 'dont', 'doin', 'ure']

    # Check characters to see if they are in punctuation

    nopunc = [char for char in text if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])





sms['clean_msg'] = sms.Message.apply(process_text)

sms.head()
# Testing the process_text function:

process_text('Hi. My name is Rhea Das, I am a Data Scientist. It\'s amazing!!')
# Visualizing the most common words occuring in Ham



ham_count = Counter(" ".join(sms[sms['Label'] == 'ham']['clean_msg']).split()).most_common(20)

ham_count = pd.DataFrame.from_dict(ham_count)

ham_count = ham_count.rename(columns={0: "words in non-spam", 1 : "count"})



ham_count.plot.bar(legend=False, figsize=(12,5),color = 'black')

y_pos = np.arange(len(ham_count["words in non-spam"]))

plt.xticks(y_pos, ham_count["words in non-spam"])

plt.title('More frequent words in non-spam messages')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
ham_count
# Visualizing the most common words occuring in Spam



spam_count = Counter(" ".join(sms[sms['Label'] == 'spam']['clean_msg']).split()).most_common(20)

spam_count = pd.DataFrame.from_dict(spam_count)

spam_count = spam_count.rename(columns={0: "words in spam", 1 : "count"})



spam_count.plot.bar(legend=False, figsize=(12,5),color = 'blue')

y_pos1 = np.arange(len(spam_count["words in spam"]))

plt.xticks(y_pos1, spam_count["words in spam"])

plt.title('More frequent words in Spam messages')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
spam_count
# Split into X and Y



X = sms['clean_msg']

Y = sms['Label'].replace({'ham':0,'spam':1})

print("X Dimension", X.shape)

print("Y Dimension", Y.shape)
# Splitting X & Y into train and test



x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1)

print('X_train Dimension:', x_train.shape)

print('X_test Dimension:', x_test.shape)

print('Y_train Dimension:', y_train.shape)

print('Y_test Dimension:', y_test.shape)
 # Initiating Vector

vect = CountVectorizer()



# Fitting the training dataset

vect.fit(x_train)



# learn training data vocabulary, then use it to create a document-term matrix(dtm)

x_train_dtm = vect.transform(x_train) 



# Combine fit and transform 

x_train_dtm = vect.fit_transform(x_train) 



# Transform test dataset into a document-term matrix(dtm)

x_test_dtm = vect.transform(x_test) 
 # Initiating Model

tfidf_transformer = TfidfTransformer()



# Fitting the training dataset

tfidf_transformer.fit(x_train_dtm)



# Transforming the test dataset

tfidf_transformer.transform(x_train_dtm)
# Initiating Model

nb = MultinomialNB()



# Train the model using X_train_dtm 

nb.fit(x_train_dtm, y_train)



# Make class predictions for X_test_dtm

y_pred_class = nb.predict(x_test_dtm)



##  Calculating accuracy of the class predictions:

print('Accuracy of Multinomial Naive-Bayes Model:',round(metrics.accuracy_score(y_test, y_pred_class)*100,2))



## Print confusion Metrics

print("\nConfusion Metrics\n", metrics.confusion_matrix(y_test, y_pred_class))



## calculation ROC/AUC

print('\nROC/AUC:',round(metrics.roc_auc_score(y_test, y_pred_class)*100,2))
# Printing the false positive predictions - The messages which actually HAM but model is predicting SPAM (#7)



x_test[y_pred_class > y_test]
# Printing the false negetive predictions - The messages which actually SPAM but model is predicting HAM (#16)



x_test[y_pred_class < y_test]
# Initiating Model

pipe = Pipeline([('bow', CountVectorizer()), 

                 ('tfid', TfidfTransformer()),  

                 ('model', MultinomialNB())])



# Fitting the training dataset

pipe.fit(x_train, y_train)



# Predicting on test dataset

y_pred_pipe = pipe.predict(x_test)



##  Calculating accuracy of the class predictions:

print('Accuracy of Pipeline:',round(metrics.accuracy_score(y_test, y_pred_pipe)*100,2))



## Print confusion Metrics

print("\nConfusion Metrics\n", metrics.confusion_matrix(y_test, y_pred_pipe))



## calculation ROC/AUC

print('\nROC/AUC:',round(metrics.roc_auc_score(y_test, y_pred_pipe)*100,2))
def detect_spam(s):

    return pipe.predict([s])[0]

detect_spam('Hi, this is Rhea.')