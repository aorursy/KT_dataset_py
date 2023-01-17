import pandas as pd

import numpy as np
# Read the data set



data = pd.read_table('....input/SMSSpamCollection', header=None, names=['Class', 'sms'])

data.head(10)
len(data)
# Get the count, unique and frequency of the data



data.describe()
# Now groupby the class column & describe it



data.groupby('Class').describe()
# Check the length of the each SMS



data['length']=data['sms'].apply(len)

data.head(10)
# Convert the class into numerical value



data['labels'] = data.Class.map({'ham':0, 'spam':1})

data.head(10)
# Now drop the class column



data = data.drop('Class', axis = 1)

data.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data['length'].plot(bins=50,kind='hist')
data.length.describe()
# A message with 910 words



data[data['length']==910]['sms'].iloc[0]
# Example using to create a function which will be use in later part of the code



import string

mess = 'sample message!...'

nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

print(nopunc)
# Now import the stopwords



from nltk.corpus import stopwords

stopwords.words('english')
nopunc.split()
# Check whether the word nopunc.split() is present in stopwords.words('english') or not



clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

clean_mess
# Now let's put both of these together in a function to apply it to our DataFrame later on:



def text_process(mess):

    nopunc =[char for char in mess if char not in string.punctuation]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
# Tokenization can be done but by not removing puctuation



from nltk.tokenize import sent_tokenize, word_tokenize

data['sms'].apply(word_tokenize).head(10)
data['sms'].apply(sent_tokenize).head(10)
# By removing puctuation



data['sms'].head(10).apply(text_process)
#create an object of class PorterStemmer



from nltk.stem import PorterStemmer

porter = PorterStemmer()
data['sms'] = data['sms'].str.lower()
print ([porter.stem(word) for word in data['sms']])
# Convert to X & y



X = data.sms

y = data.labels
print(X.shape)
print(y.shape)
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.head()
y_train.head()
# Vectorizing the sentence & removing the stopwords



from sklearn.feature_extraction.text import CountVectorizer

#vect = CountVectorizer(stop_words='english')



# or we can use the fuction text_process in place of stop_words = 'english'

vect = CountVectorizer(analyzer=text_process)
vect.fit(X_train)
# Printing the vocabulary



vect.vocabulary_
# vocab size

len(vect.vocabulary_.keys())
# transforming the train and test datasets



X_train_transformed = vect.transform(X_train)

X_test_transformed = vect.transform(X_test)
# note that the type is transformed (sparse) matrix



print(type(X_train_transformed))

print(X_train_transformed)
# training the NB model and making predictions

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()



# fit

mnb.fit(X_train_transformed,y_train)



# predict class

y_pred_class = mnb.predict(X_test_transformed)



# predict probabilities

y_pred_proba = mnb.predict_proba(X_test_transformed)
# note that alpha=1 is used by default for smoothing

mnb
# printing the overall accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
# confusion matrix

metrics.confusion_matrix(y_test, y_pred_class)

# help(metrics.confusion_matrix)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test, y_pred_class))

print(confusion_matrix(y_test, y_pred_class))
confusion = metrics.confusion_matrix(y_test, y_pred_class)

print(confusion)

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)

print("sensitivity",sensitivity)
specificity = TN / float(TN + FP)

print("specificity",specificity)
precision = TP / float(TP + FP)

print("precision",precision)

print(metrics.precision_score(y_test, y_pred_class))
print("precision",precision)

print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))
y_pred_class
y_pred_proba
# creating an ROC curve

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)
# area under the curve

print (roc_auc)
# matrix of thresholds, tpr, fpr

pd.DataFrame({'Threshold': thresholds, 'TPR': true_positive_rate, 'FPR':false_positive_rate})
# plotting the ROC curve

%matplotlib inline  

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC')

plt.plot(false_positive_rate, true_positive_rate)
print(len(X_train),len(X_test),len(y_train),len(y_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([

   ( 'bow',CountVectorizer(analyzer=text_process)),

    ('classifier',MultinomialNB()),

])
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(classification_report(predictions,y_test))