from IPython.display import Image

import os

("../input/images/Spam ham emoji.JPG")



Image("../input/images/Spam ham emoji.JPG")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#spam_ham= pd.read_csv("../input/heart.csv")
spam_ham= pd.read_csv("../input/spam-or-not-spam-dataset/spam_or_not_spam.csv")

spam_ham.head()
spam_ham.info()
spam_ham.describe()
spam_ham['email'].isnull().sum()
spam_ham= spam_ham.dropna(how='any',axis=0)

spam_ham.info()
sns.countplot(x='label', data = spam_ham)

plt.title('Number of Spam (1) & ham (0) from e-mail dataset ')

plt.show()
spam_ham['label'].sum()
def sum_func(df,column): # this function will help in counting the same group entries

    for i in range(len(column)):

        count = df[column].value_counts()

        return count



sum_func(spam_ham, 'label')

    

print( "Spam percentage is ",spam_ham['label'].sum()/len(spam_ham.index)* 100, "%")
numpy_array = spam_ham.as_matrix()

X= spam_ham.email

y= spam_ham.label

#X=numpy_array[:,0]

#y=numpy_array[:,1]

#y = y.astype('int')

print("X")

print(X)

print("y")

print(y)
import sklearn

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)

X_train
from sklearn.feature_extraction.text import CountVectorizer

vect= CountVectorizer(stop_words="english")

vect.fit(X_train)



# printing the Vocabulary texts



print(vect.vocabulary_)
X_train.shape
X_train_transformed= vect.transform(X_train)

X_test_transformed = vect.transform(X_test)

#print(X_train_transformed)

print(X_test_transformed)
#X = X_transformed.toarray()

#X
#X_train_transformed=X_transformed.toarray()

X_train_transformed
X_train_transformed.shape
# converting matrix to dataframe

#pd.DataFrame(X_train_transformed, columns=vect.get_feature_names())
y_train
y_train.shape
from sklearn.naive_bayes import BernoulliNB

bnb=BernoulliNB()

bnb.fit(X_train_transformed, y_train)

proba= bnb.predict_proba(X_test_transformed)

y_pred= bnb.predict(X_test_transformed)



#Converting array to data frame

#proba = pd.DataFrame(proba)

#proba.tail()

# printing the overall accuracy

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
from sklearn import metrics

confusion= metrics.confusion_matrix(y_test, y_pred)

print(confusion)
pd.DataFrame(proba, columns=['Ham','Spam'])

#pd.DataFrame(proba)
TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP/float(TP+FN)

print("Sensitivity= ",sensitivity)

specificity= TN/float(TN+FP)

print("Specificity= ", specificity)

print("Precision= ", TP/float(TP+FP))



print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred))
# creating an ROC curve

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)



# matrix of thresholds, tpr, fpr

print(pd.DataFrame({'Threshold': thresholds,

              'TPR': true_positive_rate,

              'FPR':false_positive_rate

             }))



# plotting the ROC curve





plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC')

plt.plot(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'k--')

plt.show()



### Bernoullis NB modelling gave better result
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
print(max(thresholds))
from sklearn.metrics import precision_recall_curve 



precision, recall, thresholds = precision_recall_curve(y_test, proba[:,1])

# create plot

plt.plot(recall, precision, label='Precision-recall curve')



_ = plt.xlabel('Recall')

_ = plt.ylabel('Precision')

_ = plt.title('Precision-recall curve')

_ = plt.legend(loc="lower left")



no_skill = len(y_test[y_test==1]) / len(y_test)

plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
from sklearn.metrics import average_precision_score

average_precision_score(y_test, proba[:, 1])