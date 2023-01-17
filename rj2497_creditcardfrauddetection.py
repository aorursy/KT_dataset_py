# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data=pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(data.head(5))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
valid=data[data["Class"]==0]

fraud=data[data["Class"]==1]

valid["Class"].count()

fraud["Class"].count()
data_corr=data.corr()

plt.figure(figsize=(20,20))

sns.heatmap(data=data_corr,annot=True)

plt.show
sns.scatterplot(x=data["V25"],y=data["V1"],hue=data["Class"])

plt.show()

sns.lmplot(data=data,x="Time",y="Amount",hue="Class")

plt.show()
from sklearn.preprocessing import StandardScaler 

data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))

y=data.loc[:,["Class"]]

X=data.drop(["Class", "Amount"],axis=1)

X.head()
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline



# define pipeline

over = SMOTE(sampling_strategy=1)

under = RandomUnderSampler(sampling_strategy=0.5)

steps = [('o', over), ('u', under)]

pipeline = Pipeline(steps=steps)

# transform the dataset

X, y = over.fit_sample(X, y)
y["Class"].value_counts()
from sklearn.metrics import classification_report, accuracy_score  

from sklearn.metrics import precision_score, recall_score 

from sklearn.metrics import f1_score, matthews_corrcoef 

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print("Number transactions X_train dataset: ", X_train.shape) 

print("Number transactions y_train dataset: ", y_train.shape) 

print("Number transactions X_test dataset: ", X_test.shape) 

print("Number transactions y_test dataset: ", y_test.shape) 


# logistic regression object 

lr = LogisticRegression() 

  

# train the model on train set 

lr.fit(X_train, y_train) 

  

pred = lr.predict(X_test) 

  

# print classification report 

print(classification_report(y_test, pred)) 

acc = accuracy_score(y_test, pred) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(y_test, pred) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(y_test, pred) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(y_test, pred) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(y_test, pred) 

print("The Matthews correlation coefficient is{}".format(MCC)) 
#confusion matrix

LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(y_test, pred) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show() 
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()



gnb.fit(X_train, y_train)



predGNB = gnb.predict(X_test)



# print classification report 

print(classification_report(y_test, predGNB)) 

acc = accuracy_score(y_test, predGNB) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(y_test, predGNB) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(y_test, predGNB) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(y_test, predGNB) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(y_test, predGNB) 

print("The Matthews correlation coefficient is{}".format(MCC))
#confusion matrix for Naive Bayes

LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(y_test, predGNB) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show()
from sklearn import svm

clf = svm.LinearSVC(C=0.5, dual=False)#or use svm.SVC() and C is always smaller number in case of LinearSVC



clf.fit(X_train,y_train)



predSVM=clf.predict(X_test)



# print classification report 

print(classification_report(y_test, predSVM)) 

acc = accuracy_score(y_test, predSVM) 

print("The accuracy is {}".format(acc)) 

  

prec = precision_score(y_test, predSVM) 

print("The precision is {}".format(prec)) 

  

rec = recall_score(y_test, predSVM) 

print("The recall is {}".format(rec)) 

  

f1 = f1_score(y_test, predSVM) 

print("The F1-Score is {}".format(f1)) 

  

MCC = matthews_corrcoef(y_test, predSVM) 

print("The Matthews correlation coefficient is{}".format(MCC))
#confusion matrix for Naive Bayes

LABELS = ['Normal', 'Fraud'] 

conf_matrix = confusion_matrix(y_test, predSVM) 

plt.figure(figsize =(12, 12)) 

sns.heatmap(conf_matrix, xticklabels = LABELS,  

            yticklabels = LABELS, annot = True, fmt ="d"); 

plt.title("Confusion matrix") 

plt.ylabel('True class') 

plt.xlabel('Predicted class') 

plt.show()