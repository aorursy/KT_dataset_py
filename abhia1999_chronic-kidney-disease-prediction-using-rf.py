import numpy as np

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import csv

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report

%matplotlib inline
data = pd.read_csv('../input/chronic-kidney-disease/new_model.csv')

data.head()
data.describe()
data.info()
corr=data.corr()

top_co=corr.index

plt.figure(figsize=(20,20))

g=sns.heatmap(data[top_co].corr(),annot=True,cmap="RdYlGn")
x = data.drop(['Class'],axis=1)

y = data['Class']

lab_enc=LabelEncoder()

y=lab_enc.fit_transform(y)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets

clf.fit(X_train,Y_train)



y_predi=clf.predict(X_test)
from sklearn.metrics import accuracy_score

print("Accuracy on training dataset:{:.1f}".format(clf.score(X_train,Y_train)))

print("Accuracy on testing dataset:{:.1f}".format(accuracy_score(Y_test, y_predi)))
c=pd.DataFrame(

    confusion_matrix(Y_test, y_predi),columns=['Predicted:0', 'Predicted:1'],

    index=['Actual:0', 'Actual:1']

)

ax= plt.subplot()

sns.heatmap(c, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
cr=classification_report(Y_test,y_predi)

print(cr)
newdf = pd.DataFrame({'Actual': Y_test, 'Predicted':y_predi})

newdf
df1 = newdf

df1.plot(kind='bar',figsize=(20,6))

plt.grid(which='major', linestyle='--', linewidth='1.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
rrc=recall_score(Y_test, y_predi)
rps=precision_score(Y_test, y_predi)

rps
from sklearn.metrics import f1_score

rfs=f1_score(Y_test,y_predi)

rfs
from sklearn.metrics import average_precision_score

from inspect import signature
precision, recall, threshold = precision_recall_curve(Y_test, y_predi)

average_precision = average_precision_score(Y_test, y_predi)

step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

plt.step(recall, precision, color='r', alpha=0.2, where='post')

plt.fill_between(recall, precision, alpha=0.2, color='r', **step_kwargs)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.0])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))