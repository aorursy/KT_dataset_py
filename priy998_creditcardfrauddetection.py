import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

sns.set()

%matplotlib inline

from pylab import rcParams



df = pd.read_csv('../input/creditcardfraud/creditcard.csv')









print(df.shape)

df.head()
df.info()
df.describe()
df.columns
fraud_cases=len(df[df['Class']==1])

print('number of fraud cases:=',fraud_cases)

Nonfraud_cases=len(df[df['Class']==0])

print('number of Nonfraud cases:=',Nonfraud_cases)
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
## Get the Fraud and the normal dataset 



fraud = df[df['Class']==1]



normal = df[df['Class']==0]



LABELS=['Normal','Fraud']
count_classes=pd.value_counts(df['Class'],sort=True)

count_classes.plot(kind='bar',color='red')

plt.title('Transaction Class Distribution')

plt.xticks(range(2),LABELS)

plt.xlabel('class')

plt.ylabel('frequency')
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(fraud.Amount, bins = bins)

ax1.set_title('Fraud')

ax2.hist(normal.Amount, bins = bins)

ax2.set_title('Normal')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 20000))

plt.yscale('log')

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud.Time, fraud.Amount)

ax1.set_title('Fraud')

ax2.scatter(normal.Time, normal.Amount)

ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
x=df.drop('Class',axis=1)

y=df['Class']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(x_train,y_train)

y_pred=log.predict(x_test)

y_pred
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test,y_pred)
#Random UnderSampling

normal_under=normal.sample(fraud_cases)

under_sample=pd.concat([normal_under,fraud],axis=0)

print('random under sampling:')

print(under_sample.Class.value_counts())

under_sample.Class.value_counts().plot(kind='bar',title='Count')
#Random OverSampling

fraud_over=fraud.sample(Nonfraud_cases,replace=True)

over_sample=pd.concat([fraud_over,normal],axis=0)

print('random under sampling:')

print(over_sample.Class.value_counts())

over_sample.Class.value_counts().plot(kind='bar',title='Count')
over_sample.head()

X=over_sample.drop('Class',axis=1)

Y=over_sample['Class']



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.2)
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(x_train,y_train)

y_pred_log=log.predict(x_test)



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score 

from sklearn.metrics import f1_score



print('Accuracy:',accuracy_score(y_test,y_pred_log))

print('Precision:',precision_score(y_test,y_pred_log))

print('Recall:',recall_score(y_test,y_pred_log))

print('F1 score:',f1_score(y_test,y_pred_log))
conf_mat_log=confusion_matrix(y_test, y_pred_log)

print('Confusion Matrix:\n',conf_mat_log)



labels=['Class 0','Class 1']

fig=plt.figure()

ax=fig.add_subplot(111)

cax=ax.matshow(conf_mat_log,cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from sklearn.naive_bayes import GaussianNB

nv = GaussianNB()

nv.fit(x_train,y_train)

y_pred_nv=nv.predict(x_test)

print('Accuracy:',accuracy_score(y_test,y_pred_nv))