import numpy as np

import pandas as pd
data=pd.read_csv("../input/creditcard.csv")
data.head()
data.describe()
data.Class.value_counts()
data.isna().sum()
from matplotlib import pyplot as plt
import seaborn as sns
sns.stripplot(data.Class[0:10000],data.Time[0:10000])

    



    
data1=data[:10000]

data1.head()
for i in data1.columns:

    print(i)

    if i!='Class':

        sns.stripplot(data1.Class,data1[i])

        plt.show()

#sns.stripplot(data.Class[0:10000],data.Time[0:10000])
####PREAPARING THE TRAINING DATA USING THE 10000 SAMPLES
data_train_x=data1[['V17','V16','V14','V12','V11','V10','V9','V4','V3']].copy()
data_train_x.shape
data_train_y=data1['Class']

data_train_y.head()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train_x, data_train_y, test_size=0.3, random_state=0)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
y_tes=data.Class[10000:]
x_tes=data[['V17','V16','V14','V12','V11','V10','V9','V4','V3']].copy()

x_tes=x_tes[10000:]
y_pre=logreg.predict(x_tes)
cnf_mat = metrics.confusion_matrix(y_tes, y_pre)
cnf_mat
logreg.score(x_tes, y_tes)
from sklearn.metrics import recall_score

print(recall_score(y_tes, y_pre, average='macro'))

print(recall_score(y_tes, y_pre, average='micro'))

print(recall_score(y_tes, y_pre, average='weighted'))

print(recall_score(y_tes, y_pre, average=None))
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_tes, y_pre)
print(average_precision)