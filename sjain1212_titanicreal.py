# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dat = pd.read_csv('../input/train.csv')
train_dat.corr()
all_vars = ['Survived','Sex','Age','SibSp','Fare']

final_train_dat = train_dat[all_vars]
final_train_dat.head(10)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

final_train_dat = final_train_dat.replace(to_replace = 'female',value=0)

final_train_dat = final_train_dat.replace(to_replace='male',value=1)



final_train_dat['Age'] = final_train_dat['Age'].replace(np.NaN, np.nanmean(final_train_dat['Age']))
final_train_dat.head()
y_train = final_train_dat['Survived']



# only features 

features = ['Age','Sex','SibSp','Fare']

x_train = final_train_dat[features]
from sklearn.metrics import auc

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression





clf = DecisionTreeClassifier()

results = cross_val_score(clf,x_train, y_train,cv=5)

results = clf.fit(x_train, y_train)
import matplotlib.pyplot as plt



# apply the model to train data

y_train_predict = clf.predict(x_train)

y_train_proba = clf.predict_proba(x_train)



print(y_train[:5],y_train_predict[:5]) 





#extract fpr and tpr to plot ROC curve and calculate AUC (Note: fpr-false positive rate and tpr -true positive rate)

fpr, tpr, threshold = metrics.roc_curve(y_train, y_train_proba[:,1])



# This is exctly the first metric you'll be evaluated on!

# Note: this will only work on the binary case -- you'll need a different method to do multi-class case

def cm_metric(y_true,y_prob):

    

    # predict the class with the greatest probability

    y_pred = [np.argmax(y) for y in y_prob]



    # calculate the confusion matrix

    cm = confusion_matrix(y_true, y_train_predict)



    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return sum(sum(np.multiply(cm_norm,np.array([[1, -2], [-2, 1]]))))



cm_metric(y_train,y_train_proba)



# Calculate the area under the ROC curve

roc_auc = metrics.auc(fpr, tpr)

print('AUC: ',roc_auc)

print("Training Acc:")

print(metrics.accuracy_score(y_train,y_train_predict))

print()



plt.title('Train Data Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
test_dat = pd.read_csv('../input/test.csv')

test_dat['Age'] = test_dat['Age'].replace(np.NaN, np.nanmean(final_train_dat['Age']))

test_dat['Fare'] = test_dat['Fare'].replace(np.NaN,np.nanmean(final_train_dat['Fare']))



test_dat = test_dat.replace(to_replace = 'female',value=0)

test_dat = test_dat.replace(to_replace='male',value=1)

test_dat.head()
x_test = test_dat[features]

passenger_ids = pd.Series(test_dat["PassengerId"])

# print(passenger_ids)

y_test_predict = clf.predict(x_test)

y_test_proba = clf.predict_proba(x_test)
y_test_predict
ser = pd.Series(y_test_predict)

df = pd.DataFrame(ser,columns=['Survived'])
df.insert(1,'PassengerId',passenger_ids)
df.head()
df.to_csv('new.csv',index=False)