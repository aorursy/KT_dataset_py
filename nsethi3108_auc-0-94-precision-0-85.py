# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



#importing the data set

df=pd.read_csv("../input/creditcard.csv")



#creating target series

target=df['Class']

target



#droppiIng the target variable from the data set

df.drop('Class',axis=1,inplace=True)

df.shape



#converting them to numpy arrays

X=np.array(df)

y=np.array(target)

X.shape

y.shape



#distribution of the target variable

len(y[y==1])

len(y[y==0])



#splitting the data set into train and test (75:25)

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#applyting SMOTE to oversample the minority class-it will produce data with equal ratio of the target

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=2)

X_sm,y_sm=sm.fit_sample(X_train,y_train)

print(X_sm.shape,y_sm.shape)

print(len(y_sm[y_sm==1]),len(y_sm[y_sm==0]))
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn import metrics



#Logistic Regression

logreg=LogisticRegression()

logreg.fit(X_sm,y_sm)

y_logreg=logreg.predict(X_test)

y_logreg_prob=logreg.predict_proba(X_test)[:,1]



#Performance metrics evaluation

print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_logreg))

print("Accuracy:\n",metrics.accuracy_score(y_test,y_logreg))

print("Precision:\n",metrics.precision_score(y_test,y_logreg))

print("Recall:\n",metrics.recall_score(y_test,y_logreg))

print("AUC:\n",metrics.roc_auc_score(y_test,y_logreg_prob))

auc=metrics.roc_auc_score(y_test,y_logreg_prob)



#plotting the ROC curve

fpr,tpr,thresholds=metrics.roc_curve(y_test,y_logreg_prob)

plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)

plt.plot([0,1],[0,1],'r-.')

plt.xlim([-0.2,1.2])

plt.ylim([-0.2,1.2])

plt.title('Receiver Operating Characteristic\nLogistic Regression')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Random Forest with 'gini' criterion

from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(random_state=3)

rf.fit(X_sm,y_sm)

y_rf=rf.predict(X_test)

y_rf_prob=rf.predict_proba(X_test)[:,1]



#Performance metrics evaluation

print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_rf))

print("Accuracy:\n",metrics.accuracy_score(y_test,y_rf))

print("Precision:\n",metrics.precision_score(y_test,y_rf))

print("Recall:\n",metrics.recall_score(y_test,y_rf))

print("AUC:\n",metrics.roc_auc_score(y_test,y_rf_prob))

auc=metrics.roc_auc_score(y_test,y_rf_prob)



#plotting the ROC curve

fpr,tpr,thresholds=metrics.roc_curve(y_test,y_rf_prob)

plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)

plt.plot([0,1],[0,1],'r-.')

plt.xlim([-0.2,1.2])

plt.ylim([-0.2,1.2])

plt.title('Receiver Operating Characteristic\nRandom Forest')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



#Random Forest with 'entropy' criterion

from sklearn.ensemble import RandomForestClassifier



rf=RandomForestClassifier(criterion='entropy',random_state=3)

rf.fit(X_sm,y_sm)

y_rf=rf.predict(X_test)

y_rf_prob=rf.predict_proba(X_test)[:,1]



#Performance metrics evaluation

print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_rf))

print("Accuracy:\n",metrics.accuracy_score(y_test,y_rf))

print("Precision:\n",metrics.precision_score(y_test,y_rf))

print("Recall:\n",metrics.recall_score(y_test,y_rf))

print("AUC:\n",metrics.roc_auc_score(y_test,y_rf_prob))

auc=metrics.roc_auc_score(y_test,y_rf_prob)



#plotting the ROC curve

fpr,tpr,thresholds=metrics.roc_curve(y_test,y_rf_prob)

plt.plot(fpr,tpr,'b', label='AUC = %0.2f'% auc)

plt.plot([0,1],[0,1],'r-.')

plt.xlim([-0.2,1.2])

plt.ylim([-0.2,1.2])

plt.title('Receiver Operating Characteristic\nRandom Forest')

plt.legend(loc='lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
