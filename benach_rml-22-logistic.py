# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Preprocessing - Data import

os.chdir(r'/kaggle/input/glass')

My_data = pd.read_csv('glass.csv')

My_data.head()
print(My_data.describe())

My_data.dtypes
#Data Cleansing and Analysis

M_Values = My_data.isnull()

M_Values.head()
#Converting into a binary feature



My_data['label'] = My_data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

My_data.head()
# Determaining Test and Train Data

In_Val = ['Na', 'Mg', 'Al', 'Si','K','Ca','Ba','Fe']

#In_Val = ['Na']

X = My_data [In_Val]

y = My_data.label

#splitting data into Training set and Test Set

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
#Feature scalling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#Fitting Logistic regression to the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=100)

classifier.fit(X_train,y_train)
#predicting the test set result

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
#visualising the Confusion Matrix



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('cm', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
# model evaluation metrics



from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# ROC Curve for regresion (Receiver Operating Characteristic)



y_pred_pr = classifier.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_pr)

auc = metrics.roc_auc_score(y_test, y_pred_pr)

plt.plot(fpr,tpr,label="ROC, auc="+str(auc), color='orange')

plt.legend(loc=4)

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curve')

plt.legend()

plt.show()
#AUC score for the case is 0.97.