# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_original = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

data_original.head()
data_original.describe()
data_original.count()
data_original.columns[data_original.isna().any()].tolist()
#drop last column -> contains NaN

data_drop_col_nan = data_original.iloc[:, :-1]
data_drop_col_nan.columns[data_drop_col_nan.isna().any()].tolist()
data_drop_col_nan[data_drop_col_nan['diagnosis'] == 'M'].count()
data_drop_col_nan[data_drop_col_nan['diagnosis'] == 'B'].count()
#convert benign = 0, malignant = 1

data_drop_col_nan['diagnosis'] = data_drop_col_nan['diagnosis'].map({'B':0, 'M':1})
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



X = data_drop_col_nan.drop('diagnosis', axis=1).values

y = data_drop_col_nan['diagnosis']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)


print("train size : ", len(X_train), "\ntest size : ", len(X_test), "\ntrain y : ", len(y_train), "\ntest y : ", len(y_test))
from sklearn.neighbors import KNeighborsClassifier



neighbors = np.arange(1,12)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i,k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors = k)

    

    knn.fit(X_train, y_train)

    

    train_accuracy[i] = knn.score(X_train, y_train)

    

    test_accuracy[i] = knn.score(X_test, y_test)

    
import matplotlib.pyplot as plt

plt.style.use('ggplot')



plt.title('k-NN varying number of neighbors')

plt.plot(neighbors, test_accuracy, label='Testing Accuracy')

plt.plot(neighbors, train_accuracy, label='Training Accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()
#so we set up number of neigbors as 6, in order the highest prediction

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
from sklearn.metrics import confusion_matrix



y_pred  = knn.predict(X_test)

confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))
y_pred_prob = knn.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1],[0,1], 'k--')

plt.plot(fpr, tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=6) ROC curve')

plt.show()
from sklearn.metrics import roc_auc_score



roc_auc_score(y_test, y_pred_prob)
from sklearn.model_selection import GridSearchCV



param_grid = {'n_neighbors' : np.arange(1,100)}



knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X_train,y_train)
knn_cv.best_score_
knn_cv.best_params_
knn_cv.score(X_test, y_test)
y_pred  = knn_cv.predict(X_test)

confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))
y_pred_prob = knn_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1],[0,1], 'k--')

plt.plot(fpr, tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=6) ROC curve')

plt.show()