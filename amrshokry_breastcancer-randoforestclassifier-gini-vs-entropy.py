# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.impute import SimpleImputer

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

BreastData = load_breast_cancer()

#X Data

X = BreastData.data



print('X shape is ' , X.shape)

print('X Features are \n' , BreastData.feature_names)



#y Data

y = BreastData.target



print('y shape is ' , y.shape)

print('y Columns are \n' , BreastData.target_names)





ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')

ImputedX = ImputedModule.fit(X)

X = ImputedX.transform(X)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=100,max_depth=2,random_state=33) 

RandomForestClassifierModel.fit(X_train, y_train)
print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)

print('----------------------------------------------------')
y_pred = RandomForestClassifierModel.predict(X_test)

y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)

print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])

print('the real value for RandomForestClassifierModel is  : ' , y_test[:10])

print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])



CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)



# drawing confusion matrix

sns.heatmap(CM, center = True)

plt.show()
RandomForestClassifierModel = RandomForestClassifier(criterion = 'entropy',n_estimators=100,max_depth=2,random_state=33) 

RandomForestClassifierModel.fit(X_train, y_train)


print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)

print('----------------------------------------------------')
y_pred = RandomForestClassifierModel.predict(X_test)

y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)

print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])

print('the real value for RandomForestClassifierModel is  : ' , y_test[:10])

print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])
#Confusion Matrix with entropy
CM = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', CM)



# drawing confusion matrix

sns.heatmap(CM, center = True)

plt.show()