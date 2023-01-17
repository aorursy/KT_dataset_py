# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from matplotlib import pyplot as plt



%matplotlib inline
train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")
train.shape, test.shape
train.columns
train.info()
train.head()
train.tail()
train["Cover_Type"].value_counts()
sns.countplot(train.Cover_Type);    
train.dtypes.any
train.count()  
test.count()
train.isna().any().any()
test.isna().any().any()
for column in train.columns:

    print(column,train[column].nunique())
for column in test.columns:

    print(column,test[column].nunique())
train[["Elevation","Aspect","Slope"]].describe()
train[["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology", \

       "Horizontal_Distance_To_Roadways","Horizontal_Distance_To_Fire_Points"]].describe()
train[["Hillshade_9am","Hillshade_Noon","Hillshade_3pm"]].describe()
train.corr()
train[["Elevation","Aspect","Slope",]].corr()
train[["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology", \

       "Horizontal_Distance_To_Roadways","Horizontal_Distance_To_Fire_Points"]].corr()
train[["Hillshade_9am","Hillshade_Noon","Hillshade_3pm"]].corr()

test.columns
ydata_train = train["Cover_Type"]



Xdata_train = train.drop('Cover_Type', axis = "columns")
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

#Split train dataset into train and validation

X_train, X_val, y_train, y_val = train_test_split(Xdata_train, ydata_train, test_size = 0.2, random_state=55 )
#knn_clf = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', metric = 'minkowski', p = 2)

#knn_clf .fit(X_train, y_train)

#knn_pred = knn_clf.predict(X_val)

#conf_matrix = confusion_matrix(y_val, knn_pred)
#accuracy_score(knn_pred, y_val)
#conf_matrix
#test_id = test['Id']

#test_data = test.drop(["Id"], axis = 1)

#knn_pred_test = knn_clf.predict(test)
#knn_pred_test.shape
#test_id.shape
#print(knn_pred_test)
#output = pd.DataFrame({'ID': test_id,

  #                     'Cover_Type': knn_pred_test})

#output.to_csv('submission.csv', index=False)

#print(pd.read_csv('submission.csv'))
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators =1000 , random_state = 55 )

rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_val)

accuracy_score(rf_pred,y_val)
confusion_matrix(y_val,rf_pred)
test_id = test['Id']

rf_pred_test = rf_clf.predict(test)
rf_pred_test.shape,test_id.shape
print(rf_pred_test)
#output = pd.DataFrame({'ID': test_id,

#                       'Cover_Type': rf_pred_test})

#output.to_csv('submission.csv', index=False)
#print(pd.read_csv('submission.csv'))
from sklearn.ensemble import ExtraTreesClassifier 
et_clf = ExtraTreesClassifier(n_estimators = 500,

                              max_depth = 50,

                              random_state = 50,

                              n_jobs = 1)

et_clf.fit(X_train,y_train)
et_pred = et_clf.predict(X_val)

accuracy_score(et_pred,y_val)
confusion_matrix(y_val,et_pred)
test_id = test['Id']

et_pred_test = et_clf.predict(test)
et_pred_test.shape,test_id.shape
output = pd.DataFrame({'ID': test_id,

                       'Cover_Type': et_pred_test})

output.to_csv('submission.csv', index=False)
#print(pd.read_csv('submission.csv'))