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
# IMPORTING PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Importing file
df = pd.read_csv("../input/svm-classification/UniversalBank.csv")
# To display the top 5 rows.
df.head()
# Total number of rows and columns.
df.shape
# Display column names.
df.columns
df.info()
# Statistics.
df.describe()
# Finding the null values.
df.isnull().sum()
# Inspecting unique values in each column.
df.nunique(axis=0)
# Histogram
df.hist(figsize=(20, 20));
df_num=df.loc[:,'ID':'Mortgage']
for i in df_num.columns:
    sns.distplot(df_num[i])
    plt.show()
del df['ID']
del df["ZIP Code"]
sns.pairplot(data = df);
plt.subplots(figsize=(10,8))
sns.heatmap(df.corr());
X = df.iloc[:,:-2].values
y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.20)
from sklearn import tree
from sklearn.metrics import accuracy_score, mean_absolute_error

# Fit / train the model
dtc = tree.DecisionTreeClassifier()
dtc.fit(trainx,trainy)
# Get the prediction for both train and test
pred_train = dtc.predict(trainx)
pred_test = dtc.predict(testx)

# Measure the accuracy of the model for both train and test sets
print("Accuracy on train is:",accuracy_score(trainy,pred_train))
print("Accuracy on test is:",accuracy_score(testy,pred_test))
# Max_depth = 3

dtc_2 = tree.DecisionTreeClassifier(max_depth=3)
dtc_2.fit(trainx,trainy)

pred_train2 = dtc_2.predict(trainx)
pred_test2 = dtc_2.predict(testx)

print("Accuracy on train is:",accuracy_score(trainy,pred_train2))
print("Accuracy on test is:",accuracy_score(testy,pred_test2))
from sklearn.svm import SVC
# Fit
svc.fit(trainx,trainy)

# Get the prediction for both train and test
train_predictions = svc.predict(trainx)
test_predictions = svc.predict(testx)
# Measure the accuracy of the model for both train and test sets
print(accuracy_score(trainy,train_predictions))
print(accuracy_score(testy,test_predictions))