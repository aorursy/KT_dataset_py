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
telco_raw = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",skipinitialspace=True)
telco_raw.head()
telco_raw.dtypes
telco_raw.isnull().mean()
custid = ['customerID']

target = ['Churn']
categorical = telco_raw.nunique()[telco_raw.nunique()<10].keys().tolist()

categorical.remove(target[0])

numerical = [col for col in telco_raw.columns if col not in custid+target+categorical]
telco_raw = pd.get_dummies(data=telco_raw, columns=categorical, drop_first=True)
# Import StandardScaler library

from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler instance

scaler = StandardScaler()

# Fit the scaler to numerical columns

scaled_numerical = scaler.fit_transform(telco_raw[numerical])
# Build a DataFrame

scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical)
# Drop non-scaled numerical columns

telco_raw = telco_raw.drop(columns=numerical, axis=1)
# Merge the non-numerical with the scaled numerical data

telco = telco_raw.merge(right=scaled_numerical,

how='left',

left_index=True,

right_index=True

)
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
cols    = [col for col in telco_raw.columns if col not in custid+target]
cols
X = telco_raw[cols]

Y = telco_raw['Churn']
# 1. Split data to training and testing

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25)
# 2. Initialize the model

mytree = tree.DecisionTreeClassifier()
# 3. Fit the model on the training data

treemodel = mytree.fit(train_X, train_Y)
# 4. Predict values on the testing data

pred_Y = treemodel.predict(test_X)
# 5. Measure model performance on testing data

accuracy_score(test_Y, pred_Y)