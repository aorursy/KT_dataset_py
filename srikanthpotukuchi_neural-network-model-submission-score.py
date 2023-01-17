# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
# Let's check a few rows

data.head()
# How many rows for each patient?

data.groupby('Patient').size()
from sklearn.preprocessing import LabelEncoder



cat_features = ['Sex','SmokingStatus']

encoder = LabelEncoder()



# Apply the label encoder to each column

encoded = data[cat_features].apply(encoder.fit_transform)
data2 = data[['FVC','Percent','Weeks','Age']].join(encoded)

data2.head()
X = data2[['SmokingStatus','Age','Sex','Weeks','Percent']]

y = data2['FVC']
import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.neural_network import MLPRegressor

from sklearn import metrics

%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regr = MLPRegressor(random_state=1, max_iter=500) # Try different options for better score

regr.fit(X_train, y_train) #training the algorithm
y_pred = regr.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
# test2 = test[['Percent','Weeks','Age']].join(encoded)

test.head()
test['Patient_Week'] = test['Patient'].astype(str)+"_"+test['Weeks'].astype(str)

test.head()
# Apply the label encoder to each column

encoded = test[cat_features].apply(encoder.fit_transform)

test2 = test[['Patient','Percent','Weeks','Age']].join(encoded)
test2.head()
X = data2[['SmokingStatus','Age','Sex','Weeks','FVC']]

y = data2['Percent']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3, random_state=0)
regr2 = MLPRegressor(random_state=1, max_iter=500) # Try different options for better score

regr2.fit(X_train2, y_train2) #training the algorithm
y_pred2 = regr2.predict(X_test2)
df = pd.DataFrame({'Actual': y_test2, 'Predicted': y_pred2})

df
test3 = test[['Patient','FVC','Weeks','Age']].join(encoded)
submission = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
submission.head(100)
submission[['Patient','Weeks']] = submission.Patient_Week.str.split("_",expand=True,)
submission.head()
submission = submission.drop('FVC',1)

submission = submission.drop('Confidence',1)

test2 = test2.drop('Weeks',1)
submission2 = pd.merge(submission,test2,on='Patient',how='left')

submission2.head(100)
test4 = test3[['Patient','FVC']]

submission2 = pd.merge(submission2,test4,on='Patient',how='left')

submission2.head(100)
X2 = submission2[['SmokingStatus','Age','Sex','Weeks','Percent']]

X3 = submission2[['SmokingStatus','Age','Sex','Weeks','FVC']]

submission2['FVC'] = regr.predict(X2)

submission2['Confidence'] = regr2.predict(X3)
submission2.head()
submission3 = submission2[['Patient_Week','FVC','Confidence']]
submission3.head(100)
# submission3['FVC'] = submission3['FVC'].astype(int)

# submission3['Confidence'] = submission3['Confidence'].astype(int)
submission3.to_csv("/kaggle/working/submission.csv",index=False)