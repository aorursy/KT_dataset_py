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
data.shape # How many rows and columns?
# Let's check a few rows

data.head()
# How many rows for each patient?

data.groupby('Patient').size()
# How many rows for each smokers vs. ex-smokers?

data.groupby('SmokingStatus')['Patient'].nunique()
data.groupby('SmokingStatus')['FVC'].mean() 

# Seems odd that the data description page says FVC is "the recorded lung capacity in ml".
data.groupby('Weeks')['Patient'].nunique()
data.groupby(['Weeks','SmokingStatus','Sex','Age'])['FVC'].mean()
from sklearn.preprocessing import LabelEncoder



cat_features = ['Sex','SmokingStatus']

encoder = LabelEncoder()



# Apply the label encoder to each column

encoded = data[cat_features].apply(encoder.fit_transform)
data2 = data[['FVC','Percent','Weeks','Age']].join(encoded)

data2.head()
X = data2[['SmokingStatus','Age','Sex','Weeks','Percent']]

y = data2['FVC']
# Let's define a function to calculate the metric

# I didn't actually use this evaluation but sharing my thoughts

# def eval_metric(FVC,FVC_Pred,sigma):

#     sigma_clipped = np.max(sigma,70)

#     delta = np.min(np.abs(FVC-FVC_Pred),1000)

#     eval_metric = -np.sqrt(2)*delta/sigma_clipped - np.ln(np.sqrt(2)*sigma_clipped)

#     return eval_metric



# We need the prediction for FVC_Pred and confidence(sigma I think?)
import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)

y_pred = regressor.predict(X_test)
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
test.groupby('SmokingStatus')['FVC'].mean() 
# Apply the label encoder to each column

encoded = test[cat_features].apply(encoder.fit_transform)

test2 = test[['Patient','Percent','Weeks','Age']].join(encoded)
test2.head(100)
submission = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
submission.head(100)
submission[['Patient','Weeks']] = submission.Patient_Week.str.split("_",expand=True,)
submission.head()
submission = submission.drop('FVC',1)

submission = submission.drop('Confidence',1)

test2 = test2.drop('Weeks',1)
submission2 = pd.merge(submission,test2,on='Patient',how='left')

submission2.head(100)
X2 = submission2[['SmokingStatus','Age','Sex','Weeks','Percent']]

submission2['FVC'] = regressor.predict(X2)
submission2.head()
submission2.shape
submission2.groupby(['SmokingStatus','Sex','Age'])['FVC'].mean()
submission2['FVC_Group'] = submission2.groupby(['SmokingStatus','Sex','Age'])['FVC'].transform('mean')
submission2.head(100)
submission2['Confidence'] = 100*submission2['FVC']/submission2['FVC_Group']
submission2.head(100)
submission3 = submission2[['Patient_Week','FVC','Confidence']]
submission3.head()
submission3['FVC'] = submission3['FVC'].astype(int)

submission3['Confidence'] = submission3['Confidence'].astype(int)
submission3.head()
submission3.to_csv("/kaggle/working/submission.csv",index=False)