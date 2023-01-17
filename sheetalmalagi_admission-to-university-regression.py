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

import numpy as np

import seaborn as sns

from math import log, sqrt, sin

import matplotlib.pyplot as plt

%matplotlib inline
adm_df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

adm_df.shape
adm_df.head()
#Data Cleaning

#Remove Serial No.



adm_df1 = adm_df.drop('Serial No.',axis=1)

adm_df1.shape
adm_df1.describe()
adm_df1.groupby('University Rating').mean()
#Rename columns to user friendly names



adm_df1.columns = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR ', 'CGPA',

       'Research', 'Chance_of_Admit']


fig = plt.figure(figsize=(6, 6))

ax = fig.gca()

ax.scatter(x=adm_df1.Chance_of_Admit, y =adm_df1.GRE_Score)
univ_counts=adm_df1['University_Rating'].value_counts()

univ_counts.plot.bar()
#Only Numeric fields

adm_df1.hist(bins=40,figsize=(20,15))
# Correlation Matrix



adm_df1_corr = adm_df1.corr()

adm_df1_corr
# Correlation bar chart - Descending 



corr_abs = pd.DataFrame(adm_df1_corr['Chance_of_Admit'].abs())

corr_abs.sort_values('Chance_of_Admit', ascending=False)[1:].plot(kind='bar', figsize=(6,6))

plt.tight_layout()
#Split Data - stratified splitting

from sklearn.model_selection import train_test_split



adm_final = adm_df1[['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR ', 'CGPA',

       'Research', 'Chance_of_Admit']]









train, test = train_test_split(adm_final, test_size = 0.2, random_state = 32,stratify=None,shuffle=False)
train_X = train.iloc[:,:-1]

test_X = test.iloc[:,:-1]



print(train_X.shape)

print(train_X.columns)
train_Y = train.iloc[:,-1]

test_Y = test.iloc[:,-1]



print(train_Y.shape)

# With all the fields

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



model = LinearRegression().fit(train_X, train_Y)

print(model.score(train_X, train_Y))

print(mean_squared_error(model.predict(train_X),train_Y))

# R like summary



import statsmodels.api as sm





x_train1 = sm.add_constant(train_X)

lm_1 = sm.OLS(train_Y, x_train1).fit()

lm_1.summary()

# Test set



test_pred = model.predict(test_X)

print(mean_squared_error(test_pred,test_Y))