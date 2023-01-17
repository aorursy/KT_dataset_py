# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
df=pd.read_csv('../input/unimelb_training.csv')
df.head()
df.info()
df.shape
df.describe()
#Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(bins=4, figsize=(30,50))
plt.show()
df['A.14'].value_counts()
df.columns
vars=['Grant.Application.ID','Grant.Status','RFCD.Code.1','RFCD.Percentage.1','RFCD.Code.2','RFCD.Percentage.2','RFCD.Code.3','RFCD.Percentage.3','RFCD.Code.4','RFCD.Percentage.4','RFCD.Code.5','RFCD.Percentage.5','SEO.Code.1','SEO.Percentage.1']
df1 = [col for col in df.columns if df[col].dtype == 'object']
df.dtypes
df1=df[vars]
df1.fillna(df1.mean(), inplace=True)
df1.columns
#For my base line model considering 14 numerical coulmns  
input = [col for col in df1.columns if df1[col].dtype != 'object']
#My Dependent variable 
output='Grant.Status'
#removing DV from My data set
input.remove(output)
from sklearn.linear_model import LogisticRegression
log1 = LogisticRegression(max_iter=20)
log1.fit(X=df1[input], y=df1[output])
from sklearn.metrics import accuracy_score
preds = log1.predict(X=df1[input])
preds
#Accuracy Checking

print(accuracy_score(y_pred=preds, y_true=df1[output]))
from sklearn.ensemble import RandomForestClassifier
random_forest1 = RandomForestClassifier()
random_forest1.fit(X=df1[input], y=df1[output])
print(accuracy_score(y_pred=random_forest1.predict(X=df1[input]), y_true=df1[output]))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(df1[output], random_forest1.predict(X=df1[input]))
print(confusion_matrix)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(df1[output], preds)
print(confusion_matrix)
from sklearn.model_selection import train_test_split