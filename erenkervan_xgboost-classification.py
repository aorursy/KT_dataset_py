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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score,roc_curve

from sklearn.preprocessing import StandardScaler, LabelEncoder

from xgboost import XGBClassifier

import time

import numpy as np 

import pandas as pd

import os
csv_files = []

for dirname, _, filenames in os.walk('/kaggle/input/cicids2017/MachineLearningCSV/MachineLearningCVE/'):

    for filename in filenames:

        csv_files.append(pd.read_csv('/kaggle/input/cicids2017/MachineLearningCSV/MachineLearningCVE/'+filename))



df = pd.concat(csv_files)

df
df.info()
old_memory_usage = df.memory_usage().sum()
#change the variable types for low memory usage

#int64 to int32,,, float64 to float32

integer = []

f = []

for i in df.columns[:-1]:

    if df[i].dtype == "int64": integer.append(i)

    else : f.append(i)



df[integer] = df[integer].astype("int32")

df[f] = df[f].astype("float32")
df.info()
new_memory_usage = df.memory_usage().sum()

old_vs_new = (old_memory_usage - new_memory_usage) / old_memory_usage * 100

print(f"%{old_vs_new} lower memory usage")
# drop one variable features 

one_variable_list = []

for i in df.columns:

    if df[i].value_counts().nunique() < 2:

        one_variable_list.append(i)

df.drop(one_variable_list,axis=1,inplace=True)
df.columns =  df.columns.str.strip()
# drop nan and infinite rows

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
# merging similar classes with low instances

df["Label"] = df["Label"].replace(["Web Attack � Brute Force","Web Attack � XSS","Web Attack � Sql Injection"],"Web Attack")
# drop duplicate rows

df =  df.drop_duplicates(keep="first")

df.reset_index(drop=True,inplace=True)
#feature reduction 

#dropping very high correlated features 

corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features 

df =  df.drop(to_drop, axis=1)

df.shape
x = df.drop(["Label"],axis=1)

y = df["Label"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
start = time.time()

xgb = XGBClassifier(random_state=42)

xgb.fit(x_train,y_train)

xgbpreds = xgb.predict(x_test)

print("Time", time.time()-start)

print("Accuracy",accuracy_score(y_test,xgbpreds))

print(classification_report(y_test,xgbpreds))