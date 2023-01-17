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
def printline(no_of_lines):

    print('_' * no_of_lines)
import plotly

print(plotly.__version__)

import plotly.plotly as py

import plotly.graph_objs as go

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
df = pd.read_csv('../input/Admission_Predict.csv')

print(df.columns)

printline(100)

print(df.info())
df1 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')

print(df1.columns)

printline(100)

print(df1.info())
dataset = pd.concat([df,df1])

print(dataset.columns)

printline(100)

print(dataset.describe())
dataset
plt.xlabel("Student Research record")

plt.ylabel("No. of Student")

sns.barplot( x=["Done Research","Not Done Research"],

            y=[len(dataset[dataset["Research"] == 1]), len(dataset[dataset["Research"] == 0])])
print(dataset.columns)

fig, ax = plt.subplots()

sns.distplot(dataset["CGPA"], bins=25, color="g", ax=ax)

plt.show()
corr_table = dataset.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr_table,annot=True)

plt.show()
X_col = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']

X_data = dataset[X_col]

y_data = dataset['Chance of Admit ']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.20, random_state=42)
reg = DecisionTreeRegressor().fit(X_train, y_train)
y_pred = reg.predict(X_val) 
r2_score(y_val, y_pred)  
from joblib import dump, load

dump(reg, 'gre.joblib') 
clf = load('../input/gre.joblib') 
y_pred = clf.predict(X_val) 

r2_score(y_val, y_pred)  
os.chdir("/kaggle/input")

path = os.getcwd()

print(path)

os.chmod(path,777)