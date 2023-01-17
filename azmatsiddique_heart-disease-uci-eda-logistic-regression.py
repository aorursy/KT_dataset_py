# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas_profiling as npp

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
report = npp.ProfileReport(df)
report
plt.scatter(df.age,df.sex)
df.isnull().sum()
df
x = df.drop("target",axis='columns')

y = df['target']
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=40)
model = LogisticRegression()
model.fit(x_train,y_train)
x_test
predicted_y = model.predict(x_test)
x_test['result']= model.predict(x_test)
x_test
x_test.drop("result",axis='columns',inplace=True)
x_test.head()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted_y)
predicted_y
y_test
cm
import seaborn as sns
sns.heatmap(cm,annot=True)

plt.xlabel("actual")

plt.ylabel("prediced")
model.score(x_test,y_test)