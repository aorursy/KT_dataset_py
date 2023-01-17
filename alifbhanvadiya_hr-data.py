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
data = pd.read_csv('../input/hr-analytics/HR_comma_sep.csv')
data.head(10)
data.describe()
data.info()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.countplot(data.salary,hue = data.left)
plt.figure(figsize=(10,6))

sns.countplot(data.Department , hue = data.left)
plt.figure(figsize=(10,6))

sns.heatmap(data.corr(),annot =True)
sns.countplot(data.number_project, hue = data.left)
sns.countplot(data.promotion_last_5years ,hue = data.left)
sns.countplot(data.time_spend_company ,hue = data.left)
sns.countplot(data.Work_accident , hue = data.left)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder
lab =LabelEncoder()

data['salary'] = lab.fit_transform(data['salary'])

data['Department'] = lab.fit_transform(data['Department'])
data.head()
X = data.drop('left',axis=1)

y = data.left
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state =42)
std =StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.fit_transform(X_test)
log = LogisticRegression()
log.fit(X_train,y_train)
y_predict = log.predict(X_test)
log.score(X_train,y_train)
log.score(X_test,y_test)