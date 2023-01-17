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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
df.head()
df.describe()
df.info()
import seaborn as sns

sns.distplot(df['Chance of Admit '])
df.corr()
sns.heatmap(df.corr())
col=df.columns

print(col)

col[:-1]
X=df[['GRE Score','TOEFL Score','SOP','LOR ','Research','CGPA']]

Y=df[col[-1]]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

lr=LinearRegression()

lr.fit(X_train,y_train)
lr.coef_
prediction=lr.predict(X_test)
plt.scatter(y_test,prediction)

sns.distplot((y_test-prediction))
lr.score(X_test,y_test)