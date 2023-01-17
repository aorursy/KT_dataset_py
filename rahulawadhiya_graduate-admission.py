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
df=pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = df.drop("Serial No.", axis =1)
df.describe()
df.info()
sns.heatmap(df.corr(), cmap = 'viridis', annot= True)
sns.pairplot(df)
# from above two plots it is clear that GRE, TOEFL and CGPA has correlation with chance of admission

sns.set_style('whitegrid')

sns.jointplot(x='GRE Score', y = 'Chance of Admit ', data = df)

sns.jointplot(x='TOEFL Score', y = 'Chance of Admit ', data = df)

sns.jointplot(x='CGPA', y = 'Chance of Admit ', data = df)
from sklearn.model_selection import train_test_split



X = df[['CGPA','TOEFL Score','GRE Score']]

y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state = 101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
pred = lm.predict(X_test)
plt.scatter(y_test,pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
# Checking the residuals to make sure everything was ok with data

sns.distplot((y_test-pred),bins=50);