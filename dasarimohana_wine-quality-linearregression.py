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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression



df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')



df.info()
# Find null values

df.isnull().mean()*100
sns.boxplot(df['fixed acidity'])
sns.boxplot(df['volatile acidity'])
sns.boxplot(df['quality'])
# Remove outliners

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3-Q1



df = df[~((df < Q1 - 1.5*IQR) | (df > Q3 + 1.5*IQR)).any(axis=1)]
# Again check for the outliners

sns.boxplot(df['fixed acidity'])
sns.boxplot(df['quality'])
# Correlation

c = df.corr()

c
# Heatmap

plt.figure(figsize=(9,6))

sns.heatmap(c, cmap='BrBG',annot = True)
# Dividing wine quality as Good and bad according to their number limits



quality_limits = (2,6.5,8)



quality_type = ['Bad','Good']



df['quality'] = pd.cut(df['quality'], bins = quality_limits ,labels = quality_type)
# Converting categorial values into numerical values using Label Encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['quality'] = le.fit_transform(df['quality'])
sns.countplot(df['quality'])
df['quality'].value_counts()
# Train and Test

from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train,df_test=train_test_split(df,train_size=0.7,test_size=0.3,random_state=100)
# Linear Regression model

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



y_train=df_train.pop('quality')

X_train=df_train



y_test=df_test.pop('quality')

X_test=df_test



regressor = LinearRegression()

regressor.fit(X_train, y_train)



print('Slope:',regressor.coef_)

print('Intercept:',regressor.intercept_)
# Predicting y values

y_pred_train = regressor.predict(X_train)

y_pred_test = regressor.predict(X_test)
# r2 square

r2_train = r2_score(y_train,y_pred_train)

r2_test = r2_score(y_test,y_pred_test)

print(r2_train)

print(r2_test)