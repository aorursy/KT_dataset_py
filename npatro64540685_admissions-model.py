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

df = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
df.head()
df.tail()
df.describe()
df.info()
df = df.rename(columns={'GRE Score': 'GRE_Score', 'TOEFL Score': 'TOEFL_Score', 'University Rating':'University_rating','LOR ': 'LOR', 'Chance of Admit ': 'Admit'})

df.head()
plt.figure(figsize = (20,10))

sns.countplot(df['GRE_Score'])
plt.figure(figsize = (20,10))

sns.countplot(df['TOEFL_Score'])
sns.countplot(df['University_rating'])
sns.pairplot(df, hue = 'University_rating', vars = ['GRE_Score', 'TOEFL_Score', 'CGPA', 'LOR', 'SOP', 'Admit'])
plt.figure(figsize = (20,10))

sns.heatmap(df.corr(), annot = True)
sns.boxplot(x = 'University_rating', y = 'CGPA', data = df)

sns.boxplot(x = 'University_rating', y = 'GRE_Score', data = df)

sns.boxplot(x = 'University_rating', y = 'Admit', data = df)

X = df[['GRE_Score', 'CGPA', 'TOEFL_Score']]



y = df['Admit']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
import statsmodels.api as sm

from statsmodels.formula.api import ols
from statsmodels.api import OLS



OLS(y_train,X_train).fit().summary()