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

        file = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv(file)

df.head()
df.info()
df.describe()
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot=True)
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
print(lm.coef_)
X_train.columns
coef = pd.DataFrame(lm.coef_,X_train.columns,columns=['coeff'])

coef
predictions = lm.predict(X_test)
predictions
plt.scatter(y_test,predictions)
sns.distplot(y_test-predictions)
from sklearn import metrics
metrics.mean_squared_error(y_test,predictions)
metrics.mean_absolute_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))