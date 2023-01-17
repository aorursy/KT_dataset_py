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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

%matplotlib inline
df=pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.info()
df.describe()
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
X=df[['age','bmi', 'children']]

y=df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y)

lm=LinearRegression()

lm.fit(X_train,y_train)
coef=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coef
predictions=lm.predict(X_test)

print(predictions)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
sns.lmplot(x='bmi',y='charges',hue='children',col='sex',data=df)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))