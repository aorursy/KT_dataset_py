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
sns.jointplot(data=df, x= "Time on Website", y= "Yearly Amount Spent")
sns.jointplot(data=df , x = "Time on App", y = "Yearly Amount Spent")
sns.jointplot(data=df,x ='Time on App', y ='Length of Membership' , kind='hex')
sns.pairplot(df)
sns.lmplot(x = 'Length of Membership',y = 'Yearly Amount Spent', data = df)
df.columns
y = df['Yearly Amount Spent']

X = df[['Avg. Session Length', 'Time on App',

       'Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size = 0.3 , random_state = 101)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train , y_train)
lr.coef_
predictions = lr.predict(X_test)
plt.scatter(y_test , predictions)

plt.xlabel("Y Test(True value)")

plt.ylabel("Prediction(predicted value)")
from sklearn import metrics

print ('MAE ',metrics.mean_absolute_error(y_test , predictions))

print ('MSE ', metrics.mean_squared_error(y_test , predictions))

print ('RMSE ', np.sqrt(metrics.mean_squared_error(y_test , predictions)))
metrics.explained_variance_score(y_test , predictions)
sns.distplot(y_test-predictions)
cdf = pd.DataFrame(lr.coef_,X.columns,columns=['Coeff'])

cdf
