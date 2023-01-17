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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/heights-and-weights-dataset/SOCR-HeightWeight.csv", index_col=0)

df.head()
df.shape
df.isnull().sum()
df.describe()
import matplotlib.style as style

style.use('fivethirtyeight')
sns.heatmap(df.corr(), annot=True, cmap='viridis', vmax=1.0, vmin=-1.0 )
plt.figure(figsize=(7,6))

plt.hist(df['Height(Inches)'], bins=20, rwidth=0.8)

plt.xlabel('Height')

plt.ylabel('Count')
plt.figure(figsize=(7,6))

plt.hist(df['Weight(Pounds)'], bins=20, rwidth=0.8)

plt.xlabel('Weight')

plt.ylabel('Count')
plt.figure(figsize=(9,7))

sns.scatterplot(df['Height(Inches)'], df['Weight(Pounds)'])
Q1 = df['Weight(Pounds)'].quantile(0.25)

Q3 = df['Weight(Pounds)'].quantile(0.75)

Q1, Q3
IQR = Q3 - Q1

IQR
lower_limit = Q1 - 1.5*IQR

upper_limit = Q3 + 1.5*IQR

lower_limit, upper_limit
df['Weight(Pounds)'].describe()
df[(df['Weight(Pounds)']<lower_limit)|(df['Weight(Pounds)']>upper_limit)]
Q1 = df['Height(Inches)'].quantile(0.25)

Q3 = df['Height(Inches)'].quantile(0.75)

Q1, Q3
IQR = Q3 - Q1

IQR
lower_limit = Q1 - 1.5*IQR

upper_limit = Q3 + 1.5*IQR

lower_limit, upper_limit
df[(df['Height(Inches)']<lower_limit)|(df['Height(Inches)']>upper_limit)]
df_no_outlier_Height = df[(df['Height(Inches)']>lower_limit)&(df['Height(Inches)']<upper_limit)]

df_no_outlier_Height
data = pd.DataFrame(df_no_outlier_Height)

data.head()
data.shape
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
LinReg = LinearRegression()

LinReg.fit(X_train,y_train)
y_pred = LinReg.predict(X_test)

y_pred
y_train