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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/weight-height/weight-height.csv')

df.head()
df.shape
df.dtypes
df.count()
df['Gender'].value_counts()
df.isnull().sum()
df.describe()
a = pd.DataFrame(df['Weight'])

b = pd.DataFrame(df['Height'])
import statsmodels.api as sms

model = sms.OLS(b,a).fit()

model.summary()
sns.heatmap(df.corr(), annot=True, cmap='viridis')
sns.countplot(df.Gender)
plt.figure(figsize=(7,6))

sns.boxplot(x='Gender', y='Height', data=df)
plt.figure(figsize=(7,6))

sns.boxplot(x='Gender', y='Weight', data=df)
sns.pairplot(df, hue='Gender', size=4)
plt.figure(figsize=(5, 4))

sns.distplot(df['Height']);

plt.axvline(df['Height'].mean(),color='blue',linewidth=2)



plt.figure(figsize=(5, 4))

sns.distplot(df['Weight']);

plt.axvline(df['Weight'].mean(),color='red',linewidth=2)
plt.figure(figsize=(7,6))

males['Height'].plot(kind='hist',bins=50, alpha=0.3,color='blue')

females['Height'].plot(kind='hist',bins=50, alpha=0.3,color='red')

plt.title('Height distribution')

plt.legend(['Males','Females'])

plt.xlabel('Height in')

plt.axvline(males['Height'].mean(),color='blue',linewidth=2)

plt.axvline(females['Height'].mean(),color='red',linewidth=2);
plt.figure(figsize=(7,6))

df.Height.plot(kind="kde", title='Univariate: Height KDE', color='c');
plt.figure(figsize=(7,6))

df.Weight.plot(kind="kde", title='Univariate: Height KDE', color='c');
sns.boxplot(df.Weight)
sns.boxplot(df.Height)
df.plot(figsize=(8,7), kind='scatter',x='Height',y='Weight');
males=df[df['Gender']=='Male']

females=df[df['Gender']=='Female']

fig,ax = plt.subplots()

males.plot(figsize=(9,8), kind='scatter', x='Height', y='Weight', ax=ax, color='blue',alpha=0.3, title='Male and Female Distribution')

females.plot(figsize=(9,8), kind='scatter', x='Height', y='Weight', ax=ax, color='red', alpha=0.3, title='Male and Female Populations');
Q1 = df.Height.quantile(0.25)

Q3 = df.Height.quantile(0.75)

Q1, Q3
IQR = Q3 - Q1

IQR
lower_limit = Q1 - 1.5*IQR

upper_limit = Q3 + 1.5*IQR

lower_limit, upper_limit
df[(df.Height<lower_limit)|(df.Height>upper_limit)]
df_no_outlier_height = df[(df.Height>lower_limit)&(df.Height<upper_limit)]

df_no_outlier_height
Q1 = df.Weight.quantile(0.25)

Q3 = df.Weight.quantile(0.75)

Q1, Q3
IQR = Q3 - Q1

IQR
lower_limit = Q1 - 1.5*IQR

upper_limit = Q3 + 1.5*IQR

lower_limit, upper_limit
df[(df.Height<lower_limit)|(df.Height>upper_limit)]
df_no_outlier_Weight = df[(df.Height>lower_limit)&(df.Height<upper_limit)]

df_no_outlier_Weight
df[['Female','Male']] = pd.get_dummies(df['Gender'])

df.head()
df.drop('Gender',axis=1,inplace=True)
df.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X = df.drop('Height',axis=1)

y = df['Height']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression

LinReg = LinearRegression()

LinReg.fit(X_train, y_train)
y_pred = LinReg.predict(X_test)

y_pred
y_test
LinReg.score(X_test, y_test)
print(LinReg.coef_)

print(LinReg.intercept_)
from sklearn.metrics import r2_score,mean_squared_error

r2_score(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))
plt.figure(figsize=(7,6))

sns.scatterplot(X_train.Weight, y_train)

plt.plot(X_train.Weight, LinReg.predict(X_train), c='r')
plt.figure(figsize=(7,6))

sns.scatterplot(X_test.Weight, y_test,color='r')

plt.plot(X_test.Weight,y_pred, c='b')
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test, y_test)
from sklearn.linear_model import LogisticRegression

Log = LogisticRegression()
Output = pd.DataFrame(X_test['Weight'], y_test)

Output