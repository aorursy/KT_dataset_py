# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.isnull().sum()
df.dtypes
df.shape
df.describe(include = [np.number])
df.agg(['mean', 'min', 'max'])
import seaborn as sns
sns.boxplot( df["age"]  )
sns.boxplot( df["bmi"]  )
sns.boxplot( df["children"]  )
sns.boxplot( df["charges"]  )
import seaborn as sns

sns.heatmap(df.corr(), annot = True,fmt='.1g',vmin=-1, vmax=1, center= 0,cmap= 'coolwarm')
# f= plt.figure(figsize=(12,5))

# ax=f.add_subplot(121)

# sns.distplot(df[(df['smoker'] == 1)]["charges"],color='c',ax=ax)

# ax.set_title('Distribution of charges for smokers')



# ax=f.add_subplot(122)

# sns.distplot(df[(df.smoker == 0)]['charges'],color='b',ax=ax)

# ax.set_title('Distribution of charges for non-smokers')
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="husl", data=df)
sns.catplot(x="sex", y="charges", hue="smoker",

            kind="violin", data=df, palette = 'magma')
plt.figure(figsize=(12,5))

plt.title("Distribution of age")

ax = sns.distplot(df["age"], color = 'g')
sns.lmplot(x="age", y="charges", hue="smoker", data=df, palette = 'inferno_r', height = 7)

ax.set_title('Smokers and non-smokers')
plt.figure(figsize=(12,5))

plt.title("Distribution of bmi")

ax = sns.distplot(df["bmi"], color = 'm')
plt.figure(figsize=(12,5))

plt.title("Distribution of charges for patients with BMI greater than 30")

ax = sns.distplot(df[(df.bmi >= 30)]['charges'], color = 'm')
plt.figure(figsize=(10,6))

ax = sns.scatterplot(x='bmi',y='charges',data=df,palette='magma',hue='smoker')

ax.set_title('Scatter plot of charges and bmi')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=df, palette = 'magma', size = 8)
sns.catplot(x="children", kind="count", palette="ch:.25", data=df, height = 6)
sns.catplot(x="smoker", kind="count", palette="rainbow",hue = "sex",

            data=df[(df.children > 0)], size = 6)
# dummy encoding of categorical features

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[['sex']])
ohe.categories_
ohe.fit_transform(df[['region']])
ohe.categories_
ohe.fit_transform(df[['smoker']])
ohe.categories_
x = df.drop(['charges'], axis = 1)

y = df.charges
x.head(2)
y.value_counts(normalize=True)
# use when different features need different preprocessing

from sklearn.compose import make_column_transformer
column_trans = make_column_transformer(

    (OneHotEncoder(), ['sex', 'smoker','region']),

    remainder='passthrough')
column_trans.fit_transform(x)
# chain sequential steps together

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
linreg =  LinearRegression()
pipe = make_pipeline(column_trans, linreg)
cross_val_score(pipe, x_train, y_train, cv=5).mean()
pipe.fit(x_train, y_train)
pipe.predict(x_test)
X = df.drop(['charges','region'], axis = 1)

Y = df.charges

X.head(2)
column_trans = make_column_transformer(

    (OneHotEncoder(), ['sex', 'smoker']),

    remainder='passthrough')

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 100,

                              criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)
pipe = make_pipeline(column_trans,rf_reg)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 0)
cross_val_score(pipe, X_train, y_train, cv=5, ).mean()
pipe.fit(X_train, y_train)
forest_test_pred = pipe.predict(X_test)
forest_train_pred = pipe.predict(x_train)

forest_test_pred = pipe.predict(x_test)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_train,forest_train_pred),

mean_squared_error(y_test,forest_test_pred)))

print('R2 train data: %.3f, R2 test data: %.3f' % (

r2_score(y_train,forest_train_pred),

r2_score(y_test,forest_test_pred)))
plt.figure(figsize=(10,6))



plt.scatter(forest_train_pred,forest_train_pred - y_train,

          c = 'black', marker = 'o', s = 35, alpha = 0.5,

          label = 'Train data')

plt.scatter(forest_test_pred,forest_test_pred - y_test,

          c = 'c', marker = 'o', s = 35, alpha = 0.7,

          label = 'Test data')

plt.xlabel('Predicted values')

plt.ylabel('Tailings')

plt.legend(loc = 'upper left')

plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')

plt.show()