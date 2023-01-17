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
import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.dtypes
df.info()
df.describe(include = 'all')
candidate_cols = df.corr().columns
labels = []

values = []



for col in candidate_cols:

    labels.append(col)

    values.append(np.corrcoef(df.charges, df[col])[0,1])
candidate_cols
values
df['sex'] = pd.Categorical(df['sex'], ordered = False)

df['smoker'] = pd.Categorical(df['smoker'], ordered = False)

df['region'] = pd.Categorical(df['region'], ordered = False)
df.info()
sns.catplot(kind = 'count', data = df, x = 'smoker', hue = 'sex')

plt.title('Count of smokers')
sns.distplot(a = df[df.smoker == 'yes']['charges'])

plt.title('Charges for Smokers')
sns.distplot(a = df[df.smoker == 'no']['charges'])

plt.title('Charges for non-Smokers')
sns.violinplot(data = df, x = 'smoker', y = 'charges', hue= 'sex')
sns.countplot(data=df, x = 'region')
sns.boxplot(data = df, x = 'region', y = 'charges', hue = 'sex')
a = df

a.children = pd.Categorical(df.children)



sns.catplot(kind = 'violin', data = a, x = 'children', y = 'charges')

plt.title('Charges incured  wrt no. of depenents')
sns.scatterplot(data = df, x = 'age', y = 'charges', hue = 'smoker')
sns.violinplot(data = df, x = 'smoker', y = 'charges', hue= 'sex')

plt.title('Smoking affecting demograph')
sns.distplot(a = df[df.smoker == 'yes']['bmi'])
sns.distplot(a = df[(df.smoker == 'yes') & (df.sex == 'female')]['bmi'], color = 'pink')

sns.distplot(a = df[(df.smoker == 'yes') & (df.sex == 'male')]['bmi'], color = 'blue')
sns.distplot(a = df[df.smoker == 'yes']['age'], bins = 30)
sns.boxplot(data = df[df.age < 20], x = 'smoker', y = 'charges')
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

print(df.sex.drop_duplicates())

le.fit(df.sex.drop_duplicates())

print(le.classes_)

df['sex'] = le.transform(df.sex)



le = LabelEncoder()

le.fit(df.smoker.drop_duplicates()) 

print(df.smoker.drop_duplicates())

print(le.classes_)

df['smoker'] = le.transform(df.smoker)



le = LabelEncoder()

le.fit(df.region.drop_duplicates())

print(df.region.drop_duplicates())

print(le.classes_)

df['region'] = le.transform(df.region)





# the dtypes of these columns become int
df.corr().loc['charges'].sort_values()
sns.heatmap(df.corr(), annot = True, fmt = '.2f', cmap="magma_r") #, cbar = False)
g = sns.jointplot(x = 'age', y = 'charges', data = df[df.smoker == 0], kind = 'kde', color = 'magenta')

g.plot_joint(plt.scatter, c = 'w', marker = '+', s = 20, linewidth = 1)

g.set_axis_labels("$Age$", "$Charges$")

plt.title('Distribution of charges and age for non-smokers')
g = sns.jointplot(x = 'age', y = 'charges', data = df[df.smoker == 1], kind = 'kde', color = 'cyan')

g.plot_joint(plt.scatter, c = 'w', marker = '+', s = 20, linewidth = 1)

g.set_axis_labels("$Age$", "$Charges$")

plt.title('Distribution of charges and age for smokers')
sns.lmplot(data=df, x = 'age', y = 'charges', hue = 'smoker', palette ='inferno_r')
plt.figure(figsize = (12,6))

plt.title('Distribution of charges for patients with BMI > 30')

sns.distplot(df[df.bmi > 30]['charges'], color = 'm')
plt.figure(figsize = (12,6))

plt.title('Distribution of charges for patients with BMI < 30')

sns.distplot(df[df.bmi < 30]['charges'], color = 'c')


g = sns.jointplot(kind = 'kde', data = df, x = 'bmi', y = 'charges', color = 'salmon', height = 7)

g.plot_joint(plt.scatter, marker = '+', s = 20, linewidth = 1, color = 'w')

g.set_axis_labels('BMI','Charges')

plt.title('Distribution of charges and BMI')
sns.scatterplot(data = df, x = 'bmi', y = 'charges', hue = 'smoker', palette = 'inferno')

sns.lmplot(data = df, x = 'bmi', y = 'charges', hue = 'smoker', palette = 'inferno')
df.children = df.children.astype('int64')
sns.catplot(data = df[df.children > 0], x = 'smoker', kind = 'count')

plt.title('Smokers and non-smokers who have children')
sns.catplot(data =df[df.smoker == 1], x = 'children', kind = 'count')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
X = df[['smoker','age','bmi']]

y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))
X = df[['smoker','age','bmi','sex']]

y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))
X = df[['smoker','age','bmi','sex','children']]

y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))
X = df[['smoker','age','bmi']]

Y = df.charges

quad = PolynomialFeatures (degree = 2) #, interaction_only = True)

x_quad = quad.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_quad,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))
X = df[['smoker','age','bmi','sex']]

Y = df.charges

quad = PolynomialFeatures (degree = 2) #, interaction_only = True)

x_quad = quad.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_quad,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)

print(model1.score(X_train,y_train))
X = df[['smoker','age','bmi','sex','children']]

Y = df.charges

quad = PolynomialFeatures (degree = 2) #, interaction_only = True)

x_quad = quad.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_quad,y)

model1 = LinearRegression()

model1.fit(X_train,y_train)



y_pred = model1.predict(X_test) 

print(model1.score(X_train,y_train))
sns.scatterplot(y_pred,y_pred - y_test, label = 'Test data')

sns.scatterplot(model1.predict(X_train),model1.predict(X_train) - y_train, label = 'Train data')

plt.hlines(xmin = 0, xmax = 60000, y = 0, color = 'red')
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



forest = RandomForestRegressor(n_estimators = 100,

                              criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)

forest.fit(X_train,y_train)

forest_train_pred = forest.predict(X_train)

forest_test_pred = forest.predict(X_test)



print('MSE train data: %.3f, MSE test data: %.3f' % (

mean_squared_error(y_train,forest_train_pred),

mean_squared_error(y_test,forest_test_pred)))

print('R2 train data: %.3f, R2 test data: %.3f' % (

forest.score(X_train,y_train),

forest.score(X_test,y_test)))
sns.scatterplot(forest_train_pred,forest_train_pred - y_train, label = 'Train data', alpha = 0.4)

sns.scatterplot(forest_test_pred,forest_test_pred - y_test, label = 'Test data', alpha = 0.4)

plt.hlines(xmin = 0, xmax = 60000, y = 0, color = 'red')