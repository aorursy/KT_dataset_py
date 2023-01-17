!pip install pyforest
from pyforest import *

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/insurance.csv')

data.head()
data.describe()
data.info()
data.isnull().sum()
data.shape
data.duplicated().sum() 
data.sort_values(['age'])
data[data.region == 'southeast']
data.mean()
data.info(memory_usage='deep')
data.memory_usage(deep=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(data.sex.drop_duplicates())

data.sex = le.transform(data.sex)

le.fit(data.smoker.drop_duplicates())

data.smoker = le.transform(data.smoker)

le.fit(data.region.drop_duplicates())

data.region = le.transform(data.region)
data.memory_usage(deep=True)
f, ax = plt.subplots(figsize=(10,8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240, 10, as_cmap=True), square=True, ax=ax)
f = plt.figure(figsize=(12,5))

ax = f.add_subplot(121)

sns.distplot(data[(data.smoker == 1)]['charges'], color='c', ax=ax)

ax.set_title('Distribution of charges for smokers')



ax = f.add_subplot(122)

sns.distplot(data[(data.smoker == 0)]['charges'], color='b', ax=ax)

ax.set_title('Distribution of charges for non-smokers')
data.smoker.value_counts() #0 means non-smokers and 1 means smokers
sns.countplot(x='smoker', hue='sex', palette='pink', data=data)
sns.catplot(x='sex', y='charges', kind='violin', hue='smoker', palette='magma', data=data)
plt.figure(figsize=(12, 5))

plt.title('Box plot for charges of women')

sns.boxplot(x='smoker', y='charges', data=data[(data.sex == 1)])
plt.figure(figsize=(12, 5))

plt.title('Box plot for charges of men')

sns.boxplot(x='smoker', y='charges', data=data[(data.sex == 0)])
plt.figure(figsize=(12, 5))

plt.title('Distribution of Age')

ax = sns.distplot(data['age'], color='b')
pd.set_option('display.max_rows', None)
data
data.loc[(data.age == 18) & (data.smoker == 1)]
data.dtypes
data.charges.mean()
data.charges.min()
data.charges.max()
data.loc[(data.charges >= 13270) & (data.smoker == 1)]
data.loc[(data.charges == 63770.42801), :]
data.loc[(data.sex == 0) & (data.smoker == 1)]
data.loc[(data.smoker == 1) & (data.region == 0)]
g = sns.jointplot(x = 'age', y = 'charges', data=data[(data.smoker == 0)], kind = 'kde', color='b')

g.plot_joint(plt.scatter, c='w', s=30, linewidth=1, marker='+')

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels('$X$', '$Y$')

ax.set_title('Distribution of charges and age for non-smokers')
sns.lmplot(x='age',y='charges', hue='smoker', data=data, palette='inferno_r', size=7)

plt.title('Smokers and Non-Smokers')
plt.figure(figsize=(12,5))

plt.title('Distribution of BMI')

ax = sns.distplot(data['bmi'], color = 'y')
data.loc[(data.bmi <=18.5)]   #Here, <=18.5 means the person is under weight.
data[data.bmi >= 30]   #If the BMI exceeds the value of 30 means, person has obesity.
data.loc[(data.bmi >= 30) & (data.age == 18)] 
data.loc[(data.bmi <= 18.5) & (data.age == 18)] 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X = data.iloc[:,:6].values

y = data.iloc[:,6].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.score(X_test, y_test))
X1 = data.iloc[:,[0,1,2,3,4]].values

y1 = data.iloc[:,6].values

X_train, X_test,y_train, y_test = train_test_split(X1, y1, test_size=0.25)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

print(regressor.score(X_test, y_test))
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25)

lin_reg = LinearRegression()

lin_reg = lin_reg.fit(X_train, y_train)

print(lin_reg.score(X_test, y_test))