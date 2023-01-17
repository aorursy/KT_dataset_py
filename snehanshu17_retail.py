import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train=pd.read_csv("../input/traines/train.csv")

test=pd.read_csv("../input/testses/test.csv")

train.head()
train['source']='train'

test['source']='test'

data = pd.concat([train, test],ignore_index=True)



data.head()

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
data.describe()
data.dtypes
#Filter categorical variables

categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude ID cols and source:

categorical_columns = [x for x in categorical_columns if x not in ['Product_ID','source']]



for col in categorical_columns:

    print ('\nFrequency of Categories for varible %s'%col)

    print (data[col].value_counts())
#box plot Age/Purchase

var = 'Age'

data = pd.concat([data['Purchase'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(9, 5))

fig = sns.boxplot(x=var, y="Purchase", data=data)

fig.axis(ymin=0, ymax=40000);
#box plot City_Category/Purchase

var = 'City_Category'

data = pd.concat([data['Purchase'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(9, 5))

fig = sns.boxplot(x=var, y="Purchase", data=data)

fig.axis(ymin=0, ymax=40000);
#box plot Gender/Purchase

var = 'Gender'

data = pd.concat([data['Purchase'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(9, 5))

fig = sns.boxplot(x=var, y="Purchase", data=data)

fig.axis(ymin=0, ymax=40000);
#box plot Stay_In_Current_City_Years/Purchase

var = 'Stay_In_Current_City_Years'

data = pd.concat([data['Purchase'], data[var]], axis=1)

f, ax = plt.subplots(figsize=(9, 5))

fig = sns.boxplot(x=var, y="Purchase", data=data)

fig.axis(ymin=0, ymax=40000);
#scatter plot Marital_Status/Purchase

var = 'Marital_Status'

data = pd.concat([data['Purchase'], data[var]], axis=1)

data.plot.scatter(x=var, y='Purchase', ylim=(0,50000));
#scatter plot Occupation/Purchase

var ='Occupation'

data = pd.concat([data['Purchase'], data[var]], axis=1)

data.plot.scatter(x=var, y='Purchase', ylim=(0,50000));
#correlation matrix

corrmat = data.corr()

f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k =7 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Purchase')['Purchase'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()


data['Product_Category_2'].fillna(data['Product_Category_2'].mean(), inplace=True)

data.head()
data['Product_Category_3'].fillna(data['Product_Category_3'].mean(), inplace=True)
data['Purchase'].fillna(data['Purchase'].mean(), inplace=True)
data.head(5)
print('data duplicated:{}'.format(data.duplicated().sum()))

data.drop_duplicates(inplace=True)


total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)


#Import library:

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

var_mod = ['Age','City_Category','Gender','Stay_In_Current_City_Years']

le = LabelEncoder()

for i in var_mod:

    data[i] = le.fit_transform(data[i])

#One Hot Coding:

data = pd.get_dummies(data, dummy_na='False', columns=['Age','City_Category','Gender','Stay_In_Current_City_Years'])

data.dtypes
#Divide into test and train:

train = data.loc[data['source']=="train"]

test = data.loc[data['source']=="test"]

#Drop unnecessary columns:

test.drop(['source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)

data.drop(['Stay_In_Current_City_Years_nan','Gender_nan','Age_nan','City_Category_nan'],axis=1,inplace=True)

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing, cross_validation, svm

from sklearn.linear_model import LinearRegression



from matplotlib import pyplot as plt

from matplotlib import style



style.use('ggplot')

X = np.array(data.drop(['Product_ID', 'source'], 1))

X = preprocessing.scale(X)

Y = np.array(data['Purchase'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
# Training Model



clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)