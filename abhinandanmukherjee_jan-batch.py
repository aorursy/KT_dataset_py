# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head(10)
train.info()
train.describe()
train.shape
train.isnull().sum()
num_cols = ['Age','SibSp','Parch','Fare']

cat_cols = ['Pclass','Sex','Embarked']

target = 'Survived'
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train['Age'] = train['Age'].fillna(train['Age'].mean())
train[num_cols].isnull().sum()
train[cat_cols].isnull().sum()
def cat_cols_info(df,col):

    print("Unique categories in {}".format(col))

    print(df[col].unique())

    print("Distribution of categories: \n")

    print(df[col].value_counts())

    print('\n')
for col in cat_cols:

    cat_cols_info(train,col)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().values[0])
train['Embarked'].mode().values[0]
sns.pairplot(train[num_cols])
train[target].unique()
colors = ['Red','Blue']
for col in num_cols:

    fig = plt.figure(figsize = (15,4))

    ax = fig.add_subplot(111)

    j = 0

    for key, df in train.groupby([target]):

        ax = sns.kdeplot(train[col][(train[target] == key)], color=colors[j], shade = True, label=key)

        ax.set_xlabel(col)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        ax.set_title('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
train['Sex'].value_counts().plot.pie()
fig = plt.figure(figsize = (15,5))

j = 1

for cat_col in cat_cols:

    ax = fig.add_subplot(1,len(cat_cols),j)

    sns.countplot(x = cat_col,

                  data = train,

                  ax = ax)

    ax.set_xlabel(cat_col)

    ax.set_ylabel("Frequency")

    ax.set_title('Frequency Distribution for individual classes in {}'.format(cat_col), fontsize = 10)

    j = j + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (15,5))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxplot(y = num_col,

                    x = cat_col, 

                    data = train, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
for num_col in num_cols:

    fig = plt.figure(figsize = (15,5))

    j = 1

    for cat_col in cat_cols:

        ax = fig.add_subplot(1,len(cat_cols),j)

        sns.boxenplot(y = num_col,

                    x = cat_col, 

                    hue = target,

                    data = train, 

                    ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
train_corr = train[num_cols].corr()

train_corr.head()
train_corr = train[num_cols].corr()

sns.heatmap(train_corr, cmap='coolwarm_r', annot_kws={'size':20})

plt.title('Correlation Matrix', fontsize=14)

plt.show()
def handle_outliers(df,var,target,tol):

    var_data = df[var].sort_values().values

    q25, q75 = np.percentile(var_data, 25), np.percentile(var_data, 75)

    

    print('Outliers handling for {}'.format(var))

    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

    iqr = q75 - q25

    print('IQR {}'.format(iqr))

    

    cut_off = iqr * tol

    lower, upper = q25 - cut_off, q75 + cut_off

    print('Cut Off: {}'.format(cut_off))

    print('{} Lower: {}'.format(var,lower))

    print('{} Upper: {}'.format(var,upper))

    

    outliers = [x for x in var_data if x < lower or x > upper]



    print('Number of Outliers in feature {}: {}'.format(var,len(outliers)))



    print('{} outliers:{}'.format(var,outliers))



    print('----' * 25)

    print('\n')

    print('\n')

        

    return list(df[(df[var] > upper) | (df[var] < lower)].index)
outliers = []

for num_col in num_cols:

    outliers.extend(handle_outliers(train,num_col,target,1.5))

outliers = list(set(outliers))

outliers
train.shape
train = train.drop(outliers)

train.shape
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(train,test_size=0.2,random_state=101)

train_data.head()
test_data.head(2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_data[num_cols])
sc.mean_
sc.var_
train_data[num_cols] = sc.transform(train_data[num_cols])
train_scaled
test_data[num_cols] = sc.transform(test_data[num_cols])
X_train = train_data[num_cols + cat_cols]

X_test = test_data[num_cols + cat_cols]
X_train = pd.get_dummies(X_train,columns=cat_cols,drop_first=True)

X_test = pd.get_dummies(X_test,columns=cat_cols,drop_first=True)
train_data.head(2)
X_train.head(2)