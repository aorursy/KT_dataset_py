import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#Acquire the datasets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df,test_df]



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(train_df.columns.values)
train_df.head()
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['LotShape','SalePrice']].groupby(['LotShape'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
train_df[['Neighborhood','SalePrice']].groupby(['Neighborhood'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
train_df[['MSZoning','SalePrice']].groupby(['MSZoning'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
g = sns.FacetGrid(train_df, col='LotShape')

g.map(plt.hist, 'PoolArea', bins=4)
print ("Before", train_df.shape, test_df.shape)



train_df = train_df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

test_df = test_df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape
train_df.GarageFinish.unique()
test_df.GarageFinish.unique()
train_df['GarageFinish'].value_counts()
test_df['GarageFinish'].value_counts()
train_df['GarageFinish'].fillna('Unf', inplace=True)
test_df['GarageFinish'].fillna('Unf', inplace=True)
combine = [train_df, test_df]
train_df['GarageFinish'].value_counts()
test_df['GarageFinish'].value_counts()
for dataset in combine:

    dataset['GarageFinish'] = dataset['GarageFinish'].map({'Unf': 1, 'Fin': 2, 'RFn': 3})
train_df['GarageFinish'].value_counts()
test_df['GarageFinish'].value_counts()
train_df['GarageFinish'] = train_df['GarageFinish'].astype('category')
train_df['GarageFinish'].unique()
test_df['GarageFinish'] = test_df['GarageFinish'].astype('category')
test_df['GarageFinish'].unique()
train_df.MSZoning.unique()
test_df.MSZoning.unique()
train_df['MSZoning'].value_counts()
test_df['MSZoning'].value_counts()
train_df['MSZoning'].fillna('RL', inplace=True)
test_df['MSZoning'].fillna('RL', inplace=True)
combine = [train_df, test_df]
train_df['MSZoning'].value_counts()
test_df['MSZoning'].value_counts()
for dataset in combine:

    dataset['MSZoning'] = dataset['MSZoning'].map({'RL': 1, 'RM': 2, 'FV': 3, 'RH': 4, 'C (all)': 5})
train_df['MSZoning'].value_counts()
test_df['MSZoning'].value_counts()
train_df['MSZoning'] = train_df['MSZoning'].astype('category')
train_df['MSZoning'].unique()
test_df['MSZoning'] = test_df['MSZoning'].astype('category')
test_df['MSZoning'].unique()
X_train = train_df[['GarageFinish', 'MSZoning']]
X_train
Y_train = train_df['SalePrice']
Y_train
X_test = test_df[['GarageFinish','MSZoning']]
X_test
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100,2)

acc_log
train_df.Street.unique()
test_df.Street.unique()
train_df.Street.value_counts()
test_df.Street.value_counts()
train_df['Street'].fillna('Pave', inplace=True)
test_df['Street'].fillna('Pave', inplace=True)
combine = [train_df, test_df]
train_df.Street.value_counts()
test_df.Street.value_counts()
for dataset in combine:

    dataset['Street'] = dataset['Street'].map({'Pave': 1, 'Grvl': 2})
train_df.Street.value_counts()
test_df.Street.value_counts()
train_df['Street'] = train_df['Street'].astype('category')
test_df['Street'] = test_df['Street'].astype('category')
train_df.Street.unique()
test_df.Street.unique()
train_df.LotShape.unique()
test_df.LotShape.unique()
train_df.LotShape.value_counts()
test_df.LotShape.value_counts()
train_df['LotShape'].fillna('Reg', inplace=True)
test_df['LotShape'].fillna('Reg', inplace=True)
combine = [train_df, test_df]
train_df.LotShape.value_counts()
test_df.LotShape.value_counts()
for dataset in combine:

    dataset['LotShape'] = dataset['LotShape'].map({'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4})
train_df.LotShape.value_counts()
test_df.LotShape.value_counts()
train_df['LotShape'] = train_df['LotShape'].astype('category')
test_df['LotShape'] = test_df['LotShape'].astype('category')
train_df.LotShape.unique()
test_df.LotShape.unique()
train_df.LandContour.unique()
test_df.LandContour.unique()
train_df.LandContour.value_counts()
test_df.LandContour.value_counts()
train_df['LandContour'].fillna('Lvl', inplace=True)
train_df['LandContour'].fillna('Lvl', inplace=True)
combine = [train_df, test_df]
for dataset in combine:

    dataset['LandContour'] = dataset['LandContour'].map({'Lvl': 1, 'HLS': 2, 'Bnk': 3, 'Low': 4})
train_df.LandContour.value_counts()
test_df.LandContour.value_counts()
train_df['LandContour'] = train_df['LandContour'].astype('category')
test_df['LandContour'] = test_df['LandContour'].astype('category')
train_df.LandContour.unique()
test_df.LandContour.unique()