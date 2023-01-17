#importing libraries

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/Boston.csv')

dataset.shape
dataset.dtypes
dataset.head()
dataset.isnull().sum()
dataset['crim'].describe()
sns.distplot(dataset['crim'])
print("Skewness: %f" % dataset['crim'].skew())

print("Kurtosis: %f" % dataset['crim'].kurt())
sns.boxplot(dataset['crim'])
dataset[dataset['crim']<dataset['crim'].quantile(0.02)]['crim']
dataset.loc[dataset['crim']<dataset['crim'].quantile(0.02),['crim']]=dataset['crim'].quantile(0.02)
dataset[dataset['crim']>dataset['crim'].quantile(0.95)]['crim']
dataset.loc[dataset['crim']>dataset['crim'].quantile(0.95),['crim']]=dataset['crim'].quantile(0.95)
sns.boxplot(dataset['crim'])
sns.distplot(dataset['crim'])
dataset['zn'].describe()
sns.boxplot(dataset['zn'])
dataset.loc[dataset['zn']<dataset['zn'].quantile(0.1),['zn']]=dataset['zn'].quantile(0.1)

dataset.loc[dataset['zn']>dataset['zn'].quantile(0.9),['zn']]=dataset['zn'].quantile(0.9)

sns.boxplot(dataset['zn'])
sns.distplot(dataset['zn'])
print("Skewness: %f" % dataset['zn'].skew())

print("Kurtosis: %f" % dataset['zn'].kurt())
dataset['indus'].describe()
sns.distplot(dataset['indus'])
sns.boxplot(dataset['indus'])
print("Skewness %f" % dataset['indus'].skew())

print("Kurtosis %f" % dataset['indus'].kurt())
dataset['chas'].describe()
dataset['chas'].value_counts()
sns.countplot(dataset['chas'])
dataset['rm'].describe()
sns.boxplot(dataset['rm'])
sns.distplot(dataset['rm'])

dataset['lstat'].describe()
sns.boxplot(dataset['lstat'])
sns.distplot(dataset['lstat'])

dataset['ptratio'].describe()
sns.boxplot(dataset['ptratio'])
dataset[dataset['ptratio']<dataset['ptratio'].quantile(0.02)]['ptratio']
dataset.loc[dataset['ptratio']<dataset['ptratio'].quantile(0.02),['ptratio']]=dataset['ptratio'].quantile(0.02)
dataset[dataset['ptratio']>dataset['ptratio'].quantile(0.98)]['ptratio']
dataset.loc[dataset['ptratio']>dataset['ptratio'].quantile(0.98),['ptratio']]=dataset['ptratio'].quantile(0.98)
sns.boxplot(dataset['ptratio'])
sns.distplot(dataset['ptratio'])

dataset['medv'].describe()
sns.boxplot(dataset['medv'])
sns.distplot(dataset['medv'])
feature_cols = ['crim', 'zn', 'indus','nox', 'rm','age', 'dis', 'rad', 'tax', 'ptratio','black', 'lstat']

target_col='medv'
X = dataset[feature_cols].values

y = dataset[target_col].values

print("X dimensions are",X.shape)

print("y dimensions are",y.shape)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print("X_train dimensions are",X_train.shape)

print("y_train dimensions are",y_train.shape)

print("X_test dimensions are",X_test.shape)

print("y_test dimensions are",y_test.shape)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print("dimensin of X_train",X_train.shape)
round(pd.Series(explained_variance),2)

#pca.components_.shape[0]

pd.DataFrame(pca.components_)



sns.heatmap(pd.DataFrame(pca.components_).corr(),annot=True)
# number of components

n_pcs= pca.components_.shape[0]



# get the index of the most important feature on EACH component

# LIST COMPREHENSION HERE

most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]



most_important_names = [feature_cols[most_important[i]] for i in range(n_pcs)]



dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}



# build the dataframe

df = pd.Series(dic.items())

dic.items()
# Fitting SVR to the dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'linear')

fitted=regressor.fit(X_train, y_train)

y_predict_train=fitted.predict(X_train)

y_predict_test=fitted.predict(X_test)
fitted.score(X_train,y_train)


from sklearn.metrics import r2_score

print('R square for train:', r2_score(y_train,y_predict_train))

print('R square for test:', r2_score(y_test,y_predict_test))