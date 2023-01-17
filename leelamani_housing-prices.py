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
X_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

X_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

X_train_original = X_train.copy(deep=True)

X_test_original = X_test.copy(deep=True)
X_train.drop('Id',inplace=True,axis=1)

X_test.drop('Id',inplace=True,axis=1)

#Shape of datasets

print("X train shape",X_train.shape)

print("X test shape",X_test.shape)
#Missing values

missing_values = X_train.isnull().sum()

missing_vales_percentage = (X_train.isnull().sum() /X_train.shape[0] ) * 100

missing_df = pd.DataFrame(index=X_train.columns,columns = ['total missing values','total missing values %'])

missing_df['total missing values'] = missing_values

missing_df['total missing values %'] = missing_vales_percentage

missing_df.sort_values('total missing values',ascending =  False,inplace=True)

missing_df.head(30)
X_train.info()
X_train.describe()
import matplotlib.pyplot as plt

X_train.hist(figsize=(20,20))



plt.show()
import seaborn as sns

numer_data_train = X_train.select_dtypes(exclude=np.object)

cat_data_train = X_train.select_dtypes(include=np.object)

corr = numer_data_train.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(X_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(X_train[cols], size = 2.5)

plt.show();
pd.set_option('display.max_colwidth', -1)

cat_data_train.describe(include='all')


sns.distplot(X_train['SalePrice'])

X_train['SalePrice'].describe()
sns.scatterplot(X_train['GrLivArea'],X_train['SalePrice'])
sns.scatterplot(X_train['OverallQual'],X_train['SalePrice'])
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([X_train['SalePrice'], X_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
for col in cat_data_train.columns:

    plt.figure()

    sns.boxplot(X_train[col],X_train['SalePrice'])

    sns.barplot
from sklearn.impute import SimpleImputer

#Impute numerical cloumns

y_train = X_train['SalePrice']

X_train.drop(['SalePrice'],inplace=True,axis=1)



numer_cols = X_train.select_dtypes(exclude=np.object).columns



numer_impute = SimpleImputer(strategy='mean')

numer_impute.fit(X_train[numer_cols])

X_train[numer_cols] = numer_impute.transform(X_train[numer_cols])

X_test[numer_cols] = numer_impute.transform(X_test[numer_cols])



#Missing values

cat_cols = X_train.select_dtypes(include=np.object).columns

missing_values = X_train[cat_cols].isnull().sum()

missing_vales_percentage = (X_train[cat_cols].isnull().sum() /X_train[cat_cols].shape[0] ) * 100

missing_df = pd.DataFrame(index=X_train[cat_cols].columns,columns = ['total missing values','total missing values %'])

missing_df['total missing values'] = missing_values

missing_df['total missing values %'] = missing_vales_percentage

missing_df.sort_values('total missing values',ascending =  False,inplace=True)

missing_df.head(30)

cat_cols = missing_df[missing_df['total missing values %'] < 15].index # Taking the columns names where missing values % is less than 15
#Impute categorical coluns with mode

cat_mode_vales = X_train[cat_cols].mode().values[0]

for i,col in enumerate(cat_cols):

    X_train[col].fillna(cat_mode_vales[i],inplace=True)

    X_test[col].fillna(cat_mode_vales[i],inplace=True)
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'] #High correlation with SalePrice



#https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data



train_set_lenth = len(X_train)

dataset = pd.concat(objs=[X_train,X_test], axis=0)

dataset_preprocessed = pd.get_dummies(dataset)

X_train = dataset_preprocessed[:train_set_lenth]

X_test = dataset_preprocessed[train_set_lenth:]

    
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)


feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['feature importance'])

feature_importances.sort_values('feature importance',ascending=False,inplace=True)



feature_importances.head(20)
golden_features = feature_importances.drop('BsmtQual_Ex').index.values.tolist()[0:20]
from catboost import CatBoostRegressor

cr = CatBoostRegressor()

cr.fit(X_train_original[golden_features],y_train)
predictions = cr.predict(X_test_original[golden_features])
to_submit = pd.DataFrame({'Id':X_test_original.Id,'SalePrice':predictions})
to_submit.to_csv('submission.csv', index=False)
cr.best_score_