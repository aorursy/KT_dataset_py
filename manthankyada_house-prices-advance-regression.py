# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

pd.pandas.set_option('display.max_columns',None)

sns.set()

%matplotlib inline

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LinearRegression



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

dataset.shape
dataset.head()
features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]

for features in features_with_na:

    print(features, np.round(dataset[features].isnull().mean(),4),' % missing values')
for feature in features_with_na:

    data = dataset.copy()

    

    data[feature] = np.where(data[feature].isnull(),1,0)

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.title(feature)

    plt.show()
print("no. of houses ", len(dataset.Id))
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtype !='O']

print("No. of numerical variables ", len(numerical_features))

dataset[numerical_features].head()
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature ]

year_feature
for feature in year_feature:

    print(feature,dataset[feature].unique() )
dataset.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('year sold')

plt.ylabel('Sale Price')

plt.title('sale price vs year chart')
for feature in year_feature:

    if feature !='YrSold':

        data = dataset.copy()

        data[feature] = data['YrSold']-data[feature]

        

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel('year')

        plt.ylabel('Sale price')

        plt.show()
#let's try to find discrete feature

discrete_feature = [feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]

print("Discrete variable Count: {}".format(len(discrete_feature)))

discrete_feature
dataset[discrete_feature].head()
for feature in discrete_feature:

    data = dataset.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_feature = [feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]

print("Continuous feature {}".format(len(continuous_feature)))
for feature in continuous_feature:

    data = dataset.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel('cont')

    plt.title(feature)

    plt.show()
for feature in continuous_feature:

    data =dataset.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data['SalePrice'] = np.log(data['SalePrice'])

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.title(feature)

        plt.show()

        



for feature in continuous_feature:

    data = dataset.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature] = np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.show()
categorical_feature = [feature for feature in dataset.columns if data[feature].dtype=='O']

dataset[categorical_feature].head()
for feature in categorical_feature:

    print('The feature is {} and the unique categories are {}'.format(feature,len(dataset[feature].unique())))
for feature in categorical_feature:

    data = dataset.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
feature_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtype=='O']

for feature in feature_nan:

    print('{} : {}% missing value'.format(feature,np.round(dataset[feature].isnull().mean(),4)))
def replace_cat_feature(dataset,feature_nan):

    data = dataset.copy()

    data[feature_nan] = data[feature_nan].fillna('Missing')

    return data

dataset = replace_cat_feature(dataset,feature_nan)

dataset[feature_nan].isnull().sum()

#for numerical values

numerical_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtype!='O']



for feature in numerical_with_nan:

    print('{} : {}% missing value'.format(feature,np.round(dataset[feature].isnull().mean(),4)))
for feature in numerical_with_nan:

    median_value = dataset[feature].median()

    

    dataset[feature+'nan'] = np.where(dataset[feature].isnull(),1,0)

    dataset[feature].fillna(median_value,inplace=True)

    

dataset[numerical_with_nan].isnull().sum()
dataset.head(50)
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    dataset[feature] = dataset['YrSold']-dataset[feature] 
dataset.head()
dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
num_feature = ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']



for feature in num_feature:

    dataset[feature] = np.log(dataset[feature])
dataset.head()
categorical_feature = [feature for feature in dataset.columns if dataset[feature].dtype=='O']

categorical_feature
for feature in categorical_feature:

    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)

    temp_df = temp[temp>0.01].index

    dataset[feature] = np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
dataset.head()
for feature in categorical_feature:

    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index

    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}

    dataset[feature]=dataset[feature].map(labels_ordered)

dataset.head()
feature_scale = [feature for feature in dataset.columns if feature not in ['Id','SalePrice']]



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(dataset[feature_scale])
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],axis=1)

dataset = data
y_train = dataset['SalePrice']

x_train = dataset.drop(['Id','SalePrice'],axis=1)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005,random_state=0))

feature_sel_model.fit(x_train,y_train)


feature_sel_model.get_support()
selected_feat = x_train.columns[(feature_sel_model.get_support())]

selected_feat

x_train = x_train[selected_feat].head()
