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
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

## Display all the columns and rows  of the dataframe

pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
train=pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.shape
features_with_nan=[features for features in train.columns if train[features].isnull().sum()>0]
len(features_with_nan)
nan_list=[]
for features in features_with_nan:
    nan_list.append([features,train[features].isnull().sum()])

nan_list.sort(key=lambda x:x[1])
#print the missing value columns based on sorted order
for i in nan_list:
    print(i[0],"--->",i[1])
#Let's plot some diagram for this relationship

data=train.copy()
#data.head()
for features in features_with_nan:
    data[features]=np.where(data[features].isnull(),1,0)
    data.groupby(features)['SalePrice'].median().plot.bar()
    plt.title(features)
    plt.show()
# list of numerical variables
numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']# 'O' means object

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
train[numerical_features].head()
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature
# let's explore the content of these year variables
for feature in year_feature:
    print(feature, train[feature].unique())
## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")

for feature in year_feature:
    if feature!='YrSold':
        plt.scatter(train[feature],train['SalePrice'],color='b')
        #sns.regplot(x=train[feature], y=train['SalePrice'],color='r')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show() 


discrete_feature=[feature for feature in numerical_features if len(train[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
train[discrete_feature].head()
#Use bar plots for discrete type of data
## Lets Find the realtionship between them and Sale PRice
for feature in discrete_feature:
    data=train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))
train[continuous_feature].head()
## Lets analyse the continuous values by creating histograms to understand the distribution

##A histogram is a plot that lets you discover and show the underlying frequency distribution
## (shape) of a set of continuous data.
##This allows the inspection of the data for its underlying distribution (e.g., normal distribution), outliers, skewness
data=train.copy()
for feature in continuous_feature:
 
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()

## We will be using logarithmic transformation

data=train.copy()
data['SalePrice']=np.log(data['SalePrice'])
for feature in continuous_feature:
    
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
#boxplot is only used for contious variables
data=train.copy()
for feature in continuous_feature:
    
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
# the black small round circles are outliers
# Categorial variables
categorical_features=[feature for feature in train.columns if train[feature].dtypes=='O']
categorical_features
train[categorical_features].head()
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(train[feature].unique())))
data=train.copy()
for feature in categorical_features:
    
    data.groupby(feature)['SalePrice'].median().plot.bar()  #the reason to take median is that we have seen there are lot outliers
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
