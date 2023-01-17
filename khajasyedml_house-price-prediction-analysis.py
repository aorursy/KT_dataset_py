# Main aim is to understand more about the data



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns', None)
dataset = pd.read_csv('../input/housepricepredictions/train.csv')



# print shape of the dataset with rows and columns

print(dataset.shape)
# print top 5 records

dataset.head()
# Here will check the percentage of NaN values present in each feature

# Step-1: Make the list of features which has missing values

features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]



# Step-2: Feature name and percentage of missing values

#for feature in features_with_na:

    #print(feature, np.round(dataset[feature].isnull().mean(), 4), ' % of missing values')



df_train_missing = pd.DataFrame(np.round(dataset[features_with_na].isnull().mean(), 4), columns=[' % missing values'])

df_train_missing
data = dataset.copy()

for feature in features_with_na:

    # let's convert all the Nan values to 1, otherwise zero for easy plotting the relationship with the SalesPrice

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    # let's calculate the mean SalePrice where the information is missing or present

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.title(feature)

    plt.show()

print('Id of houses {}'.format(len(dataset.Id)))
# Lets see the data types present in our dataset

dataset.dtypes.unique()
# list of numerical variables

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables', len(numerical_features))



# Visualize the numerical features

dataset[numerical_features].head()
# list of variables that contain year information

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature
# let's explore the content of these year variables

for feature in year_feature:

    print(feature, dataset[feature].unique())
# Lets analyze the Temporal Datetime Variables

# We will check whether there is a relation between year the house is sold and Sale Price

dataset.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year Sold')

plt.ylabel('Median House Price')

plt.title('House Price vs Year Sold')
# Lets analyze the Temporal Datetime Variables

# We will check the relationship between these year features and SalesPrice 

# and see how price is changing over the years

# Here we will compare the difference between ALL years features with Sale Price



data = dataset.copy()

for feature in year_feature:

    if feature != 'YrSold':

        # we will capture the difference between year variable and year the house was sold for

        data[feature] = data['YrSold'] - data[feature]

        

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()

        
## Numerical variables are usually of 2 type

## 1. Continous variable and Discrete Variables

discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]

print('Total discrete variables: {}'.format(len(discrete_features)))
discrete_features
data[discrete_features].head()
## Lets Find the realtionship between discrete variables and SalePice

for feature in discrete_features:

    data = dataset.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_feature = [feature for feature in numerical_features if feature not in discrete_features+year_feature+['Id']]

print('Continuous feature count {}'.format(len(continuous_feature)))
## Lets analyse the continuous values by creating histograms to understand the distribution



data = dataset.copy()

for feature in continuous_feature:

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel('Count')

    plt.title(feature)

    plt.show()
# We will be using logarithmic transformation

data = dataset.copy()

for feature in continuous_feature:

    if 0 in data[feature].unique() or feature in data['SalePrice']:

        pass

    else:

        data[feature] = np.log(data[feature])

        data['SalePrice'] = np.log(data['SalePrice'])

        plt.scatter(data[feature], data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.title(feature)

        plt.show()
data = dataset.copy()

for feature in continuous_feature:

    if 0 in data[feature].unique(): # here we pass if the unique values

        pass

    else:

        data[feature] = np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes=='O']

#len(categorical_features)

categorical_features
# checking the cardinality of categorical variables

cat_count = []

for feature in categorical_features:

    cat_count.append(len(dataset[feature].unique()))

    

data_cat = pd.DataFrame({'Feature': categorical_features, 'No of Categories': cat_count})

data_cat
## Relationship between categorical variable and dependent feature SalePrice



data = dataset.copy()

for feature in categorical_features:

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
# Capture all the nan values

# Handle categorical features which are missing

features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

feature_missing = pd.DataFrame(np.round(dataset[features_nan].isnull().mean(),4), columns=[' % missing values'])

feature_missing
# Replace missing value with a new label

def repalce_categorical_feature(data, features_nan):

    data[features_nan] = data[features_nan].fillna('Missing')

    return data



data = dataset.copy()

dataset = repalce_categorical_feature(data, features_nan)

dataset[features_nan].isnull().sum()
dataset.head()
# Checking missing values for numerical variables

numerical_with_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']



# Print all the numerical nan variables and percentage of missing values

numerical_missing = pd.DataFrame(np.round(dataset[numerical_with_nan].isnull().mean(), 4), columns=[' % missing values'])

numerical_missing
# Replacing the numerical missing values

for feature in numerical_with_nan:

    # Shall replace by using median since there are outliers

    median_value = dataset[feature].median()

    

    # create a new feature to capture nan values

    dataset[feature+'nan'] = np.where(dataset[feature].isnull(),1,0)

    dataset[feature].fillna(median_value, inplace=True)



dataset[numerical_with_nan].isnull().sum()
dataset.head(10)
# Temporal variables(Date Time variables)

for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    dataset[feature] = dataset['YrSold'] - dataset[feature]



dataset.head()
dataset[['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']].head()
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:

    dataset[feature] = np.log(dataset[feature])

    

dataset.head()


categorical_features
for feature in categorical_features:

    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)

    temp_df = temp[temp>0.01].index

    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')

    

dataset.head(20)
for feature in categorical_features:

    labels_ordered = dataset.groupby([feature])['SalePrice'].mean().sort_values().index

    labels_ordered = {k:i for i,k in enumerate(labels_ordered, 0)}

    dataset[feature] = dataset[feature].map(labels_ordered)

    

dataset.head(10)
scaling_feature = [feature for feature in dataset.columns if feature not in ['Id', 'SalePrice']]

len(scaling_feature)
scaling_feature
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(dataset[scaling_feature])
scaler.transform(dataset[scaling_feature])
# transform the train and test set, and add on the Id and SalePrice variables

data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),

                 pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)], axis=1)
data.head()
data.to_csv('X_train.csv', index=False)
dataset = pd.read_csv('X_train.csv')

dataset.head()
# capture the dependent feature

y_train = dataset[['SalePrice']]
# Drop dependent feature from dataset

X_train = dataset.drop(['Id', 'SalePrice'], axis=1)
### Apply Feature Selection

# first, I specify the Lasso Regression model, and I

# select a suitable alpha (equivalent of penalty).

# The bigger the alpha the less features that will be selected.



# Then I use the selectFromModel object from sklearn, which

# will select the features which coefficients are non-zero



## for feature slection

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel



feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) #remember to set the seed, the random state in this function

feature_sel_model.fit(X_train, y_train)
feature_sel_model.get_support()
# let's print the number of total and selected features



# this is how we can make a list of the selected features

selected_feat = X_train.columns[(feature_sel_model.get_support())]



# Print stats

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

print('features with coefficients shrank to zero: {}'.format(np.sum(feature_sel_model.estimator_.coef_ == 0)))
selected_feat
X_train = X_train[selected_feat]

X_train.head()