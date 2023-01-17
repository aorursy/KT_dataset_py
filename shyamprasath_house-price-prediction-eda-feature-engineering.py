import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import r2_score, mean_squared_log_error

from sklearn.model_selection import RandomizedSearchCV



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 100)
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head()
feature_with_na = [i for i in df.columns if df[i].isnull().sum()>1]



for i in feature_with_na:

    print('{}: {} % missing values'.format(i, round(df[i].isnull().mean(),3)))
for i in feature_with_na:

    df1 = df.copy()

    df1[i] = np.where(df1[i].isnull(),1,0)

    

    df1.groupby(i)['SalePrice'].median().plot.bar()

    plt.title(i)

    plt.show()
numerical_feature = [i for i in df.columns if df[i].dtypes != 'O']

numerical_feature
year_feature = [ i for i in  numerical_feature if 'Yr' in  i or 'Year' in i]
for i in year_feature:

    print(i, df[i].unique())

    print('-'*50)
# Lets Analyse the temporal datetime variable



df.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year sold')

plt.ylabel('Median house price')

plt.show()
# Here we will compare the difference betweeen all the year feature with salesprice



for i in year_feature:

    if i != 'YrSold':

        df1 = df.copy()

        df1[i] = df1['YrSold'] - df1[i]

        

        plt.scatter(df1[i], df1['SalePrice'])

        plt.xlabel(i)

        plt.ylabel('Saleprice')

        plt.show()
Discrete_feature = [i for i in numerical_feature if len(df[i].unique())< 25 and i not in year_feature+['Id']]

Discrete_feature
df[Discrete_feature].head()
for i in Discrete_feature:

    df.groupby(i)['SalePrice'].median().plot.bar()

    plt.xlabel(i)

    plt.ylabel('Saleprice')

    plt.title(i)

    plt.show()
cont_feature = [i for i in numerical_feature if i not in Discrete_feature+ year_feature + ['Id']]

cont_feature
for i in cont_feature:

    df[i].hist(bins = 25)

    plt.xlabel(i)

    plt.ylabel('Count')

    plt.title(i)

    plt.show()
for i in cont_feature:

    df1 = df.copy()

    if 0 in df[i].unique():

        pass

    else:

        df1[i] = np.log(df[i])

        df1[i].hist(bins = 25)

        plt.xlabel(i)

        plt.ylabel('Count')

        plt.title(i)

        plt.show()

        plt.scatter(df1[i], df1['SalePrice'])

        plt.xlabel(i)

        plt.ylabel('Count')

        plt.title(i)

        plt.show()
for i in cont_feature:

    df1 = df.copy()

    if 0 in df[i].unique():

        pass

    else:

        df1[i] = np.log(df[i])

        df1.boxplot(i)

        plt.ylabel(i)

        plt.title(i)

        plt.show()
cat_feature = [i for i in df.columns if df[i].dtypes == 'O' ]

cat_feature
df[cat_feature].head()
for i in cat_feature:

    print('The Feauture is  {} and number of unique values is {}'.format(i, len(df[i].unique())))
# Find out relation with the dependent variable

for i in cat_feature:

    df.groupby(i)['SalePrice'].median().plot.bar()

    plt.xlabel(i)

    plt.ylabel('SalePrice')

    plt.title(i)

    plt.show()
cat_missing_col = [i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtypes=='O']



for i in cat_missing_col:

    print('{}: {}% Missing values'.format(i, round(df[i].isnull().mean(),3)))
# Replace missings with new variable

def replace_cat_feauture(df, cat_missing_col):

    df[cat_missing_col] = df[cat_missing_col].fillna('Missing')

    return df



df = replace_cat_feauture(df,cat_missing_col)
df[cat_missing_col].isnull().sum()
num_missing_col = [i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtypes!='O']



for i in num_missing_col:

    print('{}: {}% Missing values'.format(i, round(df[i].isnull().mean(),3)))
# Replace the numerical missing value

for i in num_missing_col:

    df[i+'_nan'] = np.where(df[i].isnull(),1,0) # If null then label it 1 else 0

    df[i] = df[i].fillna(df[i].median())

    

df[num_missing_col].isnull().sum()
# Temporal Variable (Date variables)



for i in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:

    df[i] = df['YrSold'] - df[i]
df.head()
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']



for i in num_features:

    df[i] = np.log(df[i])
df.head()
cat_feature = [i for i in df.columns if df[i].dtypes=='O']

cat_feature
for i in cat_feature:

    temp =  df[i].value_counts()/len(df)

    temp_df = temp[temp>0.01].index

    df[i] = np.where(df[i].isin(temp_df), df[i], 'Rare_var')
df.head(35)
for i in cat_feature:

    labels_ordered=df.groupby([i])['SalePrice'].mean().sort_values().index

    labels_ordered={k:i for i,k in enumerate(labels_ordered)}    # enumerate is a iterable object

    df[i]=df[i].map(labels_ordered)
df.head()
feature_scale = df.columns.difference(['Id', 'SalePrice'])



scaler = MinMaxScaler()

scaler.fit(df[feature_scale])
scaler.transform(df[feature_scale])
# transform the train and test set, and add on the Id and SalePrice variables

df_final = pd.concat([df[['Id', 'SalePrice']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)],

                    axis=1)
df_final.head()
df_final.shape
x_train = df_final.drop(['Id', 'SalePrice'], axis = 1)

y_train = df_final['SalePrice']
# Apply feature selection, select the suitable alpha value, the bigger the alpha value the less feature will be selectes,

# then use selectfrom model, which will select the features with coefficient are non zero



feature_sel_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 0))

feature_sel_model.fit(x_train, y_train)
selected_feature = x_train.columns[feature_sel_model.get_support()]



print('The Total Features: ', len(x_train.columns))

print('The Selected Features: ', len(selected_feature))

print('Features with coefficients shrank to zero : {}'.format(np.sum(feature_sel_model.estimator_.coef_==0)))
selected_feature
x_train = x_train[selected_feature]

x_train.head()