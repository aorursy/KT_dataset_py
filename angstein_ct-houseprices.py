import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import stats

from scipy.stats import norm, skew

from statistics import mode

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from sklearn import metrics

from xgboost import XGBClassifier
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train
def missing_data(df):

    missing_values = round(df.isnull().sum()/len(df) * 100,2)

    df = missing_values.to_frame(name='MissingValuesPercentage')

    count = df[df['MissingValuesPercentage']> 0]['MissingValuesPercentage'].count()

    return df[['MissingValuesPercentage']].sort_values(by ='MissingValuesPercentage',ascending=False).head(count)
missing_data(df_train)
missing_data(df_test)
grouper= df_test.groupby(['data-location','type']).mean()
df_test['garage'] = df_test['garage'].transform(lambda x: x.fillna(0))

df_train['garage'] = df_train['garage'].transform(lambda x: x.fillna(0))
cat_cols = []

num_cols = []

for col_name in df_test.columns:

        if(df_test[col_name].dtype == 'object'):

            a = col_name

            cat_cols.append(a)

            

        else:

            a = col_name

            num_cols.append(a)
cat_cols
def conditional_impute(input_df, columns):

    df = input_df

    grouper= df.groupby(['data-location','type'])

    

    for i in columns:

        

        df[i] = grouper[i].transform(lambda x: x.fillna(round(x.median(),1)))     

    

    return df
df_train = conditional_impute(df_train, num_cols)

df_test = conditional_impute(df_test, num_cols)
def zero_imputer(df):

    

    for col_name in df.columns:

        if(df[col_name].dtype != 'object'):

            df[col_name] = df[col_name].transform(lambda x: x.fillna(round(x.mean(),1)))

    return df
zero_imputer(df_train)

zero_imputer(df_test)
cat_cols


df_test = df_test.drop(['data-url', 'data-isonshow'], 1)

df_train = df_train.drop(['data-url', 'data-isonshow'], 1)
df_train['data-date'] = df_train['data-date'].astype('datetime64[ns]')

df_test['data-date'] = df_test['data-date'].astype('datetime64[ns]')
df_train['Year_listed'] = df_train['data-date'].map(lambda x: x.year)

df_train['Month_listed'] = df_train['data-date'].map(lambda x: x.month)

df_test['Year_listed'] = df_test['data-date'].map(lambda x: x.year)

df_test['Month_listed'] = df_test['data-date'].map(lambda x: x.month)
df_train = df_train.drop('data-date', 1)

df_test = df_test.drop('data-date', 1)
df_train.head()
def type_maker(df, df1, col):

    dict1 = {'house':2,'apartment':1}

    for i in col:

        

        df = df.replace({i:dict1})

        df1 = df1.replace({i:dict1})

    

    return df, df1
df_train, df_test = type_maker(df_train, df_test,col= ['type'])
X_train = df_train.drop(['house-id', 'data-price'], axis = 1)

X_test = df_test.drop(['house-id'], axis = 1)

y_train = df_train[['data-price']]
df_train.head()
df_train_corr = df_train.corr()

top_feature = df_train_corr.index[(df_train_corr['data-price'] > 0.01)]

plt.subplots(figsize=(12, 8))

top_corr = df_train[top_feature].corr()

sns.heatmap(top_corr, annot=True, cmap = sns.color_palette("coolwarm", 10))

plt.show()
def encode_dummies(df):

    df_dummies = pd.get_dummies(df,  columns = ['data-location', 'area'] , drop_first = True)

    return df_dummies



X_ = encode_dummies(pd.concat([X_train,X_test], axis=0))

X_train = X_[:len(X_train)]

X_test = X_[len(X_train):]
(X_train.shape, X_test.shape)
y_train.shape
X_train.head()
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)
(mu, sigma) = norm.fit(y_train)



# 1. Plot Sale Price

sns.distplot(y_train , fit=norm);

plt.ylabel('Frequency')

plt.title('Price distribution')

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')



# Get the fitted parameters used by the function

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#skewness and kurtosis

print("Skewness: %f" % y_train.skew())

print("Kurtosis: %f" % y_train.kurt())
model = LinearRegression()

model.fit(X_train, y_train)
lasso = Lasso(alpha=0.02)



lasso.fit(X_train, y_train)
GBoost = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.02,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(X_train,y_train)
train_OLS = model.predict(X_train)

train_lasso = lasso.predict(X_train)

train_GBoost = GBoost.predict(X_train)

test_lm = model.predict(X_test).reshape(-1)

test_lasso = lasso.predict(X_test).reshape(-1)

test_GBoost = GBoost.predict(X_test).reshape(-1)

# Ordinary Least Squares

OLS_SalePricePredict =pd.DataFrame({'house-id': df_test['house-id'], 'price': test_lm})

OLS_SalePricePredict.to_csv('OLS_PricePredict.csv', index=False)



# Lasso Regression

Lasso_SalePricePredict =pd.DataFrame({'house-id': df_test['house-id'], 'price': test_lasso})

Lasso_SalePricePredict.to_csv('Lasso_PricePredict.csv', index=False)



# Gradient Boosting

GB_SalePricePredict =pd.DataFrame({'house-id': df_test['house-id'], 'price': test_GBoost})

GB_SalePricePredict.to_csv('GBoost_PricePredict.csv', index=False)



#Weighted

Weighted_SalePricePredict =pd.DataFrame({'house-id': df_test['house-id'], 'price': 0.5*(test_GBoost+test_lasso)})

Weighted_SalePricePredict.to_csv('Weighted_PricePredict.csv', index=False)