!pip install feature_engine
import pandas as pd 

import numpy as np 

import missingno as mno

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler

from sklearn.impute import SimpleImputer, MissingIndicator

from feature_engine import missing_data_imputers as mdi

from feature_engine.categorical_encoders import OneHotCategoricalEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from feature_engine import discretisers as dsc

from feature_engine import categorical_encoders as ce
data = pd.read_csv("../input/singapore-airbnb/listings.csv")
data.sample(4)
data.shape
x = data.iloc[:,4:16].drop(columns=['price'])

y = data.iloc[:,9]

x.info()
mno.matrix(x, figsize = (20, 6))
x=x.drop(columns=['last_review'])
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

missing_column_mean = ["reviews_per_month", ]

imputer = imputer.fit(x[missing_column_mean].values)

x[missing_column_mean] = imputer.transform(data[missing_column_mean].values)
le=LabelEncoder()

col_names=x.select_dtypes(object).columns.astype(str)

for i in col_names:

    x[i] = le.fit_transform(x[i])
from sklearn.ensemble import RandomForestRegressor

kf = KFold(10, shuffle=True, random_state=1)

score=[]

mse=[]

for l_train, l_valid in kf.split(x):

    x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

    y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]

    

    rf=RandomForestRegressor(n_estimators=100)

    rf.fit(x_train, y_train)

    pred=rf.predict(x_valid)

    m=mean_absolute_error(y_valid, pred)

    s=rf.score(x_valid, y_valid)

    score.append(s)

    mse.append(m)
sns.distplot(score)
sns.distplot(mse)
importance = pd.Series(np.abs(rf.feature_importances_))

importance.index = x.columns

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(18,6))
data = data.iloc[:,4:16]
categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables'.format(len(categorical)))
data['last_review'] = pd.to_datetime(data.last_review)

data['last_review_day'] = data['last_review'].dt.day

data['last_review_month'] = data['last_review'].dt.month

data['last_review_year'] = data['last_review'].dt.year

data['last_review_weekday'] = data['last_review'].dt.dayofweek

data=data.drop(columns=["last_review"])

categorical.remove("last_review")
numerical = [var for var in data.columns if data[var].dtype!='O']

discrete = []

for var in numerical:

    if len(data[var].unique()) < 20:

        print(var, ' values: ', data[var].unique())

        discrete.append(var)

print()

print('There are {} discrete variables'.format(len(discrete)))
numerical = [var for var in numerical if var not in discrete]

numerical.remove("price")

print('There are {} numerical and continuous variables'.format(len(numerical)))
for var in data.columns:

    if data[var].isnull().sum() > 0:

        print(var, data[var].isnull().mean())



missing = [var for var in data.columns if data[var].isnull().sum() > 0]
for var in numerical:

    plt.figure(figsize=(6,4))

    plt.subplot(1, 2, 1)

    fig = data.boxplot(column=var)

    fig.set_title('')

    fig.set_ylabel(var)

    

    plt.subplot(1, 2, 2)

    fig = data[var].hist(bins=20)

    fig.set_ylabel('Number of houses')

    fig.set_xlabel(var)



    plt.show()
for var in discrete:

    data.groupby(var)['price'].median().plot()

    plt.ylabel('Median Price per label')

    plt.title(var)

    plt.show()
data[categorical].nunique().plot.bar(figsize=(10,6))

plt.title('CARDINALITY: Number of categories in categorical variables')

plt.xlabel('Categorical variables')

plt.ylabel('Number of different categories')
for col in numerical:

    if data[col].isnull().mean() > 0:

        print(col, data[col].isnull().mean())
x = data.drop(columns=['price'])

y = data.iloc[:,6]

x[discrete] = x[discrete].astype('O')
pipe = Pipeline([

    ('missing_ind', mdi.AddNaNBinaryImputer(

        variables=["reviews_per_month", "last_review_day"]+discrete)),

    ('imputer_num', mdi.MeanMedianImputer(imputation_method='median',

                                          variables=["reviews_per_month", "last_review_day"])),

    ('imputer_cat', mdi.CategoricalVariableImputer(variables=categorical)),

    ('freq_ca', mdi.FrequentCategoryImputer(variables=discrete))

])



pipe.fit(x)

x = pipe.transform(x)

x.isnull().mean()
x[discrete] = x[discrete].astype('O')
pipe = Pipeline([

    ('categorical_enc', ce.OrdinalCategoricalEncoder(

        encoding_method='ordered', variables=categorical+discrete)),

    ('discretisation', dsc.EqualFrequencyDiscretiser(

        q=5, return_object=True, variables=numerical)),

    ('encoding', ce.OrdinalCategoricalEncoder(

        encoding_method='ordered', variables=numerical))])



pipe.fit(x, y)

x = pipe.transform(x)
from sklearn.ensemble import RandomForestRegressor

kf = KFold(10, shuffle=True, random_state=1)

score=[]

mse=[]

for l_train, l_valid in kf.split(x):

    x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

    y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]

    rf=RandomForestRegressor(n_estimators=100)

    rf.fit(x_train, y_train)

    pred=rf.predict(x_valid)

    m=mean_absolute_error(y_valid, pred)

    s=rf.score(x_valid, y_valid)

    score.append(s)

    mse.append(m)
sns.distplot(score)
sns.distplot(mse)
importance = pd.Series(np.abs(rf.feature_importances_))

importance.index = x.columns

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(18,6))