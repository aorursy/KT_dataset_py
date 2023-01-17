import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

import missingno as msno



%matplotlib inline

style.use('fivethirtyeight')

sns.set(style='whitegrid', color_codes=True)
# classification

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



# regression

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
# model selection

from sklearn.model_selection import train_test_split, cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# evaluation metrics: regression

from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error



# evaluation metrics: classification

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



df = train.copy()

test_df = test.copy()
df.head()
df.columns
df.info()
df.isnull().sum()
msno.matrix(df)
df['season'].value_counts()
sns.factorplot(x='season', data=df, kind='count', size=5, aspect=1.5)
df['holiday'].value_counts()
sns.factorplot(x='holiday', data=df, kind='count')
df["workingday"].value_counts()
sns.factorplot(x='workingday', data=df, kind='count')
df['weather'].value_counts()
sns.factorplot(x='weather', data=df, kind='count')
df.describe()
sns.boxplot(data=df[["temp",'atemp','humidity','windspeed']])

fig = plt.gcf()

fig.set_size_inches(6,3)
a = df['windspeed']

b = a[a > 17].value_counts()

b
df.temp.unique()

fig, axes = plt.subplots(2,2)



axes[0,0].hist(x='temp', data=df, bins=20, edgecolor='black', linewidth=2)

axes[0,0].set_title('variation of temp')



axes[0,1].hist(x='atemp', data=df, bins=20, edgecolor='black', linewidth=2)

axes[0,1].set_title('variation of atemp')



axes[1,0].hist(x='windspeed', data=df, bins=20, edgecolor='black', linewidth=2)

axes[1,0].set_title('variation of windspeed')



axes[1,1].hist(x='humidity', data=df, bins=20, edgecolor='black', linewidth=2)

axes[1,1].set_title('variation of humidity')



fig.set_size_inches(8,6)
cor_mat = df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig = plt.gcf()

fig.set_size_inches(10,8)

sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)
season = pd.get_dummies(df['season'], prefix='season')

df = pd.concat([df,season], axis=1)



season = pd.get_dummies(df['season'], prefix='season')

test_df = pd.concat([test_df,season], axis=1)
df.head()
test_df.head()
weather = pd.get_dummies(df['weather'], prefix='weather')

df = pd.concat([df,weather], axis=1)



weather = pd.get_dummies(df['weather'], prefix='weather')

test_df = pd.concat([test_df,weather], axis=1)
df.head()
test_df.head()
df.drop(['season','weather'], inplace=True, axis=1)

test_df.drop(['season','weather'], inplace=True, axis=1)
df.head()
test_df.head()
df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]

df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]

df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]

df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]

df['year'] = df['year'].map({2011:0, 2012:1})
df.head()
test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]

test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]

test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = test_df['year'].map({2011:0, 2012:1})
df.drop('datetime', axis=1, inplace=True)

df.head()
cor_mat = df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig = plt.gcf()

fig.set_size_inches(14,14)

sns.heatmap(cor_mat, mask=mask, annot=True, cbar=True, square=True)
df.drop(['casual','registered'], inplace=True, axis=1)

df.head()
sns.factorplot(x='hour', y='count', data=df, kind='bar', size=5, aspect=1.5)
sns.factorplot(x='month', y='count', data=df, kind='bar', size=5, aspect=1.5)
new_df = df.copy()

new_df['temp_bin'] = np.floor(new_df['temp'])//5

new_df['temp_bin'].unique()
new_df.head()
sns.factorplot(x='temp_bin', y='count', data=new_df, kind='bar')
df.columns.to_series().groupby(df.dtypes).groups
x_train, x_test, y_train, y_test = train_test_split(df.drop('count',axis=1), df['count'], test_size=0.25, random_state=42)
models = [RandomForestRegressor(), AdaBoostRegressor(), BaggingRegressor(), SVR(), KNeighborsRegressor()]

model_names = ['RandomForestRegressor', 'AdaBoostRegressor', 'BaggingRegressor', 'SVR', 'KNeighborsRegressor']



rmsle = []

d = {}



for model in range(len(models)):

    clf = models[model]

    clf.fit(x_train, y_train)

    test_pred = clf.predict(x_test)

    rmsle.append(np.sqrt(mean_squared_log_error(test_pred, y_test)))

    

d = {'Modeling Algorithm':model_names, 'RMSLE':rmsle}
d
rmsle_frame = pd.DataFrame(d)

rmsle_frame
# lets tune a bit for random forest regression



params_dict = { 'n_estimators':[500], 'n_jobs':[-1], 'max_features':['auto','sqrt','log2'] }

clf_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params_dict, scoring='neg_mean_squared_log_error')



clf_rf.fit(x_train, y_train)

pred = clf_rf.predict(x_test)



print((np.sqrt(mean_squared_log_error(pred, y_test))))
clf_rf.best_params_