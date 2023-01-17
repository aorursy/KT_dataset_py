# basic

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline



# split data

from sklearn.model_selection import train_test_split



# scale data

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



# ML

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb

from sklearn.naive_bayes import GaussianNB





# score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn import metrics





pd.pandas.set_option('display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/weatherAUS.csv')

data.head(3)
data.drop(['Location','RISK_MM'], axis=1, inplace=True)
data['RainTomorrow'] = data['RainTomorrow'].map( {'No': 0, 'Yes': 1} ).astype(int)
data.dtypes
len(data.dtypes)
data.describe()
data['DateNew']= pd.to_datetime(data.Date)
data['Month'] = data['DateNew'].dt.month

data['Day'] = data['DateNew'].dt.day

data['Year'] = data['DateNew'].dt.year
data.drop(['Date','DateNew'], axis=1, inplace=True)
categorical = [var for var in data.columns if data[var].dtype=='O']

#list(set(categorical))

categorical
numerical = [var for var in data.columns if data[var].dtype!='O']

#list(set(categorical))

numerical
data.isnull().mean()
plt.figure(figsize=(12,8))

data.boxplot(column=['MinTemp','MaxTemp','Evaporation','Sunshine'])
plt.figure(figsize=(12,8))

data.boxplot(column=['WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm'])
plt.figure(figsize=(12,8))

data.boxplot(column=['Pressure9am','Pressure3pm'])
plt.figure(figsize=(12,8))

data.boxplot(column=['Cloud9am','Temp3pm','Temp9am'])
plt.figure(figsize=(12,8))

data.boxplot(column=['Rainfall'])
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.MinTemp.hist(bins=20)

fig.set_ylabel('Temp')

fig.set_xlabel('MinTemp')



plt.subplot(1, 2, 2)

fig = data.MaxTemp.hist(bins=20)

fig.set_ylabel('Temp')

fig.set_xlabel('MaxTemp')
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.Rainfall.hist(bins=20)

fig.set_ylabel('Rainfall')

fig.set_xlabel('mm')



plt.subplot(1, 2, 2)

fig = data.Evaporation.hist(bins=20)

fig.set_ylabel('Evaporation')

fig.set_xlabel('mm')
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.WindSpeed9am.hist(bins=20)

fig.set_ylabel('WindSpeed9am')

fig.set_xlabel('WindSpeed9am')



plt.subplot(1, 2, 2)

fig = data.WindSpeed3pm.hist(bins=20)

fig.set_ylabel('WindSpeed3pm')

fig.set_xlabel('WindSpeed3pm')
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.WindGustSpeed.hist(bins=20)

fig.set_ylabel('WindGustSpeed')

fig.set_xlabel('WindGustSpeed')



plt.subplot(1, 2, 2)

fig = data.Humidity9am.hist(bins=20)

fig.set_ylabel('Humidity9am')

fig.set_xlabel('Humidity9am')
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.Pressure9am.hist(bins=20)

fig.set_ylabel('Pressure9am')

fig.set_xlabel('Pressure9am')



plt.subplot(1, 2, 2)

fig = data.Pressure3pm.hist(bins=20)

fig.set_ylabel('Pressure3pm')

fig.set_xlabel('Pressure3pm')
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = data.Temp9am.hist(bins=20)

fig.set_ylabel('Temp9am')

fig.set_xlabel('Temp9am')



plt.subplot(1, 2, 2)

fig = data.Temp3pm.hist(bins=20)

fig.set_ylabel('Temp3pm')

fig.set_xlabel('Temp3pm')
Upper_boundary = data.MinTemp.mean() + 3* data.MinTemp.std()

Lower_boundary = data.MinTemp.mean() - 3* data.MinTemp.std()

print('MinTemp outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.MaxTemp.mean() + 3* data.MaxTemp.std()

Lower_boundary = data.MaxTemp.mean() - 3* data.MaxTemp.std()

print('MaxTemp outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.WindSpeed3pm.mean() + 3* data.WindSpeed3pm.std()

Lower_boundary = data.WindSpeed3pm.mean() - 3* data.WindSpeed3pm.std()

print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.WindGustSpeed.mean() + 3* data.WindGustSpeed.std()

Lower_boundary = data.WindGustSpeed.mean() - 3* data.WindGustSpeed.std()

print('WindGustSpeed outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.Pressure9am.mean() + 3* data.Pressure9am.std()

Lower_boundary = data.Pressure9am.mean() - 3* data.Pressure9am.std()

print('Pressure9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.Pressure3pm.mean() + 3* data.Pressure3pm.std()

Lower_boundary = data.Pressure3pm.mean() - 3* data.Pressure3pm.std()

print('Pressure3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.Temp9am.mean() + 3* data.Temp9am.std()

Lower_boundary = data.Temp9am.mean() - 3* data.Temp9am.std()

print('Temp9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
Upper_boundary = data.Temp3pm.mean() + 3* data.Temp3pm.std()

Lower_boundary = data.Temp3pm.mean() - 3* data.Temp3pm.std()

print('Temp3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
IQR = data.Rainfall.quantile(0.75) - data.Rainfall.quantile(0.25)

Lower_fence = data.Rainfall.quantile(0.25) - (IQR * 3)

Upper_fence = data.Rainfall.quantile(0.75) + (IQR * 3)

print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = data.Evaporation.quantile(0.75) - data.Evaporation.quantile(0.25)

Lower_fence = data.Evaporation.quantile(0.25) - (IQR * 3)

Upper_fence = data.Evaporation.quantile(0.75) + (IQR * 3)

print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = data.WindSpeed9am.quantile(0.75) - data.WindSpeed9am.quantile(0.25)

Lower_fence = data.WindSpeed9am.quantile(0.25) - (IQR * 3)

Upper_fence = data.WindSpeed9am.quantile(0.75) + (IQR * 3)

print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = data.Humidity9am.quantile(0.75) - data.Humidity9am.quantile(0.25)

Lower_fence = data.Humidity9am.quantile(0.25) - (IQR * 3)

Upper_fence = data.Humidity9am.quantile(0.75) + (IQR * 3)

print('Humidity9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
for var in ['WindGustDir',  'WindDir9am', 'WindDir3pm']:

    print(data[var].value_counts() / np.float(len(data)))

    print()
for var in categorical:

    print(var, ' contains ', len(data[var].unique()), ' labels')
X = data.drop('RainTomorrow', axis=1)
y = data[['RainTomorrow']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,

                                                    random_state=0)

X_train.shape, X_test.shape
numerical = [var for var in X_train.columns if data[var].dtype!='O']
for col in numerical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
for col in numerical:

    X_train[col] = X_train[col].fillna((X_train[col].mean()))
for col in numerical:

    X_test[col] = X_test[col].fillna((X_test[col].mean()))
for col in categorical:

    if X_train[col].isnull().mean()>0:

        print(col, X_train[col].isnull().mean())
for df in [X_train, X_test]:

    df['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)

    df['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)

    df['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

    df['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
X_train.isnull().sum()
X_test.isnull().sum()
to_describe = ['MinTemp','MaxTemp','WindSpeed3pm','WindGustSpeed','Pressure9am','Pressure3pm',

               'Temp9am','Temp3pm','Rainfall','Evaporation','WindSpeed9am','Humidity9am']
X_train[to_describe].describe()
def top_code(df, variable, top):

    return np.where(df[variable]>top, top, df[variable])
def bottom_code(df, variable, bottom):

    return np.where(df[variable]<bottom, bottom, df[variable])
for df in [X_train, X_test]:

    df['MinTemp'] = top_code(df, 'MinTemp', 31.38)

    df['MinTemp'] = bottom_code(df, 'MinTemp', -7.02)

    df['MaxTemp'] = top_code(df, 'MaxTemp', 44.57)

    df['MaxTemp'] = bottom_code(df, 'MaxTemp', 1.87)

    df['WindSpeed3pm'] = top_code(df, 'WindSpeed3pm', 45.04)

    df['WindGustSpeed'] = top_code(df, 'WindGustSpeed', 80.75)

    df['Pressure9am'] = top_code(df, 'Pressure9am', 1038.97)

    df['Pressure9am'] = bottom_code(df, 'Pressure9am', 996.33)

    df['Pressure3pm'] = top_code(df, 'Pressure3pm', 1036.36)

    df['Pressure3pm'] = bottom_code(df, 'Pressure3pm', 994.14)

    df['Temp9am'] = top_code(df, 'Temp9am', 36.46)

    df['Temp9am'] = bottom_code(df, 'Temp9am', -2.49)

    df['Temp3pm'] = top_code(df, 'Temp3pm', 42.50)

    df['Temp3pm'] = bottom_code(df, 'Temp3pm', 0.87)

    

    df['Rainfall'] = top_code(df, 'Rainfall', 3.20)

    df['Evaporation'] = top_code(df, 'Evaporation', 21.80)

    df['WindSpeed9am'] = top_code(df, 'WindSpeed9am', 55.00)

    df['Humidity9am'] = top_code(df, 'Humidity9am', 161.00)
X_train[to_describe].describe()
X_test[to_describe].describe()
categorical
for df in [X_train, X_test]:

    df['WindGustDir']  = pd.get_dummies(df.WindGustDir, drop_first=False)

    df['WindDir9am']  = pd.get_dummies(df.WindDir9am, drop_first=False)

    df['WindDir3pm']  = pd.get_dummies(df.WindDir3pm, drop_first=False)

    df['RainToday']  = pd.get_dummies(df.RainToday, drop_first=False)
mx = MinMaxScaler()
X_train_mx = mx.fit_transform(X_train)
X_test_mx = mx.fit_transform(X_test)
knn = KNeighborsClassifier()
knn.fit(X_train_mx, y_train)
predictions = knn.predict(X_test_mx)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
logreg = LogisticRegression()
logreg.fit(X_train_mx, y_train)
predictions = logreg.predict(X_test_mx)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
gaussian = GaussianNB()
gaussian.fit(X_train_mx, y_train)
predictions = gaussian.predict(X_test_mx)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))