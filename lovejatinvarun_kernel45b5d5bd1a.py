# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')
data="/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv"

df=pd.read_csv(data)
df.shape
df.head()
columname=df.columns

columname
df.drop(['RISK_MM'], axis=1, inplace=True)
df.head()
df.info()
df.describe()
df.describe(include='all')
df['RainTomorrow'].isnull().sum()
df['RainTomorrow'].unique()
x=df['RainTomorrow'].value_counts()/len(df)

x=x*100

x
f, ax=plt.subplots(figsize=(4,8))

ax=sns.countplot(x='RainTomorrow', data=df, palette='Set1')

plt.show()
cat=[var for var in df.columns if df[var].dtype=='O']

cat
df[cat].head()
df[cat].isnull().sum()
for var in cat:

    print(df[var].value_counts())
df['Date']=pd.to_datetime(df['Date'])

df['Year']=df['Date'].dt.year

df['Month']=df['Date'].dt.month

df['Day']=df['Date'].dt.day

df['Day'].head()
df.head()

pd.get_dummies(df.Location, drop_first=True).head()
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
df.head()
plt.figure(figsize=(15,10))





plt.subplot(2, 2, 1)

fig=df.boxplot(column='Rainfall')

fig.set_title('')

fig.set_ylabel('Rainfall')





plt.subplot(2, 2, 2)

fig = df.boxplot(column='Evaporation')

fig.set_title('')

fig.set_ylabel('Evaporation')





plt.subplot(2, 2, 3)

fig = df.boxplot(column='WindSpeed9am')

fig.set_title('')

fig.set_ylabel('WindSpeed9am')





plt.subplot(2, 2, 4)

fig = df.boxplot(column='WindSpeed3pm')

fig.set_title('')

fig.set_ylabel('WindSpeed3pm')
plt.figure(figsize=(15,10))





plt.subplot(2, 2, 1)

fig = df.Rainfall.hist(bins=10)

fig.set_xlabel('Rainfall')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 2)

fig = df.Evaporation.hist(bins=10)

fig.set_xlabel('Evaporation')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 3)

fig = df.WindSpeed9am.hist(bins=10)

fig.set_xlabel('WindSpeed9am')

fig.set_ylabel('RainTomorrow')





plt.subplot(2, 2, 4)

fig = df.WindSpeed3pm.hist(bins=10)

fig.set_xlabel('WindSpeed3pm')

fig.set_ylabel('RainTomorrow')
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)

Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)

Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)

print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)

Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)

Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)

print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)

Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)

Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)

print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)

Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)

Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)

print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
correlation = df.corr()

plt.figure(figsize=(16,12))

plt.title('Correlation Heatmap of Rain in Australia Dataset')

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']

sns.pairplot(df[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')

plt.show()


X = df.drop(['RainTomorrow'], axis=1)



y = df['RainTomorrow']

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X_train.shape, X_test.shape
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical.pop(0)

numerical


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
for df1 in [X_train, X_test]:

    for col in numerical:

        col_median=X_train[col].median()

        df1[col].fillna(col_median, inplace=True) 
for df2 in [X_train, X_test]:

    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)

    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)

    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
def max_value(df3, variable, top):

    return np.where(df3[variable]>top, top, df3[variable])



for df3 in [X_train, X_test]:

    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)

    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)

    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)

    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
X_test[numerical].describe()
X_train[numerical].describe()
categorical
X_train[categorical].head()
import category_encoders as ce



encoder = ce.BinaryEncoder(cols=['RainToday'])



X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)

X_train.head()
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],

                     pd.get_dummies(X_train.Location), 

                     pd.get_dummies(X_train.WindGustDir),

                     pd.get_dummies(X_train.WindDir9am),

                     pd.get_dummies(X_train.WindDir3pm)], axis=1)



X_train.head()
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],

                     pd.get_dummies(X_test.Location), 

                     pd.get_dummies(X_test.WindGustDir),

                     pd.get_dummies(X_test.WindDir9am),

                     pd.get_dummies(X_test.WindDir3pm)], axis=1)



X_test.head()
cols = X_train.columns



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



X_train = scaler.fit_transform(X_train)



X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])

X_test = pd.DataFrame(X_test, columns=[cols])

X_train.describe()
from sklearn.linear_model import LogisticRegression





# instantiate the model

logreg = LogisticRegression(solver='liblinear', random_state=0)





# fit the model

logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)



y_pred_test

from sklearn.metrics import accuracy_score



print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))