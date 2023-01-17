import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
df.head(2)
df.describe().transpose()
df.info()
df.shape
df.isnull().sum()
100*df.isnull().sum()/df.shape[0]
for col in df.columns:

  df[col] = df[col].interpolate(method='linear',limit_direction='both')
df.isnull().sum()
df.columns
fig, axes = plt.subplots(7,2,figsize=(5,25))

df.boxplot(column='Population', ax=axes[0,0])



df.boxplot(column='Schooling',ax=axes[0,1])



df.boxplot(column='Income composition of resources',ax=axes[1,0])

df.boxplot(column='GDP',ax=axes[1,1])



df.boxplot(column='Total expenditure',ax=axes[2,0])

df.boxplot(column='Polio',ax=axes[2,1])



df.boxplot(column='Adult Mortality',ax=axes[3,0])

df.boxplot(column='Alcohol',ax=axes[3,1])



df.boxplot(column='Hepatitis B',ax=axes[4,0])

df.boxplot(column=' thinness 5-9 years',ax=axes[4,1])



df.boxplot(column=' BMI ',ax=axes[5,0])

df.boxplot(column='under-five deaths ',ax=axes[5,1])



df.boxplot(column=' HIV/AIDS',ax=axes[6,0])

df.boxplot(column='Diphtheria ',ax=axes[6,1])
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df_clean = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df_clean.shape
df_clean.corr()
plt.figure(figsize=(12,10))

sns.heatmap(df_clean.corr(),cmap='viridis',annot=True)
sns.pairplot(df_clean)
df_clean.corr()['Life expectancy '].sort_values(ascending=False)
df_clean.corr()['Life expectancy '].sort_values(ascending=False).plot(kind='bar')
less_than_65 = df_clean[df_clean['Life expectancy '] < 65]

sns.lmplot(x='percentage expenditure',y='Life expectancy ',data=less_than_65)
sns.lmplot(x='Schooling',y='Life expectancy ',data=df_clean)
df_clean['Status'].unique()
dmap = {'Developed':1,'Developing':0}

df_clean['Status'] = df_clean['Status'].map(dmap)
df_clean.head(4)
y = df_clean['Life expectancy '].values

X = df_clean.drop(['Country','Life expectancy '],axis=1).values # Too many countries to create dummy variables on
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print('Coefficients:\n',lm.coef_)

print('\n')

print('Intercept:\n',lm.intercept_)
predictions = lm.predict(X_test)
plt.figure(figsize=(12,8))

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions),bins=50);