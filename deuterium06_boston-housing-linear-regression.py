import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

import sklearn.datasets as datasets
data = datasets.load_boston()
data.keys() #to check the object attributes
print(data.DESCR) #to familiarize the data loaded
data.data.shape, data.target.shape #check if our data have the same array index
input = pd.DataFrame(data=data.data, columns=data.feature_names) #place input arrays into table

target = pd.DataFrame(data=data.target, columns=['Target']) #place target array into table
raw_data = pd.concat([input, target], axis=1) #merge input & target variables into one table

df = raw_data.copy()
df.info() #check the data types and whether if each of them has no non-null values
df.sample(random_state=20, n=10) #taking a glimpse of the data
columns_sorted = df.corr().abs().nlargest(14, 'Target').index

correlation_sorted = np.corrcoef(df[columns_sorted].values.T)



f, ax = plt.subplots(figsize = (9,7.5))

hm = sns.heatmap(abs(correlation_sorted), annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns_sorted.values, xticklabels=columns_sorted.values, cmap='summer')

plt.show()
sns.pairplot(df[columns_sorted], height=4)

sns.set(font_scale=3)

plt.show()
df_transformed = df.copy()
sns.set()

sns.distplot(df['Target'])
sns.set()

sns.distplot(np.log(df['Target']))
print ('Skew Before Transformation: ' + str(df['Target'].skew()))

print ('Skew After Transformation: ' + str(np.log(df['Target']).skew()))
df_transformed['Target'] = np.log(df_transformed['Target']) #replace the Target values with Logged Target values
df_transformed.head()
sns.scatterplot(df['LSTAT'], df['Target'])
sns.scatterplot(df['LSTAT'], df_transformed['Target'])
sns.scatterplot(np.log(df['LSTAT']), df_transformed['Target'])

sns.set()

plt.show()
sns.residplot(np.log(df['LSTAT']), df_transformed['Target'])

sns.set()

plt.show()
print (np.corrcoef(df[['LSTAT', 'Target']].values.T))

print (np.corrcoef(df_transformed[['LSTAT', 'Target']].values.T))
df_transformed['LSTAT'] = np.log(df['LSTAT']) #replace the Target values with Logged Target values

df_transformed.head()
df['RAD'].value_counts()
sns.boxplot(df['RAD'], df['Target'])
RAD_array = []



for i in df['RAD']:

    if i == 24.0:

        RAD_array.append(1)

    else:

        RAD_array.append(0)
df_transformed['RAD'] = RAD_array
df_transformed.sample(6)
df_transformed['RAD'].value_counts()
sns.boxplot(df_transformed['RAD'], df_transformed['Target'])
print (np.corrcoef(df[['RAD', 'Target']].values.T))

print (np.corrcoef(df_transformed[['RAD', 'Target']].values.T))
df['CHAS'].value_counts()
print (sns.boxplot(df['CHAS'], df['Target']))
df_scaled = preprocessing.scale(df_transformed)
df_scaled = pd.DataFrame(data=df_scaled, columns=df_transformed.columns)
X_train, X_test, y_train, y_test = train_test_split(df_scaled.iloc[:, :-1], df_scaled.iloc[:, -1:], test_size=0.20, random_state=50)
X = sm.add_constant(X_train)

results = sm.OLS(y_train, X).fit()

results.summary()
X_trainraw, X_testraw, y_trainraw, y_testraw = train_test_split(df.iloc[:, :-1], df.iloc[:, -1:], test_size=0.20, random_state=50)



reg_raw = LinearRegression()

reg_raw.fit(X_trainraw, y_trainraw)



print ('Train Accuracy Before Transformation: ' + str(round(reg_raw.score(X_trainraw, y_trainraw)*100, 2)) + '%')

print ('Test Accuracy Before Transformation: ' + str(round(reg_raw.score(X_testraw, y_testraw)*100, 2)) + '%')
reg = LinearRegression()

reg.fit(X_train, y_train)



print ('Train Accuracy: ' + str(round(reg.score(X_train, y_train)*100, 2)) + '%')

print ('Test Accuracy: ' + str(round(reg.score(X_test, y_test)*100, 2)) + '%')
input_columns = df.columns.drop(['Target'])



reg_coef = pd.DataFrame(reg.coef_.T, index=input_columns, columns=['Coefficients'])

reg_coef.sort_values(by='Coefficients')