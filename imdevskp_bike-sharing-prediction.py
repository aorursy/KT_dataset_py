import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



sns.set_palette('husl')

sns.set_style("whitegrid")



from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, PowerTransformer

from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV



from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("../input/bike-sharing-demand/train.csv")

df.head()
df.info()
df.describe()
df['datetime'] = pd.to_datetime(df['datetime'])



df['day'] = pd.DatetimeIndex(df['datetime']).day

df['month'] = pd.DatetimeIndex(df['datetime']).month

df['hour'] = pd.DatetimeIndex(df['datetime']).hour

df['day_of_week'] = pd.DatetimeIndex(df['datetime']).weekday







df['year'] = pd.DatetimeIndex(df['datetime']).year

df['year'] =df['year']-2011



df["season"] = df['season'].map({1 : "Spring", 

                                 2 : "Summer", 

                                 3 : "Fall", 

                                 4 : "Winter" })



df["day_of_week"] = df['day_of_week'].map({0 : "Sunday", 

                                           1 : "Monday", 

                                           2 : "Tuesday", 

                                           3 : "Wednesday", 

                                           4 : "Thursday", 

                                           5 : "Friday",

                                           6 : "Saturday" })
df.head()
# missing values

df.isna().sum()
# # target column distribution



# plt.figure(figsize=(16,6))

# sns.countplot(df['count'], kde=False)
# date vs count plot



plt.figure(figsize=(16,6))

plt.plot(df['datetime'], df['count'], alpha=0.8)

plt.show()
# date vs temp plot



plt.figure(figsize=(16,6))

plt.plot(df['datetime'], df['atemp'], alpha=0.8, color='orange')
sns.set_palette('RdBu_r')
fig, axes = plt.subplots(figsize=(15, 4), ncols=3)

sns.barplot(x='season', y='count', data=df, ax=axes[0])

sns.barplot(x='workingday', y='count', data=df, ax=axes[1])

sns.barplot(x='holiday', y='count', data=df, ax=axes[2])

plt.show()
fig, axes = plt.subplots(figsize=(15, 4), ncols=2)

sns.barplot(x='month', y='count', color='Grey', data=df, ax=axes[0])

sns.barplot(x='day_of_week', y='count', color='Grey', data=df, ax=axes[1])

plt.show()
fig, axes = plt.subplots(figsize=(15, 4))

sns.barplot(x='day', y='count', data=df, color='Grey')

plt.show()
fig, axes = plt.subplots(figsize=(15, 4))

sns.barplot(x='hour', y='count', data=df, color='Grey')

plt.show()
fig, axes = plt.subplots(figsize=(15, 10), ncols=2, nrows=2)

sns.lineplot(x='hour', y='count', hue='season', data=df, ax=axes[0][0])

sns.lineplot(x='hour', y='count', hue='day_of_week', data=df, ax=axes[0][1])

sns.lineplot(x='hour', y='count', hue='weather', data=df, ax=axes[1][0])

sns.lineplot(x='hour', y='count', hue='holiday', data=df, ax=axes[1][1])
plt.figure(figsize=(10,6))

sns.boxplot(data=df[['atemp','humidity','windspeed']], palette="Set2")

plt.show()
# min-max scaling

features=['temp', 'atemp', 'humidity', 'windspeed']

for i in features:

    scaler = MinMaxScaler()

    df[i] = scaler.fit_transform(df[[i]])

    

# one hot encoding using pandas get_dummies

features=['weather', 'season']

for i in features:

    temp=pd.get_dummies(df[i], prefix=i, prefix_sep='_')

    df=pd.concat([df,temp], axis=1)

    df=df.drop(i, axis=1)

    

# cyclic encoding cyclic variables

def cyc_enc(df, col, max_vals):

    df[col+'_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col+'_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df

df = cyc_enc(df, 'hour', 24)

df = cyc_enc(df, 'month', 12)

df = cyc_enc(df, 'day', 31)
# PCA to reduce components

pca = PCA(n_components=1)

df['temperature'] = pca.fit_transform(df[['temp','atemp']])

df = df.drop(columns=['temp', 'atemp'])
# # final correlation matrix



plt.figure(figsize=(14, 14))

sns.heatmap(df.corr(), annot=True, fmt='.2f')

plt.show()
df.head()
X = df.drop(['datetime', 'count', 'day_of_week', 'registered', 'casual'], axis=1)

y = df['count']



X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
# linear regression



lr=LinearRegression()

lr.fit(X_train, y_train)

y_pred=lr.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()
# k nearest neighbours regressor



knn=KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred=knn.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()
# decision tree regressor



dt=DecisionTreeRegressor()

dt.fit(X_train, y_train)

y_pred=dt.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))

# print(y_test[:10])

# print(y_pred[:10])



plt.scatter(y_test, y_pred)

plt.show()
# random forest regressor



rf=RandomForestRegressor(n_estimators=10)

rf.fit(X_train, y_train)

y_pred=rf.predict(X_test)



print(np.sqrt(mean_squared_error(y_pred, y_test)))

print(mean_absolute_error(y_pred, y_test))



plt.scatter(y_test, y_pred)

plt.show()