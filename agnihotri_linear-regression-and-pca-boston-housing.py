# Import Libraries needed to load the data

import pandas as pd

from sklearn.datasets import load_boston
# Load the data from sklearn module

df = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)

df['MEDV'] = pd.DataFrame(load_boston().target)

print('Shape of Data is : {} rows and {} columns'.format(df.shape[0],df.shape[1]))
df.head()
# Lets look at the null values of the data

df.isna().sum()
# Lets look at the datatype of the features

df.dtypes
# import libraries needed to do EDA

import matplotlib.pyplot as plt

import seaborn as sns
# Lets look at the distribution plot of the features

pos = 1

fig = plt.figure(figsize=(16,24))

for i in df.columns:

    ax = fig.add_subplot(7,2,pos)

    pos = pos + 1

    sns.distplot(df[i],ax=ax)
# lets look at some descriptive stats of our features

df.describe()
# Lets look at the correlation matrix of our data.

fig = plt.figure(figsize=(16,12))

ax = fig.add_subplot(111)

sns.heatmap(df.corr(),annot=True)
# import libraries needed for this.

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
# lets get the VIF value to understand the multi collinearity

vifdf = []

for i in df.columns:

    X = np.array(df.drop(i,axis=1))

    y = np.array(df[i])

    lr = LinearRegression()

    lr.fit(X,y)

    y_pred = lr.predict(X)

    r2 = r2_score(y,y_pred)

    vif = 1/(1-r2)

    vifdf.append((i,vif))



vifdf = pd.DataFrame(vifdf,columns=['Features','Variance Inflation Factor'])

vifdf.sort_values(by='Variance Inflation Factor')
# Lets build our function which will perform the normaliztion

def rescale(X):

    mean = X.mean()

    std = X.std()

    scaled_X = [(i - mean)/std for i in X]

    return pd.Series(scaled_X)
# We will build a new dataframe

df_std = pd.DataFrame(columns=df.columns)

for i in df.columns:

    df_std[i] = rescale(df[i])
# Lets look at the descriptive stats now

df_std.describe().iloc[1:3:]
# lets look at the shape of data after scaling

pos = 1

fig = plt.figure(figsize=(16,24))

for i in df_std.columns:

    ax = fig.add_subplot(7,2,pos)

    pos = pos + 1

    sns.distplot(df_std[i],ax=ax)
# import libraries for PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=13)

X = df_std.drop('MEDV',axis=1)

X_pca = pca.fit_transform(X)

df_std_pca = pd.DataFrame(X_pca,columns=['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10','PCA11','PCA12','PCA13'])

df_std_pca['MEDV'] = df_std['MEDV']
# Lets look at the correlation matrix now.

fig = plt.figure(figsize=(16,12))

ax = fig.add_subplot(111)

sns.heatmap(df_std_pca.corr(),annot=True)
# Lets look at the distribution of our features after applying PCA

pos = 1

fig = plt.figure(figsize=(16,24))

for i in df_std_pca.columns:

    ax = fig.add_subplot(7,2,pos)

    pos = pos + 1

    sns.distplot(df_std_pca[i],ax=ax)
# import libraires needed to perform our Regression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
# Split data into Training and testing

X = np.array(df_std_pca.drop('MEDV',axis=1))

y = np.array(df_std_pca['MEDV'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

for i in [X_train,X_test,y_train,y_test]:

    print("Shape of Data is {}".format(i.shape))
# Lets train our model on training data and predict also on training to see results

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_train)

r2 = r2_score(y_train,y_pred)

rmse = np.sqrt(mean_squared_error(y_train,y_pred))

print('R-Squared Score is : {} | Root Mean Square Error is : {}'.format(r2,rmse))
# Lets train our model on training data and predict on testing to see results

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test,y_pred)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print('R2 Score is : {} | Root Mean Square Error is : {}'.format(r2,rmse))