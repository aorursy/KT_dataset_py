# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import math

import matplotlib

import tensorflow as tf



# Print versions of libraries

print(f"Numpy version : Numpy {np.__version__}")

print(f"Pandas version : Pandas {pd.__version__}")

print(f"Matplotlib version : Matplotlib {matplotlib.__version__}")

print(f"Seaborn version : Seaborn {sns.__version__}")

print(f"Tensorflow version : Tensorflow {tf.__version__}")



#Magic function to display In-Notebook display

%matplotlib inline



# Setting seabon style

sns.set(style='darkgrid', palette='deep')
# df = pd.read_csv("kc_house_data.csv", encoding = 'latin-1')

df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head(8).T
df.columns
df.info()
df.describe().T
# Finging unique values for each column

# TO understand which column is categorical and which one is Continuous

df.nunique()
# Dealing with missing data

df.isnull().sum()
df.shape
df = df.drop_duplicates()
df.shape
plt.figure(figsize=(8,6))

sns.distplot(df['price'])

plt.show()
plt.figure(figsize=(8,6))

sns.boxplot(df['price'])

plt.show()
# Create a function to return the outliers

def detect_outliers(x, c = 1.5):

    q1, q3 = np.percentile(x, [25,75])

    #print("q1 - ",q1, " q3 - ", q3)

    

    iqr = (q3 - q1)

    #print("iqr --", iqr)

    

    lob = q1 - (iqr * c)

    #print("lob - ",lob)

    

    uob = q3 + (iqr * c)

    #print("uob - ",uob)

    

    # Generate outliers

    indicies = np.where((x > uob) | (x < lob))



    return indicies
# Detect all Outliers 

priceOutliers = detect_outliers(df['price'])

print("Total Outliers count : ",len(priceOutliers[0]))
df.shape
# Remove outliers

df = df.drop(priceOutliers[0])
df.shape
df.corr()['price'].sort_values(ascending=False).head(10)
features = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

       'lat', 'long', 'sqft_living15', 'sqft_lot15']



mask = np.zeros_like(df[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="viridis",

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});



plt.show()
df.corr()['sqft_living'].sort_values(ascending=False).head(10)
fig, axs = plt.subplots( nrows=2, ncols=3, sharey=True, figsize=(18,12))



sns.scatterplot(y='price',x='sqft_living',data=df, hue='condition', palette='viridis', ax=axs[0,0])

sns.scatterplot(y='price',x='sqft_above',data=df, hue='condition', palette='viridis', ax=axs[0,1])

sns.scatterplot(y='price',x='sqft_living15',data=df, hue='condition', palette='viridis', ax=axs[0,2])



sns.scatterplot(y='price',x='sqft_basement',data=df, hue='condition', palette='viridis', ax=axs[1,0])

sns.scatterplot(y='price',x='sqft_lot',data=df, hue='condition', palette='viridis', ax=axs[1,1])

sns.scatterplot(y='price',x='sqft_lot15',data=df, hue='condition', palette='viridis', ax=axs[1,2])





plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='view',data=df, palette='Set2', ax=axs[0])

sns.boxplot(y='price',x='view',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.plot()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



ax = sns.countplot(x='grade',data=df, palette='Set2', ax=axs[0])

ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")

sns.boxplot(y='price',x='grade',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='condition',data=df, palette='Set2', ax=axs[0])

sns.boxplot(y='price',x='condition',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='waterfront',data=df, palette='Set2', ax=axs[0])

sns.boxplot(y='price',x='waterfront',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='bedrooms',data=df, palette='Set2',ax=axs[0])

sns.boxplot(y='price',x='bedrooms',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



chart1 = sns.countplot(x='bathrooms',data=df, palette='Set2',ax=axs[0])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=90, horizontalalignment='right')



chart2 = sns.boxplot(y='price',x='bathrooms',data=df, palette='Set2', ax=axs[1])

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90, horizontalalignment='right')



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='floors',data=df, palette='Set2',ax=axs[0])

sns.boxplot(y='price',x='floors',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='long',data=df)

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='price',y='lat',data=df)

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='long',y='lat',

                data=df,hue='price',

                palette='RdYlGn',edgecolor=None,alpha=0.2)

plt.show()
df.info()
df = df.drop('id',axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date : date.month)
df['year'] = df['date'].apply(lambda date : date.year)
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='year',data=df, palette='Set2',ax=axs[0])

sns.boxplot(y='price',x='year',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='month',data=df, palette='Set2',ax=axs[0])

sns.boxplot(y='price',x='month',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
plt.figure(figsize=(8,6))

df.groupby('month').mean()['price'].plot()
plt.figure(figsize=(8,6))

df.groupby('year').mean()['price'].plot()
# Drop the date columns after doing feature engineering.

df = df.drop('date',axis=1)
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
# could make sense due to scaling, higher should correlate to more value

df['yr_renovated'].value_counts()
df['renovated'] = df['yr_renovated'].apply(lambda yr : 0 if yr==0 else 1)
df['renovated'].value_counts()
fig, axs = plt.subplots(ncols=2, figsize=(12,6))



sns.countplot(x='renovated',data=df, palette='Set2',ax=axs[0])

sns.boxplot(y='price',x='renovated',data=df, palette='Set2', ax=axs[1])



plt.tight_layout()

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='yr_built', y='price',hue='renovated' ,data=df)

plt.show()
# Separate Target Variable and Predictor Variables

# Also please note that to call the values here because tensor flow may work with numeric array, and it can't work with pandas dataframes.

X = df.drop('price',axis=1).values

y = df['price'].values
from sklearn.model_selection import train_test_split
# Split the data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Quick sanity check with the shapes of Training and testing datasets

print("X_train - ",X_train.shape)

print("y_train - ",y_train.shape)

print("X_test - ",X_test.shape)

print("y_test - ",y_test.shape)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train= scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
np.min(X_train)
np.max(X_train)
X_train.shape[1]
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.optimizers import Adam
model = Sequential()



model.add(Dense(20,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(20,activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train,

          validation_data=(X_test,y_test),

          batch_size=128,epochs=400)
model.summary()
losses = pd.DataFrame(model.history.history)
losses.plot(figsize=(12,8))
from sklearn import metrics



def measure_accuracy(original, predicted, train=True):  

    mae = metrics.mean_absolute_error(original, predicted)

    mse = metrics.mean_squared_error(original, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(original, predicted))

    #rmsle = np.sqrt(metrics.mean_squared_log_error(original, predicted))

    r2_square = metrics.r2_score(original, predicted)

    evs = metrics.explained_variance_score(original,predicted)

    

    if train:

        print("Training Result : ")

        print('------------------')

        print('MAE: {0:0.3f}'.format(mae))

        print('MSE: {0:0.3f}'.format(mse))

        print('RMSE: {0:0.3f}'.format(rmse))

        #print('RMSLE: {0:0.3f}'.format(rmsle))

        print('Explained Variance Score: {0:0.3f}'.format(evs))

        print('R2 Square: {0:0.3f}'.format(r2_square))

        print('\n')

    elif not train:

        print("Testing Result : ")

        print('------------------')

        print('MAE: {0:0.3f}'.format(mae))

        print('MSE: {0:0.3f}'.format(mse))

        print('RMSE: {0:0.3f}'.format(rmse))

        #print('RMSLE: {0:0.3f}'.format(rmsle))

        print('Explained Variance Score: {0:0.3f}'.format(evs))

        print('R2 Square: {0:0.3f}'.format(r2_square))
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
plt.figure(figsize=(8,6))

# Our predictions

plt.scatter(y_test,y_test_pred)



# Perfect predictions

plt.plot(y_test,y_test,'r')
errors = y_test.reshape(6141, 1) - y_test_pred
plt.figure(figsize=(8,6))

sns.distplot(errors)
measure_accuracy(y_train, y_train_pred, train=True)

measure_accuracy(y_test, y_test_pred, train=False)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



measure_accuracy(y_train, y_train_pred, train=True)

measure_accuracy(y_test, y_test_pred, train=False)