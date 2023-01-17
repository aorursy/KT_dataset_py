import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_palette("husl")



from sklearn.preprocessing import MinMaxScaler



import warnings

warnings.filterwarnings('ignore')



# Read in the data using pd.read_csv

# The first column is the index column, so we will set the index to that column

data = pd.read_csv('../input/diamonds.csv',index_col=0)

data.sample(3)
data.info()
data['volume'] = data['x']*data['y']*data['z']

data.drop(['x','y','z'],axis=1,inplace=True)

data = data[data['volume']<1000]

data.columns
plt.figure(figsize=[12,12])



# First subplot showing the diamond carat weight distribution

plt.subplot(221)

plt.hist(data['carat'],bins=20,color='b')

plt.xlabel('Carat Weight')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Carat Weight')



# Second subplot showing the diamond depth distribution

plt.subplot(222)

plt.hist(data['depth'],bins=20,color='r')

plt.xlabel('Diamond Depth (%)')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Depth')



# Third subplot showing the diamond price distribution

plt.subplot(223)

plt.hist(data['price'],bins=20,color='g')

plt.xlabel('Price in USD')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Price')



# Fourth subplot showing the diamond volume distribution

plt.subplot(224)

plt.hist(data['volume'],bins=20,color='m')

plt.xlabel('Volume in mm cubed')

plt.ylabel('Frequency')

plt.title('Distribution of Diamond Volume')
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.regplot(x = 'carat', y = 'price', data=data, ax = saxis[0,0])

sns.regplot(x = 'volume', y = 'price', data=data, ax = saxis[0,1])



# Order the plots from worst to best

sns.barplot(x = 'cut', y = 'price', order=['Fair','Good','Very Good','Premium','Ideal'], data=data, ax = saxis[1,0])

sns.barplot(x = 'color', y = 'price', order=['J','I','H','G','F','E','D'], data=data, ax = saxis[1,1])

sns.barplot(x = 'clarity', y = 'price', order=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], data=data, ax = saxis[1,2])

# These look very ugly, but I'm not sure how else to do this

# I tried using the LabelEncoder method from scikit-learn preprocessing

# but I'm not sure how to label them based on order

data['cut'] = data['cut'].apply(lambda x: 1 if x=='Fair' else(2 if x=='Good' 

                                           else(3 if x=='Very Good' 

                                           else(4 if x=='Premium' else 5))))



data['color'] = data['color'].apply(lambda x: 1 if x=='J' else(2 if x=='I'

                                          else(3 if x=='H'

                                          else(4 if x=='G'

                                          else(5 if x=='F'

                                          else(6 if x=='E' else 7))))))



data['clarity'] = data['clarity'].apply(lambda x: 1 if x=='I1' else(2 if x=='SI2'

                                          else(3 if x=='SI1'

                                          else(4 if x=='VS2'

                                          else(5 if x=='VS1'

                                          else(6 if x=='WS2'

                                          else 7 if x=='WS1' else 8))))))
scaler = MinMaxScaler()

data[['cut','color','clarity']] = scaler.fit_transform(data[['cut','color','clarity']])



data['diamond score'] = data['cut'] + data['color'] + data['clarity']



sns.regplot(x = 'diamond score', y = 'price', data=data)
plt.figure(figsize=(12, 12))

correlation = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

sns.heatmap(correlation, vmax=1, annot=True,square=True)
test_data = data.iloc[-round(len(data)*.1):].copy()

data.drop(data.index[-round(len(data)*.1):],inplace=True)

test_data.drop('price',1,inplace=True)

print(data.shape)

print(test_data.shape)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.cross_validation import train_test_split



X = data.drop(['price'],1)

y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



linear_regression = LinearRegression()

linear_regression.fit(X_train,y_train)

print('Linear regression accuracy: ', linear_regression.score(X_test,y_test))



ridge = Ridge(normalize=True)

ridge.fit(X_train,y_train)

print('Ridge regression accuracy: ',ridge.score(X_test,y_test))



lasso = Lasso(normalize=True)

lasso.fit(X_train,y_train)

print('Lasso regression accuracy: ',ridge.score(X_test,y_test))



elastic_net = ElasticNet()

elastic_net.fit(X_train,y_train)

print('Elastic net accuracy: ',elastic_net.score(X_test,y_test))
plt.figure(figsize=[12,12])



# Linear regression model

plt.subplot(221)

plt.scatter(test_data['diamond score'],linear_regression.predict(test_data),color='lightcoral')

plt.ylim(0,8000)

plt.xlabel('Diamond Score')

plt.ylabel('Price in USD')

plt.title('Linear Regression Model')



# Ridge regression model

plt.subplot(222)

plt.scatter(test_data['diamond score'],ridge.predict(test_data),color='royalblue')

plt.ylim(0,8000)

plt.xlabel('Diamond Score')

plt.ylabel('Price in USD')

plt.title('Ridge Regression Model')



# Lasso regression model

plt.subplot(223)

plt.scatter(test_data['diamond score'],lasso.predict(test_data),color='lightgreen')

plt.ylim(0,8000)

plt.xlabel('Diamond Score')

plt.ylabel('Price in USD')

plt.title('Lasso Regression Model')



# Elastic net model

plt.subplot(224)

plt.scatter(test_data['diamond score'],elastic_net.predict(test_data),color='orange')

plt.ylim(0,8000)

plt.xlabel('Diamond Score')

plt.ylabel('Price in USD')

plt.title('Elastic Net Model')