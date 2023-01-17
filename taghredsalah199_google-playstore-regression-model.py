import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_google= pd.read_csv('../input/google-play-store-cleaned-data/Goog_out1.csv')

df_google
df_google.info()
plt.figure(figsize=(20,10))

sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )
df_google=df_google.fillna(value=df_google['Size'].mean())

df_google=df_google.fillna(value=df_google['Installs'].mean())

# Fill nan value with the mean of col
df_google=df_google.drop('Unnamed: 0', axis=1)
plt.figure(figsize=(20,10))

sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )
figure= plt.figure(figsize=(10,10))

sns.heatmap(df_google.corr(), annot=True)

#To show the correlation between variables
figure= plt.figure(figsize=(10,10))



sns.distplot(df_google['Rating']) #My predicted col
sns.pairplot(df_google)
sns.lmplot(x='Rating',y='Size',data=df_google)
x= df_google[['Reviews','Size','Installs']]

y=df_google['Rating']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=108)
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

LR.fit(x_train,y_train)
# The coefficients

print('Coefficients: \n', LR.coef_)
prediction = LR.predict(x_test)

plt.scatter(y_test,prediction) #NOT CORRECT MODEL
from sklearn import metrics

metrics.mean_absolute_error(y_test,prediction)
metrics.mean_squared_error(y_test,prediction)
np.sqrt(metrics.mean_squared_error(y_test,prediction))
df_google_KNN= df_google.drop(['App','Category','Type','Price','Content Rating','Genres','Last Updated','Current Ver','Android Ver'], axis=1)

#Select the numiric cols only
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Data=df_google_KNN.drop('Rating',axis=1)

Data
scaler.fit(Data)

scaled_features= scaler.transform(df_google_KNN.drop('Rating',axis=1))
df_accepted_feat= pd.DataFrame(scaled_features,columns=df_google_KNN.columns[1:])

df_accepted_feat
from sklearn.model_selection import train_test_split

x= df_accepted_feat

y=df_google_KNN['Rating']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=109)
from sklearn.neighbors import KNeighborsRegressor

knn= KNeighborsRegressor(n_neighbors=3)

knn.fit(x_train,y_train)

pred= knn.predict(x_test)
from sklearn import metrics

metrics.mean_absolute_error(y_test,pred)
metrics.mean_squared_error(y_test,pred)
np.sqrt(metrics.mean_squared_error(y_test,pred))
error_rate= []

for i in range(1,50):

    knn=KNeighborsRegressor(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i= knn.predict(x_test)

    error_rate.append(np.mean(pred_i!= y_test))
plt.figure(figsize=(12,8))

plt.plot(range(1,50), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',markersize=8)

plt.title('Error rate vs K values')

plt.xlabel('K Values')

plt.ylabel('Error rate')