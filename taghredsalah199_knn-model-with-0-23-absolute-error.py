import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_attack= pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

df_attack
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Data=df_attack.drop('target',axis=1)

scaler.fit(X=Data)

scaled_features= scaler.transform(df_attack.drop('target',axis=1))
df_accepted_feat= pd.DataFrame(scaled_features,columns=df_attack.columns[:13])

df_accepted_feat
from sklearn.model_selection import train_test_split

x= df_accepted_feat

y=df_attack['target']

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