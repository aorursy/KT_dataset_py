# Import libraries here

# import numpy as np

# from sklearn import linear_model

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

df = pd.read_csv("../input/spa-data/training-SpaData/SpaData.csv")

df

X = df.drop(['GTOccupancy','TimeStamp'], axis = 1)

y = df['GTOccupancy']

X.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

scaled_features = scaler.transform(X)

df_feat = pd.DataFrame(scaled_features,columns=X.columns[0:])

df_feat.head()
from sklearn.metrics import classification_report,confusion_matrix



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.2) # 70% training and 30% test





#Create KNN Classifier

knn = KNeighborsClassifier(n_neighbors=2)



#Train the model using the training sets

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))



import numpy as np

error_rate = []

# Might take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#import required packages

from sklearn import neighbors

from sklearn.metrics import mean_squared_error 

from math import sqrt

import matplotlib.pyplot as plt

%matplotlib inline

rmse_val = [] #to store rmse values for different k

for K in range(20):

    K = K+1

    model = neighbors.KNeighborsRegressor(n_neighbors = K)



    model.fit(X_train, y_train)  #fit the model

    pred=model.predict(X_test) #make prediction on test set

    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse

    rmse_val.append(error) #store rmse values

    print('RMSE value for k= ' , K , 'is:', error)

curve = pd.DataFrame(rmse_val) #elbow curve 

curve.plot()
from sklearn.model_selection import GridSearchCV

params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]}



knn = neighbors.KNeighborsRegressor()



model = GridSearchCV(knn, params, cv=5)

model.fit(X_train,y_train)

model.best_params_
pd.plotting.scatter_matrix(df, alpha=0.6)