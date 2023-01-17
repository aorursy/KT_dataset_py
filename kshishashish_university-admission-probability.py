
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

dataset.head()

dataset.drop('Serial No.',axis=1,inplace=True)
dataset.head()
dataset.groupby(by='University Rating').mean()
dataset.describe()
sns.distplot(dataset['GRE Score'])
fig = plt.figure(figsize = (10,10))
ax = fig.gca()
dataset.hist(ax=ax)
plt.show()
sns.pairplot(dataset)
corr_matrix=dataset.corr()
plt.figure(figsize=(12,12))

sns.heatmap(corr_matrix,annot=True)
plt.show()
X=dataset.iloc[:,[0,1,2,3,4,5,6]].values
y=dataset.iloc[:,7].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X_train, y_train)
y_prediction = LinearRegression_model.predict(X_test)

print('Accuracy', 1-np.sqrt(mean_squared_error(y_test, y_prediction)))
plt.scatter(y_test, y_prediction, alpha=0.2)
plt.xlabel('Targets', size = 18)
plt.ylabel('Predictions', size = 18)

plt.show()
from sklearn.metrics import r2_score,mean_absolute_error 
from math import sqrt

k=X_test.shape[1]
n=len(X_test)

RMSE=float(format(np.sqrt(mean_squared_error(y_test,y_prediction))))
MSE=mean_squared_error(y_test,y_prediction)
MAE=mean_absolute_error(y_test,y_prediction)
r_2=r2_score(y_test,y_prediction)
adj_r2=1-(1-r_2)*(n-1)/(n-k-1)

print('RMSE: ',RMSE)
print('MSE: ',MSE)
print('MAE: ',MAE)
print('R2: ',r_2)
print('Adjusted R2: ',adj_r2)
import tensorflow as tf
from keras.layers import Dense,Activation,Dropout

ANN_model=keras.Sequential()
ANN_model.add(Dense(50,input_dim=7))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))

ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
from keras.optimizers import Adam
ANN_model.compile(loss='mse',optimizer='adam')
ANN_model.summary()
from sklearn.metrics import mean_squared_error
ANN_model.compile(optimizer='adam',loss='mean_squared_error')
epoch_hist=ANN_model.fit(X_train,y_train,epochs=100,batch_size=20,validation_split=0.2)
result= ANN_model.evaluate(X_test,y_test)
accuracy_ANN=1-result
accuracy_ANN

plt.plot(epoch_hist.history['loss'])
plt.title('Model Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend('Training loss')
