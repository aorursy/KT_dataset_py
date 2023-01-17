import pandas as pd

data=pd.read_csv('../input/diamonds/diamonds.csv')
data.head()
data.drop('Unnamed: 0',axis=1,inplace=True)
data.isna().sum()
data.describe()
#checking for columns with 0 

data.loc[(data['x']==0)|(data['y']==0)|(data['z']==0)]

#deleting values having zero 

data=data[(data[['x','y','z']]!=0).all(axis=1)]
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



X=data.drop('price',axis=1)

y=data['price']

   

plt.figure(figsize=(15,8))

sns.countplot(x='cut',data=X,order=X['cut'].value_counts().index)

plt.figure(figsize=(15,8))

sns.countplot(x='color',data=X,order=X['color'].value_counts().index)
plt.figure(figsize=(15,8))

sns.countplot(x='clarity',data=X,order=X['clarity'].value_counts().index)
object_col=[col for col in X.columns if X[col].dtype=="object"]

oh_encoder=pd.get_dummies(X[object_col])



num_X_train=X.drop(object_col,axis=1)

scale=MinMaxScaler()

scale_X_train=pd.DataFrame(scale.fit_transform(num_X_train),index=num_X_train.index,columns=['carat','depth','table','x','y','z'])

diamond_data=pd.concat([scale_X_train,oh_encoder],axis=1)
#checking correlation 

plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(diamond_data.corr(), annot=True,cmap='RdYlGn')
X_train,X_test,y_train,y_test=train_test_split(diamond_data,y,test_size=0.2,random_state=42)



#linear Regression 

lnr_model=LinearRegression()

lnr_model.fit(X_train,y_train)

y_pred=lnr_model.predict(X_test)





print("Accuracy :- "+ str(lnr_model.score(X_test,y_test)*100) +' %')

print("R Squared :- "+ str(metrics.r2_score(y_test,y_pred)))

print("Mean Absolute Error :- {}".format(mean_absolute_error(y_test,y_pred)))

#Decision tree

regressor = DecisionTreeRegressor(random_state = 0)  



regressor.fit(X_train, y_train)



y_pred_dt=regressor.predict(X_test)





print("Accuracy :- " + str(regressor.score(X_test,y_test)*100) +' %')

print("R Squared :- " + str(metrics.r2_score(y_test,y_pred_dt)))

print("Mean absolute error :- {}".format(mean_absolute_error(y_test,y_pred_dt)))
#random forest with default parameters 

rf=RandomForestRegressor(random_state=42)

rf.fit(X_train,y_train)

rfy_pred=rf.predict(X_test)

metrics.r2_score(y_test,rfy_pred)



print("Accuracy :- "+ str(rf.score(X_test,y_test)*100) +' %')

print("R Squared :- "+ str(metrics.r2_score(y_test,rfy_pred)))

print("Mean Absolute Error :- {}".format(mean_absolute_error(y_test,rfy_pred)))
#random forest with estimators



rf = RandomForestRegressor(n_estimators=100,random_state = 42)



rf.fit(X_train,y_train)



y_pred_rf=rf.predict(X_test)



print("Accuracy :- "+ str(rf.score(X_test,y_test)*100) +' %')

print("R Squared :- "+ str(metrics.r2_score(y_test,y_pred_rf)))

print("Mean Absolute Error :- {}".format(mean_absolute_error(y_test,y_pred_rf)))