import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sb
data=pd.read_csv('/kaggle/input/world-happiness-report-2019.csv')

data.head()
data = data.rename(columns={'Country (region)':'Country','SD of Ladder':'SD',

                         'Positive affect':'Positive','Negative affect':'Negative','Social support':'Social',

                         })
data.head()
data.info()
data.isnull().sum()
data=data.fillna(method='ffill')
data.head()
fig, ax = plt.subplots(figsize=(10,10))

sb.heatmap(data.corr(),ax=ax,annot=True,linewidth=5,fmt='.2f',cmap=None)

plt.show()

plt.scatter(data['Ladder'],data['Social'])

plt.title('Ladder compare with Social')

plt.xlabel('Ladder')

plt.ylabel('Social')

plt.show()
plt.scatter(data['Ladder'],data['Log of GDP\nper capita'])

plt.title('Ladder compare with Log of GDP')

plt.xlabel('Ladder')

plt.ylabel('Social')

plt.show()
plt.scatter(data['Healthy life\nexpectancy'],data['Log of GDP\nper capita'])

plt.title('life expectancy with Gdp')

plt.xlabel('Health lif Expectancy')

plt.ylabel('Log of Gdp')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

# x=data.drop(['SD'])

data.head()
data.columns
x=data.drop(['SD','Country'],axis=1)

y=data.SD

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
lr=LinearRegression()

lr.fit(x_train,y_train)

lr_predict=lr.predict(x_test)

r2_score(y_test,lr_predict)
plt.scatter(y_test,lr_predict,color='c')

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('LinearRegression')
rf=RandomForestRegressor()

rf.fit(x_train,y_train)

rf_predict=rf.predict(x_test)

r2_score(y_test,rf_predict)

plt.scatter(y_test,rf_predict)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('RandomForestRegressor')
dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

dt_predict=dt.predict(x_test)

print(r2_score(y_test,dt_predict))

plt.scatter(y_test,dt_predict)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('DecisionTreeRegressor')
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
# sv=SVR(kernel='rbf')

# sv.fit(x_train,y_train)

# sv_predict=sv.predict(x_test)

# r2_score(y_test,sv_predict)

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(x_train,y_train)

knn_predict=knn.predict(x_test)

print(r2_score(y_test,knn_predict))

plt.scatter(y_test,knn_predict)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('KNN Regressor')
y = np.array([r2_score(y_test,lr_predict),r2_score(y_test,rf_predict),r2_score(y_test,dt_predict),r2_score(y_test,knn_predict)])

x = ['Linear','RandomForest','DecisionTree','K-Neighbours']



plt.bar(x,y)

plt.title('comparision')

plt.xlabel('Regressor')

plt.ylabel('r2_score')