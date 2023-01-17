import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
data.head()
data = data.drop(['Serial No.'], axis = 1)
data.head()
data.info()
data.describe()
sns.pairplot(data)
plt.show()
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(),annot=True)
plt.show()
dataX=data.drop(['Chance of Admit '],axis=1)
dataY=data['Chance of Admit ']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
pd.DataFrame({"Actual": y_test, "Predict": y_pred}).head()
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state = 101)
rfr.fit(X_train,y_train)
y_head_rfr = rfr.predict(X_test)
y_head_rfr
from sklearn.metrics import r2_score
r2_score(y_test, y_head_rfr)
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 101)
dtr.fit(X_train,y_train)
y_head_dtr = dtr.predict(X_test) 
r2_score(y_test,y_head_dtr)
