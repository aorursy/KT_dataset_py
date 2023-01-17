

import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv('../input/50-startups/50_Startups.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,4].values
x
plt.scatter(x['R&D Spend'],y,color='r')
plt.scatter(x['Administration'],y,color='g')

plt.scatter(x['Marketing Spend'],y,color='orange')
sns.barplot(x=data['State'],y=data['Profit'],data=data)
sns.distplot(data['Profit'],color='red',rug=True)
sns.distplot(data['R&D Spend'],color='g',rug=True)
sns.distplot(data['Administration'],color='orange',rug=True)
sns.distplot(data['Marketing Spend'],color='b',rug=True)
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(x).toarray()
x.astype(int)
y.astype(int)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 0)
svr=SVR()
svr.fit(x_train,y_train)
y_pred=svr.predict(x_test)
y_pred
y_test
score=svr.score(x_test,y_test)
score
