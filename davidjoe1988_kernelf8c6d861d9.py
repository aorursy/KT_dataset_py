import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
data=pd.read_csv('../input/50_Startups.csv')

data.head()
X=data.iloc[:,:-1].values

Y=data.iloc[:,4].values
f,axes=plt.subplots(2,2,figsize=(7,7))

sb.distplot(data['R&D Spend'],color='skyblue',ax=axes[0,0])

sb.distplot(data['Administration'],color='olive',ax=axes[0,1])

sb.distplot(data['Marketing Spend'],color='gold',ax=axes[1,0])

sb.distplot(data['Profit'],color='teal',ax=axes[1,1])
X=data.iloc[:,:-1].values

Y=data.iloc[:,4].values
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X=LabelEncoder()

X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])

X=onehotencoder.fit_transform(X).toarray()
#Avoiding the Dummy variable trap

X=X[:,1:]
#spliting the dataset into training and testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

regessor=LinearRegression()

regessor.fit(x_train,y_train)

regessor.score(x_test,y_test)
y_pred=regessor.predict(x_test)

y_pred