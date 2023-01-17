import pandas as pd
df=pd.read_csv('../input/one-hot-encoding/dummy.csv')
df
dummies=pd.get_dummies(df.town)
df2=pd.concat([df,dummies],axis=1) # axis =1 ='columns'
df2
final_df=df2.drop(['town','west windsor'],axis=1)
final_df
from sklearn import linear_model
reg=linear_model.LinearRegression()
x=final_df.drop(['price'],axis=1)
x
y=final_df.price
y
reg.fit(x,y)
reg.predict([[2600,0,1]]) #to find price in robbinsville
reg.predict([[3400,0,0]]) #to find price in west widsor
reg.score(x,y)
#another way
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
dfle=df
dfle.town=le.fit_transform(df.town)
dfle
x=dfle[['town','area']].values
x
y=dfle.price
y
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dfle = df

dfle.town = le.fit_transform(dfle.town)

dfle
X = dfle[['town','area']].values
X
y = dfle.price.values

y
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

X
X = X[:,1:]
X
reg.fit(X,y)
reg.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor

reg.predict([[1,0,2800]]) # 2800 sqr ft home in robbinsville
