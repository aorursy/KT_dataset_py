import pandas as pd

df = pd.read_csv(r'../input/Home_price.csv')

df.head(9)
dummies = pd.get_dummies(df.town) #creating dummies variable

dummies
df_dummies= pd.concat([df,dummies],axis='columns')

df_dummies
df_dummies.drop('town',axis='columns',inplace=True)

df_dummies
'''Dummy Variable Trap

When you can derive one variable from other variables, they are known to be multi-colinear. Here if you know values of california and georgia then you can easily infer value of new jersey state, i.e. california=0 and georgia=0. There for these state variables are called to be multi-colinear. In this situation linear regression won't work as expected. Hence you need to drop one column.



NOTE: sklearn library takes care of dummy variable trap hence even if you don't drop one of the state columns it is going to work, however we should make a habit of taking care of dummy variable trap ourselves just in case library that you are using is not handling this for you'''
df_dummies.drop('west windsor',axis='columns',inplace=True)

df_dummies
X = df_dummies.drop('price',axis='columns')

y = df_dummies.price



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,y)
X
model.predict(X)
model.score(X,y)
model.predict([[3400,0,0]])
model.predict([[2800,0,1]])
'''Using sklearn OneHotEncoder

First step is to use label encoder to convert town names into numbers'''
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

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()

X
X = X[:,1:]

X
model.fit(X,y)
model.predict([[0,1,3400]])
model.predict([[1,0,2800]])