import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv("../input/car-price/carprices.csv")

df
df.rename(columns={"Sell Price($)":"Price","Age(yrs)":"Age" },inplace = True)
df
plt.scatter(df['Mileage'],df['Price'],color='red',marker="X")

plt.xlabel("MILEAGE")

plt.ylabel("PRICE")
plt.scatter(df['Age'],df['Price'],color='green',marker='x')

plt.xlabel("AGE")

plt.ylabel("PRICE")
# TWO TYPES OF CATEGORICAL VARIABLES:



# 1) NOMINAL : THEY DO NOT HAVE ANY GENERAL VALUE .  

# 2) ORDINAL : THEY CARRY NUMERICAL VALUE ON THEM .
# ONE HOT ENCODING



# METHOD 1 :
dummies = pd.get_dummies(df['Car Model'])

dummies
merg = pd.concat([df,dummies],axis='columns')

merg
final = merg.drop(['Car Model','Mercedez Benz C class'],axis='columns')

final
#CREATING YOUR REGRESSION MODEL



from sklearn.linear_model import LinearRegression

model = LinearRegression()
X = final.drop(['Price'],axis='columns')

X
y = final.Price
model.fit(X,y)
model.score(X,y)
model.predict([[3400,0,1,1]])
# MEHTOD 2 :
df
from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
dfle=df

dfle['Car Model']=le.fit_transform(dfle['Car Model'])

dfle
X = df[['Car Model','Mileage','Age']].values

X
y =dfle['Price']

y
from sklearn.preprocessing import OneHotEncoder

ohe= OneHotEncoder(categorical_features=[0])
X=ohe.fit_transform(X).toarray()

X
X=X[:,1:]

X
model.fit(X,y)
model.score(X,y)