import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv("../input/80-cereals/cereal.csv")
df.head()
df=pd.read_csv("../input/80-cereals/cereal.csv")
print(df[:])
df=pd.read_csv("../input/80-cereals/cereal.csv")
df.describe()
df=pd.read_csv("../input/80-cereals/cereal.csv")
print(df[50:71])
df=pd.read_csv("../input/80-cereals/cereal.csv")
df.head()
cols=['calories','protein','fat','carbo','vitamins']
sns.pairplot(data=df[cols])
df=pd.read_csv("../input/80-cereals/cereal.csv")
x=df.carbo.values.reshape(-1,1)
y=df.calories.values.reshape(-1,1)
sns.lmplot(x='carbo',y='calories',data=df)
df=pd.read_csv("../input/80-cereals/cereal.csv")
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
df=pd.read_csv("../input/80-cereals/cereal.csv")
x_input=[[100]]
y_predict=model.predict(x_input)
y_predict