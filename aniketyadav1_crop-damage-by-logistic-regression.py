import pandas as pd

import numpy as np

df=pd.read_csv("../input/crop-damage/train_yaOffsB.csv")
df
df['Number_Weeks_Used'].unique()        # unique function is used to find the unique values 
df.isnull().sum()                    #  isnull() function is used to find the null values,   sum() is used here to summ the total null values in a perticular column
df.drop(['ID'],axis=1,inplace=True)                  # drop function is used here to drop a perticular column
df["Number_Weeks_Used"].fillna(method ='ffill', inplace = True)  
df1=pd.read_csv("../input/crop-damage/test_pFkWwen.csv")
df1['Number_Weeks_Used'].unique()
df1.isnull().sum()
df1.drop(['ID'],axis=1,inplace=True)
df1["Number_Weeks_Used"].fillna(method ='bfill', inplace = True)
df1.head()
X=df.drop(['Crop_Damage'],axis=1)
X.isnull().sum()
y=df.Crop_Damage
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=150)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0,1))



scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
lr = LogisticRegression(solver='liblinear',multi_class='ovr')

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
y_pred = lr.predict(df1)
pred = pd.DataFrame(y_pred)
pred.head()