import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error

import pickle as pkl

import seaborn as sns

from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/beer-consumption-sao-paulo/Consumo_cerveja.csv")

df.head()
cols = ['Date','Temp_Mean','Temp_Min','Temp_Max','Rainfall','Weekend','Consumption/Week']

df.columns = cols

df.head()
df.fillna(method='ffill',inplace=True)

df['Rainfall'] = df['Rainfall'].apply(lambda x : float(str(x).replace(',','.')))

df['Temp_Mean'] = df['Temp_Mean'].apply(lambda x : float(str(x).replace(',','.')))

df['Temp_Min'] = df['Temp_Min'].apply(lambda x : float(str(x).replace(',','.')))

df['Temp_Max'] = df['Temp_Max'].apply(lambda x : float(str(x).replace(',','.')))
y = df['Consumption/Week']

X = df.drop(columns=['Consumption/Week'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
X_train.corr()
def preprocess(df):

    df['Difference'] = df['Temp_Max']-df['Temp_Min']

    df.drop(columns=['Date','Temp_Min','Temp_Max'],inplace=True)

    return df



def scaling(df,scaler=None):

    if scaler==None:

        sc = StandardScaler()

        sc.fit(df)

        df = sc.transform(df)

        pkl.dump(sc,open("beer_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return pd.DataFrame(df,columns=['Temp_Mean', 'Rainfall', 'Weekend', 'Difference'])
X_train = preprocess(X_train)

X_train = scaling(X_train)

X_test = preprocess(X_test)

X_test = scaling(X_test,pkl.load(open("beer_scaler.pkl",'rb')))
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
sns.residplot(y_test, y_pred, lowess=True, color="g")
import math

math.sqrt(mean_squared_error(y_test,y_pred))
math.sqrt(mean_squared_error(y_test,np.repeat(y_test.mean(),len(y_test))))