import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

df
df.isnull().any()
def sqft(x):

    if "-" in str(x):

        x=str(x).split(" - ")

        x=(float(x[0])+float(x[1]))/2

    return (x)

df["total_sqft"]=df["total_sqft"].apply(sqft)
df["size"].fillna(df["size"].mode()[0],inplace=True)
df["type"]=df["size"].apply(lambda x:str(x).split(" ")[1])

df["size"]=df["size"].apply(lambda x:str(x).split(" ")[0])
def ready(x):

    if(str(x)=="Ready To Move"):

        return 1

    else:

        return 0

df.availability=df["availability"].apply(ready)
df.drop("society",axis=1,inplace=True)
df["bath"].fillna(df["bath"].mean(),inplace=True)

df["balcony"].fillna(df["balcony"].mean(),inplace=True)

df["location"].fillna(df["location"].mode()[0],inplace=True)
def change(x):

    if("Sq. Meter" in str(x)):

        y=x.split("S")

        z=float(y[0])*10.7639

        return z

    elif("Sq. Yards" in str(x)):

        y=x.split("S")

        z=float(y[0])*9

        return z

    elif("Guntha" in str(x)):

        y=x.split("G")

        z=float(y[0])*1088.9848169

        return z

    elif("Acres" in str(x)):

        y=x.split("A")

        z=float(y[0])*43560

        return z

    elif("Perch" in str(x)):

        y=x.split("P")

        z=float(y[0])*272.25

        return z

    elif("Cents" in str(x)):

        y=x.split("C")

        z=float(y[0])*435.6

        return z

    elif("Grounds" in str(x)):

        y=x.split("G")

        z=float(y[0])*2400

        return z

    else:

        return x

df["total_sqft"]=df["total_sqft"].apply(change)
a=["balcony","size","bath"]

for i in a:

    df[i]=df[i].astype("int64")

df["total_sqft"]=df["total_sqft"].astype("float64")
df['price'] = df['price']*100000/df['total_sqft']
df
sns.boxplot("total_sqft",data=df,orient="vertical")
sns.boxplot("size",data=df,orient="vertical")
sns.boxplot("bath",data=df,orient="vertical")
sns.boxplot("bath",data=df,orient="vertical")
maxi=df[["bath","balcony","size","total_sqft","price"]].quantile(0.95)

maxi
df = df.drop(df[df['bath']>5].index)

df = df.drop(df[df['size']>5].index)

df=df.drop(df[df["price"]>15288.21].index)
sns.boxplot("price",data=df,orient="vertical")
sns.boxplot("bath",data=df,orient="vertical")
sns.boxplot("balcony",data=df,orient="vertical")
sns.boxplot("size",data=df,orient="vertical")
sns.boxplot("total_sqft",data=df,orient="vertical")
def outliers(df1):

    new_dataframe = pd.DataFrame()

    for key, df2 in df1.groupby('location'):

        m = np.mean(df2.price)

        st = np.std(df2.price)

        reduced_df = df2[(df2.price>(m-st)) & (df2.price<=(m+st))]

        new_dataframe = pd.concat([new_dataframe,reduced_df],ignore_index=True)

    return new_dataframe

df=outliers(df)

df=outliers(df)
df.shape
df.location = df.location.str.strip()

location_count = df['location'].value_counts(ascending=False)

location_8 = location_count[location_count<=8]

df.location = df.location.apply(lambda x: 'other' if x in location_8 else x)

df = df[df.location != 'other']
df.shape
X=df.drop("price",axis=1)

y=df.price
X=pd.get_dummies(X,drop_first=True)

X
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor().fit(X_train,y_train)

model.score(X_test,y_test)
from sklearn.linear_model import LinearRegression

model1=LinearRegression().fit(X_train,y_train)

model1.score(X_test,y_test)
from sklearn.neighbors import KNeighborsRegressor

model2=KNeighborsRegressor(n_neighbors=5)

model2.fit(X_train,y_train)

model2.score(X_test,y_test)