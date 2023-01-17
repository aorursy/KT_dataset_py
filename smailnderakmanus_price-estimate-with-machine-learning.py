import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df=pd.read_csv("/kaggle/input/housepricing/HousePrices_HalfMil.csv")

print(df.isnull().sum())
area=float(input("Area:"))
garage=int(input("Number of garages:"))
fireplace=int(input("Number of fireplaces:"))
baths=int(input("Number of baths:"))

marble=input("Marble kind (white,black or indian):")
if marble.lower().strip()=="white":
    df=df[df["White Marble"]==1]
if marble.lower().strip()=="black":
    df=df[df["Black Marble"]==1]
if marble.lower().strip()=="indian":
    df=df[df["Indian Marble"]==1]

floors=int(input("House have floor (yes=1,no=0):"))
if floors==1:
    df=df[df["Floors"]==1]
else:
    df=df[df["Floors"]==0]

city=int(input("City number (1,2,3):"))
if city==1:
    df=df[df["City"]==1]
if city == 2:
    df = df[df["City"] == 2]
if city == 3:
    df = df[df["City"] == 3]

electric=int(input("House have electric (yes=1,no=0):"))
if electric==1:
    df=df[df["Electric"]==1]
else:
    df = df[df["Electric"] == 0]

fiber=int(input("House have fiber (yes=1,no=0):"))
if fiber==1:
    df=df[df["Fiber"]==1]
else:
    df = df[df["Fiber"] == 0]

solar=int(input("House have solar (yes=1,no=0):"))
if solar==1:
    df=df[df["Solar"]==1]
else:
    df = df[df["Solar"] == 0]

glass_doors=int(input("House have glass door (yes=1,no=0):"))
if glass_doors==1:
    df=df[df["Glass Doors"]==1]
else:
    df = df[df["Glass Doors"] == 0]

pool=int(input("House have swimming pool (yes=1,no=0):"))
if pool==1:
    df=df[df["Swiming Pool"]==1]
else:
    df = df[df["Swiming Pool"] == 0]

garden=int(input("House have garden (yes=1,no=0):"))
if garden==1:
    df=df[df["Garden"]==1]
else:
    df = df[df["Garden"] == 0]

df.drop(["White Marble", "Black Marble", "Indian Marble", "Floors", "City","Electric","Fiber","Solar","Glass Doors","Swiming Pool","Garden"], axis=1, inplace=True)



y=df[["Prices"]]
X=df.drop("Prices",axis=1)

from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(X,y)

values=[[area],[garage],[fireplace],[baths]]

values=pd.DataFrame(values).T

model.predict(values)
#succes of model
from sklearn.metrics import mean_squared_error

MSE=mean_squared_error(y,model.predict(X))        
RMSE=np.sqrt(MSE) 

print(MSE,RMSE)
#verification
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)

train_model=LinearRegression().fit(X_train,y_train)


#RMSE for train values
RMSE_train=np.sqrt(mean_squared_error(y_train,train_model.predict(X_train)))
#RMSE for test values
RMSE_test=np.sqrt(mean_squared_error(y_test,train_model.predict(X_test)))


RMSE_train
RMSE_test
#cross validation
from sklearn.model_selection import cross_val_score


MSE_crossval=np.mean(-cross_val_score(train_model,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))          
MSE_crossval
RMSE_crossval=np.sqrt(MSE_crossval)
RMSE_crossval
#verification
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)

train_model=LinearRegression().fit(X_train,y_train)


#RMSE for train values
print(np.sqrt(mean_squared_error(y_train,train_model.predict(X_train))))

#RMSE for test values
print(np.sqrt(mean_squared_error(y_test,train_model.predict(X_test))))
