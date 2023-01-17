import pandas as pd

import numpy as np

df= pd.read_csv("../input/VehicleInformation.csv",sep=";")

#here 'sep' means seperator, it is ',' by default.

# to learn more about it run pd.read_csv? on your notebook 
df.head()

#head() method,by default, gives us the first 5 rows of dataframe or series.

#and also we have tail(), which gives the last 5 rows by default. we can write any negative or positive number in ().

#try df.head(100000) or df.head(-50000)
df.drop_duplicates(inplace=True)

len(df)
df.info()
print(list(df["FUEL_TYPE"].unique()))

print(np.min(df["MAX_SPEED"]))

print(np.max(df["MAX_SPEED"]))
df.drop(["FUEL_TYPE","MAX_SPEED","BRAND_CODE","VEHICLE_CODE"],inplace=True,axis=1)  #axis=1 means columns. it is axis=0 by default, which is rows.
#lets check again our DataFrame

print(len(df))

df.head()
y=df["VEHICLE_CLASS"]

x=df.drop(["VEHICLE_CLASS"],axis=1)

#Dont run the drop method first, you must either run these two together, or y=df["VEHICLE_CLASS"] first. 

#Otherwise, you may lose "VEHICLE_CLASS" by dropping it.
print(x.head())

print("\n")

print(y.head())
from sklearn.model_selection import train_test_split

x_train,x_test =train_test_split(x,test_size=0.2,random_state=42)

y_train,y_test=train_test_split(y,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=len(y.unique()))

knn.fit(x_train,y_train)
print("score with test_size=0.2 : ",knn.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier

x_train_new,x_test_new =train_test_split(x,test_size=0.3,random_state=42)

y_train_new,y_test_new=train_test_split(y,test_size=0.3,random_state=42)

knn=KNeighborsClassifier(n_neighbors=len(y.unique()))

knn.fit(x_train_new,y_train_new)

print("score with test_size=0.3 : ",knn.score(x_test_new,y_test_new))
