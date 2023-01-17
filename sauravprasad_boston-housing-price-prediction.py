#importing libraries 

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn.model_selection import train_test_split

import numpy as np

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor



%matplotlib inline
#storing the data into variable

boston_dataset=load_boston()

#creating pandas dataframe

data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)



#adding target variable

data['PRICE']=boston_dataset.target
#will return 5 values

data.head()
#this function will return the information about each column

data.info()
#PRICE DISTRIBUTION

#setting plot size

plt.figure(figsize=(20,7))

sns.set(style="whitegrid")



#plot using matplotlib

plt.subplot(1,2,1)   #arguments no of rows,columns, and index

plt.hist(data.PRICE,bins=50,ec="black",color="#FFEB3B")  #ec- edge color

plt.xlabel("Price in Thousands",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Price Distribution (Matplot)",fontsize=16)





#plot using seaborn

plt.subplot(1,2,2)

sns.distplot(data.PRICE,bins=50,color="#512DA8")

plt.xlabel("Price in Thousands",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Price Distribution (Seaborn)",fontsize=16)

#there are outliers in price
#Rooms plot 

plt.figure(figsize=(20,7))

sns.set(style="whitegrid")



plt.subplot(1,2,1)

plt.hist(data.RM,bins=50,ec="black",color="#33691e") #ec- edge color

plt.xlabel("Number of rooms",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("RM Distribution (Matplot)",fontsize=16)





plt.subplot(1,2,2)

sns.distplot(data.RM,bins=50,color="#dd2c00")

plt.xlabel("Number of Rooms",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("RM Distribution (Seaborn)",fontsize=16)

#Index of accessibility of highways 



#Setting figure size

plt.figure(figsize=(20,7))

sns.set(style="whitegrid")





#plot using matplotlib

plt.subplot(1,2,1)

plt.hist(data.RAD,bins=24,ec="black",color="#880e4f")

plt.xlabel("Radial Index",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("RAD Disrtibution (Matplot)",fontsize=16)



#plot using seaborn

plt.subplot(1,2,2) 

sns.distplot(data.RAD,bins=24,color="#c62828")

plt.xlabel("Radial Index",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("RAD Disrtibution (Seaborn)",fontsize=16)

#checking RAD variable

freq=data.RAD.value_counts()

freq
#creating a barchart

#Setting figure size

plt.figure(figsize=(7,7))

sns.set(style="whitegrid")





#plot using matplotlib



plt.bar(freq.index,height=freq,color="#bf360c")

plt.xlabel("Radial Index",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("RAD Disrtibution (Matplot)",fontsize=16)

#this will return statistical summary

data.describe()
#corrleation of price with attributes using pearson method

corr_mat=data.corr()
#creating correlation heatmap

plt.figure(figsize=(15,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(corr_mat,annot=True,annot_kws={"size":14},linewidth=.5)
#plot between NOX(measure of pollution) and DIS(distance to employement centres)

data.ND=round((data.NOX.corr(data.DIS)),2)





plt.figure(figsize=(10,6))



plt.scatter(x=data.DIS,y=data.NOX,color="green",alpha=0.5)

plt.xlabel("DIS-Distance from employement center",fontsize=12)

plt.ylabel("NOX-Measure of Pollution",fontsize=12)

plt.title(f"Distance V/S Pollution (correlation{data.ND})",fontsize=12)

#plot between NOX(measure of pollution) and RADdistance to employement centres)

data.ND=round((data.TAX.corr(data.RAD)),2)





plt.figure(figsize=(10,6))



plt.scatter(x=data.TAX,y=data.RAD,color="blue",alpha=0.5)

plt.xlabel("TAX",fontsize=12)

plt.ylabel("RAD",fontsize=12)

plt.title(f"TAX V/S RAD (correlation{data.ND})",fontsize=12)



#running linear regression and plotting it, for this we use seaborn

sns.lmplot(x="TAX",y="RAD",data=data,size=7)

plt.show()
#price and RM 

data.ND=round((data.PRICE.corr(data.RM)),2)

 

plt.figure(figsize=(10,6))



plt.scatter(x=data.RM,y=data.PRICE,color="darkgreen",alpha=0.5)

plt.xlabel("No. Of rooms per dwelling",fontsize=12)

plt.ylabel("Price of the House",fontsize=12)

plt.title(f"Price Vs No. of Rooms (correlation{data.ND})",fontsize=12)



#regression plot using seaborn

sns.lmplot(x="RM",y="PRICE",data=data,size=7)

plt.title("Regresion line btw RM and PRICE")

plt.xlabel("RM")

plt.ylabel("Price")

plt.show()
#creating scatter plot of LSTAT and NO. of rooms

data.ND=round((data.PRICE.corr(data.LSTAT)),2)

 

plt.figure(figsize=(10,6))



plt.scatter(x=data.LSTAT,y=data.PRICE,color="darkgreen",alpha=0.5)

plt.xlabel("proportion of Lower status",fontsize=12)

plt.ylabel("Price of the House",fontsize=12)

plt.title(f"LSTAT Vs No. of Rooms (correlation{data.ND})",fontsize=12)
#regression plot using seaborn

sns.lmplot(x="LSTAT",y="PRICE",data=data,size=7)

plt.title("Regresion line btw LSTAT and PRICE")

plt.xlabel("LSTAT")

plt.ylabel("Price")

plt.show()
#creating target and features

prices=data.PRICE

features=data.drop("PRICE",axis=1)





#tuple unpacking / dividing dataset

X_train,X_test,Y_train,Y_test=train_test_split(features,prices,test_size=0.2,random_state=10)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
reg=LinearRegression()

model_1=reg.fit(X_train,Y_train)

print("Train data R squared value is :",model_1.score(X_train,Y_train))

print("Test data R squared values is :",model_1.score(X_test,Y_test))



y=Y_train

x=sm.add_constant(X_train)

mod=sm.OLS(y,x)

res=mod.fit()

res.summary()
residuals=res.resid
plt.figure(figsize=(10,4))

mean=np.mean(residuals)

sns.distplot(residuals)

plt.title(f"Residual plot, Mean of Residuals {mean}")
#checking for the assumption of Homoscadasticity

from statsmodels.stats.diagnostic import het_goldfeldquandt 

het_goldfeldquandt(Y_train,X_train)
mc=pd.Series([variance_inflation_factor(X_train, i) 

               for i in range(data.drop("PRICE",axis=1).shape[1])], 

              index=data.drop("PRICE",axis=1).columns)

mc=round(mc,2)

mc
#removing all the variables having multi-collinearity greater than 10

data=data.drop(["INDUS","NOX","RM","AGE","RAD","TAX","PTRATIO","B"],axis=1)
#creating target and features

prices=data.PRICE

features=data.drop("PRICE",axis=1)





#tuple unpacking / dividing dataset

X_train,X_test,Y_train,Y_test=train_test_split(features,prices,test_size=0.2,random_state=10)





X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
y=Y_train

x=sm.add_constant(X_train)

mod=sm.OLS(y,x)

res=mod.fit()

res.summary()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
#preparing data

#creating target and features

prices=data.PRICE.values

features=data.drop("PRICE",axis=1).values





#splitting the data

X_train,X_test,y_train,y_test=train_test_split(features,prices,test_size=0.2,random_state=10)



#scaling

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

model=Sequential()#creating model



#adding layer

model.add(Dense(12,activation="relu"))

model.add(Dense(6,activation="relu"))

model.add(Dense(3,activation="relu"))



model.add(Dense(1))



#N.N will evaluate the model on the basis of compile

model.compile(optimizer="adam",loss="mse")
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=128,epochs=500)
loss=pd.DataFrame(model.history.history)

plt.plot(loss)
predictions=model.predict(X_test)

mean_absolute_error(predictions,y_test)