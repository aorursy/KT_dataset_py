#Installing ibraries

!pip install regressors
import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from regressors import stats

from sklearn import linear_model as lm

import statsmodels.formula.api as sm

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import classification_report

print(os.listdir("../input"))
#Data Preprocessing

gp=pd.read_csv("../input/googleplaystore.csv")

#print(gp.head()) #understanding what information is in the data



google=pd.read_csv("../input/googleplaystore.csv") #to be used later when creating scatterplots, need general data

google = google[["Size", "Installs", "Reviews", "Price", "Rating"]] # Have to list out specific columns, because if I don't when I convert to float and then remove null values, the entire dataframe is emptry

google = google[google.Size != 'Varies with device'] #removing all rows where there is this string

google = google[~google.Size.str.contains("k")] #removing all rows where there is a k

google['Installs'] = google['Installs'].str.replace(',', '') #Removed commas

google['Installs'] = google['Installs'].str.replace('+', '') #Removed + signs

google['Size'] = google['Size'].str.replace('M', '') #Removed M's

google['Size'] = google['Size'].str.replace('k', '') #Removed k's 

google['Price'] = google['Price'].str.replace('$', '') #Removed $ signs



for x in google:

    google[x] = pd.to_numeric(google[x], errors='coerce') #converted everything to float

print(google.dtypes) #checking to make sure it all converted correctly

print(google.head())

google = google.dropna() #removed rows will null values

print(google.isnull().values.any()) #data does have null values



#Creating and cleaning Dataframe for linear modeling

gp = gp[["Size", "Installs"]]

gp = gp[gp.Size != 'Varies with device'] #removing all rows where there is this string

gp = gp[~gp.Size.str.contains("k")] #removing all rows where there is a k

gp['Installs'] = gp['Installs'].str.replace(',', '') #Removed commas

gp['Installs'] = gp['Installs'].str.replace('+', '') #Removed + signs

gp['Size'] = gp['Size'].str.replace('M', '') #Removed M signs

gp['Size'] = gp['Size'].str.replace('k', '') #Removed K signs



for x in gp:

    gp[x] = pd.to_numeric(gp[x], errors='coerce') #converted everything to float



print(gp.dtypes) #checking to make sure it all converted correctly



#print(gp) #making sure that all the data looks good

#print(gp.count()) #Making sure that we have enought points of data

#print(gp.isnull().values.any()) #data does have null values

gp = gp.dropna() #removed rows will null values

print(gp.isnull().values.any()) #data does have null values

print(gp.count()) #Making sure that we have enought points of data
print(google.head())
#Linear Regression Interaction - Size and Number of Reviews

inter = sm.ols(formula="Installs ~ Size*Reviews",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - Size and Price

inter = sm.ols(formula="Installs ~ Size*Price",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - Size and Rating

inter = sm.ols(formula="Installs ~ Size*Rating",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - 

inter = sm.ols(formula="Installs ~ Rating*Price",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - 

inter = sm.ols(formula="Installs ~ Reviews*Rating",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - Number of Reviews, Ratings, and App Size

inter = sm.ols(formula="Installs ~ Reviews*Rating*Size",data=google).fit()

print(inter.summary())
#Linear Regression Interaction - 

inter = sm.ols(formula="Installs ~ Reviews*Rating*Size*Price",data=google).fit()

print(inter.summary())
#Simple Linear Regression of Number of App Reviews

#Adjusted R-Squared & P-vlaue genarated using Statsmodels

res = sm.ols(formula="Installs ~ Reviews",data=google).fit()

print(res.summary())
#Polynomial Transformation of Number of App Reviews

#Adjusted R-Squared & P-vlaue genarated for cubic polynomial transformation

res = sm.ols(formula="Installs ~ Reviews + I(Reviews*Reviews)+ I(Reviews*Reviews*Reviews)",data=google).fit()

print(res.summary())
#Logarithmic Transformation of Number of App Reviews

res = sm.ols(formula = "Installs ~ np.log(Reviews)",data=google).fit()

print(res.summary())
#Simple Linear Regression of App Size

#Adjusted R-Squared & P-vlaue genarated using Statsmodels

res = sm.ols(formula="Installs ~ Size",data=google).fit()

print(res.summary())
#Polynomial Transformation of App Size

#Adjusted R-Squared & P-vlaue genarated for cubic polynomial transformation

res = sm.ols(formula="Installs ~ Size + I(Size*Size)+ I(Size*Size*Size)",data=google).fit()

print(res.summary())
#Logarithmic Transformation of App Size

res = sm.ols(formula = "Installs ~ np.log(Size)",data=google).fit()

print(res.summary())
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
DF=pd.read_csv("../input/googleplaystore.csv") #to be used later when creating scatterplots, need general data



FS = DF

FS['TypeFree'] = FS['Type'].map({'Free': 1, 'Paid': 0})

FS = FS[["Size", "Reviews", "Installs","Price", "Rating", "TypeFree", "Category", "Genres", "Content Rating"]] # Have to list out specific columns, because if I don't when I convert to float and then remove null values, the entire dataframe is emptry

FS = FS.rename(columns={"Content Rating":"Content_Rating"})

FS = FS[FS.Size != 'Varies with device'] #removing all rows where there is this string

FS = FS[~FS.Size.str.contains("k")] #removing all rows where there is a k

FS['Installs'] = FS['Installs'].str.replace(',', '') #Removed commas

FS['Installs'] = FS['Installs'].str.replace('+', '') #Removed + signs

FS['Size'] = FS['Size'].str.replace('M', '') #Removed M's

FS['Size'] = FS['Size'].str.replace('k', '') #Removed k's 

FS['Price'] = FS['Price'].str.replace('$', '') #Removed $ signs

FS = FS.dropna()



for x in FS:

    FS[x] = pd.to_numeric(FS[x], errors='coerce')

    

FS['Category'] = pd.get_dummies(DF['Category'])

FS['Genres'] = pd.get_dummies(DF['Genres'])

FS['Content_Rating'] = pd.get_dummies(DF['Content Rating'])



outputDF = FS["Installs"].copy()



inputDF = FS[["Size","Reviews","Price","Rating","TypeFree","Category","Genres","Content_Rating"]].copy()



inter = sm.ols(formula="Installs ~ Reviews*Rating*Size*Category*Content_Rating",data=FS).fit()

print(inter.summary())



# from mlxtend.feature_selection import SequentialFeatureSelector as sfs



# forwardModel = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

# forwardModel.fit(inputDF,outputDF)



# print (forwardModel.k_feature_idx_)



# print(forwardModel.k_feature_names_)
DF=pd.read_csv("../input/googleplaystore.csv") #to be used later when creating scatterplots, need general data



FS = DF

FS['TypeFree'] = FS['Type'].map({'Free': 1, 'Paid': 0})

FS = FS[["Size", "Reviews", "Installs","Price", "Rating", "TypeFree", "Category", "Genres", "Content Rating"]] # Have to list out specific columns, because if I don't when I convert to float and then remove null values, the entire dataframe is emptry

FS = FS.rename(columns={"Content Rating":"Content_Rating"})

FS = FS[FS.Size != 'Varies with device'] #removing all rows where there is this string

FS = FS[~FS.Size.str.contains("k")] #removing all rows where there is a k

FS['Installs'] = FS['Installs'].str.replace(',', '') #Removed commas

FS['Installs'] = FS['Installs'].str.replace('+', '') #Removed + signs

FS['Size'] = FS['Size'].str.replace('M', '') #Removed M's

FS['Size'] = FS['Size'].str.replace('k', '') #Removed k's 

FS['Price'] = FS['Price'].str.replace('$', '') #Removed $ signs

FS = FS.dropna()



for x in FS:

    FS[x] = pd.to_numeric(FS[x], errors='coerce')

    

FS['Category'] = pd.get_dummies(DF['Category'])

FS['Genres'] = pd.get_dummies(DF['Genres'])

FS['Content_Rating'] = pd.get_dummies(DF['Content Rating'])



outputDF = FS["Installs"].copy()



inputDF = FS[["Size","Reviews","Price","Rating","TypeFree","Category","Genres","Content_Rating"]].copy()



from mlxtend.feature_selection import SequentialFeatureSelector as sfs



backwardModel = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

backwardModel.fit(inputDF,outputDF)



print (backwardModel.k_feature_idx_)



print(backwardModel.k_feature_names_)

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

from sklearn.linear_model import LinearRegression 



DF=pd.read_csv("../input/googleplaystore.csv") #to be used later when creating scatterplots, need general data



FS = DF

FS['TypeFree'] = FS['Type'].map({'Free': 1, 'Paid': 0})

FS = FS[["Size", "Reviews", "Installs","Price", "Rating", "TypeFree", "Category", "Genres", "Content Rating"]] # Have to list out specific columns, because if I don't when I convert to float and then remove null values, the entire dataframe is emptry

FS = FS.rename(columns={"Content Rating":"Content_Rating"})

FS = FS[FS.Size != 'Varies with device'] #removing all rows where there is this string

FS = FS[~FS.Size.str.contains("k")] #removing all rows where there is a k

FS['Installs'] = FS['Installs'].str.replace(',', '') #Removed commas

FS['Installs'] = FS['Installs'].str.replace('+', '') #Removed + signs

FS['Size'] = FS['Size'].str.replace('M', '') #Removed M's

FS['Size'] = FS['Size'].str.replace('k', '') #Removed k's 

FS['Price'] = FS['Price'].str.replace('$', '') #Removed $ signs

FS = FS.dropna()



for x in FS:

    FS[x] = pd.to_numeric(FS[x], errors='coerce')

    

FS['Category'] = pd.get_dummies(DF['Category'])

FS['Genres'] = pd.get_dummies(DF['Genres'])

FS['Content_Rating'] = pd.get_dummies(DF['Content Rating'])







#Backward Selection

#From the backward selection we noticed that Size, Reviews, Ratings, Category and Content_rating have an effect on the installs.

#rmse using Leave one out

outputDF = FS["Installs"].copy()



inputDF = FS[["Size","Reviews","Rating","Category","Content_Rating"]].copy()



model = LinearRegression()

loocv = LeaveOneOut()



rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())



#rmse using Kfold - 5 folds

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())



#rmse using Kfold - 10 folds

kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())