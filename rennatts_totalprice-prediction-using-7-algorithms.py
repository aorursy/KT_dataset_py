#import packages and dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy import stats

from scipy.stats import norm, skew

import sklearn.metrics as metrics

import os



df= pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
#Let's check for missing data

df.isnull().sum()
#totalprice correlation matrix

k = 10 #number of variables for heatmap

plt.figure(figsize=(16,8))

corrmat = df.corr()

# picking the top 15 correlated features

cols = corrmat.nlargest(k, 'total (R$)')['total (R$)'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

#finding outliers

fig, ax = plt.subplots()

ax.scatter(x = df['hoa (R$)'], y = df['total (R$)'])

plt.ylabel('price', fontsize=13)

plt.xlabel('hora', fontsize=13)

plt.show()
#Deleting outliers

df= df.drop(df[(df['hoa (R$)']>400000) & (df['total (R$)']>800000)].index)

#checking for outliers again

fig, ax = plt.subplots()

ax.scatter(x = df['hoa (R$)'], y = df['total (R$)'])

plt.ylabel('price', fontsize=13)

plt.xlabel('hora', fontsize=13)

plt.show()
#deleting outliers

df= df.drop(df[(df['hoa (R$)']>100000) & (df['total (R$)']>200000)].index)
#finding outliers

fig, ax = plt.subplots()

ax.scatter(x = df['hoa (R$)'], y = df['total (R$)'])

plt.ylabel('price', fontsize=13)

plt.xlabel('hora', fontsize=13)

plt.show()
#deleting outliers

df= df.drop(df[(df['hoa (R$)']>60000) & (df['total (R$)']>90000)].index)
#finding outliers

fig, ax = plt.subplots()

ax.scatter(x = df['hoa (R$)'], y = df['total (R$)'])

plt.ylabel('price', fontsize=13)

plt.xlabel('hora', fontsize=13)

plt.show()
#deleting outliers

df= df.drop(df[(df['total (R$)']>300000)].index)

df= df.drop(df[(df['hoa (R$)']>30000)].index)
#finding outliers

fig, ax = plt.subplots()

ax.scatter(x = df['hoa (R$)'], y = df['total (R$)'])

plt.ylabel('price', fontsize=13)

plt.xlabel('hora', fontsize=13)

plt.show()
#target variable- sale price

sns.distplot((df['total (R$)']), fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit((df['total (R$)']))

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df['total (R$)'], plot=plt)

plt.show()

sns.pairplot(df)
#hora x total

sns.lmplot(x='hoa (R$)',y='total (R$)',data=df) #hour is very correlated to total price.

plt.figure(figsize=(13,8))

sns.boxplot(x= 'bathroom',y='total (R$)',data=df)

plt.show()
plt.figure(figsize=(13,8))

sns.boxplot(x= 'rooms',y='total (R$)',data=df)

plt.show()
#histogram of the number of rooms

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.hist(df['bathroom'])

plt.title("number of rooms")

plt.xlabel("quantity")

plt.ylabel("number of rooms")

plt.grid()

plt.show()

#casas x preço

plt.scatter(df['area'],df['total (R$)'])

plt.title("area x price")

plt.xlabel("area")

plt.ylabel("price")

plt.show()
plt.figure(figsize=(10,10))

sns.boxplot(x="city", y= 'rooms', palette=["m", "g"], data=df)

plt.title('City and number of rooms')
plt.figure(figsize=(13,8))

sns.boxplot(x= 'city',y='total (R$)',data=df)

plt.show()
sns.countplot(df['animal'],hue = df['city']).set_title('animals allowed per city')
sns.violinplot(x ='furniture', y ='rent amount (R$)', data = df, hue ='city').set_title=("furniture per city and total price")
#parking spaces

plt.figure(figsize =(6,6))

plt.subplot(2,1,1)

ax = sns.regplot(df['parking spaces'],df['rent amount (R$)'])

plt.subplot(2,1,2)

sns.distplot(df['parking spaces'],kde =False)
#fire insurance x total price per city

plt.figure(figsize =(12,6))

sns.violinplot(x ='city', y ='fire insurance (R$)', data = df,hue ='city')

#fire insurance is very related to total price

ax = sns.regplot(df['fire insurance (R$)'],df['rent amount (R$)'])

# Categorical boolean mask

categorical_feature_mask = df.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = df.columns[categorical_feature_mask].tolist()





from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df[categorical_cols] = df[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))

#selecting dependent and independent variables

X= df.drop(["total (R$)"], axis=1)

y= df.loc[:,["total (R$)"]]

#split the dataset

from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(X,y,test_size =0.3)
#building the machine learning models

acc= []





#Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor as regr

model =regr()

model.fit(X_train,y_train)

from sklearn.metrics import r2_score

print(r2_score(y_test,model.predict(X_test)))

acc.append(['DTR',r2_score(y_test,model.predict(X_test))])





#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['RFN',r2_score(y_test,model.predict(X_test))])





#Linear regression

from sklearn.linear_model import LinearRegression as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['LIR',r2_score(y_test,model.predict(X_test))])





#SVM Regression

from sklearn.svm import SVR as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['SVM',r2_score(y_test,model.predict(X_test))])







#K Nearest Neighbour Regression

from sklearn.neighbors import KNeighborsRegressor as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['KNNR',r2_score(y_test,model.predict(X_test))])



#Lasso Regression

from sklearn.linear_model import Lasso as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['LaR',r2_score(y_test,model.predict(X_test))])



#Ridge Regression

from sklearn.linear_model import Ridge as regr

model =regr()

model.fit(X_train,y_train)

print(r2_score(y_test,model.predict(X_test)))

acc.append(['RiR',r2_score(y_test,model.predict(X_test))])





#Different Algorithms and their performance

acc.sort(key = lambda y:y[1],reverse =True)

#print all the models accurancy score

print(acc)
#As the RiR tops the list we will use it as our final model!!!

from sklearn.linear_model import Ridge as regr

model =regr()

model.fit(X_train,y_train)
#making the predictions

y_pred = model.predict(X_test)
#ploting the model prediction with the y_test values the check the model prediction power

ax1 = sns.distplot(y_test,hist=False,kde =True,color ="r",label ="Actual Value")

sns.distplot(model.predict(X_test),color ="b",hist = False,kde =True, label = "Preicted Value",ax =ax1)
