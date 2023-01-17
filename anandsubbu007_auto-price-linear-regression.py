%%time

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/AutoData.csv")

df.head(3)
df.isnull().sum()

#No null values
df.columns
for i in df.columns:

    print(i)

    print(df[i].unique())
df["make"].unique()

#it has both brand name & model name
#seperating make name & model from data

n= df["make"].str.split(" ",n = 1, expand = True)

n.head(2)
df["make"]=n[0].str.upper()

df["model"]=n[1]

df["make"].unique()

#df["model"].unique()

df["model"].value_counts()

df.head()
#maker is seperated 

df1 = df.drop(["model"],axis=1)

df1.head()
df1.info()
df1["make"].unique()

df1["make"]= df1["make"].replace(to_replace='TOYOUTA',value='TOYOTA')

df1["make"]= df1["make"].replace(to_replace='VOKSWAGEN',value='VOLKSWAGEN')

#volkswagen

df1["make"].unique()
make = df1["make"].value_counts().reset_index()

make.head()
make.head()
plt.figure(figsize=[20,10])

sns.set(style="whitegrid")

chart=sns.barplot("index","make",data = make)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.title("Number of vehicles by make")

plt.ylabel('Number of vehicles')

plt.xlabel('Make');
sns.distplot(df1["symboling"]).set(xlim=(-3,3))
sns.countplot(df1["fueltype"])
#df1.dtypes(include=[object])

obj_col = df1.select_dtypes(object).columns

obj_col = obj_col.drop("make")
for i in obj_col:

    #plt.figure(figsize=[8,8])

    sns.set(style="whitegrid")

    sns.countplot(x = i,data=df1)

    plt.show()
int_col = df1.select_dtypes(exclude=object).columns

int_col
for i in int_col:

    plt.figure(figsize=[8,8])

    sns.set(style="whitegrid")

    sns.distplot(df1[i])

    plt.show()
sns.pairplot(df1)
corr = df1.corr()

plt.figure(figsize = (20,15))

sns.heatmap(corr,annot=True,linewidths=1)

#plt.figure(figsize=[20,20])
corr["price"].reset_index().sort_values(["price"], ascending=False)
plt.figure(figsize=[15,10])

a=sns.boxplot(df1["make"],df1["price"])

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
sns.regplot(df1["enginesize"],df1["price"])
#plt.subplot(figsize=[5,5])

sns.pairplot(data = df1,x_vars=df1.columns,y_vars="price")
from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split
df1.head(3)
# create training and testing vars

y = df1[["price"]]

X = df1[["enginesize"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40, random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

#X.head()
model = linear_model.LinearRegression()

model.fit(X_train,y_train)
preds = model.predict(X_test)
from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))
#curbweight

y = df1[["price"]]

X = df1[["curbweight"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40, random_state=100)



model = linear_model.LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_test)



from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))
#horsepower



y = df1[["price"]]

X = df1[["horsepower"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40, random_state=100)



model = linear_model.LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_test)



from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))
#symboling: -3, -2, -1, 0, 1, 2, 3

#(Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less),

# this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling".

# A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.)

#rep_val = {3:"Risky",2:"",1:"",0:"",}

df1["symboling"] = df1["symboling"].astype(str)

df1.head()
df_dum = pd.get_dummies(df1)

df_dum.head()



test_dum = pd.get_dummies(df1)

test_dum.head(3)

#test_dum.info()
y = test_dum[["price"]]

X = test_dum.drop("price",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
#VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(X_train.values, ind) for ind in range(3)]
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = True)

vif[:12]
vif_col = vif[:12]

col = list(vif_col["Features"])

col
y = test_dum[["price"]]

X = test_dum[col]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
model = linear_model.LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_test)



from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))
vif_col = vif[:8]

col = list(vif_col["Features"])



y = test_dum[["price"]]

X = test_dum[col]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

model = linear_model.LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_test)



from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))
y = test_dum[["price"]]

X = test_dum.drop("price",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
model = linear_model.LinearRegression()

model.fit(X_train,y_train)

preds = model.predict(X_test)



from sklearn.metrics import r2_score

print("R2 score : %.2f" % r2_score(y_test,preds))