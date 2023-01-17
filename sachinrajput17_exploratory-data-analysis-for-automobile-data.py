import os

import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy import stats

import pandas_profiling as pp

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")

df.head()
# numbers of columns and rows in dataset.

df.shape
# Name of the columns of the dataset.

df.columns
# Datatypes of every column for the dataset.

df.dtypes
# Information(no of rows and columns, datatypes for the columns , null values in dataframe memory usage) about the dataset.

df.info()
#replace ? with the nan

df.replace("?",np.nan,inplace =True)

df.head()
df.info()
# sum of null values in every columns.

df.isnull().sum()
# no of duplicated rows in data frames

df.duplicated().value_counts()
#Count the unique values in num-of-doors

Counter(df["num-of-doors"])
miss_col= ["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]

for col in miss_col:

    df[col].replace(np.nan,df[col].astype("float").mean(axis=0),inplace=True)

    

df["num-of-doors"].replace(np.nan,df["num-of-doors"].value_counts().idxmax(),inplace=True)

df.head().T

    
print("Data Types of Variables \n",df.dtypes)
# correct the data format.

df[["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]]=df[["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]].astype("float")

df.dtypes
pp.ProfileReport(df)
#Statistical discription of the data for numerical features.

df.describe()
#Statistical discription of the data for categorical features.

df.describe(include='object')
df["city-L/100km"]=235/df["city-mpg"]

df["highway-L/100km"]=235/df["highway-mpg"]
df.drop(["city-mpg","highway-mpg"],axis=1)
for col in ["length","width","height"]:

    df[col]=df[col]/df[col].max()

    

df[["length","width","height"]].head()


df[["horsepower"]].hist()

plt.show()
df["horsepower_binned"]=pd.cut(df["horsepower"],bins=np.linspace(min(df["horsepower"]),max(df["horsepower"]),4),

                               labels=["low","medium","high"],include_lowest=True)

df[["horsepower","horsepower_binned"]].head()
plt.bar(["low","medium","high"],df["horsepower_binned"].value_counts())

plt.xlabel("horsepower",fontsize=15)

plt.ylabel("count",fontsize=15)

plt.title("Horsepower Bins",fontsize=15)

plt.show()
df.hist(bins=3,figsize=(15,12))

plt.tight_layout()
df.columns
dummy_var=pd.get_dummies(df[["fuel-type","aspiration"]])

dummy_var
# rename the dummy variable column names

dummy_var.rename(columns = {'fuel-type_gas':'gas','fuel-type_diesel':'diesel','aspiration_std':'std','aspiration_turbo':'turbo'},

                 inplace = True)

dummy_var
df=pd.concat([df,dummy_var],axis=1)



df.head()
#drop the unwanted columns as we create the dummy variables for them. 

data=df.drop(df[["fuel-type","aspiration"]],axis=1)

data.head()
dt=data.corr()

# correlation matrix where correlation of price with other variables is greater than 0.5

dt[dt["price"]>0.5]
dt[dt["price"]<-0.5].T
data.describe()
data.describe(include='object')
data["drive-wheels"].value_counts().to_frame().rename(columns={"drive-wheels":"value_counts"})
group_data=df[["drive-wheels","body-style","price"]].groupby(by=["drive-wheels","body-style"],as_index=False).mean()

group_data
group_data.pivot(index="drive-wheels",columns="body-style")


cols=["wheel-base","bore","horsepower","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]

for i in cols:

    pearson_coef,p_value=stats.pearsonr(df[i],df["price"])

    print("For {} :  pearson coefficient= {} and p value={} ".format(i,pearson_coef,p_value))
df_gptest = data[['drive-wheels','body-style','price']]

grouped_test=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])

grouped_test.head()

grouped_test.get_group('4wd')['price']
f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('rwd')['price'],grouped_test.get_group('fwd')['price'])

print("F-value is = ",f_val," P-Values is = ",p_val)
f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('rwd')['price'])

print("F-value is = ",f_val," P-Values is = ",p_val)
f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('fwd')['price'])

print("F-value is = ",f_val," P-Values is = ",p_val)
f_val,p_val=stats.f_oneway(grouped_test.get_group('fwd')['price'],grouped_test.get_group('rwd')['price'])

print("F-value is = ",f_val," P-Values is = ",p_val)
lm=LinearRegression()

X=df[['highway-mpg']]

Y=df['price']

lm.fit(X,Y)

y_pred=lm.predict(X)

y_pred[0:5]
print("intercept:",lm.intercept_)

print("coefficient:",lm.coef_)
plt.scatter(Y,y_pred)

plt.title("Predicted price by Linear Regression with one variable (Highway-mpg)",fontsize=15)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.show()
#regression plot

plt.figure(figsize=(8,8))

sns.regplot(x='highway-mpg',y='price',data=df)

plt.title("Regression Plot",fontsize=20)

#plt.ylim(0)

plt.show()



# Residual plot

plt.figure(figsize=(8,8))

sns.residplot(x='highway-mpg',y='price',data=df)

plt.title("Residual Plot",fontsize=20)

plt.show()
plt.scatter(Y,y_pred)

plt.title("Relation between Actual and Predicted values",fontsize=15)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.show()
r2_score(Y,y_pred)
x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

y= df["price"]

mlr=LinearRegression()

mlr.fit(x,y)

pred_mlr=mlr.predict(x)

pred_mlr[0:5]
print("intercept:",mlr.intercept_)

print("coefficient:",mlr.coef_)
plt.scatter(y,pred_mlr)

plt.title("Predicted price by Multiple Linear Regression with more than one variables",fontsize=15)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.show()
# function to plot the data.

def PollyPlot(model,x,y,name):

    x_new=np.linspace(15,55,100)

    y_new=model(x_new)

    plt.plot(x,y,'.',x_new,y_new,'-')

    plt.xlabel(name)

    plt.ylabel('price of cars')

    plt.show()
x=df["highway-mpg"]

y=df['price']

f=np.polyfit(x,y,3)

p=np.poly1d(f)

print(p)
PollyPlot(p,x,y,'highway-mpg')
z=df[['horsepower','curb-weight','engine-size','highway-mpg']]

pf=PolynomialFeatures(degree=2)

pf

z.shape
z_pf=pf.fit_transform(z)

z_pf
plr=LinearRegression()

plr.fit(z_pf,y)

pred_plr=plr.predict(z_pf)
plt.scatter(y,pred_plr)

plt.title("Predicted price by Polynomial Linear Regression with more than one variables",fontsize=15)

plt.xlabel("Actual Value")

plt.ylabel("Predicted Value")

plt.show()
plt.scatter(Y,y_pred)

plt.title("Predicted price by Linear Regression with one variable (Highway-mpg)",fontsize=15)

plt.xlabel("Actual_price")

plt.ylabel("Predicted Price")

plt.show()

plt.scatter(y,pred_mlr)

plt.title("Predicted price by Multiple Linear Regression with more than one variables",fontsize=15)

plt.xlabel("Actual_price")

plt.ylabel("Predicted Price")

plt.show()

plt.scatter(y,pred_plr)

plt.title("Predicted price by Polynomial Linear Regression with more than one variables",fontsize=15)

plt.xlabel("Actual_price")

plt.ylabel("Predicted Price")

plt.show()

# comparison of  r2 score  for MLR and PLR for same data .

print("Multiple Linear Regression R2 Score:\n",r2_score(y,pred_mlr))

print("Polynomial Regression R2 Score:\n",r2_score(y,pred_plr))