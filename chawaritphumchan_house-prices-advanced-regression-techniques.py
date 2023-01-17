# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

from pandas import DataFrame,Series 

import matplotlib.pyplot as plt 

import seaborn as sns 

%matplotlib inline 

from sklearn.preprocessing import StandardScaler 
#read_csv file 



df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#check columns , shpe ...

print("shape:", df_train.shape)

print("columns", df_train.columns)
df_train["SalePrice"].describe()



#min is larger than zero 
#distributing 



sns.distplot(df_train["SalePrice"])



# we can see that this has normal distribution .

# show peakedness 
var1 = "TotalBsmtSF"

var2 = "GrLivArea"

plt.figure(figsize = (10,4))

plt.subplot(1,3,1)



plt.scatter(df_train[var1], df_train["SalePrice"])

plt.xlabel("TotalBsmtSF")

plt.ylabel("SalePrice")



plt.subplot(1,3,3)

plt.scatter(df_train[var2], df_train["SalePrice"])

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")



#both have powerfull positive linearity 
var = "YearBuilt"

data = pd.concat([df_train["SalePrice"], df_train[var]], axis =1)

f , ax = plt.subplots(figsize= (16, 8))



fig = sns.boxplot(x= var , y = "SalePrice" , data = data )

plt.xticks(rotation =90)



# new bubuilt has tendency that price has wide range
var = "OverallQual"



data = pd.concat([df_train["SalePrice"], df_train[var]], axis =1)



f , ax = plt.subplots(figsize=(8,6))

fig = sns.boxplot(x=var , y= "SalePrice" , data =data)





#positive reration to sale price
corrmatrix = df_train.corr()

plt.figure(figsize = (12,9))

sns.heatmap(corrmatrix , square =True)
k =10 

cols = corrmatrix.nlargest(k,"SalePrice")["SalePrice"].index

cm =np.corrcoef(df_train[cols].values.T)

hm = sns.heatmap(cm , cbar =True , annot=True , square =True,

                annot_kws={"size":10},yticklabels=cols.values,

                xticklabels =cols.values)



plt.show()



## plot heatmap  by top10 numericalfeature  related to "SalePrice" 
#pairplot to see overview of corelationship 



cols = ["SalePrice","OverallQual","GrLivArea", "GarageCars",

       "TotalBsmtSF","FullBath", "YearBuilt"] #features which we adopt.



sns.pairplot(df_train[cols])
#missing data 

total = df_train.isnull().sum().sort_values(ascending=False) 

percent = (df_train.isnull().sum()/df_train.count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent], axis=1 )

missing_data.columns = ["Total", "Percent"]

missing_data.loc["GarageCars",:]

missing_data.head(20)



#sum()=>count True by columns 

#count()=> all caunt by columns
delete_index = missing_data[missing_data["Total"] >1].index

df_train= df_train.drop(delete_index.values , axis =1)

df_train =df_train.drop(df_train.loc[df_train["Electrical"].isnull()].index)



df_train.isnull().sum().max()
df_train.columns
#to define a threshhold to define an outlier , we use StandardScaler



from sklearn.preprocessing import StandardScaler



saleprice_scaled = StandardScaler().fit_transform(df_train["SalePrice"][:,np.newaxis])



sorted_price = np.sort(saleprice_scaled ,axis =0)

#np.array(df_train["SalePrice"]).reshape(-1,1)



low_range = sorted_price.flatten()[:10].reshape(-1,1)

high_range = sorted_price.flatten()[-10:].reshape(-1,1)



print("out range (low) of distribution:" )

print(low_range)

print("\n\n_____________________________________________\n\n")

print("out range(high) of distribution:")

print(high_range)
var ="GrLivArea"

data = pd.concat([df_train["SalePrice"],df_train[var]],axis=1)

data.plot.scatter(x=var , y = "SalePrice" , ylim=(0.8000000))

plt.grid()
#delete points 

#ascending => sort upper 

#axis =1



delete_index = df_train.sort_values(by = var ,ascending=False)[:2].index

df_train.drop(delete_index , axis =0)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

plt.grid()
import scipy.stats as stats



plt.grid(True)

plt.title("SalePrice-distribution")

sns.distplot(df_train["SalePrice"]  ,bins =100)

fig = plt.figure()



res = stats.probplot(df_train["SalePrice"] , plot=plt)
# to nmalize , we use transformation to data by log.

df_train["SalePrice"] = np.log(df_train["SalePrice"])
sns.distplot(df_train['SalePrice']);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train
df_train["Electrical"].value_counts() #categorical feature 
df_test= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv",

                     

                    )



df_train = pd.concat([df_train , pd.get_dummies(df_train["Electrical"])], axis =1)

df_test = pd.concat([df_test , pd.get_dummies(df_test["Electrical"])], axis =1) 
from sklearn.linear_model import LinearRegression 

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge 

from sklearn.linear_model import Lasso



#dummy_electrical = pd.get_dummies(df_train["Electrical"],drop_first =True )

#df_train = pd.concat([df_train , dummy_electrical] , axis =1)





features =  ["OverallQual","GrLivArea", "GarageCars",

                   "TotalBsmtSF","FullBath", "YearBuilt",'1stFlrSF','FuseA', 'FuseF', 'FuseP', 'SBrkr']



train_data = df_train[features]

X = train_data.values

y = df_train["SalePrice"]



X_train,X_val , y_train, y_val = train_test_split(X, y , random_state =0)



model1 = RandomForestRegressor(random_state =0 , max_depth=1000)

model2 = LinearRegression() 

model3 = Ridge()

model4 = Lasso()

models = [model1 , model2 , model3 , model4]





for i , model in enumerate(models):

    model.fit(X_train , y_train )

    y_pred_train = model.predict(X_train)

    y_pred_val = model.predict(X_val)

    print("Model{}'s RMSE for train:{}".format(i+1 , np.sqrt(mean_squared_error(y_train, y_pred_train))))

    print("Model{}'s RMSE for test:{}".format(i+1 , np.sqrt(mean_squared_error(y_val, y_pred_val))))

    print("\n____________________\n")
train_data
# we decide to use ridge_model , we shoud decide alpha 



accuracy_list = []

alpha_range= np.arange(10,1000)

for alpha in alpha_range:

    model = Ridge(alpha = alpha).fit(X_train, y_train)

    y_pred = model.predict(X_val) 

    accuracy_list.append(np.sqrt(mean_squared_error(y_pred, y_val)))

    

plt.plot(alpha_range , accuracy_list , label= "accuracy_test")

plt.xlabel("alpha")

plt.ylabel("RMSE")

plt.title("accuracy by alpha")

plt.legend()

alpha_reasonable = accuracy_list.index(min(accuracy_list))
# last model we use



Ridge_full_model = Ridge(alpha =alpha_reasonable).fit(X,y)
df_test[features].isnull().sum()  #there are missing data (´;ω;｀)
X_test = df_test[features].fillna(df_test[features].mean())

X_test = X_test.values
prediction = Ridge_full_model.predict(X_test)



output = pd.DataFrame({"Id":df_test.Id , "SalePrice":prediction})



output.to_csv("Submission_Group6.csv" , index =False)