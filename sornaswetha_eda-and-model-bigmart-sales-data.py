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
train_data = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')

test_data = pd.read_csv('/kaggle/input/bigmart-sales-data/Test.csv')
print('\nShape of training data :',train_data.shape)

print('\nShape of testing data :',test_data.shape)
train_data["source"]="train"

test_data["source"]="test"

data = pd.concat([train_data,test_data], sort= True)
print (train_data.shape, test_data.shape, data.shape , sep = "\n")
data.head()
data.tail()
data.describe()
data.info()
cat_col = data.select_dtypes(include="object")



for c in cat_col:

    if c not in( 'Item_Identifier','Outlet_Identifier','source'):

        print( "\n Feature:",c)

        print(data[c].value_counts())
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(train_data['Item_Outlet_Sales'])

plt.show()



print('Skewness: %f' % train_data['Item_Outlet_Sales'].skew())

print('Kurtsis: %f' %train_data['Item_Outlet_Sales'].kurt())
train_data['Item_Weight'].hist(bins = 100);

plt.show()
train_data['Item_Visibility'].hist(bins = 100);

plt.show()
train_data['Item_MRP'].hist(bins = 100);

plt.show()
sns.catplot(x= "Item_Type", data= train_data, kind = "count", aspect=4)

plt.show()
sns.catplot(x= "Outlet_Size", data= train_data, kind = "count")

plt.show()
sns.catplot(x= "Item_Fat_Content", data= train_data, kind = "count")

plt.show()
sns.catplot(x= "Outlet_Location_Type", data= train_data, kind = "count")

plt.show()
sns.catplot(x= "Outlet_Type", data= train_data, kind = "count",aspect=2)

plt.show()
data["Outlet_Establishment_Year"]=data["Outlet_Establishment_Year"].astype("category")

sns.catplot(x= "Outlet_Establishment_Year", data= train_data, kind = "count", aspect=4)

plt.show()
sns.scatterplot(x = "Item_Weight" , y ="Item_Outlet_Sales" , data = train_data, alpha = 0.3, color = "r")

plt.show()
sns.scatterplot(x = "Item_Visibility" , y ="Item_Outlet_Sales" , data = train_data, alpha = 0.3, color = "y")

plt.show()
sns.scatterplot(x = "Item_MRP" , y ="Item_Outlet_Sales" , data = train_data, alpha = 0.3, color = "g")

plt.show()
sns.catplot(x= "Outlet_Size", y = "Item_Outlet_Sales" ,data= train_data, kind = "box")

plt.show()
sns.catplot(x= "Outlet_Establishment_Year", y = "Item_Outlet_Sales" ,data= train_data, kind = "box", aspect=4)

plt.show()
sns.catplot(x= "Outlet_Type", y = "Item_Outlet_Sales" ,data= train_data, kind = "box", aspect=3)

plt.show()
sns.catplot(x= "Item_Fat_Content", y = "Item_Outlet_Sales" ,data= train_data, kind = "box")

plt.show
sns.catplot(x= "Item_Type", y = "Item_Outlet_Sales" ,data= train_data, kind = "violin", aspect=3)

plt.show()
sns.catplot(x= "Outlet_Type", data= train_data, kind = "count", aspect = 2, hue="Outlet_Size")

plt.show()
sns.catplot(x= "Outlet_Type", data= train_data, kind = "count", aspect = 2, hue="Outlet_Location_Type")

plt.show()
data.isnull().sum()
item_mean_weight = data.pivot_table( index = "Item_Identifier" , values = "Item_Weight",aggfunc='mean')



print("Missing Item_Weight : " , data['Item_Weight'].isnull().sum()  )

data.loc[data['Item_Weight'].isnull(), "Item_Weight"] =   data.loc[data['Item_Weight'].isnull(), "Item_Identifier"]. apply ( lambda x : item_mean_weight.loc[x])

print("Missing Item_Weight : " , data['Item_Weight'].isnull().sum()  )
from scipy.stats import mode



outlet_mode= data.pivot_table( index = "Outlet_Type" , values = "Outlet_Size",aggfunc= lambda x : mode(x).mode[0]  )



print("Missing Outlet_Size : " , data['Outlet_Size'].isnull().sum()  )

data.loc[data['Outlet_Size'].isnull(), "Outlet_Size"] =   data.loc[data['Outlet_Size'].isnull(), "Outlet_Type"]. apply ( lambda x : outlet_mode.loc[x])

print("Missing Outlet_Size : " , data['Outlet_Size'].isnull().sum()  )
data.isnull().sum()
item_mean_visibility = data.pivot_table( index = "Item_Identifier" , values = "Item_Visibility",aggfunc='mean')
print("Rows with '''0''' visbility " , (data['Item_Visibility'] == 0).sum()  )

data.loc[data['Item_Visibility'] == 0, "Item_Visibility"] =   data.loc[data['Item_Visibility'] == 0, "Item_Identifier"]. apply ( lambda x : item_mean_visibility.loc[x])

print("Rows with '''0''' visbility " , (data['Item_Visibility'] == 0).sum()  )
data["Item_combined"] = data ["Item_Identifier"].apply( lambda  x : x[0:2])

data["Item_combined"]  = data["Item_combined"]. map ( { "FD" : "Food", "DR" : "Drink",  "NC" : "Non_Edible"} )
print(data["Item_Fat_Content"].unique())

data["Item_Fat_Content"].replace ( {"low fat": "Low Fat" , "LF": "Low Fat", "reg": "Regular"} , inplace = True)

print(data["Item_Fat_Content"].unique())
data.loc[data["Item_combined"] == "Non_Edible" , "Item_Fat_Content" ] = "Non_Edible"

print(data["Item_Fat_Content"].value_counts())
data.pivot_table(values= "Item_Outlet_Sales", index = "Outlet_Type",aggfunc='mean' )
data["year"] = data["Outlet_Establishment_Year"].apply( lambda x : 2013-x )

data["year"]=data["year"].astype("int8")

data["year"].describe()
data["Outlet_Identifier"].unique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(data["Outlet_Identifier"])

data["Outlet"]= le.transform(data["Outlet_Identifier"])

data["Outlet"] = data["Outlet"].astype("category")

data["Outlet"].unique()
cor = data.corr()

sns.heatmap(cor,cmap="bone" )

plt.show()
sns.pairplot(data[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'year', 'Item_Outlet_Sales']] )

plt.show()
data.dtypes
data= pd.get_dummies (data = data , columns = ['Item_Fat_Content','Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_combined',  'Outlet'] , drop_first = True)
data.dtypes
data.head(5)
data.drop ( ["Item_Type","Outlet_Establishment_Year"] , inplace = True , axis = 1)
train = data.loc [data["source"] == "train"]

test = data.loc [data["source"] == "test"]



#Drop unnecessary columns:

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

train.drop(['source'],axis=1,inplace=True)



#Export files as modified versions:

train.to_csv("train_modified.csv",index=False)

test.to_csv("test_modified.csv",index=False)
import pandas as pd

import numpy as np

import warnings 

warnings.filterwarnings('ignore')



train = pd.read_csv('train_modified.csv')

test= pd.read_csv('test_modified.csv')



print('\nShape of training data :',train.shape)

print('\nShape of testing data :',test.shape)
base_model = test[["Item_Identifier", "Outlet_Identifier"]]

base_model["Item_Outlet_Sales"] =  train["Item_Outlet_Sales"].median()

base_model.to_csv("base_model.csv",index=False)
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression , Lasso ,Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_validate,cross_val_score

from sklearn.metrics import mean_squared_error

from math import sqrt

#Function 

def models(algorithm,X_val,y_val,X_train,y_train ,file_name,X_test):

    model = algorithm

    model.fit(X_train,y_train)

    

    ytrain_pred = model.predict(X_train)

    yval_pred = model.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, ytrain_pred) )

    rmse_val = np.sqrt(mean_squared_error(y_val, yval_pred) )

    

    scores_train = model.score(X_train,y_train)

    scores_val = model.score(X_val,y_val)

    

    accuracy = cross_val_score(estimator=model, X=X_train, y=y_train,cv=10)

#     print(f"The accuracy of the Polynomial Regression Model is \t {accuracy.mean()}")

#     print(f"The deviation in the accuracy is \t {accuracy.std()}")





    score.loc[file_name] = [ scores_train ,  scores_val,rmse_train, rmse_val,accuracy.mean(),accuracy.std()  ]

    

    #submission

    submission = test[["Item_Identifier", "Outlet_Identifier"]]

    submission["Item_Outlet_Sales"] =  model.predict(X_test)

    file_name = file_name + ".csv"

    submission.to_csv(file_name,index=False)


score = pd.DataFrame ( columns = ["Train_Score", "Validate_Score", "Train_RMSE","Validate_RMSE", "Accuuracy_Mean", "Accuracy_Std"])

y_train = train ["Item_Outlet_Sales"]



X= train.drop(["Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"] , axis = 1)

y= train["Item_Outlet_Sales"]



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2,random_state = 0)



X_test= test.drop(["Item_Identifier","Outlet_Identifier"] , axis = 1)



models (LinearRegression() , X_val,y_val, X_train,y_train ,"LinearRegression",X_test)

models (Lasso() , X_val,y_val, X_train,y_train ,"Lasso",X_test)

models (Ridge() , X_val,y_val, X_train,y_train ,"Ridge",X_test)

models(DecisionTreeRegressor(max_depth=15, min_samples_leaf=100) ,  X_val,y_val, X_train,y_train ,"DecisionTreeRegressor",X_test)

models(DecisionTreeRegressor(max_depth=8, min_samples_leaf=150) ,  X_val,y_val, X_train,y_train ,"DecisionTreeRegressor2",X_test)

alg_RFR = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4) 

models (alg_RFR , X_val,y_val, X_train,y_train ,"RandomForestRegressor",X_test)

alg_RFR2 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)

models (alg_RFR2 , X_val,y_val, X_train,y_train ,"RandomForestRegressor2",X_test)



score
from sklearn import ensemble



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)

models (clf , X_val,y_val, X_train,y_train ,"GradientBoostingRegressor",X_test)





params = {'n_estimators': 750, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.00999, 'loss': 'ls', 'criterion':'mse', 'random_state' : 1}

clf = ensemble.GradientBoostingRegressor(**params )

models (clf , X_val,y_val, X_train,y_train ,"GradientBoostingRegressor2",X_test)

score



from sklearn.linear_model import ElasticNetCV, ElasticNet



cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 

                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 

                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')

cv_model.fit(X_train, y_train)



e_net = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)



models (e_net , X_val,y_val, X_train,y_train ,"ElasticNet",X_test)

score