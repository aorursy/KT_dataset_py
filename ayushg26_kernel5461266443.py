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

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams.update({'font.size': 12})

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm

from statsmodels.formula.api import ols

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score  

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.feature_selection import RFE





df = pd.read_csv("../input/train1/train.csv")

df.head()



print('Product category 2')



# check

print('Pre-Clean')

print(df.Product_Category_2.isnull().sum())



# median

vMedian = int(df['Product_Category_2'].median())

print('Median')

print(vMedian)



# update

df['Product_Category_2'] = np.where(df['Product_Category_2'].isnull(), vMedian, df['Product_Category_2'])



# recheck

print('PostClean')

print(df.Product_Category_2.isnull().sum())

print('')



# median

vMedian = int(df['Product_Category_2'].median())

print('median')

print(vMedian)



#missing values in product_Category_3

print(df.Product_Category_3.isnull().sum())



print(df)



s = 383247/550067

print (s)



#missing values >30%

df.drop(['Product_Category_3'],1,inplace=True)

df.head()





#label encoding to convert categorical columns into numerical columns

le = LabelEncoder()

le.fit(df.Gender.drop_duplicates()) 

df.Gender = le.transform(df.Gender)

le.fit(df.User_ID.drop_duplicates())

df.User_ID = le.transform(df.User_ID)

le.fit(df.Product_ID.drop_duplicates())

df.Product_ID = le.transform(df.Product_ID)

print(df.Age)

le.fit(df.Age.drop_duplicates()) 

df.Age = le.transform(df.Age)

le.fit(df.City_Category.drop_duplicates()) 

df.City_Category = le.transform(df.City_Category)

le.fit(df.Stay_In_Current_City_Years.drop_duplicates()) 

df.Stay_In_Current_City_Years = le.transform(df.Stay_In_Current_City_Years)





#applying decision tree

from sklearn.model_selection import train_test_split as tts, cross_val_score, GridSearchCV

y=df["Purchase"]

x=df.drop(["Purchase"],axis=1)

x_train, x_test, y_train, y_test = tts(x,y,random_state = 42, test_size = 0.50)

dtr=DecisionTreeRegressor(criterion='mse',max_depth=17,max_leaf_nodes=1000,min_samples_leaf=7, random_state=24, min_samples_split=195)

dtr.fit(x_train,y_train)

y_pred=dtr.predict(x_test)

r2_score(y_test,y_pred)

#Applying xgboost

from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.99,learning_rate = 0.1,

                max_depth =14,alpha = 12, n_estimators = 1000)

xgb.fit(x_train,y_train)

xgb_preds = xgb.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))

print(rmse)
r2_score(y_test,xgb_preds)
#estimating values of purchase

df2 = pd.read_csv("../input/test01/test.csv")
#cleaning data

print('Product category 2')





# check

print('Pre-Clean')

print(df2.Product_Category_2.isnull().sum())



# mean

vMedian = int(df2['Product_Category_2'].median())

print('Median')

print(vMedian)



# update

df2['Product_Category_2'] = np.where(df2['Product_Category_2'].isnull(), vMedian, df2['Product_Category_2'])



# recheck

print('PostClean')

print(df2.Product_Category_2.isnull().sum())

print('')



# mean

vMedian = int(df2['Product_Category_2'].median())

print('median')

print(vMedian)







#missing values in product_category_3

print(df2.Product_Category_3.isnull().sum())
s = 162562/550067

print (s)

df2.drop(['Product_Category_3'],1,inplace=True)

df2.head()



#label encoding categorical columns

le.fit(df2.Gender.drop_duplicates()) 

df2.Gender = le.transform(df2.Gender)

le.fit(df2.User_ID.drop_duplicates())

df2.User_ID = le.transform(df2.User_ID)

le.fit(df2.Product_ID.drop_duplicates())

df2.Product_ID = le.transform(df2.Product_ID)

print(df2.Age)

le.fit(df2.Age.drop_duplicates()) 

df2.Age = le.transform(df2.Age)

le.fit(df2.City_Category.drop_duplicates()) 

df2.City_Category = le.transform(df2.City_Category)

le.fit(df2.Stay_In_Current_City_Years.drop_duplicates()) 

df2.Stay_In_Current_City_Years = le.transform(df2.Stay_In_Current_City_Years)



#predicting purchase values

a_pred = xgb.predict(df2)
print(a_pred)
#writing predicted values into csv file

df2['Purchase'] = a_pred

df2.to_csv('output.csv',index=False)
#implementing linear models

import pandas as pd

import numpy as np

df1 = pd.read_csv("../input/train1/train.csv")

df1.head()



df1.drop(['User_ID'],1,inplace=True)

df1.drop(['Product_ID'],1,inplace=True)

df1.head()



print('Product category 2')



# check

print('Pre-Clean')

print(df1.Product_Category_2.isnull().sum())



# median

vMedian = int(df1['Product_Category_2'].median())

print('Median')

print(vMedian)



# update

df1['Product_Category_2'] = np.where(df1['Product_Category_2'].isnull(), vMedian, df1['Product_Category_2'])



# recheck

print('PostClean')

print(df1.Product_Category_2.isnull().sum())

print('')



# median

vMedian = int(df1['Product_Category_2'].median())

print('median')

print(vMedian)



#finding % of null values in product_category_3

print(df1.Product_Category_3.isnull().sum())



print(df1)



s = 383247/550067

print (s)



df1.drop(['Product_Category_3'],1,inplace=True)

df1.head()



#one hot encoding

df1.Stay_In_Current_City_Years=np.where(df1.Stay_In_Current_City_Years=="4+",4,df1.Stay_In_Current_City_Years)

df1.Occupation=df1.Occupation.astype(str)

df1.Stay_In_Current_City_Years=df1.Stay_In_Current_City_Years.astype(int)

df1.Product_Category_1=df1.Product_Category_1.astype(str)

df1.Product_Category_2=df1.Product_Category_2.astype(int)

df1.Product_Category_2=df1.Product_Category_2.astype(str)

df1.Marital_Status=df1.Marital_Status.astype(str)

df1.info()



df1 = pd.get_dummies(df1)

df1.info()



X=df1.drop(["Purchase"],axis=1)

y=df1["Purchase"]

print(X)

print(y) 



# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, 

                                                    random_state=1) 

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score  

# create linear regression object 

reg = linear_model.LinearRegression() 



# train the model using the training sets 

reg.fit(X_train, y_train) 

reg.score(X_train, y_train)

y_pred = reg.predict(X_test)

print(r2_score(y_test,y_pred))





import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.model_selection import GridSearchCV, cross_val_score



#predicting r2 score using lasso

lasso = Lasso()

lasso.fit(X_train,y_train)

lasso_pred=lasso.predict(X_test)

print(r2_score(y_test,lasso_pred))



#predicting r2 score using ridge

ridge = Ridge()

ridge.fit(X_train,y_train)

ridge_pred=ridge.predict(X_test)

print(r2_score(y_test,ridge_pred))



#predicting r2 score using elastic net and using tuned hyper parameters got from the next line of codes written from line 112

elastic = ElasticNet(alpha=0.01, copy_X=True, fit_intercept=True, l1_ratio=0.5,

           max_iter=1000, normalize=False, positive=False, precompute=False,

           random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

elastic.fit(X_train,y_train)

elastic_pred=elastic.predict(X_test)

print(r2_score(y_test,elastic_pred))



#hyper parameter tuning for elastic net

#params= {"alpha":np.arange(0.01, 0.05, 0.001)}

#elastic_cv = GridSearchCV(estimator=elastic, param_grid=params, cv = 5)

#elastic_cv.fit(X_train,y_train)

#elastic_cv.best_estimator_

#elastic_cv.best_score_

#elastic_cv.best_estimator_.coef_



#finding feature ranking

#ranking = elastic_cv.best_estimator_.coef_.tolist()

#ranking = [abs(t) for t in ranking]

#features = list(X)

#d = dict(zip(features, ranking))

#d = pd.DataFrame(list(d.items()), columns = ["Features", "Ranking"])

#d.sort_values(["Ranking"], ascending=False)



#hyper parameter tuning for ridge

#ridge.get_params()

#params1={"alpha":np.arange(0.01, 0.05, 0.001)}

#ridge_cv = GridSearchCV(estimator=ridge, param_grid=params1, cv = 5)

#ridge_cv.fit(X_train,y_train)

#ridge_cv.best_estimator_

#ridge_cv.best_score_

#ridge_cv.best_estimator_.coef_



#finding r2_score using ridge and by using hyper parameters

ridge=Ridge(alpha=1.0, copy_X=True, fit_intercept=True,

                             max_iter=None, normalize=False, random_state=None,

                             solver='auto', tol=0.001)

ridge.fit(X_train,y_train)

ridge_pred=ridge.predict(X_test)

print(r2_score(y_test,ridge_pred))
