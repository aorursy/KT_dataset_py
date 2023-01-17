#importing of basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#importing of the dataset as dataframe

ins_df=pd.read_csv("../input/insurance/insurance.csv")



ins_df.info()
ins_df.head(10)
ins_df.age.min(),ins_df.age.max()
#To change the age values in age column to a categorical values

l=[]

for i,j in ins_df.iteritems():

    if(i=='age'):

        for m in range(len(j)):

            if(j[m]<=35):

                l.append('Young Adult')

            elif((j[m]>=36)&(j[m]<=55)):

                l.append('Senior Adult')

            elif(j[m]>=56):

                l.append('Elder')    

            
#To check first few elements

l[0:10]
ins_df['age_cat']=l
ins_df.head(10)
#Heatmap showing correlation between the feature variables

sns.heatmap(ins_df.corr(),annot=True,cmap='YlGnBu')

plt.title("Correlaton between feature variables")
#To transform the categorical variables to dummy variables

#And we drop one column for each categorical dummy variable

in_df=pd.get_dummies(ins_df,drop_first=True)



#To drop the age column as we have considered it as categorical variable

in_df=in_df.drop(columns=['age'])

in_df.info()
in_df.head()
X=in_df.drop(columns=['charges']).values

y=in_df['charges'].values
#Fitting of linear regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)

reg=LinearRegression()

reg.fit(X_train,y_train)

y_predict=reg.predict(X_test)
# R-square value

reg.score(X_test,y_test)
#To find the coefficient and the intercept

coefficient=reg.coef_

intercept=reg.intercept_
print(f"The values of coefficient and intercept are {coefficient} and { intercept} .")
#Now using cross validation with n=5

from sklearn.model_selection import cross_val_score

reg=LinearRegression()

scores=cross_val_score(reg,X,y,cv=5)

np.mean(scores)
# Ridge Regression

# Hyperparameter tuning using gridsearch

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

param_grid={'alpha':np.arange(0.1,1,0.1)}

ridge=Ridge()

ridge_cv=GridSearchCV(ridge,param_grid,cv=5)

ridge_cv.fit(X,y)



#Best value of alpha parameter

ridge_cv.best_params_
#Best score value 

ridge_cv.best_score_
# Lasso Regression

# Hyperparameter tuning using gridsearch

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

param_grid={'alpha':np.arange(0.1,1,0.1)}

lasso=Ridge()

lasso_cv=GridSearchCV(lasso,param_grid,cv=5)

lasso_cv.fit(X,y)



#Best value of alpha parameter

lasso_cv.best_params_
lasso_cv.best_score_
#To check for alpha =0.7

#ridge regression

ridge=Ridge(alpha=0.7)

ridge.fit(X_train,y_train)

ridge.score(X_test,y_test)
#To find the coefficient and the intercept of ridge regression

coefficient=ridge.coef_

intercept=ridge.intercept_
print(f"The values of coefficient and intercept of ridge regression are {coefficient} and { intercept} .")
#To check for alpha=0.7

#lasso regression

lasso=Lasso(alpha=0.7)

lasso.fit(X_train,y_train)

lasso.score(X_test,y_test)
#To find the coefficient and the intercept of lasso regression

coefficient=lasso.coef_

intercept=lasso.intercept_
print(f"The values of coefficient and intercept for lasso regression are {coefficient} and { intercept} .")