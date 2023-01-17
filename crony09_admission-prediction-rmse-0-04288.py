import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn
df=pd.read_csv('../input/admission/Admission_Predict_Ver1.1.csv')

df.head()
# Checking for Null Values

df.drop('Serial No.',axis=1,inplace=True)

df.isnull().sum()
df=df.rename(columns={'GRE Score':'gre_score'})

df=df.rename(columns={'TOEFL Score':'toefl_score'})

df=df.rename(columns={'University Rating':'uni_rating'})

df=df.rename(columns={'Chance of Admit ':'chance'})

df=df.rename(columns={'LOR ':'LOR'})

df.describe()

#By looking at the minimum and maximum values for each feature we can conclude that there are no invalid entries.
sns.jointplot(x='gre_score',y='chance',data=df,kind='reg')

# we can see a direct and linear relationship between GRE Scores and Chances of Admission
sns.jointplot(x='toefl_score',y='chance',data=df,kind='reg')

# we observe a similar relationship for TOEFL Scores
sns.jointplot(x='uni_rating',y='chance',data=df,kind='scatter')

# this plot is kind of difficult to interpret, hence we try a different approach
#computing the average chances for each university rating category

for i in set(df.uni_rating):

    plt.bar(i,df[df.uni_rating==i].chance.mean()*100)

plt.ylabel('% chance of admission')

plt.xlabel('University Rating (Out of 5)')

ax=plt.gca()

ax.set_ylim([40,100])

#We can say that the chance of admission is higher with higher university rating.
for i in set(df.SOP):

    plt.bar(i,df[df.SOP==i].chance.mean()*100,width=0.45)

plt.ylabel('% chance of admission')

plt.xlabel('SOP Rating (Out of 5)')

ax=plt.gca()

ax.set_ylim([40,100])

#We can say that the chance of admission is higher with higher SOP rating.
for i in set(df.LOR):

    plt.bar(i,df[df.LOR==i].chance.mean()*100,width=0.45)

plt.ylabel('% chance of admission')

plt.xlabel('LOR Rating (Out of 5)')

ax=plt.gca()

ax.set_ylim([40,100])

#We can say that the chance of admission is higher with higher LOR rating.
sns.jointplot(x='CGPA',y='chance',data=df,kind='scatter')

# we can see a direct and linear relationship between CGPA Scores and Chances of Admission
for i in set(df.Research):

    plt.bar(i,df[df.Research==i].chance.mean()*100)

plt.ylabel('% chance of admission')

plt.xlabel('LOR Rating (Out of 5)')

ax=plt.gca()

ax.set_ylim([50,100])
df.head(7)
#the GRE score rating starts from 260 (base score), so to get a more accurate measure of a student's performance we transform it

df['gre_score']=df['gre_score']-260

df.head(3)
#making the GRE score within the range of 0 to 5

df['gre_score']=df['gre_score']*5/80

df.head(4)
#we take the minimum TOEFL score to be 60

df['toefl_score']=df['toefl_score']-60

df.head(3)
df['toefl_score']=df['toefl_score']*5/60

df.head(3)
df['CGPA']=df['CGPA']/2

df.head(3)
df.describe()
X=df.drop('chance',axis=1)

y=df['chance']
X_train=X.loc[0:399,:]

X_test=X.loc[400:,:]

y_train=y.loc[0:399]

y_test=y.loc[400:]



print('X Train Data Count:',X_train.shape[0])

print('X Test Data Count:',X_test.shape[0])

print('y Train Data Count:',y_train.shape[0])

print('y Test Data Count:',y_test.shape[0])
from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from math import sqrt
reg=linear_model.LinearRegression()

reg.fit(X_train,y_train)

predictions=reg.predict(X_test)

print('For Linear Regression model, RMSE=',sqrt(mean_squared_error(predictions,y_test)))

plt.scatter(predictions,y_test)

plt.plot([0,1],[0,1],color='red')

ax=plt.gca()

ax.set_ylim([0.35,1])

ax.set_xlim([0.35,1])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Actual vs Predicted (For Test Data)')

# Closer the points to the red line, better the prediction of the model
from sklearn.ensemble import RandomForestRegressor

rf_random=RandomForestRegressor()

rf_random.fit(X_train,y_train)

predictions=rf_random.predict(X_test)

print('For RandomForest Regression model, RMSE=',sqrt(mean_squared_error(predictions,y_test)))

plt.scatter(predictions,y_test)

plt.plot([0,1],[0,1],color='red')

ax=plt.gca()

ax.set_ylim([0.35,1])

ax.set_xlim([0.35,1])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Actual vs Predicted (For Test Data)')
#We observe that the linear regression model is better. But, we can try to improve the performance using hyperparameter tuning.

estimators=[]

for i in range(1,70,6):

    estimators.append(i)

for i in estimators:

    rf=RandomForestRegressor(n_estimators=i)

    rf.fit(X_train,y_train)

    print('For RandomForest Regression model with',i,'estimators,','RMSE=',sqrt(mean_squared_error(rf.predict(X_test),y_test)))
from sklearn.linear_model import Ridge

ridge=Ridge()

ridge.fit(X_train,y_train)

predictions=ridge.predict(X_test)

print('For Ridge Regression model, RMSE=',sqrt(mean_squared_error(predictions,y_test)))

plt.scatter(predictions,y_test)

plt.plot([0,1],[0,1],color='red')

ax=plt.gca()

ax.set_ylim([0.35,1])

ax.set_xlim([0.35,1])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Actual vs Predicted (For Test Data)')
from sklearn.model_selection import GridSearchCV

ridge=Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,35,40,45,50,55]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X_train,y_train)

predictions=ridge_regressor.predict(X_test)

print('For Ridge Regression model, RMSE=',sqrt(mean_squared_error(predictions,y_test)))

print('Best Alpha:',ridge_regressor.best_params_)

plt.scatter(predictions,y_test)

plt.plot([0,1],[0,1],color='red')

ax=plt.gca()

ax.set_ylim([0.35,1])

ax.set_xlim([0.35,1])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Actual vs Predicted (For Test Data)')
from sklearn.linear_model import Lasso

lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,35,40,45,50,55]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)

predictions=lasso_regressor.predict(X_test)

print('For Lasso Regression model, RMSE=',sqrt(mean_squared_error(predictions,y_test)))

print('Best Alpha:',lasso_regressor.best_params_)

plt.scatter(predictions,y_test)

plt.plot([0,1],[0,1],color='red')

ax=plt.gca()

ax.set_ylim([0.35,1])

ax.set_xlim([0.35,1])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.title('Actual vs Predicted (For Test Data)')
results=pd.DataFrame(columns=['Actual Chance'],data=y_test)

results['Actual Chance']=y_test

results['Predicted Chance']=predictions

results.reset_index(inplace=True,drop=True)

results.head(4)
rows,columns=results.shape

for i in range(0,rows):

    results.iloc[i,1]=float(format(results.iloc[i,1],'.2f'))

results.head(4)
print('RMSE(Test):',sqrt(mean_squared_error(results['Actual Chance'],results['Predicted Chance'])))