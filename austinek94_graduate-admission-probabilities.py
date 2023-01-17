#Import the usual packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.describe()
df.info()

df.shape
df['gpa'] = df.CGPA / 2.5

df.head()
sns.boxplot(x=df['University Rating'],y=df['GRE Score'])

sns.swarmplot(x=df['University Rating'],y=df['GRE Score'],color='black',alpha=0.25)
sns.boxplot(x=df['University Rating'],y=df['gpa'])

sns.swarmplot(x=df['University Rating'],y=df['gpa'],color='black',alpha=0.25)
sns.boxplot(x=df['University Rating'],y=df['SOP'])

sns.swarmplot(x=df['University Rating'],y=df['SOP'],color='black',alpha=0.25)
sns.boxplot(x=df['University Rating'],y=df['LOR '])

sns.swarmplot(x=df['University Rating'],y=df['LOR '],color='black',alpha=0.25)
sns.boxplot(x=df['University Rating'],y=df['Chance of Admit '])

sns.swarmplot(x=df['University Rating'],y=df['Chance of Admit '],color='black',alpha=0.25)
sns.barplot(x=df['University Rating'],y=df['Research'])

sns.swarmplot(x=df['University Rating'],y=df['Research'],color='black',alpha=0.25)
df.columns
sns.heatmap(df[['GRE Score','University Rating','SOP','LOR ','gpa','Research','Chance of Admit ']].corr(),cmap='Blues')
#Import the usual data modeling packages

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression,Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



X = df.drop('Chance of Admit ',axis=1)

y = df['Chance of Admit '].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_sm = X = sm.add_constant(X)

model = sm.OLS(y,X_sm)

model.fit().summary()
lm = LinearRegression()

lm.fit(X_train,y_train)



cross_val_score(lm,X_train,y_train,scoring='neg_mean_absolute_error',cv=5)
lm_l = Lasso()

lm_l.fit(X_train,y_train)

cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=5)
rf = RandomForestRegressor()

cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=5)
parameters = {'n_estimators':range(10,300,10),'criterion':('mse','mae'),'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=5)

gs.fit(X_train,y_train)
print(gs.best_score_)

print(gs.best_estimator_)
pred_lm = lm.predict(X_test)

pred_lml = lm_l.predict(X_test)

pred_rf = gs.best_estimator_.predict(X_test)



print('MAE for Linear Regression:',mean_absolute_error(y_test,pred_lm))

print('MAE for Lasso Regression:',mean_absolute_error(y_test,pred_lml))

print('MAE for Random Forest Regressor:',mean_absolute_error(y_test,pred_rf))



#See if two models together perform better

print('MAE of Linear Regression combined with random Forest Regressor:',mean_absolute_error(y_test, (pred_lm+pred_rf)/2))