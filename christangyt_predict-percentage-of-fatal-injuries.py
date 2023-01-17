# import libraries 
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt 
data = pd.read_csv("/kaggle/input/aviation-accident-database-synopses/AviationData.csv",engine='python')  # engine for avoiding unicode error 
# pick relavant data 
data = data[['Investigation.Type','Aircraft.Damage','Aircraft.Category',
            'Number.of.Engines','Engine.Type','Purpose.of.Flight',
            'Total.Fatal.Injuries','Total.Serious.Injuries','Total.Minor.Injuries',
            'Total.Uninjured','Weather.Condition','Broad.Phase.of.Flight','Event.Date']]

data.head()
data.dtypes
data.shape
data[data['Total.Fatal.Injuries']==0]
data = data[data['Total.Fatal.Injuries'].notna()]
data
def check_null_values(data):
    print('=====Number of Missing Values for each Column=====')
    for col in data.columns:
        print(col,'=',data[col].isnull().sum())

check_null_values(data)
data['Total.Uninjured'].fillna(data['Total.Uninjured'].mean(), inplace=True)
data['Total.Minor.Injuries'].fillna(data['Total.Minor.Injuries'].mean(), inplace=True)
data['Total.Serious.Injuries'].fillna(data['Total.Serious.Injuries'].mean(), inplace=True)
data['Number.of.Engines'].fillna(data['Number.of.Engines'].mean(), inplace=True)
data['Aircraft.Category'].fillna('Unknown',inplace=True)

data['Engine.Type'].fillna('Others',inplace=True)
data['Engine.Type'].replace(['None','Unknown'],'Others')

data['Purpose.of.Flight'].fillna('Unknown',inplace=True)
data['Weather.Condition'].fillna('UNK',inplace=True)
data['Broad.Phase.of.Flight'].fillna('UNKNOWN',inplace=True)
# data['Investigation.Type'] = data['Investigation.Type'].astype('category').cat.codes
# data['Aircraft.Damage'] = data['Aircraft.Damage'].astype('category').cat.codes
# data['Aircraft.Category'] = data['Aircraft.Category'].astype('category').cat.codes
# data['Engine.Type'] = data['Engine.Type'].astype('category').cat.codes
# data['Purpose.of.Flight'] = data['Purpose.of.Flight'].astype('category').cat.codes
# data['Weather.Condition'] = data['Weather.Condition'].astype('category').cat.codes
# data['Broad.Phase.of.Flight'] = data['Broad.Phase.of.Flight'].astype('category').cat.codes
data['year'] = [int(i.split('-')[0]) for i in data['Event.Date']]
data['month'] = [int(i.split('-')[1]) for i in data['Event.Date']]
data['day'] = [int(i.split('-')[2]) for i in data['Event.Date']]
del data['Event.Date']
data  # cleaned data set 
check_null_values(data)
data['Total.Fatal.Injuries'].describe()
f,axes = plt.subplots(1,1,figsize=(20,10))
sb.boxplot(x=data['Total.Fatal.Injuries'], ax=axes)
data['ratio'] = data['Total.Fatal.Injuries']/(data['Total.Uninjured']+
                                             data['Total.Serious.Injuries']+
                                             data['Total.Minor.Injuries']+
                                             data['Total.Fatal.Injuries'])
data[data['ratio'].isnull()]
data = data[data['ratio'].notna()]
data['ratio'].describe()
f,axes = plt.subplots(1,1,figsize=(20,10))
sb.boxplot(x=data['ratio'], ax=axes)
data['Total.Uninjured'].describe()
f,axes = plt.subplots(1,1,figsize=(40,10))
sb.boxplot(x=data['Total.Uninjured'], ax=axes)
sb.jointplot(x='Total.Uninjured',y='Total.Fatal.Injuries',data=data)
data['Total.Serious.Injuries'].describe()
f,axes = plt.subplots(1,1,figsize=(40,10))
sb.boxplot(x='Total.Serious.Injuries',data=data, ax=axes)
sb.jointplot(x='Total.Serious.Injuries',y='Total.Fatal.Injuries',data=data)
data['Total.Minor.Injuries'].describe()
f,axes = plt.subplots(1,1,figsize=(40,10))
sb.boxplot(x='Total.Minor.Injuries',data=data, ax=axes)
sb.jointplot(x='Total.Minor.Injuries',y='Total.Fatal.Injuries',data=data)
data['Investigation.Type'].value_counts()
sb.catplot(x='Investigation.Type',y='Total.Fatal.Injuries',data=data)
data['Aircraft.Damage'].value_counts()
sb.catplot(x='Aircraft.Damage',y='Total.Fatal.Injuries',data=data)
data['Aircraft.Category'].value_counts()
f,axes = plt.subplots(1,1,figsize=(30,5))
sb.catplot(x='Aircraft.Category',y='Total.Fatal.Injuries',data=data,ax=axes)
data['Aircraft.Category'].value_counts()
f,axes = plt.subplots(1,1,figsize=(25,10))
sb.catplot(x='Engine.Type',y='Total.Fatal.Injuries',data=data,ax=axes)
data['Purpose.of.Flight'].value_counts()
f,axes = plt.subplots(1,1,figsize=(30,5))
sb.catplot(x='Purpose.of.Flight',y='Total.Fatal.Injuries',data=data,ax=axes)
data['Weather.Condition'].value_counts()
sb.catplot(x='Weather.Condition',y='Total.Fatal.Injuries',data=data)
data['Broad.Phase.of.Flight'].value_counts()
f,axes = plt.subplots(1,1,figsize=(30,5))
sb.catplot(x='Broad.Phase.of.Flight',y='Total.Fatal.Injuries',data=data,ax=axes)
sb.set(style="darkgrid")
plt.subplot(211)
g = sb.countplot(x="year", palette="GnBu_d", data=data,order=np.arange(1982,2020))
a = plt.setp(g.get_xticklabels(), rotation=90)
data.corr()
def plot_correlation_map( df ):
    corr = df.corr()
    f,axes = plt.subplots(figsize=(12,10))
    g = sb.heatmap(corr,annot=True,ax=axes)
    bottom, top = g.get_ylim()
    g.set_ylim(bottom+0.5,top-0.5)    # fix annotation not centered 

    
plot_correlation_map(data)

# integer encoding 
data['Investigation.Type'] = data['Investigation.Type'].astype('category').cat.codes
data['Aircraft.Damage'] = data['Aircraft.Damage'].astype('category').cat.codes
data['Aircraft.Category'] = data['Aircraft.Category'].astype('category').cat.codes
data['Engine.Type'] = data['Engine.Type'].astype('category').cat.codes
data['Purpose.of.Flight'] = data['Purpose.of.Flight'].astype('category').cat.codes
data['Weather.Condition'] = data['Weather.Condition'].astype('category').cat.codes
data['Broad.Phase.of.Flight'] = data['Broad.Phase.of.Flight'].astype('category').cat.codes
# prepare pre
X = data[['Investigation.Type','Aircraft.Damage','Aircraft.Category',
            'Number.of.Engines','Engine.Type','Purpose.of.Flight',
            'Total.Serious.Injuries','Total.Minor.Injuries',
            'Total.Uninjured','Weather.Condition','Broad.Phase.of.Flight',
            'year','month','day']]

y = data['ratio']
# split data 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
# Import LinearRegression model from Scikit-Learn
from sklearn.linear_model import LinearRegression


# Linear Regression using Train Data
linreg = LinearRegression()         # create the linear regression object
linreg.fit(X_train, y_train)        # train the linear regression model

# Coefficients of the Linear Regression line
print('Intercept of Regression \t: b = ', linreg.intercept_)
print('Coefficients of Regression \t: a = ', linreg.coef_)
print()
# Print the Coefficients against Predictors
pd.DataFrame(list(zip(X_train.columns, linreg.coef_)), columns = ["Predictors", "Coefficients"])
# Predict the Total values from Predictors
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# Plot the Predictions vs the True values
f, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].scatter(y_train, y_train_pred, color = "blue")
axes[0].plot(y_train, y_train, 'r-', linewidth = 1)
axes[0].set_xlabel("True values of the Response Variable (Train)")
axes[0].set_ylabel("Predicted values of the Response Variable (Train)")
axes[1].scatter(y_test, y_test_pred, color = "green")
axes[1].plot(y_test, y_test, 'r-', linewidth = 1)
axes[1].set_xlabel("True values of the Response Variable (Test)")
axes[1].set_ylabel("Predicted values of the Response Variable (Test)")
plt.show()
from math import sqrt
def evaluate(predicted,actual):
    size = actual.size
    mse = ((predicted-actual)**2).sum()/size
    print('MSE =',mse)
    rmse = sqrt(mse)
    print('RMSE =',rmse)
    mae = abs(predicted-actual).sum()/size
    print('MAE =',mae)
    var = ((actual-np.mean(actual))**2).sum()/size
    R2 = 1-mse/var
    print('R^2 =',R2)

print('Train Set')
evaluate(linreg.predict(X_train),y_train)
print('Test Set')
evaluate(linreg.predict(X_test),y_test)
# base model 
from sklearn.ensemble import RandomForestRegressor

base = RandomForestRegressor(n_estimators=100)
base.fit(X_train,y_train)
print('Train Set')
evaluate(base.predict(X_train),y_train)
print('Test Set')
evaluate(base.predict(X_test),y_test)
# Randomized Search to find optimal parameters 
from sklearn.model_selection import RandomizedSearchCV


rf = RandomForestRegressor()

param_grid = {
    'min_samples_split':[2,5,7],
    'max_depth':[5,10,15,20],
    'max_features':['auto','sqrt','log2'],
    'min_samples_leaf': [2, 3, 4],
    'n_estimators': [100, 500, 1000, 1500]
}

search = RandomizedSearchCV(estimator=rf,param_distributions=param_grid,cv=3,verbose=2,n_jobs = -1)
search.fit(X_train, y_train)
search.best_params_
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1500, bootstrap=True,
                           max_features='auto', min_samples_split=5,
                           min_samples_leaf=2, max_depth=15)
# training 
rf.fit(X_train,y_train)
print('Train Set')
evaluate(rf.predict(X_train),y_train)
print('Test Set')
evaluate(rf.predict(X_test),y_test)
from sklearn.tree import export_graphviz
import graphviz
import os 

treedot = export_graphviz(rf.estimators_[5],                                      # the model
                          feature_names = X_train.columns,          # the features 
                          filled = True,                                # node colors
                          rounded = False,                               # make pretty
                          special_characters = True)                    # postscript

graphviz.Source(treedot)
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train.columns, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances];
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, X_train.columns, rotation='vertical')
plt.ylabel('Importance'); 
plt.xlabel('Variable'); 
plt.title('Variable Importances');