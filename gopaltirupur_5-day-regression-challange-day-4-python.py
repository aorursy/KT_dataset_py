import pandas as pd
import numpy as np
import scipy as sc
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline
%pylab inline
bmi_data = pd.read_csv("../input/eating-health-module-dataset//ehresp_2014.csv")
nyc_census = pd.read_csv("../input/new-york-city-census-data/nyc_census_tracts.csv")
bmi_data.head()
bmi_data.describe()
columnsRequired = ['erbmi','euexfreq','euwgt','euhgt','ertpreat']
bmi_data = bmi_data[columnsRequired]
bmi_data.head(5)
bmi_data.plot(kind='hist',subplots=True,figsize=(15,15),sharex=False)
bmi_data.plot(kind='scatter',x='erbmi',y='euwgt')
# There are some records with negative emi, can be removed
print(len(bmi_data))
bmi_data = bmi_data[bmi_data['erbmi']>0]
print(len(bmi_data))
bmi_data.plot(kind='scatter',x='erbmi',y='euwgt')
bmi_data.plot(kind='scatter',x='euexfreq',y='euwgt')
bmi_data.plot(kind='scatter',x='euhgt',y='euwgt')
bmi_data.plot(kind='scatter',x='ertpreat',y='euwgt')
sns.pairplot(bmi_data)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
columnsRequired = ['euexfreq','euwgt','euhgt','ertpreat']

y = bmi_data['erbmi']
x = bmi_data[columnsRequired]

print(y.head())
print(x.head())

reg.fit(x,y)
y_pred = reg.predict(x)
residual = y - y_pred
plt.scatter(y_pred,residual)
# MY TURN - PHASE 1
# TRYING TO PREDICT THE INCOME WITH USING OTHER TWO INDEPENDENT VARIABLES

census_block = pd.read_csv('../input/new-york-city-census-data/census_block_loc.csv')
display(census_block.head())
#UNDERSTANDING THE RELATIONSHIP BETWEEN  INCOME AND POVERTY
nyc_census.plot(kind = 'scatter',x='Income',y='Poverty')
#UNDERSTANDING THE RELATIONSHIP BETWEEN THE POPULATION AND THE MEN COUNT
nyc_census.plot(kind = 'scatter',x='TotalPop',y='Men')
#PAIR PLOT FOR UNDERSTANDING THE POSSIBLE RELATIONSHIP BETWEEN THE SELECTED VARIABLES
nyc_census_new = nyc_census.fillna(nyc_census.mean())
sns.pairplot(nyc_census_new[['Poverty','Income','IncomePerCap','Professional','SelfEmployed','White']])
#PAIR PLOT FOR UNDERSTANDING THE POSSIBLE RELATIONSHIP BETWEEN ALL THE VARIABLES
nyc_census_new = nyc_census.fillna(nyc_census.mean())
corr = nyc_census_new.corr()
corr.head()
f,ax = plt.subplots(figsize=(40,40))
sns.heatmap(corr,annot=True,fmt='g',cmap='viridis',square=True,ax=ax,xticklabels=1, yticklabels=1)
print('No. of columns :',len(nyc_census.columns))
nyc_corr_income = nyc_census[["Income","SelfEmployed","Employed","WorkAtHome","CensusTract","White","Citizen","Professional","Drive","Walk","OtherTransp","Unemployment","MeanCommute","Hispanic","Black","Poverty","ChildPoverty","Service","Office","Construction","Production","Transit"]]
print('No. of columns :',len(nyc_corr_income.columns))
corr = nyc_corr_income.corr()
corr.head()
f,ax = plt.subplots(figsize=(40,40))
sns.heatmap(corr,annot=True,fmt='g',cmap='viridis',square=True,ax=ax,xticklabels=1, yticklabels=1)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def drawModel(reg,x,y):  
    reg.fit(x,y)
    y_pred = reg.predict(x)
    residual = y - y_pred
    plt.scatter(y_pred,residual)       
# Now Let's Try to predict the Income using the selected Variables
# Removing ' CensusTract '
x = nyc_census_new[["SelfEmployed","Employed","WorkAtHome","White","Citizen","Professional","Drive","Walk","OtherTransp","Unemployment","MeanCommute","Hispanic","Black","Poverty","ChildPoverty","Service","Office","Construction","Production","Transit"]]
y=nyc_census_new['Income']
display(x.head())
drawModel(LinearRegression(),x,y)
#PAIR PLOT FOR UNDERSTANDING THE POSSIBLE RELATIONSHIP BETWEEN THE SELECTED VARIABLES
# SELECTING ONLY THE HIGHLY CORELATED VARIABLES FOR ANALYSIS
nyc_census_new = nyc_census.fillna(nyc_census.mean())
corr = nyc_census_new[['Poverty','Income','IncomePerCap','Professional','SelfEmployed','White']].corr()
corr.head()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True,fmt='g',cmap='viridis',square=True,ax=ax,xticklabels=1, yticklabels=1)
x = nyc_census_new[['Professional','Poverty','IncomePerCap']]
y=nyc_census_new['Income']
from sklearn.linear_model import LinearRegression
drawModel(LinearRegression(),x,y)
from sklearn.linear_model import ARDRegression
drawModel(ARDRegression(),x,y)
from sklearn.linear_model import HuberRegressor
drawModel(HuberRegressor(),x,y)
from sklearn.linear_model import LogisticRegressionCV
drawModel(LogisticRegressionCV(),x,y)
#WRONG ONE TO BE ANALYZED
from sklearn.linear_model import PassiveAggressiveRegressor
drawModel(PassiveAggressiveRegressor(),x,y)
#WRONG ONE TO BE ANALYZED
display(nyc_census_new.describe())
nyc_census_new.hist(figsize=(15,15))
nyc_census.head()
nyc_census.columns
# MY TURN - PHASE 2
my_columns=['Unemployment','Hispanic','White','Black','Native']
selected_nyc_censes = nyc_census[my_columns]
print(' Length of nyc_census : ',len(nyc_census))
selected_nyc_censes = selected_nyc_censes.dropna()
print(' Length of selected_nyc_censes : ',len(selected_nyc_censes))
selected_nyc_censes.head()