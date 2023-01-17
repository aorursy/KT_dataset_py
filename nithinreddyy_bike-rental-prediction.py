import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import chi2_contingency

import os

import statistics

from sklearn.metrics import r2_score

from scipy import stats

from sklearn.model_selection import train_test_split,RandomizedSearchCV



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve,auc,roc_auc_score



%matplotlib inline

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')
#Setting working directory

os.chdir("C:/Users/Click/Desktop/Bike rental")

print(os.getcwd())
#Loading Dataset

data = pd.read_csv('Bike_Rental.csv')
data = pd.DataFrame(data)
#Creating Duplicate instances of data for Preprocessing and exploration

df = data.copy()
data.head(5)
#Checking info of data -> data types and rows n cols

data.info()



#This shows that we have no Missing Values for any column.
data.describe()
#calculating number of unique values for all df columns

data.nunique()
data.columns
##We know that 'cnt' which is our target variable is sum of two other variables - 'registered' and 'casusal'. 

#'instant' variable is of no use and can be dropped

#'dteday' variable is a date column which is not significant in our analysis and can be excluded

#So we will drop these variables now itself

drop1 = ['casual', 'registered', 'instant', 'dteday']

data = data.drop(drop1, axis = 1)
# Variables are " Continuos" and "Categorical"

con = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']



cat = ['season','yr','mnth',

                     'holiday','weekday', 'workingday', 'weathersit']
#Target Variable probability data distribution

plt.figure(figsize=(8,6))

plt.hist(data['cnt'], normed=True, bins=30)

plt.ylabel('Probability', fontsize= 15)

plt.xlabel('Number of Users', fontsize= 15)

plt.savefig("Count of Users.png")

plt.title("Bike Rental Statistics",fontsize= 20)
#Function to view the categories present in each categorical feature and thier values

def view_feature_cat(obj):

    for i in range(len(obj)):

        print('*******************************************')

        print('Feature:',obj[i])

        print('-----------------------')

        print(data[str(obj[i])].value_counts())

        print('*******************************************')
view_feature_cat(cat)
sns.catplot(x="weekday", y="cnt", data=data)

plt.savefig('days_bikecnt.png')



sns.catplot(x="mnth", y="cnt", data=data)

plt.savefig('mnth_bikecnt.png')



sns.catplot(x="season", y="cnt", data=data)

plt.savefig('season_bikecnt.png')



sns.catplot(x="weathersit", y="cnt", data=data)

plt.savefig('hol_bikecnt.png')
# Checking the distribution of values for variables in data

for i in con:

    if i == 'cnt':

        continue

    sns.distplot(data[i],bins = 'auto')

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Density")

    plt.savefig('{i}_Vs_Density.png'.format(i=i))

    plt.show()
"""def box_plot(x):

    plt.boxplot(data[x])

    plt.xlabel(x,fontsize= 15)

    plt.ylabel('Values',fontsize= 15)

    plt.xticks(fontsize=10, rotation=90)

    plt.yticks(fontsize=10)

    plt.title("Boxplot for {X}".format(X=x),fontsize = 20)

    plt.savefig("Boxplot for {X}.png".format(X=x))

    plt.show()

    box_plot('windspeed')

    box_plot('temp')

    box_plot('atemp')

    box_plot('hum')"""
box=plt.boxplot([data['temp'], data['atemp'], data['hum'], data['windspeed']],patch_artist=True)

plt.xlabel(['1. Temperature', '2. Feeling Temperature', '3. Humidity', '4. Windspeed'])

plt.title("BoxPlot of the Variables for Weather Conditions")

colors = ['cyan', 'lightblue', 'lightgreen', 'tan']

for patch, color in zip(box['boxes'], colors):

    patch.set_facecolor(color)

plt.ylabel('Values')

plt.savefig('BoxPlot of the Variables for Weather Conditions')
box2=plt.boxplot([data['cnt']],patch_artist=True)

plt.xlabel(['1. Total Count'])

plt.title("BoxPlot of the Variables for user count")

colors = ['red']

for patch, color in zip(box2['boxes'], colors):

    patch.set_facecolor(color)

plt.ylabel('Values')

plt.savefig('BoxPlot of the Variables for user count')
# From the above boxplot we can conclude that there are outliers windspeed variables
# Getting 75 and 25 percentile of variable "windspeed"

q75, q25 = np.percentile(data['windspeed'], [75,25])

# Calculating Interquartile range

iqr = q75 - q25

    

# Calculating upper extream and lower extream

minimum = q25 - (iqr*1.5)

maximum = q75 + (iqr*1.5)

    

# Replacing all the outliers value to NA

data.loc[data['windspeed']< minimum,'windspeed'] = np.nan

data.loc[data['windspeed']> maximum,'windspeed'] = np.nan



# Checking % of missing values

data.isnull().sum().sum()
#Checking missing values in train dataset

print(data.isnull().sum())

#result shows there are missing values in the dataset
##we will impute the missing values which was outlier values by using mean imputation

# we chose mean imputation because median imputation is majorly suitable for the data having outliers

## as we dont have outliers so we will choose mean imputation over KNN.



data['windspeed'] = data['windspeed'].fillna(data['windspeed'].mean())
print(data.isnull().sum())
#Code for plotting pairplot

sns_plot = sns.pairplot(data=data[con])

plt.plot()

plt.savefig('Pairplot')
##Correlation analysis for continuous variables

#Correlation plot

data_corr = data.loc[:,con]



#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(10, 10))



#Generate correlation matrix

corr = data_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 

            cmap=sns.diverging_palette(220, 50, as_cmap=True),

            square=True, ax=ax, annot = True)

plt.plot()

plt.savefig('Heatmap')
label = 'cnt'

obj_dtype = cat

drop_feat = []



## ANOVA TEST FOR P VALUES

import statsmodels.api as sm

from statsmodels.formula.api import ols



anova_p = []

for  i in obj_dtype:

    buf = label + ' ~ ' + i

    mod = ols(buf,data=data).fit()

    anova_op = sm.stats.anova_lm(mod, typ=2)

    print(anova_op)

    anova_p.append(anova_op.iloc[0:1,3:4])

    p = anova_op.loc[i,'PR(>F)']

    if p >= 0.05:

        drop_feat.append(i)
drop_feat
#As a result of correlation analysis and ANOVA, we have concluded that we should remove 6 columns

#'temp' and 'atemp' are correlated and hence one of them should be removed

#'holiday', 'weekday' and 'workingday' have p>0.05 and hence should be removed
# Droping the variables which has redundant information

to_drop = ['atemp', 'holiday', 'weekday', 'workingday']

data = data.drop(to_drop, axis = 1)
data.info()
# Updating the Continuous and Categorical Variables after droping correlated variables

con = [i for i in con if i not in to_drop]

cat = [i for i in cat if i not in to_drop]
# Checking the distribution of values for variables in data

for i in con:

    if i == 'data':

        continue

    sns.distplot(data[i],bins = 'auto')

    plt.title("Checking Distribution for Variable "+str(i))

    plt.ylabel("Density")

    plt.savefig('{i}_Vs_Density.png'.format(i=i))

    plt.show()
#Data before scaling

data.head()
# Since our data is normally distributed, we will use Standardization for Feature Scalling

# #Standardization

for i in con:

    if i == 'cnt':

        continue

    data[i] = (data[i] - data[i].mean())/(data[i].std())
#Data after scaling

data.head()
dummy_data = pd.get_dummies(data = data, columns = cat)



#Copying dataframe

bike_data = dummy_data.copy()
dummy_data.head()
#Using train test split functionality for creating sampling

X_train, X_test, y_train, y_test = train_test_split(dummy_data.iloc[:, dummy_data.columns != 'cnt'], 

                         dummy_data.iloc[:, 3], test_size = 0.33, random_state=101)
(X_train.shape),(y_train.shape)
# Importing libraries for Decision Tree 

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error



# Building model on top of training dataset

fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)



# Calculating RMSE for test data to check accuracy

pred_test = fit_DT.predict(X_test)

rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))



def MAPE(y_true,y_pred):

    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100

    return mape



DT_rmse = rmse_for_test

DT_mape = MAPE(y_test,pred_test)

DT_r2 = r2_score(y_test,pred_test)



print('Decision Tree Regressor Model Performance:')

print("Root Mean Squared Error For Test data = "+str(rmse_for_test))

print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))

print("MAPE(Mean Absolute Percentage Error) = "+str(DT_mape))





#Decision Tree Regressor Model Performance:

#Root Mean Squared Error For Test data = 997.3873927346699

#R^2 Score(coefficient of determination) = 0.7073525764693427

#MAPE(Mean Absolute Percentage Error) = 25.707144204754727
# Importing libraries for Random Forest

from sklearn.ensemble import RandomForestRegressor



# Building model on top of training dataset

fit_RF = RandomForestRegressor(n_estimators = 500).fit(X_train,y_train)



# Calculating RMSE for test data to check accuracy

pred_test = fit_RF.predict(X_test)

rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))



RF_rmse = rmse_for_test

RF_mape = MAPE(y_test,pred_test)

RF_r2 = r2_score(y_test,pred_test)



print('Random Forest Regressor Model Performance:')

print("Root Mean Squared Error For Test data = "+str(rmse_for_test))

print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))

print("MAPE(Mean Absolute Percentage Error) = "+str(RF_mape))



#Random Forest Regressor Model Performance:

#Root Mean Squared Error For Test data = 567.4712836267795

#R^2 Score(coefficient of determination) = 0.9052662486980746

#MAPE(Mean Absolute Percentage Error) = 13.33175245911665
# Importing libraries for Linear Regression

from sklearn.linear_model import LinearRegression



# Building model on top of training dataset

fit_LR = LinearRegression().fit(X_train , y_train)



# Calculating RMSE for test data to check accuracy

pred_test = fit_LR.predict(X_test)

rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))



LR_rmse = rmse_for_test

LR_mape = MAPE(y_test,pred_test)

LR_r2 = r2_score(y_test,pred_test)



print('Linear Regression Model Performance:')

print("Root Mean Squared Error For Test data = "+str(rmse_for_test))

print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))

print("MAPE(Mean Absolute Percentage Error) = "+str(LR_mape))



#Linear Regression Model Performance:

#Root Mean Squared Error For Test data = 736.2047259447531

#R^2 Score(coefficient of determination) = 0.8405538055300172

#MAPE(Mean Absolute Percentage Error) = 17.217590042129938
# Importing library for Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor



# Building model on top of training dataset

fit_GB = GradientBoostingRegressor().fit(X_train, y_train)



# Calculating RMSE for test data to check accuracy

pred_test = fit_GB.predict(X_test)

rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))



GBR_rmse = rmse_for_test

GBR_mape = MAPE(y_test,pred_test)

GBR_r2 = r2_score(y_test,pred_test)



print('Gradient Boosting Regressor Model Performance:')

print("Root Mean Squared Error For Test data = "+str(rmse_for_test))

print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))

print("MAPE(Mean Absolute Percentage Error) = "+str(GBR_mape))



#Gradient Boosting Regressor Model Performance:

#Root Mean Squared Error For Test data = 575.7689853723047

#R^2 Score(coefficient of determination) = 0.9024755542385117

#MAPE(Mean Absolute Percentage Error) = 13.039727726693526
dat = {'Model_name': ['Decision tree default', 'Random Forest Default', 'Linear Regression',

                   'Gradient Boosting Default'], 

          'RMSE': [DT_rmse, RF_rmse, LR_rmse, GBR_rmse], 

         'MAPE':[DT_mape, RF_mape, LR_mape, GBR_mape],

        'R^2':[DT_r2, RF_r2, LR_r2, GBR_r2]}

results = pd.DataFrame(data=dat)
results
#Importing essential libraries

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression
##Random Search CV

from sklearn.model_selection import RandomizedSearchCV



RRF = RandomForestRegressor(random_state = 0)

n_estimator = list(range(1,20,2))

depth = list(range(1,100,2))



# Create the random grid

rand_grid = {'n_estimators': n_estimator,

               'max_depth': depth}



randomcv_rf = RandomizedSearchCV(RRF, param_distributions = rand_grid, n_iter = 5, cv = 5, random_state=0)

randomcv_rf = randomcv_rf.fit(X_train,y_train)

predictions_RRF = randomcv_rf.predict(X_test)

predictions_RRF = np.array(predictions_RRF)



view_best_params_RRF = randomcv_rf.best_params_



best_model = randomcv_rf.best_estimator_



predictions_RRF = best_model.predict(X_test)





#R^2

RRF_r2 = r2_score(y_test, predictions_RRF)



#Calculating MSE

RRF_mse = np.mean((y_test - predictions_RRF)**2)



#Calculate MAPE

RRF_mape = MAPE(y_test, predictions_RRF)



print('Random Search CV Random Forest Regressor Model Performance:')

print('Best Parameters = ',view_best_params_RRF)

print('R-squared = {:0.2}.'.format(RRF_r2))

print('MSE = ',round(RRF_mse))

print('MAPE = {:0.4}%.'.format(RRF_mape))

print('**********************************************')
### END OF CODE ###