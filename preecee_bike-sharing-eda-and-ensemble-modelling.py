## ----------- Part 1: Define and categorize the problem statement --------------
#### The problem statement is to "Predict the daily bike rental count based on the environmental and seasonal settings"
##### This is clearly a 'Supervised machine learning regression problem' to predict a number based on the input features

## ----------- Part 1 ends here ----------------- 

##------------- Import all the required libraries--------------

## Import all the required libraries
import os
import pandas as pd
import numpy as np

#---- for model building
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.cross_validation import train_test_split

#---- for visualization---
import matplotlib.pyplot as plt 
import seaborn as sn

#------ for model evaluation -----
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#---- For handling warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
## ------------------- Part 2: Gather the data -----------------

### Here data is provided as .csv file with the problem.
### Let's import the data 

#bike = pd.read_csv("/input../day.csv")
bike =pd.read_csv('../input/train.csv')
bike_test =pd.read_csv('../input/test.csv')
bike.head()

##---------- Part 2 ends here --------------------------
# ------------Part 3 : Prepare the data for consumption(Data Cleaning) ---------------
#### 3a.) Check the shape/properties of the data
#### 3b.) Completing -- Perform missing value analysis and impute missing values if necessary
#### 3c.) Correcting -- Check for any invalid data inputs , for outliers or for any out of place data
#### 3d.) Creating -- Feature extraction . Extract any new features from existing features if required
#### 3e.) Converting -- Converting data to proper formats
#### --------3a.) Check the shape/properties of the data
## Check the shape of the data
bike.shape

# what we can infer:
## ->the dataset has 731 observations and 16 features
## Check the properties of the data
bike.info()
# what we can infer:
# ->There are no null values in the dataset
# -> The datatypes are int,float and object 
# -------------- 3b.) Completing -- Perform missing value analysis and impute missing values if necessary
# Although we have already seen above thatthere are no null values in the dataset. Lets try other way to confirm this
#Checking nulls
bike.isnull().sum().sort_values(ascending=False)

# what we can infer:
# ->There are no null values in the dataset.If it had, then eithere the rows/columns had to be dropped or the null values be imputed based on the % of null values
#### ------------------3c.) Correcting -- Check for any invalid data inputs , for outliers or for any out of place data
# From above observations data doesnot seem to have any invalid datatypes to be handled
# Let's check for the outliers in EDA step
#### -----------------3d.) Creating -- Feature extraction . Extract any new features from existing features if required
bike.head()
bike.datetime.describe()
## We can see that here we have 'datetime', which gives us the exact date. This features has 2 years of data(2011, 2012), all through 12 months(1 to 12) of a year
## However, date(day of month) information is not saperately given.
## Lets extract 'date','mnth','weekday' and 'yr' from 'datetime' column
bike['datetime'] = pd.to_datetime(bike['datetime'])
bike['date'] = bike['datetime'].dt.day
bike['mnth'] = bike['datetime'].dt.month
bike['yr'] = bike['datetime'].dt.year
bike['weekday'] = bike['datetime'].dt.weekday

#--convert year 2011 : 1 and 2012 : 2
bike['yr']=bike.yr.replace({2011,2012},{1,2})


## Now, 'dteday' column is not required, since we already have year, month, date info in other columns. So lets drop it.
bike = bike.drop(columns=['datetime'])

#--------repeating the same operation for test data ------------
bike_test
bike_test['datetime'] = pd.to_datetime(bike_test['datetime'])
bike_test['date'] = bike_test['datetime'].dt.day
bike_test['mnth'] = bike_test['datetime'].dt.month
bike_test['yr'] = bike_test['datetime'].dt.year
bike_test['weekday'] = bike_test['datetime'].dt.weekday
bike_test['yr']=bike_test.yr.replace({2011,2012},{1,2})
bike_test = bike_test.drop(columns=['datetime'])
#---------------------------------------------------------------

bike.tail()
#### 3e.) ------- Converting -- Converting data to proper formats
#We can clearly see that "season", "yr","mnth","holiday","weekday","workingday","weather","date" are categories,rather than continous variable.
#Let them convert to categories
categoryFeatureList = ["season", "yr","mnth","holiday","weekday","workingday","weather","date"]
for var in categoryFeatureList:
    bike[var] = bike[var].astype("category")
    bike_test[var] = bike[var].astype("category")
bike.info()
# ------------Part 3 : Prepare the data for consumption(Data Cleaning) ENDS here---------------
# ------------Part 4 : Exploratory Data Analysis(EDA) STARTS here -----------
#----- 4 a.) Outlier Analysis -----------
## -- Lets do the outlier analysis ----
## -- Visualize continous variables(cnt,temp,atemp,humidity,windspeed) and 
##  count with respect to categorical variables("season", "yr","mnth","holiday","weekday","workingday","weathersit","date")with boxplots ---
fig, axes = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(20,15)

#-- Plot total counts on y bar
sn.boxplot(data=bike, y="count",ax=axes[0][0])

#-- Plot temp on y bar
sn.boxplot(data=bike, y="temp",ax=axes[0][1])

#-- Plot atemp on y bar
sn.boxplot(data=bike, y="atemp",ax=axes[0][2])

#-- Plot hum on y bar
sn.boxplot(data=bike, y="humidity",ax=axes[0][3])

#-- Plot windspeed on y bar
sn.boxplot(data=bike, y="windspeed",ax=axes[1][0])

#-- Plot total counts on y-bar and 'yr' on x-bar
sn.boxplot(data=bike,y="count",x="yr",ax=axes[1][1])

#-- Plot total counts on y-bar and 'mnth' on x-bar
sn.boxplot(data=bike,y="count",x="mnth",ax=axes[1][2])

#-- Plot total counts on y-bar and 'date' on x-bar
sn.boxplot(data=bike,y="count",x="date",ax=axes[1][3])

#-- Plot total counts on y-bar and 'season' on x-bar
sn.boxplot(data=bike,y="count",x="season",ax=axes[2][0])

#-- Plot total counts on y-bar and 'weekday' on x-bar
sn.boxplot(data=bike,y="count",x="weekday",ax=axes[2][1])

#-- Plot total counts on y-bar and 'workingday' on x-bar
sn.boxplot(data=bike,y="count",x="workingday",ax=axes[2][2])

#-- Plot total counts on y-bar and 'weathersit' on x-bar
sn.boxplot(data=bike,y="count",x="weather",ax=axes[2][3])
# what we can infer from above boxplots:
# -> There are many outliers.
# Lets keep these outliers for now, till we complete full EDA(will remove the outliers in next update of kernel)
#---- 4b.) Correlation Analysis
#--- Explore continous features
#--- Explore categorical features
#------------- Explore continous features -----------------
##Explore the correlation btwn the independent continous features with target variabe
corr=bike[['temp','atemp','humidity','windspeed']].corrwith(bike['count'])
corr.plot.bar(figsize=(8,8), title='Correlation of features with the response variable count_of_rented_bikes', grid=True, legend=False, style=None, fontsize=None, colormap=None, label=None)
##------heatmap for correlation matrix---------##
##to check multicollinearity ---##

#correlation matrix
sn.set(style='white')
#compute correlation matrix
corr =bike.drop(columns=['count']).corr()
#generate a mask for upper triangle#
mask =np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)]=True
#setuop the matplotlab figure
f,ax=plt.subplots(figsize=(10,10))
#generate a custom diverging colormap
cmap=sn.diverging_palette(220, 10, s=75, l=50, sep=10, n=6, center='light', as_cmap=True)
#heatmap
sn.heatmap(corr, vmin=None, vmax=None, cmap=cmap, center=0, robust=False, fmt='.2g', linewidths=0, linecolor='white', square=True, mask=mask, ax=None)
#Clearly, from above heatmap, we can se that the dataset has multicolinearity. 'temp' and 'atemp' are highly correlated.
#Will need to drop one of them.
#Visualize the relationship among all continous variables using pairplots
NumericFeatureList=["temp","atemp","humidity","windspeed"]
sn.pairplot(bike,hue = 'yr',vars=NumericFeatureList)
#Lets explore some more, the relationship btwn independent continous variables and dependent variable using JOINT PLOTs
#graph individual numeric features by count of rented bikes
for i in NumericFeatureList:
    sn.jointplot(i, "count", data=bike, kind='reg', color='g', size=4, ratio=2, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None)
# Check the distribution plot of target variable 'count'
sn.distplot(bike["count"],color ='r')
# what we can infer from above analysis of continous variables:
# -> Target variable 'cnt' is almost normally distributed, which is a good thing.
# -> From correlation with dependent variable cnt, we can see that 'casual','registered' are very highly correlated to cnt. These are actually 'leak variablles'. Needs to be dropped from the dataset
# -> 'hum' has low correlation with 'cnt'. For ow, lets keep it.
# -> atemp and temp has good correlation with 'cnt'
# -> From heatmap, we can see that atemp and temp are highly correlated. So we need to drop 1 to remove multicollinearity.
# -> Since, as seen from jointplot,p(atemp) < p(temp), we can drop 'temp' and retain 'atemp' in the dataset
#------------- Explore categorical features ------------------
##checking the pie chart distribution of categorical variables
#bike_piplot = bike.drop(columns=['instant','dteday','temp','atemp','hum','windspeed','casual','registered','cnt'])
bike_piplot=bike[categoryFeatureList]
plt.figure(figsize=(15,12))
plt.suptitle('pie distribution of categorical features', fontsize=20)
for i in range(1,bike_piplot.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(bike_piplot.columns.values[i-1])
    values=bike_piplot.iloc[:,i-1].value_counts(normalize=True).values
    index=bike_piplot.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index,autopct='%1.1f%%')
#plt.tight_layout()
#What we can infer from above piplot:
#-> Most of the categorical variables are uniformally distributed, except 'holiday','weathersit','workingday'
#-> This makes sense for 'weathersit', as extreme weather is rare and hence %percentage of extreme weather in whole dataset is low
#-> This makes sense for 'holiday', as number of holidays are less in comparison to working days
#-> This makes sense for 'workingday' for the same reason as above
#-> So, categorical data seems o be pretty much uniformly distributed
#graph individual categorical features by count
fig, saxis = plt.subplots(3, 3,figsize=(16,12))

sn.barplot(x = 'season', y = 'count',hue= 'yr', data=bike, ax = saxis[0,0], palette ="Blues_d")
sn.barplot(x = 'yr', y = 'count', order=[0,1,2,3], data=bike, ax = saxis[0,1], palette ="Blues_d")
sn.barplot(x = 'mnth', y = 'count', data=bike, ax = saxis[0,2])
sn.barplot(x = 'holiday', y = 'count',  data=bike, ax = saxis[1,0])
sn.barplot(x = 'weekday', y = 'count',  data=bike, ax = saxis[1,1])
sn.barplot(x = 'workingday', y = 'count', data=bike, ax = saxis[1,2])
sn.barplot(x = 'weather', y = 'count', data=bike, ax = saxis[2,0])
sn.barplot(x = 'date', y = 'count' , data=bike, ax = saxis[2,1])
#sn.pointplot(x = 'weathersit', y = 'cnt', data=bike, ax = saxis[2,0])
sn.pointplot(x='date', y='count', hue='yr', data=bike, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[2,2])
#sn.pointplot()
#--- Lets see how these categorical variables individually ffects the count of rented bikes
# Does 'yr' affects count of rented bikes
#--> YES. the count have an upward trend wrt year

#Does 'season' affects count of rented bikes
#--> YES, it seems ppl rent more bikes during season 3 and 2, i.e. highest in fall and summer and less in winter and springs. This makes sense as weather is good to ride during summer and fall.

#Does 'month' affects count of rented bikes
#-->YES.ppl are likely to rent bikes more btwn the months May- October and lowest in month of Jan,Feb and Dec(in that order). This again makes sense, as this trend is in sync with favourable weather conditions

#Does 'holiday' affects count of rented bikes
#--> YES. ppl rent more bikes on non-holiday than holiday. It makes sense as bikers who commute to work/school will be less on holiday.

#Does 'weekday' affects count of rented bikes
#--> To some extent Yes. ppl seems to rent lesser bikes on Sat/ Sun. ie. over the weekend. Again makes sense as school and offices are closed on weekend.
#Monday also has lesser count of rented bikes. It may be possible the ppl visit to other places/cities over weekend and travel back in car on Monday, istead of renting bikes.

#Does 'weather' affects count of rented bikes
#--> Most definately YES. noone rented bike on extreme weather(season=4). ppl rent maximum bikes during a clear day (weathersit=1)

#Does 'date' affects count of rented bikes
#--> Well there is no set trends. It seems to be random. Let explore bit more of it over the 12 months using pointplot
#-->
#-- exploring some more pairplots, to see the trends over the years
fig, saxis = plt.subplots(2, 2,figsize=(16,12))
sn.pointplot(x='season', y='count', hue='yr', data=bike, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[0,0])
sn.pointplot(x='holiday', y='count', hue='yr', data=bike, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[0,1])
sn.pointplot(x='weekday', y='count', hue='mnth', data=bike, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[1,0])
sn.pointplot(x='workingday', y='count', hue='yr', data=bike, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[1,1])

#------ Exploratory Data Analysis ENDS Here------------------
# Final observations:
#1.) 'casual' and 'registered' are leak variables. They need to be dropped from the dataset
#2.) 'atemp' and 'temp' are very strongly correlated . Drop 'atemp' from the dataset(since it has higher p-value than 'temp')
#3.) 'date' does not seem to have any affect on count of bikes, it can be dropped from the dataset
#------------------------------------------------------------
#---- Drop the features mentioned above(as part of feature engineering)
train = bike.drop(columns=['temp','casual','registered'])
test = bike_test.drop(columns=['temp'])
train.head()
#----------Part 5 : Model Builing starts here ----------------------
#Train the models with both datasets(before and after feature engineering)
#Note: Just to show how feature engineering improves the result, I am going to train and test 1st model(linear regression model) with both 'before feature engineering' and 'after feature engineering' data and compare the results
# For subsequent models,I'll only use the dataset with feature engineering implemented
# 1.) I am selecting 3 models to test and evaluate
 #   -> Linear Regression Model
 #   -> Random Forrest (ensemble method using bagging technique)
 #   -> Gradient Boosting (ensemble method using boosting technique)
#2.) Cross validation    
#3.) All these 3 models will be compared and evaluated(with and without feature engineering)
#4.) We'll choose the best out of 3
#----- 5a.) -- Selecting train and test datasets for cross validations
#split train data in to test and train(after featr engineering)
#train, test = train_test_split(bike_aftr_ftr_eng, test_size=0.20, random_state = 5)

train_data = train[:80]
test_data = train[20:]
X_train = train_data.drop(columns=['count'])
Y_train = train_data['count']
X_test = test_data.drop(columns=['count'])
Y_test = test_data['count']

#--- *AFT <=> After Feature Engineering------
#------- 5b.) Define a dataframe to store performance metrices of the models 
#--- define a function which takes model, predicted and test values and returns evalution matrix: R-squared value,RootMeanSquared,MeanAbsoluteError
def model_eval_matrix(model,X_test,Y_test,Y_predict):
    r_squared = model.score(X_test, Y_test)
    mse = mean_squared_error(Y_predict, Y_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_predict, Y_test)
    return r_squared,mse,rmse, mae
#-------- 5c.) Define and fit models ---------------
#--Define Linear regession model --
lrm_regressor = LinearRegression()
lrm_regressor.fit(X_train, Y_train)
Y_predict_lrm =lrm_regressor.predict(X_test)
#------- Random Forest Model (Ensemble method using Bagging technique) --------------
forest_reg = RandomForestRegressor(random_state=1)
forest_reg.fit(X_train, Y_train)
Y_predict_forest =forest_reg.predict(X_test)
## ----------- Building XGBoost Model (Ensemble method using Boosting technique) ---------------
#xgb_reg = GradientBoostingRegressor(random_state=1) # without parameter hypertuning
# Following model is with parameter hypertuning
xgb_reg = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=300, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=1, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=100, warm_start=False, presort='auto')
xgb_reg.fit(X_train, Y_train)
Y_predict_xgb = xgb_reg.predict(X_test)

#-------Part 5 ENDS here ------------------------------------------------
#-------Part 6 : Model comparisions STARTS here---------------------------
#---Stroring all model performances in dataframe to compare----
metric=[]
ml_models=['Linear Regression','Random Forest','Gradient Boosting']
fitted_models= [lrm_regressor,forest_reg,xgb_reg]
Y_Predict =[Y_predict_lrm,Y_predict_forest,Y_predict_xgb]
i=0
for mod in ml_models:
    R_SQR,MSE,RMSE,MAE = model_eval_matrix(fitted_models[i],X_test,Y_test,Y_Predict[i])
    metric.append([mod,R_SQR,MSE,RMSE,MAE])
    i=i+1
df_mod_performance=pd.DataFrame(metric,columns =['Model','R-Squared','MeanSquaredError','RootMeanSquaredError','MeanAbsoluteError'])
df_mod_performance
#------ Comparing the performance matrix values of the models-----
#fig, saxis = plt.subplots(2, 2,figsize=(16,12))
#a=sn.pointplot(y='Model', x='R-Squared', rotate =90,data=df_mod_performance, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[0,0])
#a.set_xticklabels(a.get_xticklabels(), rotation=45)
#sn.pointplot(y='Model', x='MeanSquaredError', data=df_mod_performance, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[0,1])
#sn.pointplot(y='Model', x='RootMeanSquaredError', data=df_mod_performance, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[1,0])
#sn.pointplot(y='Model', x='MeanAbsoluteError', data=df_mod_performance, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[1,1])
#plt.tight_layout()
#What can be inferred from above observations:
#-->It is evidently clear that gradient boost gives the best performance out of all the models
#-->Hence we'll consider Gradient Boosting as our final model
#---------Part 6 : Model comparisions ENDS here ---------------------
#---------Part 7 : Hypertune the selected model starts here ------------

#Now, Gradient Boosting is the final model, parameter hypertuning can be performed on the model to find the best parameters which will give the maximum performance.
#Functions like GRIDSearchCV from GridSearch library of python can be used for this.

#However, I tried here simple approach of ‘hit and trial’, where I changed parameter few times and found a set which gave me maximum performance.

#Before parameter tuning:
#-----> Gradient Boosting
#-----> R-Squared :0.897838
#-----> MSE: 387939.616482
#-----> RMSE: 622.847988
#-----> MAE: 460.576495

#Before parameter tuning:
#-----> Gradient Boosting
#-----> R-Squared :0.913779
#-----> MSE: 327408.191428
#-----> RMSE: 572.195938
#-----> MAE: 415.264316

#Evident here, hypertuning the parameter boosted the model performance. So, we lock the parameters as below:
#-->loss='ls',
#-->learning_rate=0.1, 
#-->n_estimators=300, 
#-->subsample=1.0, 
#-->criterion='friedman_mse', 
#-->min_samples_split=2, 
#-->min_samples_leaf=1, 
#-->min_weight_fraction_leaf=0.0, 
#-->max_depth=3, 
#-->min_impurity_decrease=0.0, 
#-->min_impurity_split=None, 
#-->init=None, 
#-->random_state=1, 
#-->max_features=None, 
#-->alpha=0.9, 
#-->verbose=0, 
#-->max_leaf_nodes=100, 
#-->warm_start=False, 
#-->presort='auto'

#Lets produce the output using this model

#---------Part 7 : Hypertune the selected model ENDS here ------------
#--------Part 8 : Produce sample output with tuned model STARTS here----------------------

Y_predict_xgb_final = xgb_reg.predict(test)
final_bike_prediction_df=test
#final_bike_prediction_df['ActualCount'] = Y_test
final_bike_prediction_df['PredictedCount'] = Y_predict_xgb_final
final_bike_prediction_df['PredictedCount'] = round(final_bike_prediction_df['PredictedCount'])
#--- Sample output(with actual counts and predicted counts) ---
#final_bike_prediction_df
final_bike_prediction_df.head()
#-----Plotting the distributions of 'ActualCount' and 'PredictedCount'
#fig, saxis = plt.subplots(2, 2,figsize=(16,12))
#sn.distplot(final_bike_prediction_df["ActualCount"],color ='r', ax = saxis[0,0])
#sn.distplot(final_bike_prediction_df["PredictedCount"],color ='g',ax = saxis[0,1])

#--- As clearly evident from the below charts the distributions of both the counts are very similar.
#--This seems a fair model


























