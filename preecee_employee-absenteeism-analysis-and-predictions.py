## ----------- Part 1: Define and categorize the problem statement --------------
#### The problem statement is to analyze the cause of absenteeism and predict the every month losses in 2011 due to absenteism.
##### This is clearly a 'Supervised machine learning regression problem' to predict a number based on the input features

## ----------- Part 1 ends here ----------------- 
##------------- Import all the required libraries--------------

## Import all the required libraries
import os
import pandas as pd
import numpy as np
from scipy import stats


#------ for model evaluation -----
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#----- for preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

#---- for model building
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#---- for cross validation
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
emp_abntsm=pd.read_excel('../input/Absenteeism_at_work_Project.xls')
emp_abntsm.head()
##---------- Part 2 ends here --------------------------
# For ease of operations, lets change the names of the columns to short versions
#emp_abntsm.rename(columns=lambda x: x.replace(' ', '_'))
emp_abntsm=emp_abntsm.rename(columns = {'Reason for absence':'Absence_Reason','Month of absence':'Absence_Month','Day of the week':'Absence_Day','Transportation expense': 'Transportation_Expense','Distance from Residence to Work':'Work_Distance','Service time':'Service_Time','Work load Average/day ':'Average_Workload','Hit target': 'Hit_Target','Disciplinary failure':'Disciplinary_Failure','Social drinker':'Drinker','Social smoker':'Smoker','Body mass index':'BMI','Absenteeism time in hours':'Absent_Hours'})
emp_abntsm.head()
#-- Here the target feature is: 'Absenteeism time in hour' and other 20 columns, which are mix of continous and categorical(although defined as int/floats) features are predictors.
#-- Lets analyze them further
# ------------Part 3 : Prepare the data for consumption(Data Cleaning) ---------------
#### 3a.) Check the shape/properties of the data
#### 3b.) Completing -- Perform missing value analysis and impute missing values if necessary
#### 3c.) Correcting -- Check for any invalid data inputs , for outliers or for any out of place data
#### 3d.) Creating -- Feature extraction . Extract any new features from existing features if required
#### 3e.) Converting -- Converting data to proper formats
#### --------3a.) Check the shape/properties of the data
## Check the shape of the data
emp_abntsm.shape

# what we can infer:
## ->the dataset has 740 observations and 21 features
## Check the properties of the data
emp_abntsm.info()
# what we can infer:
# ->There are null values in the dataset
# -> The datatypes are int and float
#### ------------------3b.) Correcting -- Check for any invalid data inputs 
# From above observations data doesnot seem to have any invalid datatypes to be handled

# However feature 'Absence_Month' have an imvalid value 0. Lets drop it.
# ALso, as we can see, 'Absent_Hours' are 0 in some places.
# This could be result of cancelled or withdrwan leaves. Lets drop these

emp_abntsm = emp_abntsm[(emp_abntsm.Absent_Hours > 0)]
emp_abntsm = emp_abntsm[(pd.notnull(emp_abntsm.Absence_Month)) & ~(emp_abntsm.Absence_Month == 0)] 
# Let's check for the outliers in EDA step
# -------------- 3c.) Completing -- Perform missing value analysis and impute missing values if necessary
#-- Calculating % of nulls
(emp_abntsm.isna().sum() / emp_abntsm.shape[0])*100
# what we can infer:
# ->There are  null values in almost all the columns of the dataset, although in small amount.
# -> We'll drop all the null value rows for target variable and 
# -> We'll will impute null values for all other features.
#-- impute missing values in all the independent featues(exept Average_Workload)
#-- Replace missing of any any employee with  information of same employee from other instances
#-- example if 'Age' of employee 1 is missing, then impute it with 'Age' from other instance of employee 1.
final_col = ['Transportation_Expense','Work_Distance','Service_Time','Age','BMI','Drinker','Smoker','Height','Weight','Pet','Son','Education','Disciplinary_Failure','Hit_Target']
#----impute missing values and Nas --------
for i in emp_abntsm['ID'].unique(): 
    for j in final_col :
        emp_abntsm.loc[((emp_abntsm['ID'] == i) & (emp_abntsm[j].isna())), j] = emp_abntsm[(emp_abntsm.ID==i)][j].max()
#--- Now for 'Average_Workload' missing values, let's analyze which is the best way to impute 
emp_abntsm[['ID','Absence_Month','Average_Workload']].sort_values(['Absence_Month','ID','Average_Workload'])
plt.scatter(x='ID', y='Average_Workload', s=None, c=None, marker=None, data=emp_abntsm)
plt.scatter(x='Absence_Month', y='Average_Workload', s=None, c=None, marker=None, data=emp_abntsm)
#From above, we can deduce that 'Average_Workload' is distributed mostly by month.
#So, let's impute missing 'Average_Workload' by mode of that month
# update workload with the mode of corresponding month's workload
for i in emp_abntsm['Absence_Month'].unique(): 
    frequent_wrkld = stats.mode(emp_abntsm[emp_abntsm['Absence_Month']==i]['Average_Workload'])[0][0]
    emp_abntsm.loc[((emp_abntsm['Absence_Month']==i) & pd.isna(emp_abntsm['Average_Workload'])),'Average_Workload'] = frequent_wrkld
#Fill missing values of 'Absent_Hours' with 0
emp_abntsm.Absent_Hours = emp_abntsm.Absent_Hours.fillna(0)
#---- Missing Value handling ENDS here ------------------
#### 3d.) ------- Converting -- Converting data to proper formats
# features like 'Absence_Month','Education' are categories here. Lets convert to categories
categorical_var = ['Absence_Reason','Absence_Month','Absence_Day','Seasons','Disciplinary_Failure','Education','Son','Drinker','Smoker','Pet']
continous_var = ['ID','Transportation_Expense','Work_Distance','Service_Time','Age','Average_Workload','Hit_Target','Weight','Height','BMI']
target_var = ['Absent_Hours']

for i in categorical_var:
    emp_abntsm[i] = emp_abntsm[i].astype("category")
emp_abntsm.info()
#### -----------------3e.) Creating -- Feature extraction . Extract any new features from existing features if required

# Here we do not need any feature extraction.
# However, before feeding to model, we might need to aggregate the data
# ------------Part 3 : Prepare the data for consumption(Data Cleaning) ENDS here------------------------------------------------
# ---------------------------Part 4 : Exploratory Data Analysis(EDA) STARTS here -----------------------------------------------
#----- 4 a.) Outlier Analysis -----------
box = ['Transportation_Expense','Work_Distance','Service_Time','Age','Average_Workload','Hit_Target','Weight','Height','BMI','Absent_Hours']
row = 5
col = 2
r = 0
c=0
i=0
fig,ax = plt.subplots(nrows=row,ncols=col)
fig.set_size_inches(20,20)

while r < row:
    c =0
    while c < col:
        sn.boxplot(x=box[i], y=None, hue=None, data=emp_abntsm, order=None, hue_order=None, orient=None, color=None, palette=None,ax=ax[r,c])
        c=c+1
        i=i+1
    r=r+1
fig,ax = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(10,8)
sn.boxplot(x=emp_abntsm['Absence_Month'], y='Absent_Hours', hue=None, data=emp_abntsm, order=None, hue_order=None, orient=None, color=None, palette=None,ax=ax)
# what we can infer from above boxplots:
# --> Target feature 'Absent_hours', has many outliers. It needs to be handled( will handle it after exploratory analysis)
# -> Not many outliers in independent features. Data seems balanced.

#---- 4b.) Correlation Analysis
#--- Explore continous features
#--- Explore categorical features
#------------- Explore continous features -----------------
##Explore the correlation btwn the independent continous features with target variabe
corr=emp_abntsm[continous_var].corrwith(emp_abntsm.Absent_Hours)
corr.plot.bar(figsize=(8,6), title='Correlation of features with the response variable Absent_Hours', grid=True, legend=False, style=None, fontsize=None, colormap=None, label=None)
##------heatmap for correlation matrix---------##
##to check multicollinearity ---##

corr = emp_abntsm[continous_var].corr()
#correlation matrix
sn.set(style='white')
#compute correlation matrix
#corr =bike.drop(columns=['cnt']).corr()
#generate a mask for upper triangle#
mask =np.zeros_like(corr, dtype=np.bool)
mask[np.tril_indices_from(mask)]=True
#setuop the matplotlab figure
f,ax=plt.subplots(figsize=(10,10))
#generate a custom diverging colormap
cmap=sn.diverging_palette(220, 10, s=75, l=50, sep=10, n=6, center='light', as_cmap=True)
#heatmap
sn.heatmap(corr, vmin=None, vmax=None, cmap=cmap, center=0, robust=False, fmt='.2g', linewidths=0, linecolor='white', square=True, mask=mask, ax=None)
#correlation matrix
# This shows that there is multicollinearity in the dataset. BMI and Weight are highly correlated. 'Service_Time' and 'Age' are also correlated
#Will have to deal with multi collinearity by removing few features from the dataset.
#Visualize the relationship among all continous variables using pairplots
NumericFeatureList=['Transportation_Expense','Work_Distance','Service_Time','Age','Average_Workload','Hit_Target','Weight','Height','BMI']
sn.pairplot(emp_abntsm,vars=NumericFeatureList)
#Lets explore some more, the relationship btwn independent continous variables and dependent variable using JOINT PLOTs
#graph individual numeric features by 'Absent_Hours'
for i in continous_var:
    sn.jointplot(i, "Absent_Hours", data=emp_abntsm, kind='reg', color='g', size=4, ratio=2, space=0.2, dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None, annot_kws=None)
#--- Checking the effect of 'Age' on 'Absence'
#--- Aggregate data by 'Age' and by total hours of absence
emp_hours = emp_abntsm[['ID','Absent_Hours']].groupby('ID').sum().reset_index()
emp_age = emp_abntsm[['ID','Age']].groupby('ID').max().reset_index()
absence_by_age = emp_hours.merge(emp_age, how='inner',left_on='ID', right_on='ID')

plt.scatter('Age', 'Absent_Hours', data=absence_by_age)
# Clearly, people over 40+ years of age tends to take less leaves compare to others
#--- Checking the effect of 'Transportation_Expense' on 'Absence'
#--- Aggregate data by 'Transportation_Expense' and by total hours of absence

emp_transport = emp_abntsm[['ID','Transportation_Expense']].groupby('ID').max().reset_index()
absence_by_transport = emp_hours.merge(emp_transport, how='inner',left_on='ID', right_on='ID')

plt.scatter('Transportation_Expense', 'Absent_Hours', data=absence_by_transport)
# This clearly shows concentration of leaves more whre the Transportation_Expense is between 150-300
#--- Checking the effect of 'Work_Distance' on 'Absence'
#--- Aggregate data by 'Work_Distance' and by total hours of absence

emp_distance = emp_abntsm[['ID','Work_Distance']].groupby('ID').max().reset_index()
absence_by_distance = emp_hours.merge(emp_distance, how='inner',left_on='ID', right_on='ID')

plt.scatter('Work_Distance', 'Absent_Hours', data=absence_by_distance)
# This clearly shows concentration of leaves more where the distance from work is between 10-30 km
#--- Checking the effect of 'Service_Time' on 'Absence'
#--- Aggregate data by 'Service_Time' and by total hours of absence

emp_service = emp_abntsm[['ID','Service_Time']].groupby('ID').max().reset_index()
absence_by_service = emp_hours.merge(emp_service, how='inner',left_on='ID', right_on='ID')

plt.scatter('Service_Time', 'Absent_Hours', data=absence_by_service)
# Evident from above, employees with service years < 8 and >18 tends to take less leaves
# Checking the distribution of target feature
sn.distplot(emp_abntsm['Absent_Hours']/1000, bins=None, hist=True, kde=True, rug=False, fit=None, hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None, vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)
# what we can infer from above analysis of continous variables:
# -> Target variable 'Absent_Hours' is not normally distributed, which is not a good thing. 
# -> We have to look in to this, before feeding the data to model.

# -> 'Work_Distance','Age','Average_Workload' has good correlation with target feature 'Absent_Hours'.
# -> Let's drop others from further analysis.

# -> There is multi collinearity in dataset. 'Work_Distance' and 'Transportation_Expense' are correlated. 
# -> However, since p(Transportation_Expense) > p(Work_Distance), we'll drop Transportation_Expense from further analysis.
#------------- Explore categorical features ------------------
##checking the pie chart distribution of categorical variables
emp_piplot=emp_abntsm[categorical_var]
plt.figure(figsize=(15,12))
plt.suptitle('pie distribution of categorical features', fontsize=20)
for i in range(1,emp_piplot.shape[1]+1):
    plt.subplot(4,3,i)
    f=plt.gca()
    f.set_title(emp_piplot.columns.values[i-1])
    values=emp_piplot.iloc[:,i-1].value_counts(normalize=True).values
    index=emp_piplot.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index,autopct='%1.1f%%')
#plt.tight_layout()
# --- Now let's analyze the absence by total hours of absence (not by frequency) ----
#--- Checking for the reason of Absence---
#checking the top reasons for absence as per the total numbers of absence
emp_reason_tot_hours = emp_abntsm[['Absence_Reason','Absent_Hours']].groupby('Absence_Reason').sum().sort_values('Absent_Hours').reset_index()
fig,ax = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(10,5)
sn.barplot(x='Absence_Reason', y='Absent_Hours', hue=None, data=emp_reason_tot_hours, order=None, hue_order=None, units=None, orient=None, color=None, palette=None,errcolor='.26', ax=ax)
#Analyzing absence dependency of no of kids
emp_son_tot = emp_abntsm[['Son','Absent_Hours']].groupby('Son').sum().sort_values('Absent_Hours').reset_index()
fig,ax = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(10,5)
sn.barplot(x='Son', y='Absent_Hours', hue=None, data=emp_son_tot, order=None, hue_order=None, units=None, orient=None, color=None, palette=None,errcolor='.26', ax=ax)
# Clearly, employee with 3-4 kids tend to take less hours of absence
#Analyzing absence dependency of month of year

#--leaves by frquency 
emp_month_frequent = emp_abntsm[['Absence_Month','Absent_Hours']].groupby('Absence_Month').count().sort_values('Absent_Hours').reset_index()

#--Leaves by total hours
emp_month_tot = emp_abntsm[['Absence_Month','Absent_Hours']].groupby('Absence_Month').sum().sort_values('Absent_Hours').reset_index()
fig,ax = plt.subplots(nrows=2,ncols=1)
fig.set_size_inches(10,6)
sn.barplot(x='Absence_Month', y='Absent_Hours', hue=None, data=emp_month_frequent, order=None, hue_order=None, units=None, orient=None, color=None, palette=None,errcolor='.26', ax=ax[0])
sn.barplot(x='Absence_Month', y='Absent_Hours', hue=None, data=emp_month_tot, order=None, hue_order=None, units=None, orient=None, color=None, palette=None,errcolor='.26', ax=ax[1])
#------ Exploratory Data Analysis ENDS Here------------------
# Final observations:
#1.) 
#------------------------------------------------------------
#----------------------Prepare data for modelling ------------------

#---- Drop the features which are not very relevant based on above analyses
emp_df  = emp_abntsm[['ID','Absence_Month','Son','Drinker','Work_Distance','Service_Time','Age','Average_Workload','Absent_Hours']]
#----Lets aggregate the data on 'Month' and 'Id'
emp_num = emp_df[['ID','Absence_Month','Work_Distance','Service_Time','Age','Average_Workload']].groupby(['ID','Absence_Month']).max().reset_index()
emp_tgt = emp_df[['ID','Absence_Month','Absent_Hours']].groupby(['ID','Absence_Month']).sum().reset_index()
emp_cat = emp_abntsm[['ID','Absence_Month','Son','Drinker']].groupby(['ID','Absence_Month']).max().reset_index()
emp = emp_num.merge(emp_cat, how='inner',left_on=['ID','Absence_Month'], right_on=['ID','Absence_Month']).merge(emp_tgt, how='inner',left_on=['ID','Absence_Month'], right_on=['ID','Absence_Month'])
emp.head()
#--- Lets deal with Nans introduced(same way already done above, by imputing)

#---- imputing Nan values with max each value present for a particular id. eg. Age will always be same for any id.
final_col = ['Work_Distance','Service_Time','Age','Drinker','Son']
#----impute missing values and Nas --------
for i in emp['ID'].unique(): 
    for j in final_col :
        emp.loc[((emp['ID'] == i) & (emp[j].isna())), j] = emp[(emp.ID==i)][j].max()
        
# update workload with the mode of corresponding month's workload
for i in emp['Absence_Month'].unique(): 
    frequent_wrkld = stats.mode(emp[emp['Absence_Month']==i]['Average_Workload'])[0][0]
    emp.loc[((emp['Absence_Month']==i) & pd.isna(emp['Average_Workload'])),'Average_Workload'] = frequent_wrkld

#update NA 'Absent_Hours' with 0
emp.Absent_Hours = emp.Absent_Hours.fillna(0)
emp.head()
#----- Lets check for any outliers in the aggregated data -----
continous_var = ['Work_Distance','Service_Time','Age','Average_Workload','Absent_Hours','Son']
row = 3
col = 2
r = 0
c=0
i=0
fig,ax = plt.subplots(nrows=row,ncols=col)
fig.set_size_inches(20,20)

while r < row:
    c =0
    while c < col:
        sn.boxplot(x=continous_var[i], y=None, hue=None, data=emp, order=None, orient=None, ax=ax[r,c])
        c=c+1
        i=i+1
    r=r+1
# Clearly, 'Absent_Hours' has so many outliers, this will affect model. So, extreme outliers needs to be removed to make the model more generic.
# We are not removing outliers in service time, since the input data for 2011 is going to be same as 2010(except 'Age' and 'ServiceTime')
#----- Create a function to remove outliers from any column, from any database
def remove_outlier(df_in, col_name):
    q1 = np.percentile(df_in[col_name],25)
    q3 = np.percentile(df_in[col_name],75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
#--- remove out liers
continous_var = ['Service_Time','Age','Absent_Hours']
for i in continous_var:
    emp = remove_outlier(emp,i)
# Check the distribution of target feature.
#It seems better distributed then previous
sn.distplot(emp['Absent_Hours']/1000, bins=None, hist=True, kde=True)
emp.head()
#--- As we can clearly see that the dataset has different features of differenr range/scale.
#--- Lets standardise the range/scale for better performance of model
#--We can use scikit-learn preprocessing library functions StandardScaler/Normalizer for the same.
# -- However, here I am using simple formula to standardize the scale
# ---------->>value(new) = (value(max) - value) / (value(max) - value(min))
def Standardize_Values(df):
    df_new = df
    var = ['Work_Distance','Service_Time','Age','Average_Workload']
    for i in var:
        df_new[i] = (np.max(df_new[i]) - df_new[i]) / (np.max(df_new[i]) - np.min(df_new[i]))
    return df_new
#--- Standardize the values ---
emp_final = Standardize_Values(emp)
emp_final.head()
#------------ Done preparing data for modelling -------------
#----------Part 5 : Model Builing starts here ----------------------
# 1.) I am selecting 3 models to test and evaluate
 #   -> Linear Regression Model
 #   -> Random Forrest (ensemble method using bagging technique)
 #   -> Gradient Boosting (ensemble method using boosting technique)
#2.) Cross validation    
#3.) All these 3 models will be compared and evaluated
#4.) We'll choose the best out of 3
#--- define a function which takes model, predicted and test values and returns evalution matrix: R-squared value,RootMeanSquared,MeanAbsoluteError
def model_eval_matrix(model,X_test,Y_test,Y_predict):
    r_squared = model.score(X_test, Y_test)
    mse = mean_squared_error(Y_predict, Y_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_predict, Y_test)
    return r_squared,mse,rmse, mae
#train,test = train_test_split(emp_final, test_size=0.20, random_state=1)
train = emp_final[:80]
test = emp_final[20:]
X_train = train.drop(columns = ['Absent_Hours','ID'])
#Y_train = np.log(train.Absent_Hours)
Y_train = train.Absent_Hours/1000
X_test = test.drop(columns = ['Absent_Hours','ID'])
#Y_test = np.log(test.Absent_Hours)
Y_test = test.Absent_Hours/1000
#--Define Linear regession model --
lrm_regressor = LinearRegression()
lrm_regressor.fit(X_train, Y_train)
Y_predict_lrm =lrm_regressor.predict(X_test)
#------- Random Forest Model (Ensemble method using Bagging technique) --------------
forest_reg = RandomForestRegressor(n_estimators=2000, criterion='mse', max_depth=10, min_samples_split=5, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=20, min_impurity_decrease=0.00, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=1, verbose=0, warm_start=False)
forest_reg.fit(X_train, Y_train)
Y_predict_forest =forest_reg.predict(X_test)
## ----------- Building XGBoost Model (Ensemble method using Boosting technique) ---------------
#xgb_reg = GradientBoostingRegressor(random_state=1) # without parameter hypertuning
# Following model is with parameter hypertuning
xgb_reg = GradientBoostingRegressor(loss='ls', learning_rate=0.2, n_estimators=2000, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=1, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=15, warm_start=False, presort='auto')
xgb_reg.fit(X_train, Y_train)
Y_predict_xgb = xgb_reg.predict(X_test)
#---Stroring all model performances in dataframe to compare----
metric=[]
ml_models=['Linear Reg','Random Forest','GradientBoost']
fitted_models= [lrm_regressor,forest_reg,xgb_reg]
Y_Predict =[Y_predict_lrm,Y_predict_forest,Y_predict_xgb]
i=0
for mod in ml_models:
    R_SQR,MSE,RMSE,MAE = model_eval_matrix(fitted_models[i],X_test,Y_test,Y_Predict[i])
    metric.append([mod,R_SQR,MSE,RMSE,MAE])
    i=i+1
df_mod_performance=pd.DataFrame(metric,columns =['Model','R-Squared','MeanSquaredError','RootMeanSquaredError','MeanAbsoluteError'])
df_mod_performance[['Model','RootMeanSquaredError']]
#Clearly, Random Forest proves to be best model here -----
# We'll use Random forest as our final model to predict 2011 losses due to absence
# FINAL MODE :: RANDOM FOREST
absence_prediction=X_test
absence_prediction['Absent_Hours'] = 1000*Y_test
absence_prediction['Predicted_Absent_Hours'] = 1000*Y_predict_forest
#final_bike_prediction_df['Predicted_Absent_Hours'] = round(final_bike_prediction_df['Predicted_Absent_Hours'])
#--- Sample output(with actual counts and predicted counts) ---
absence_prediction
#Predicted absence hours of 2010
absence_prediction.Predicted_Absent_Hours.sum()
#Actual absence hours of 2010
absence_prediction.Absent_Hours.sum()
#--- Predicted absence hours per month 
absence_prediction.groupby('Absence_Month').sum().reset_index()[['Absence_Month','Absent_Hours','Predicted_Absent_Hours']]
#--------data fo 2011
#--- service and age will be added by 1

emp_2011 = emp
emp_2011.Service_Time = emp.Service_Time + 1
emp_2011.Age = emp.Age + 1
emp_2011= emp_2011.drop(columns = ['Absent_Hours','ID'])
#-------- Standardise the scale, before passing the input to model
emp_2011 = Standardize_Values(emp_2011)
predict_2011_absence =forest_reg.predict(emp_2011)
absence_prediction_2011=emp_2011
absence_prediction_2011['Predicted_Absent_Hours'] = predict_2011_absence*1000

absence_prediction_2011
monthly_absence= absence_prediction_2011.groupby('Absence_Month').sum().reset_index()[['Absence_Month','Predicted_Absent_Hours']]
monthly_absence
#lets say in a month excluding weekend 22 days are working days. total working hours of 36 employees will be 22*8*36.
# total losses % = (absent_hours / Total_Hours)*100
tot_Monthly_hours = 22*8*36
monthly_absence['monthly_loss_percentage'] = (monthly_absence['Predicted_Absent_Hours']/tot_Monthly_hours) * 100
#---------MONTHLY LOSSES PREDICTED FOR YEAR 2011 PER MONTH -----------
monthly_absence
#---------------------------------------------------------------------
# Hereby, concluding the project with above predictions 






