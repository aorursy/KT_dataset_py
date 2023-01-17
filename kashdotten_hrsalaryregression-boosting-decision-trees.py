import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import Libraries

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sb

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor #machine learning model 1

from sklearn.ensemble import RandomForestRegressor #machine learning model 2

from sklearn.metrics import mean_squared_error #regression evaluation

from matplotlib import rcParams #Plotting params.

%matplotlib inline
#Read the data.

data = pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')

emp_survey = pd.read_csv('/kaggle/input/hr-analytics-case-study/employee_survey_data.csv')

man_survey = pd.read_csv('/kaggle/input/hr-analytics-case-study/manager_survey_data.csv')

#in_time = pd.read_csv('/Users/kashs/Datasets/hr-analytics-case-study/in_time.csv')

#out_time = pd.read_csv('/Users/kashs/Datasets/hr-analytics-case-study/out_time.csv')
#Inspect the dataframe.

data.sample(5)
#Lets check the columns in the dataset.

data.columns
#Lets combine the dataset from the employee survey, management survey and employee data.

#Merge 1

#Merging Employee Survey Data to main df.

combined = data.merge(emp_survey, on= 'EmployeeID')

#Merge 2

#Merging Management Survey Data to main df.

df = combined.merge(man_survey, on= 'EmployeeID')

del combined
#Checking Null Values

df.isnull().sum()



#There are some null values in NumCompaniesWorked, TotalWorkingYears, EnvironmentSatisfaction, JobSatisfaction and

#WorkLifeBalance.
#Drop Na's. (Since they are <5% of total observations).

df.dropna(inplace=True)
#Checking number of rows and columns.

df.shape 
#Check Data Types and Count.

df.info()
#Average salary across all departments.

print ("Average Salary across all Departments: $",df['MonthlyIncome'].mean())

#Average salary across Education levels. (1-Below College, 5-Doctor)

print ("Average Salary for Education Level (Below College):$ ",df[df['Education'] == 1].MonthlyIncome.mean() )

print ("Average Salary for Education Level (Doctor):$ ",df[df['Education'] == 5].MonthlyIncome.mean())

#There isnt much difference between average salaries of Doctor education level and college!
#Checking monthly income by education level and distance from home.

rcParams['figure.figsize']=12,10

ax1=df.plot.scatter(x='DistanceFromHome',

                      y='MonthlyIncome', c='Education', colormap = 'viridis')

plt.xlabel('Distance From Home')

plt.ylabel('MonthlyIncome', fontsize = 12)

plt.suptitle ('Income by Distance from Home and Education Level')



#Education Levels (As given in data dictionary).

# 1 'Below College'

# 2 'College'

# 3 'Bachelor'

# 4 'Master'

# 5 'Doctor'
#Checking Outliers (Although the outliers have been removed here, the exploration will be done using the original 'df').

df1 = df.copy(deep = True) 

# In this technique, floor (e.g., the 10th percentile) the lower values and cap (e.g., the 90th percentile) the higher values. 

# Print the 10th and 90th percentiles which will be used for quantile-based flooring/capping.

print("10th Percentile: ",df1['MonthlyIncome'].quantile(0.10))

print("90th Percentile: ",df1['MonthlyIncome'].quantile(0.95))

df1["MonthlyIncome"] = np.where(df1["MonthlyIncome"] <23176, 23176,df1['MonthlyIncome']) #floor values lower than 10th pcntile.

df1["MonthlyIncome"] = np.where(df1["MonthlyIncome"] >137755, 137755,df1['MonthlyIncome']) #floor values higher than 90th pcntile.

#Check the skew.

print("The skewness of capped data is: ", df1['MonthlyIncome'].skew())
#Monthly Income by Department Type. 

plt.figure(figsize=(7,4))

sb.boxplot(x='Department', y='MonthlyIncome', data=df, palette='hls')
#Monthly Income by EducationField.

plt.figure(figsize=(12,5))

sb.boxplot(x='EducationField', y='MonthlyIncome', data=df, palette='hls')
#Monthly Income By Job Role.

plt.figure(figsize=(19,4))

sb.boxplot(x='JobRole', y='MonthlyIncome', data=df, palette='hls')
#Monthly Income by Sex.

plt.figure(figsize=(5,4))

sb.boxplot(x='Gender', y='MonthlyIncome', data=df, palette='hls')
#Monthly Income by Attrition.

plt.figure(figsize=(5,4))

sb.boxplot(x='Attrition', y='MonthlyIncome', data=df, palette='hls')
#Mapping numerical values to attrition.

df1['Attrition'] = df1['Attrition'].map({'No':0, 'Yes':1})
#Get a count of objects (categorical columns).

object_col = []

for column in df.columns:

    if df[column].dtype == object and len(df[column].unique()) <= 30:

        object_col.append(column)

        print(f"{column} : {df[column].unique()}")

        print(df[column].value_counts())

        print("==================================")
#Dropping Employee Count and StandardHours features (sd=0/have just one value in column).

df1.drop(['EmployeeCount','StandardHours','EmployeeID','Over18'],axis=1 , inplace=True)
#Check correlation.

plt.figure(figsize=(15,10))

sb.heatmap(df1.corr(), annot=True, cmap="YlGnBu", annot_kws={"size":8})
#Highly correlated features cause overfitting.

#Will drop the features - TotalWorkingYear/Age, TotalWorkingYears/YearsAtCompany

df1.drop(['TotalWorkingYears','YearsAtCompany'],axis=1, inplace=True)
#Check Correlation Again.

df1.corr().sort_values(by = 'MonthlyIncome' ,ascending=False)
#Dummifying DF.

df1 = pd.get_dummies(df1)
df1.head(2)
#Checking distribution of target feature (Monthly Income).

plt.figure(figsize=(10,5))

sb.distplot(df1['MonthlyIncome']) #The data is right skewed.
#Remove Target Feature: Monthly Income.

X = df1.drop('MonthlyIncome',axis=1)

y = df1['MonthlyIncome']
#### Snappy Functions to check cross validation and print the result using best hyperparams. (Grid Search CV)
#Function to get cross validation scores

from sklearn.model_selection import cross_val_score

def get_cv_scores(model):

    scores = cross_val_score(model,

                             X_train,

                             y_train,

                             cv=5,

                             scoring='r2')

    

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')
#Function to print best hyperparams.

def print_results(results):

    print('BEST hyperparams: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#Split the dataset into 80-20.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



#Applying Scaler.

scaler = preprocessing.Normalizer()

X_train = scaler.fit_transform(X_train) #notice how the target feature (y) is untouched.

X_test = scaler.fit_transform(X_test)
# #Lets instantiate the model and use cross validation to find the best hyperparams.

# from sklearn.model_selection import GridSearchCV

# clf = GradientBoostingRegressor()

# parameters = {'n_estimators': [500,750,1000], 'max_depth': [4,8,None], 'learning_rate':[.1, .01]} 

# #smaller value of c is stronger regularization. (it is the inverse of regularization strength)

# cv = GridSearchCV(clf, parameters, cv=5)

# cv.fit(X_train, y_train.ravel()) #fit it on the train data to find best hyperparams.

# print_results(cv)
#The best identified hyperparams are - 0.811 (+/-0.079) for {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 750}



# Fit regression model using best hyperparams.

params = {'n_estimators': 750, 'max_depth': 8, 'min_samples_split': 20,

          'learning_rate': 0.1, 'loss': 'ls'}

clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)
#Model Evaluation

from sklearn import metrics

print ('MAE:',metrics.mean_absolute_error(y_test,y_pred_clf))

print ('MSE:',metrics.mean_squared_error(y_test,y_pred_clf))

print ('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_clf)))

print ('Mean Abs % Error:', mean_absolute_percentage_error(y_test, y_pred_clf)) #Under 4% error.
# Plot training deviance. Check Bias-Variance Trade-off.



# Compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
#Plotting the most important features. (This is the magic of boosting algorithm!).

plt.figure(figsize=(15,10))

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
#Split the dataset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)



from sklearn import preprocessing

#Applying Scaler.

scaler = preprocessing.Normalizer()

X_train = scaler.fit_transform(X_train) #notice how the target feature (y) is untouched.

X_test = scaler.fit_transform(X_test)
# #Lets instantiate the model and use cross validation to find the best hyperparams.

# from sklearn.model_selection import GridSearchCV

# rf = RandomForestRegressor()

# parameters = {'n_estimators': [100, 250,500]} 

# cv = GridSearchCV(rf, parameters, cv=5)

# cv.fit(X_train, y_train.ravel()) #fit it on the train data to find best hyperparams.

# print_results(cv)



#The reason the above code is commented to reduce the run time, since cross validation takes hold out sets, it take a while to run.
#Reapplying with best hyperparams.

from sklearn.ensemble import RandomForestRegressor #Regressor

rf = RandomForestRegressor(n_estimators = 500,

                              criterion = 'mse',

                              random_state = 1,

                              n_jobs = -1)
#Fit model

rf.fit(X_train,y_train.values.ravel())

y_pred_rf = rf.predict(X_train)

y_pred_rf = rf.predict(X_test)



#Another way to issue print.

print('Forest train score %.3f, Forest test score: %.3f' % (rf.score(X_train,y_train), rf.score(X_test, y_test)))
#Model Evaluation - 

from sklearn import metrics

print ('MAE:',metrics.mean_absolute_error(y_test,y_pred_rf))

print ('MSE:',metrics.mean_squared_error(y_test,y_pred_rf))

print ('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf)))

print ('Mean Abs % Error:', mean_absolute_percentage_error(y_test, y_pred_rf))
#Taking a subet of the data to show actuals vs predictions.

sample_data = df1.drop('MonthlyIncome',axis=1)[30:35]

sample_salary = df1['MonthlyIncome'][30:35]
#Now lets predict and check the monthly income value for the observations.

#Scale the data.

scaler = preprocessing.Normalizer()

sample_data = scaler.fit_transform(sample_data)



#Predict on the new sample.

sample_prediction = clf.predict(sample_data)

print ('\n''--------------')

print ("Predicted Monthly Income" '\n', sample_prediction.reshape(-1,1))

print ('\n''--------------')

print ("Actual Monthly Income" '\n' ,sample_salary)
#Thats all for now. Next steps feature engineering and hyperparams tuning. Always open for feedback.

#Happy Learning!