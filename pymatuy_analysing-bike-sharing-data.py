# importing librarys

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set(style="darkgrid")



from scipy import stats



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor









# Laoding daily data set

bike_day = pd.read_csv('../input/day.csv')
# expand number of columns

pd.set_option('display.max_columns', 30)

# Viewing day data

bike_day.head()
# View shape of sets

print(bike_day.dtypes)

# Renaming Columns



bike_day.rename(columns={'instant':'rec_id',

                        'dteday':'datetime',

                        'weathersit':'weather',

                        'mnth':'month',

                        'hum':'humidity',

                        'cnt':'total_cnt'},inplace=True)



import calendar

from datetime import datetime



# Creating new Columns



bike_day["weekday"] = bike_day.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

bike_day["month"] = bike_day.datetime.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

bike_day["season_num"] = bike_day.season 

bike_day["season"] = bike_day.season.map({1: 'Winter', 2 : 'Spring', 3 : 'Summer', 4 : 'Fall' })

bike_day["weather_num"] = bike_day.weather

bike_day["weather"] = bike_day.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })



bike_day['weekday_num'] = bike_day.weekday.map({'Monday': 1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday': 7})

bike_day['month_num'] = bike_day.month.map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June' : 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})







# defining categorical variables



bike_day['season'] = bike_day.season.astype('category')

bike_day['holiday'] = bike_day.holiday.astype('category')

bike_day['weekday'] = bike_day.weekday.astype('category')

bike_day['weather'] = bike_day.weather.astype('category')

bike_day['month'] = bike_day.month.astype('category')

bike_day['workingday'] = bike_day.workingday.astype('category')





bike_day.head()
# checking data set once more 



print(bike_day.dtypes)
bike_day.isnull().any()
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sns.boxplot(data=bike_day,y="total_cnt",x="month",orient="v",ax=axes[0][0])

sns.boxplot(data=bike_day,y="total_cnt",x="season",orient="v",ax=axes[0][1])

sns.boxplot(data=bike_day,y="total_cnt",x="weekday",orient="v",ax=axes[1][0])

sns.boxplot(data=bike_day,y="total_cnt",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Total Count vs. Month")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Total Count vs. Season")

axes[1][0].set(xlabel='Weekday', ylabel='Count',title="Total Count vs. Weekday")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Total Count vs. Working Day")
bike_day_WithoutOutliers = bike_day[np.abs(bike_day["total_cnt"]-bike_day["total_cnt"].mean())<=(3*bike_day["total_cnt"].std())] 



print ("Shape Of The Before Ouliers: ",bike_day.shape)

print ("Shape Of The After Ouliers: ",bike_day_WithoutOutliers.shape)
corrMatt = bike_day[['temp','atemp','windspeed','humidity','registered','casual','total_cnt']].corr()



mask = np.array(corrMatt)

# Turning the lower-triangle of the array to false

mask[np.tril_indices_from(mask)] = False

fig,ax = plt.subplots()

sns.heatmap(corrMatt, 

            mask=mask,

            vmax=0.7, 

            square=True,

            annot=True,

            cmap="YlGnBu")

fig.set_size_inches(8,10)
fig, axes = plt.subplots(nrows=2,ncols=3)

fig.set_size_inches(20, 13)

sns.scatterplot(data=bike_day,y="total_cnt",x="temp",ax=axes[0][0])

sns.scatterplot(data=bike_day,y="total_cnt",x="humidity",ax=axes[0][1])

sns.scatterplot(data=bike_day,y="total_cnt",x="windspeed",ax=axes[0][2])

sns.scatterplot(data=bike_day,y="total_cnt",x="month",ax=axes[1][0])

sns.barplot(data=bike_day,y="total_cnt",x="season",ax=axes[1][1])

sns.scatterplot(data=bike_day,y="total_cnt",x="weather_num",ax=axes[1][2])







axes[0][0].set(xlabel='Temp Norm.',ylabel='Count',title="Total Count vs. Temp")

axes[0][1].set(xlabel='Humidity', ylabel='Count',title="Total Count vs. Humidity")

axes[0][2].set(xlabel='Windspeed', ylabel='Count',title="Total Count vs. Windspeed")

axes[1][0].set(xlabel=' ', ylabel='Count',title="Total Count vs. Weekday")

axes[1][1].set(xlabel=' ', ylabel='Count',title="Total Count vs. Working Day")

axes[1][2].set(xlabel='Weather Condition', ylabel='Count',title="Total Count vs. Working Day")
fig,axes = plt.subplots(ncols=2,nrows=1)

fig.set_size_inches(12, 8)

sns.distplot(bike_day["total_cnt"],ax=axes[0]) # was macht distplot?

stats.probplot(bike_day["total_cnt"], dist='norm', fit=True, plot=axes[1])

#sns.distplot(np.log(bike_day_WithoutOutliers["total_cnt"]),ax=axes[1][0])

#stats.probplot(np.log1p(bike_day_WithoutOutliers["total_cnt"]), dist='norm', fit=True, plot=axes[1][1])
import statsmodels.api as sm



#fit the model

regressors = bike_day[['temp','atemp', 'windspeed', 'humidity']] 

reg_const = sm.add_constant(regressors)

mod = sm.OLS(bike_day['total_cnt'], reg_const)

res = mod.fit()

#print the summary

print(res.summary())
import statsmodels.formula.api as smf



#fit the model

mixed = smf.mixedlm("total_cnt ~ atemp+windspeed+humidity", bike_day, groups = bike_day['month_num'],re_formula='~windspeed+humidity')

mixed_fit = mixed.fit()

#print the summary

print(mixed_fit.summary())
# load data

bike_hour = pd.read_csv('../input/hour.csv')
# Renaming Columns



bike_hour.rename(columns={'instant':'rec_id',

                        'dteday':'datetime',

                        'weathersit':'weather',

                        'mnth':'month',

                        'hr':'hour',

                        'hum':'humidity',

                        'cnt':'total_cnt'},inplace=True)



import calendar

from datetime import datetime



# Creating new Columns



bike_hour["weekday"] = bike_hour.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

bike_hour["month"] = bike_hour.datetime.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

bike_hour["season_num"] = bike_hour.season

bike_hour["season"] = bike_hour.season.map({1: 'Winter', 2 : 'Spring', 3 : 'Summer', 4 : 'Fall' })

bike_hour["weather_num"] = bike_hour.weather

bike_hour["weather"] = bike_hour.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })







bike_hour['weekday_num'] = bike_hour.weekday.map({'Monday': 1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday': 7})

bike_hour['month_num'] = bike_hour.month.map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June' : 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})







# defining categorical variables



bike_hour['season'] = bike_hour.season.astype('category')

bike_hour['holiday'] = bike_hour.holiday.astype('category')

bike_hour['weekday'] = bike_hour.weekday.astype('category')

bike_hour['weather'] = bike_hour.weather.astype('category')

bike_hour['month'] = bike_hour.month.astype('category')

bike_hour['workingday'] = bike_hour.workingday.astype('category')







bike_hour.head()
bike_hour.isnull().any()
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sns.boxplot(data=bike_hour,y="total_cnt",x="month",orient="v",ax=axes[0][0])

sns.boxplot(data=bike_hour,y="total_cnt",x="season",orient="v",ax=axes[0][1])

sns.boxplot(data=bike_hour,y="total_cnt",x="weekday",orient="v",ax=axes[1][0])

sns.boxplot(data=bike_hour,y="total_cnt",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Total Count vs. Month")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Total Count vs. Season")

axes[1][0].set(xlabel='Weekday', ylabel='Count',title="Total Count vs. Weekday")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Total Count vs. Working Day")
bike_hour_WithoutOutliers = bike_hour[np.abs(bike_hour["total_cnt"]-bike_hour["total_cnt"].mean())<=(3*bike_hour["total_cnt"].std())] 



# how many outliers are removed?

print ("Shape Of The Before Ouliers: ",bike_hour.shape)

print ("Shape Of The After Ouliers: ",bike_hour_WithoutOutliers.shape)
fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sns.distplot(np.log(bike_hour["total_cnt"]),ax=axes[0][0])

stats.probplot(bike_hour["total_cnt"], dist='norm', fit=True, plot=axes[0][1])

sns.distplot(np.log(bike_hour_WithoutOutliers["total_cnt"]),ax=axes[1][0])

stats.probplot((bike_hour_WithoutOutliers["total_cnt"]), dist='norm', fit=True, plot=axes[1][1])



axes[0][0].set(xlabel='log(Count)',title="With Outliers")

axes[1][0].set(xlabel='log(Count)',title="Without Outliers")
# renaming bike_hour_WithoutOutliers to bike_hour

bike_hour = bike_hour_WithoutOutliers

print ("Shape Of Data after Cleaning: ",bike_hour.shape)
fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)

fig.set_size_inches(8,15)

sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]

hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]



# Plotting average count vs. month

month_h_Aggregated = pd.DataFrame(bike_hour.groupby("month")["total_cnt"].mean()).reset_index()

month_h_Sorted = month_h_Aggregated.sort_values(by="total_cnt",ascending=False)

sns.barplot(data=month_h_Sorted,x="month",y="total_cnt",ax=ax1,order=sortOrder)

ax1.set(xlabel='Month', ylabel='Count',title="Count vs. Month")



# Plotting average count vs. season

hour_h_Aggregated = pd.DataFrame(bike_hour.groupby(["hour","season"],sort=True)["total_cnt"].mean()).reset_index()

sns.pointplot(x=hour_h_Aggregated["hour"], y=hour_h_Aggregated["total_cnt"],hue=hour_h_Aggregated["season"], data=hour_h_Aggregated, join=True,ax=ax2)

ax2.set(xlabel='Hour Of The Day', ylabel='Count',title="Hourly Count vs. Season",label='big')



# Plotting average count vs. weekday

weekday_h_Aggregated = pd.DataFrame(bike_hour.groupby(["hour","weekday"],sort=True)["total_cnt"].mean()).reset_index()

sns.pointplot(x=weekday_h_Aggregated["hour"], y=weekday_h_Aggregated["total_cnt"],hue=weekday_h_Aggregated["weekday"], data=weekday_h_Aggregated, join=True,ax=ax3)

ax3.set(xlabel='Hour', ylabel='Count',title="Count vs. Weekday",label='big')



# Create X by defining features 

features = ['month','weather','temp','windspeed','season','humidity']

X_hour = bike_hour[features]



# Define y

y_hour = bike_hour.total_cnt
# Split into training, test and validaion set



X_hour_train, X_hour_test, y_hour_train, y_hour_test  = train_test_split(X_hour, y_hour, test_size=0.2, random_state=1)



X_hour_train, X_hour_val, y_hour_train, y_hour_val = train_test_split(X_hour_train, y_hour_train, test_size=0.25, random_state=1)
# checking for categorical variables

s = (X_hour_train.dtypes == 'category')

object_cols_h = list(s[s].index)



print("Categorical variables:")

print(object_cols_h)
# Encoding Categoricals using One Hot Encoding

from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 

OH_cols_train_h = pd.DataFrame(OH_encoder.fit_transform(X_hour_train[object_cols_h])) #training set

OH_cols_valid_h = pd.DataFrame(OH_encoder.transform(X_hour_val[object_cols_h])) # validation set

OH_cols_test_h = pd.DataFrame(OH_encoder.transform(X_hour_test[object_cols_h])) # validation set

OH_cols_X_h = pd.DataFrame(OH_encoder.transform(X_hour[object_cols_h])) #complete set of X





# One-hot encoding removed index; put it back

OH_cols_train_h.index = X_hour_train.index

OH_cols_valid_h.index = X_hour_val.index

OH_cols_test_h.index = X_hour_test.index

OH_cols_X_h.index = X_hour.index



# Remove categorical columns (will replace with one-hot encoding)

num_train_X_h = X_hour_train.drop(object_cols_h, axis=1)

num_val_X_h = X_hour_val.drop(object_cols_h, axis=1)

num_test_X_h = X_hour_test.drop(object_cols_h, axis=1)

num_X_h = X_hour.drop(object_cols_h, axis=1)







# Add one-hot encoded columns to numerical features

OH_train_X_h = pd.concat([num_train_X_h, OH_cols_train_h], axis=1)

OH_val_X_h = pd.concat([num_val_X_h, OH_cols_valid_h], axis=1)

OH_test_X_h = pd.concat([num_test_X_h, OH_cols_test_h], axis=1)

OH_X_h = pd.concat([num_X_h, OH_cols_X_h], axis=1) 
OH_X_h.head()
#predictions using the Random Forest Regressor



#Define the model. Set random_state to 1

rf_model = RandomForestRegressor(n_estimators=100,random_state=1)

rf_model.fit(OH_train_X_h, y_hour_train)

rf_val_predictions = rf_model.predict(OH_val_X_h)

rf_val_mae = mean_absolute_error(rf_val_predictions, y_hour_val)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

#apply model on test set:



rf_model.fit(OH_test_X_h, y_hour_test)



test_preds_h = rf_model.predict(OH_test_X_h)

errors_h = abs(y_hour_test-test_preds_h)



# Calculate mean absolute percentage error (MAPE)

MAPE_h = 100 * (errors_h / test_preds_h)





# Calculate and display accuracy

accuracy = 100 - np.mean(MAPE_h)

print("Accuracy: {:,.2f}%".format(accuracy))





# apply model on all data



rf_model.fit(OH_X_h, y_hour)



preds_all_h = rf_model.predict(OH_X_h)

errors_all_h = abs(y_hour-preds_all_h)



# Calculate mean absolute percentage error (MAPE)

MAPE_all_h = 100 * (errors_all_h / preds_all_h)





# Calculate and display accuracy

accuracy_all_h = round(100 - np.mean(MAPE_all_h),2)

print("Accuracy: {:,.2f}%".format(accuracy_all_h))





# Create X by defining features 

features_d = ['month','weather','temp','windspeed','season','humidity']

X_day = bike_day[features_d]



# Define y

y_day = bike_day.total_cnt





# Split into training, test and validaion set



X_day_train, X_day_test, y_day_train, y_day_test  = train_test_split(X_day, y_day, test_size=0.2, random_state=1)



X_day_train, X_day_val, y_day_train, y_day_val = train_test_split(X_day_train, y_day_train, test_size=0.25, random_state=1)
X_day_train.head()
X_day_train.isnull().any()
# checking for categorical variables

s = (X_day_train.dtypes == 'category')

object_cols_d = list(s[s].index)



print("Categorical variables:")

print(object_cols_d)
# Encoding Categoricals using One Hot Encoding





# Apply one-hot encoder to each column with categorical data

OH_encoder_d = OneHotEncoder(handle_unknown='ignore', sparse=False) 

OH_cols_train_d = pd.DataFrame(OH_encoder_d.fit_transform(X_day_train[object_cols_d])) #training set

OH_cols_valid_d = pd.DataFrame(OH_encoder_d.transform(X_day_val[object_cols_d])) # validation set

OH_cols_test_d = pd.DataFrame(OH_encoder_d.transform(X_day_test[object_cols_d])) # validation set

OH_cols_X_d = pd.DataFrame(OH_encoder_d.transform(X_day[object_cols_d])) #complete set of X





# One-hot encoding removed index; put it back

OH_cols_train_d.index = X_day_train.index

OH_cols_valid_d.index = X_day_val.index

OH_cols_test_d.index = X_day_test.index

OH_cols_X_d.index = X_day.index



# Remove categorical columns (will replace with one-hot encoding)

num_train_X_d = X_day_train.drop(object_cols_d, axis=1)

num_val_X_d = X_day_val.drop(object_cols_d, axis=1)

num_test_X_d = X_day_test.drop(object_cols_d, axis=1)

num_X_d = X_day.drop(object_cols_d, axis=1)







# Add one-hot encoded columns to numerical features

OH_train_X_d = pd.concat([num_train_X_d, OH_cols_train_d], axis=1)

OH_val_X_d = pd.concat([num_val_X_d, OH_cols_valid_d], axis=1)

OH_test_X_d = pd.concat([num_test_X_d, OH_cols_test_d], axis=1)

OH_X_d = pd.concat([num_X_d, OH_cols_X_d], axis=1) 
#predictions using the Random Forest Model without specifying leaf nodes



#Define the model. Set random_state to 1

rf_model_d = RandomForestRegressor(n_estimators=100, random_state=1)

rf_model_d.fit(OH_train_X_d, y_day_train)

rf_val_predictions_d = rf_model_d.predict(OH_val_X_d)

rf_val_mae_d = mean_absolute_error(rf_val_predictions_d, y_day_val)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae_d))



#apply model on test set:



rf_model_d.fit(OH_test_X_d, y_day_test)



test_preds_d = rf_model_d.predict(OH_test_X_d)

errors_d = abs(y_day_test-test_preds_d)



# Calculate mean absolute percentage error (MAPE)

MAPE_d = 100 * (errors_d / test_preds_d)





# Calculate and display accuracy

accuracy_d = 100 - np.mean(MAPE_d)

print("Accuracy: {:,.2f}%".format(accuracy_d))
#apply model on complete data set:



model = rf_model_d.fit(OH_X_d, y_day)



preds_all_d = rf_model_d.predict(OH_X_d)

errors_all_d = abs(y_day-preds_all_d)



# Calculate mean absolute percentage error (MAPE)

MAPE_all_d = 100 * (errors_all_d / preds_all_d)





# Calculate and display accuracy

accuracy_all_d = 100 - np.mean(MAPE_all_d)

print("Accuracy: {:,.2f}%".format(accuracy_all_d))
feat_imp = model.feature_importances_



df = pd.DataFrame(feat_imp)



df['ind'] = np.arange(len(feat_imp))



df['feat_imp'] = df[0]



df['labels'] = df.ind.map({0: 'temp', 1 : 'windspeed', 2 : 'humidity', 3 : 'Jan',  4 : 'Feb' , 5 : 'Mar', 6 : 'Apr' , 7 : 'May', 8 : 'Jun' , 9 : 'Jul', 10 : 'Aug', 11 : 'Sep', 12 : 'Oct', 13 : 'Nov', 14 : 'Dec' , 15 : 'Weat1', 16 : 'Weat2', 17 : 'Weat3', 18 : 'Sea1', 19 : 'Sea2', 20 : 'Sea3', 21 : 'Sea4',})

print(df.head())



width = 0.1

ind = np.arange(len(feat_imp))

ax = df['feat_imp'].plot(kind='bar', xticks=df.index)

ax.set_xticklabels(df['labels'])





plt.title('Feature Importance')

plt.xlabel('Relative importance')

plt.ylabel('feature')
