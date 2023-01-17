# Importing libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

%matplotlib inline





#my_details 

__author__ = "sreetam dev"

__email__  = "sreetamkumardev@gmail.com"
#loading the data

df_bike_sharing_day = pd.read_csv("../input/bike-share-dataset/day.csv")

df_bike_sharing_day.head()
df_bike_sharing_day.info() # fetching data types and length of data entries
df_bike_sharing_day.isnull().any() # searching if there are any null values that require imputation.
# examining for duplicate instances within our dataset

print("The total no of duplicate records within our dataset are:", df_bike_sharing_day.duplicated().sum())
df_bike_sharing_day.describe(include = [np.number]) # descriptive statistics for numerical features
df_bike_sharing_day.describe(include = 'O') # descriptive features for categorical features
# converting into datetime format so that late we can work with "dteday" feature

df_bike_sharing_day["dteday"] = pd.to_datetime(df_bike_sharing_day["dteday"])
# examining potential outlier 

df_bike_sharing_day_nr_features = ["instant","season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","cnt"]



for feature in df_bike_sharing_day_nr_features:

    y = df_bike_sharing_day[feature]

    plt.figure(figsize = (15,6))

#     plt.subplot(1,2,1)

    sns.boxplot(y)

#     plt.subplot(1,2,2)

#     sns.distplot(y, bins =20)

    plt.show()
df_bike_sharing_day_nr_features = ["instant","season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","cnt"]



for feature in df_bike_sharing_day_nr_features:

    stat_feature = df_bike_sharing_day[feature].describe()

#     print(stat_feature)

    IQR   = stat_feature['75%'] - stat_feature['25%']

    upper = stat_feature['75%'] + 1.5 * IQR

    lower = stat_feature['25%'] - 1.5 * IQR

    print('For the feature {} the upper boundary is {} and lower boundary is {}'.format(feature,upper, lower))
df_bike_sharing_day.describe()
df_bike_sharing_day[df_bike_sharing_day.holiday > 0.8]
len(df_bike_sharing_day[df_bike_sharing_day.holiday == 1])
df_bike_sharing_day[df_bike_sharing_day.hum < 0.2]
df_bike_sharing_day[df_bike_sharing_day.windspeed > 0.38]
df_bike_sharing_day.describe()
# removing humidity outliers which were causing skewness



df_bike_sharing_day=(df_bike_sharing_day[df_bike_sharing_day.hum > 0.2])
# removing windspeed outliers which were causing skewness



df_bike_sharing_day= (df_bike_sharing_day[df_bike_sharing_day.windspeed < 0.41])
df_bike_sharing_day_nr_features = ["instant","season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","cnt"]



# def boxplots_features_target(size, target, features, data):

#     plt.figure(figsize = size)

#     for each in range(len(df_bike_sharing_day_nr_features)-1):

#         plt.subplot(5,3, each+1)

#         sns.boxplot(x = target, y = features[each], data = data)



def crossCorrelation(data):

    corr = data.corr()

    plt.figure(figsize = (10,6))

    sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

    print(corr)



# # creating boxplots with respect to target variables :

# boxplots_features_target((30,10),'cnt', df_bike_sharing_day_nr_features[0:-1], df_bike_sharing_day)
crossCorrelation(df_bike_sharing_day)
#Note: Faced issues in executing this cell block, but this works within Jupyter cell

# Nr_features = df_bike_sharing_day_nr_features[0:-1]

# target_feature = df_bike_sharing_day_nr_features[-1]



# def creating_plots(features, targetfeature):

#     plt.figure(figsize=(16,16))

#     for feature in Nr_features:

#         plt.subplot(7,2, Nr_features.index(feature)+1)

#         sns.distplot(df_bike_sharing_day[feature], label= feature, color="b")

#         plt.axvline(df_bike_sharing_day[feature].mean(), linestyle = '--', color="b")

#         plt.legend()

        

# creating_plots(Nr_features, target_feature)
df_bike_sharing_day.windspeed= np.log(df_bike_sharing_day.windspeed + 1)
plt.figure(figsize = (10,6))

sns.distplot(df_bike_sharing_day["windspeed"], bins =20, color = "y")
scaler =  MinMaxScaler() #initiating a scaler and applying features to it

df_bike_sharing_day[Nr_features] = scaler.fit_transform(df_bike_sharing_day[Nr_features]) # applying noramlisation to numerical variables
df_bike_sharing_day.head()
Nr_features[1:]
# Visualize and Examine 

#to find out are there any linear relationship between features and response using scatterplots



fig, axs = plt.subplots(1,14, sharey = True) #Controls sharing of properties among x (sharex) or y (sharey)

                                            #1 row and 5 columns

i= 0

for each in Nr_features[1:]:

    df_bike_sharing_day.plot(kind = 'scatter',x = each, y = 'cnt', ax= axs[i],figsize= (100,7))

    i= i+1

import statsmodels.api as sm

x_stats = sm.add_constant(df_bike_sharing_day.drop(['instant','dteday','casual','registered','cnt'],axis = 1))

y_stats = df_bike_sharing_day.cnt



#applying OLS to our X and Y

lm = sm.OLS(y_stats,x_stats).fit()

lm.summary()
# Now using VIF to detect multicollinearity

# if VIF =1 not corelated, VIF>5 High corelated, 1<VIF<5 Moderately correlated



from statsmodels.stats.outliers_influence import variance_inflation_factor



df_bike_sharing_day_vif = df_bike_sharing_day.drop(['instant','dteday','casual','registered','cnt'],axis = 1).assign(const = 1)

pd.Series([variance_inflation_factor(df_bike_sharing_day_vif.values, i) for i in range(df_bike_sharing_day_vif.shape[1])],index = df_bike_sharing_day_vif.columns)

# Now, getting necessary model parameters( will try for both with and without Log transformation)

X_new = df_bike_sharing_day.drop(['season','atemp','instant','dteday','casual','registered','cnt','hum','windspeed'], axis =1)

Y_new = df_bike_sharing_day.cnt

Y_log_new = np.log(df_bike_sharing_day.cnt+1)



lm_1 = sm.OLS(Y_new, X_new).fit()

# lm_2 = sm.OLS(Y_log_new, X_new).fit()

lm_1.summary()

#creating residual plot of cnt vs residuals



plot_lm_cnt_1 = plt.figure(1)

plot_lm_cnt_1.set_figheight(5)

plot_lm_cnt_1.set_figwidth(8)



model_fitted_y_cnt_bikes_day = lm_1.fittedvalues



plot_lm_cnt_1.axes[0] = sns.residplot(model_fitted_y_cnt_bikes_day,'cnt', data = df_bike_sharing_day, scatter_kws = {'alpha':0.5})

plot_lm_cnt_1.axes[0].set_title('Residuals vs Fitted')

plot_lm_cnt_1.axes[0].set_xlabel('Fitted Values')

plot_lm_cnt_1.axes[0].set_ylabel('Residuals')
import seaborn as sns; sns.set()

#check for Linearity

f = plt.figure(figsize = (14,5))

ax = f.add_subplot(121)

sns.regplot(x = df_bike_sharing_day.cnt, y = model_fitted_y_cnt_bikes_day, line_kws = {"color":"r","alpha": 0.7,"lw":2})

plt.title('Check for linearity')

plt.xlabel('Actual value')

plt.ylabel('predicted value')



#checkinf for residual normality and mean

ax  = f.add_subplot(122)

sns.distplot((df_bike_sharing_day.cnt- model_fitted_y_cnt_bikes_day),ax=ax,color='b')

plt.axvline((df_bike_sharing_day.cnt - model_fitted_y_cnt_bikes_day).mean(),color = 'k', linestyle ='--')

plt.title("Check for Residual normality and mean")

plt.xlabel('Residual error')

plt.ylabel('$p(x)$');


#model_residuals



model_norm_residuals = lm_1.get_influence().resid_studentized_internal



# Detecting Normal Distribution of Residuals - QQ plot

#If residuals do not have a normal distribution then while performing significance test siuch as t test for beta parameter may not perform.



from statsmodels.graphics.gofplots import ProbPlot



QQ = ProbPlot(model_norm_residuals)

plot_lm = QQ.qqplot(line = '45', alpha = 0.5, color ='#4C72B0', lw =1) 

plot_lm.set_figheight(5)

plot_lm.set_figwidth(10)



plot_lm.axes[0].set_title('Normal Q-Q')

plot_lm.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm.axes[0].set_ylabel('Standardized Residuals')
# performing gradient descent  for the temperature feature

x = df_bike_sharing_day.temp

y = df_bike_sharing_day.cnt



# Initial random theta values 

theta = np.zeros(2)



#creating X_day_temp feature matirx, first column = intercept values

X_day_temp = np.ones(shape=(len(x),2)) #numpy.ones(shape, dtype=None, order='C')

X_day_temp[:,1] = x



Y = (df_bike_sharing_day.cnt[:,np.newaxis])



# Cost function



def cost(theta, X_day_temp,y):

    predictions = X_day_temp @ theta

    squared_errors = np.square(predictions - y)

    return np.sum(squared_errors)/(2*len(y))



print('The initial cost is:', cost(theta, X_day_temp, Y))





# Gradient Descent function:

def gradientDescent(X_day_temp, y, theta, alpha, num_iters):

    m = y.size #number of training examples

    for i in range(num_iters):

        y_hat = np.dot(X_day_temp, theta)

        theta = theta - alpha * (1.0/m) * np.dot(X_day_temp.T, y_hat-y)

    return theta



theta = gradientDescent(X_day_temp, Y, 0, 0.008,3000)

gd_predictions = X_day_temp @ theta



#plotting the regression line

plt.scatter(x,y)

plt.plot(x, gd_predictions, 'g')

plt.title("count of bike as per day and varying with temperature",fontsize =16)

plt.xlabel("Temperature", fontsize = 14)

plt.ylabel("Cost function", fontsize = 14)

plt.show()
# Define gradient descent function again



def gradientDescent(X_day_temp, y, theta, alpha, num_iters):

    cost_bike_day = np.zeros(num_iters) # create a vector

    m = y.size #number of training examples

    for i in range(num_iters):

        y_hat = np.dot(X_day_temp,theta)

        theta = theta - alpha * (1.0/m) * np.dot(X_day_temp.T,y_hat-y)

        cost_bike_day[i] = cost(theta, X_day_temp, y)

    

    return theta,cost_bike_day





# Gradient Descent with Different Learning Rates



num_iters = 3000

learning_rates = [0.000008, 0.000005, 0.00001]

for lr in learning_rates:

    _, cost_bike_day = gradientDescent(X_day_temp, Y, 0, lr, num_iters)

    plt.plot(cost_bike_day, linewidth = 2)

plt.title("Gradient descent with different learning rates", fontsize = 16)

plt.xlabel("no of iterations for cost function of bike_day", fontsize = 14)

plt.ylabel("cost", fontsize = 14)

plt.legend(list(map(str, learning_rates)))
#Now, performing multiple linear regression

X = df_bike_sharing_day.drop(['season','atemp','instant','dteday','casual','registered','cnt','hum','windspeed'], axis = 1)

Y = df_bike_sharing_day[['cnt']]
#Now, splitting the data into train/test data

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state =42)
# Initialize LR model

lr = LinearRegression()



#Fitting the model

lr.fit(X_train, y_train)



#finding intercept (B0)

lr.intercept_



#finding the coefficcient parameter (B1)

lr.coef_



#Make predictions 

predictions_lr = lr.predict(X_test)
#Now validation of model



r2 = format(r2_score(y_test, predictions_lr),'.3f')

rmse = format(np.sqrt(mean_squared_error(y_test, predictions_lr)), '.3f') #Here we specify 3 digits of precision and f is used to represent floating point number.

mae = format(mean_absolute_error(y_test, predictions_lr),'.3f')

std_lr = format(mean_absolute_error(y_test, predictions_lr),'.3f')





#printing out the result 

print("R squared Score:", r2)

print("Root Mean Squared Error:", rmse)

print("Mean Absolute Error:", mae)
result_1 =  pd.DataFrame({'Model':['Multiple'], 'R Squared': [r2], 'RMSE': [rmse], 'MAE':[mae]})

result_1
# Retrieved from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74



gcv = GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': [80,90,100] ,'n_estimators': [10, 50, 100, 1000]},cv=5, scoring='neg_mean_squared_error', verbose=0,n_jobs=1)

model_gcv = gcv.fit(X_train,y_train)

gcv_param = model_gcv.best_params_

rfr = RandomForestRegressor(max_depth = gcv_param["max_depth"], n_estimators= gcv_param["n_estimators"],random_state= 1, verbose=False)

cv_neg_mse = cross_val_score(rfr, X_train, y_train, cv=2, n_jobs= 1, scoring='neg_mean_squared_error')

mean_neg_mse = -1.0*np.mean(cv_neg_mse)

Root_mean_neg_mse = np.sqrt(mean_neg_mse)

cv_std = np.std(cv_neg_mse)



rfr = RandomForestRegressor(max_depth = gcv_param["max_depth"], n_estimators= gcv_param["n_estimators"],random_state= 1, verbose=False)

rfr = rfr.fit(X_train,y_train)

predictions_rfr = rfr.predict(X_test)
# validating resutls via R2, mse and mae. Further, collecting features that were considered important by the model.



R_sq_rfr = r2_score(y_test,predictions_rfr)

Mean_sq_rfr = mean_squared_error(y_test,predictions_rfr)

Mean_abs_rfr = mean_absolute_error(y_test,predictions_rfr)



imp      = rfr.feature_importances_

feat_col = X_train.columns

feat_imp = pd.DataFrame({'feature':feat_col, 'importance':imp })

feat_imp_sorted = feat_imp.sort_values(by='importance', ascending=False)
feat_imp_sorted
# plotting date time feature with respect to cnt

fig, ax = plt.subplots()

ax.scatter(df_bike_sharing_day.dteday,df_bike_sharing_day.cnt)

ax.set_xlabel('datet')

ax.set_ylabel('count of bikes ')

plt.show()
# Retrieved from https://scikit-learn.org/0.18/auto_examples/plot_cv_predict.html

# plotting predicted values from linear model with y test

fig, ax = plt.subplots()

ax.scatter(y_test, predictions_lr)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted as per linear model')

plt.show()
# plotting predicted values from Random Forest Regressor model with y test



fig, ax = plt.subplots()

ax.scatter(y_test, predictions_rfr)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted as per Random Forest Regressor')

plt.show()