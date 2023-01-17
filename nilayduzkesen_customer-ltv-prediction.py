import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import seaborn as sns
import scipy.stats as stats
from scipy.stats.mstats import winsorize
from datetime import datetime
import json
from wordcloud import WordCloud

%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format

import warnings
warnings.filterwarnings(action="ignore")
df = pd.read_csv('Marketing-Customer-Value-Analysis.csv')
df.sort_values('Customer Lifetime Value')
df.info()
df.head().T
#lets edit date format
df['Effective To Date']= df['Effective To Date'].astype('datetime64[ns]')
df.describe()

df.isnull().sum()

# Looking at outliers of continuos variables

significant_cont = ['Income','Monthly Premium Auto','Total Claim Amount']

sns.set(color_codes=True)
plt.figure(figsize=(15,20))
plt.subplots_adjust(hspace=0.5)

for i in range(len(significant_cont)):
    plt.subplot(3,2,i+1)
    plt.boxplot(df[significant_cont[i]])
    plt.title(significant_cont[i])
    
plt.show()
#checking all categorical variables to determine significant ones.

cat_df = df.select_dtypes(include='object')
cat_df = cat_df.drop(['Customer'], axis = 1)
cols = cat_df.columns
cols
sns.set(color_codes=True)
plt.subplots_adjust(hspace=0.5)
plt.figure(figsize=(20,40))

for i in range(len(cols)):
    plt.subplot(7,2,i+1)
    sns.barplot(x = cols[i],y='Customer Lifetime Value',data = df)
    plt.title(cols[i])
    
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(15,6))
ax = sns.violinplot(x="Number of Policies", y="Customer Lifetime Value", data=df)

# Test whether Gender differences are significant or not.
gender = df[['Customer Lifetime Value','Gender']].groupby('Gender')
female = gender['Customer Lifetime Value'].get_group('F')
male = gender['Customer Lifetime Value'].get_group('M')
stats.ttest_ind(female,male)
# Test whether Covarage differences are significant or not.
Coverage = df[['Customer Lifetime Value','Coverage']].groupby('Coverage')
Basic = Coverage['Customer Lifetime Value'].get_group('Basic')
Extended = Coverage['Customer Lifetime Value'].get_group('Extended')
Premium =Coverage['Customer Lifetime Value'].get_group('Premium')
stats.f_oneway(Basic,Extended,Premium)
# Test whether Marital Status differences are significant or not.

Marital = df[['Customer Lifetime Value','Marital Status']].groupby('Marital Status')
married = Marital['Customer Lifetime Value'].get_group('Married')
single = Marital['Customer Lifetime Value'].get_group('Single')

stats.ttest_ind(married,single)
# Test whether Vehicle Class differences are significant or not.

Vehicleclass = df[['Customer Lifetime Value','Vehicle Class']].groupby('Vehicle Class')
fourdoor = Vehicleclass['Customer Lifetime Value'].get_group('Four-Door Car')
twodoor = Vehicleclass['Customer Lifetime Value'].get_group('Two-Door Car')
suv = Vehicleclass['Customer Lifetime Value'].get_group('SUV')
luxurysuv =Vehicleclass['Customer Lifetime Value'].get_group('Luxury SUV')
luxurycar =Vehicleclass['Customer Lifetime Value'].get_group('Luxury Car')
sportscar =Vehicleclass['Customer Lifetime Value'].get_group('Sports Car')


stats.f_oneway(fourdoor,twodoor,suv,luxurysuv,luxurycar,sportscar)
# Test whether Renew Offer Type differences are significant or not.

Renewoffer = df[['Customer Lifetime Value','Renew Offer Type']].groupby('Renew Offer Type')
offer1 = Renewoffer['Customer Lifetime Value'].get_group('Offer1')
offer2 = Renewoffer['Customer Lifetime Value'].get_group('Offer2')
offer3 = Renewoffer['Customer Lifetime Value'].get_group('Offer3')
offer4 =Renewoffer['Customer Lifetime Value'].get_group('Offer4')


stats.f_oneway(offer1,offer2,offer3,offer4)
# Test whether EmploymentStatus differences are significant or not.


EmploymentStatus = df[['Customer Lifetime Value','EmploymentStatus']].groupby('EmploymentStatus')
employed = EmploymentStatus['Customer Lifetime Value'].get_group('Employed')
unemployed = EmploymentStatus['Customer Lifetime Value'].get_group('Unemployed')
medleave = EmploymentStatus['Customer Lifetime Value'].get_group('Medical Leave')
disabled = EmploymentStatus['Customer Lifetime Value'].get_group('Disabled')
retired = EmploymentStatus['Customer Lifetime Value'].get_group('Retired')
stats.f_oneway(employed,unemployed,medleave,disabled,retired)

# Test whether Education differences are significant or not.

Education = df[['Customer Lifetime Value','Education']].groupby('Education')
bachelor = Education['Customer Lifetime Value'].get_group('Bachelor')
college = Education['Customer Lifetime Value'].get_group('College')
highschool = Education['Customer Lifetime Value'].get_group('High School or Below')
master = Education['Customer Lifetime Value'].get_group('Master')
doctor = Education['Customer Lifetime Value'].get_group('Doctor')
stats.f_oneway(bachelor,college,highschool,master,doctor)
df2 =df.copy()
df2.drop(['State','Coverage','Renew Offer Type','Vehicle Class','Customer','Response','Gender','Location Code','Vehicle Size','Policy','Policy Type','Sales Channel','Effective To Date'],axis=1,inplace = True)
df2['Number of Policies'] = np.where(df2['Number of Policies']>2,3,df2['Number of Policies'])
new = pd.get_dummies(df2,columns=['Marital Status','Number of Policies','Education','EmploymentStatus'],drop_first=True)
new

ax = sns.scatterplot(x="Income", y="Customer Lifetime Value", hue="State",
                     data=df)


maritalstts = sns.scatterplot(x="Income", y="Customer Lifetime Value", hue="EmploymentStatus",
                     data=df)


ax = sns.scatterplot(x="Total Claim Amount", y="Customer Lifetime Value", hue="Marital Status",
                     data=df)

import statsmodels.api as sm

y = new['Customer Lifetime Value']
x = new.drop('Customer Lifetime Value',axis=1)


x = sm.add_constant(x)
results = sm.OLS(y, x).fit()
results.summary()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(x_train.shape[0]))
print('Test Data Count: {}'.format(x_test.shape[0]))

x_train = sm.add_constant(x_train)
results = sm.OLS(y_train, x_train).fit()
results.summary()
# Model graph to see predictions


x_test = sm.add_constant(x_test)

y_preds = results.predict(x_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV")
plt.show()
#lets see their errors

print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
all_score = []

all_score.append((results.rsquared,
                  mean_absolute_error(y_test, y_preds),
                 mse(y_test, y_preds),rmse(y_test, y_preds),
                 np.mean(np.abs((y_test - y_preds) / y_test)) * 100))

#duplicate the original data and get the log version of it to be able to reach higher R2(with outliers)
df3 = new.copy()

df3['Monthly Premium Auto'] = np.log(df2['Monthly Premium Auto'])
df3['Total Claim Amount'] = np.log(df2['Total Claim Amount'])
y = np.log(df3['Customer Lifetime Value'])

import statsmodels.api as sm


x1 =  df3.drop('Customer Lifetime Value',axis=1)
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(x1_train.shape[0]))
print('Test Data Count: {}'.format(x1_test.shape[0]))

x1_train = sm.add_constant(x1_train)
results_log = sm.OLS(y_train, x1_train).fit()
results_log.summary()
# Model graph to see predictions


x1_test = sm.add_constant(x1_test)

y_preds = results_log.predict(x1_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV-Log Transformation with outliers")
plt.show()
print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
exp_ypreds = np.exp(y_preds)
exp_ytest = np.exp(y_test)


print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(exp_ytest, exp_ypreds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(exp_ytest, exp_ypreds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(exp_ytest, exp_ypreds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((exp_ytest - exp_ypreds) / exp_ytest)) * 100))
all_score.append((results.rsquared,
                  mean_absolute_error(exp_ytest, exp_ypreds),
                 mse(exp_ytest, exp_ypreds),rmse(exp_ytest, exp_ypreds),
                 np.mean(np.abs((exp_ytest - exp_ypreds) / exp_ytest)) * 100))
#duplicate the original data and winsorize the data at %5
df4 = new.copy()

df4['Monthly Premium Auto'] = winsorize(df4['Monthly Premium Auto'],(0, 0.05))
df4['Total Claim Amount'] = winsorize(df4['Total Claim Amount'],(0, 0.05))


y = df4['Customer Lifetime Value']
x3 =  df4.drop('Customer Lifetime Value',axis=1)

x3_train, x3_test, y_train, y_test = train_test_split(x3, y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(x3_train.shape[0]))
print('Test Data Count: {}'.format(x3_test.shape[0]))


x3_train = sm.add_constant(x3_train)
results_wins = sm.OLS(y_train, x3_train).fit()
results_wins.summary()
# Model graph to see predictions


x3_test = sm.add_constant(x3_test)

y_preds = results_wins.predict(x3_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV-5% Winsorize")
plt.show()
print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
all_score.append((results_wins.rsquared,
                  mean_absolute_error(y_test, y_preds),
                 mse(y_test, y_preds),rmse(y_test, y_preds),
                 np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
#duplicate the original data and take log of the data without outlier

df5 = df4.copy()


df5['Monthly Premium Auto'] = np.log(df5['Monthly Premium Auto'])
df5['Total Claim Amount'] = np.log(df5['Total Claim Amount'])


y = np.log(df5['Customer Lifetime Value'])
x7 =df5.drop('Customer Lifetime Value',axis=1)

x7_train, x7_test, y_train, y_test = train_test_split(x7, y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(x7_train.shape[0]))
print('Test Data Count: {}'.format(x7_test.shape[0]))


x7_train = sm.add_constant(x7_train)
results_logwins = sm.OLS(y_train, x7_train).fit()
results_logwins.summary()
# Model graph to see predictions


x7_test = sm.add_constant(x7_test)

y_preds = results_logwins.predict(x7_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV- Both Log Transformation & 5% Winsorize")
plt.show()
print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
exp_ypreds = np.exp(y_preds)
exp_ytest = np.exp(y_test)

all_score.append((results_logwins.rsquared,
                  mean_absolute_error(exp_ytest, exp_ypreds),
                 mse(exp_ytest, exp_ypreds),rmse(exp_ytest, exp_ypreds),
                 np.mean(np.abs((exp_ytest - exp_ypreds) / exp_ytest)) * 100))
#the best model is the one with log transformation and outliers included

#Let's use polynomial features to see if we can do better


from sklearn.preprocessing import PolynomialFeatures


y = np.log(df3['Customer Lifetime Value'])
x5 =df3.drop('Customer Lifetime Value',axis=1)


pol = PolynomialFeatures()


array = pol.fit_transform(x5)

df_pol = pd.DataFrame(array)
df_pol.columns = pol.get_feature_names(x5.columns)

df_pol_train, df_pol_test, y_train, y_test = train_test_split(df_pol, y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(df_pol_train.shape[0]))
print('Test Data Count: {}'.format(df_pol_test.shape[0]))

df_pol_train = sm.add_constant(df_pol_train)
results_pol = sm.OLS(y_train, df_pol_train).fit()
results_pol.summary()
# Model graph to see predictions


df_pol_test = sm.add_constant(df_pol_test)

y_preds = results_pol.predict(df_pol_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV-Polynomial Features")
plt.show()
# Model graph to see predictions


df_pol_test = sm.add_constant(df_pol_test)

y_preds = results_pol.predict(df_pol_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV-Polynomial Features")
plt.show()
print("Mean Absolute Error (MAE)     : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)    : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE)  : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
exp_ypreds = np.exp(y_preds)
exp_ytest = np.exp(y_test)

all_score.append((results_pol.rsquared,
                  mean_absolute_error(exp_ytest, exp_ypreds),
                 mse(exp_ytest, exp_ypreds),rmse(exp_ytest, exp_ypreds),
                 np.mean(np.abs((exp_ytest - exp_ypreds) / exp_ytest)) * 100))
# Model graph to see exponential version of predictions


df_pol_test = sm.add_constant(df_pol_test)

y_preds = np.exp(results_pol.predict(df_pol_test))
sns.set(color_codes=True)
plt.scatter(exp_ytest, y_preds)
plt.plot(exp_ytest, exp_ytest, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv", )
plt.title("Actual vs Estimated Customer LTV-Polynomial Features-Exp")
plt.show()

mse( y_test[y_test<10],y_preds[y_test<10])
significant_features = list(results_pol.pvalues[results_pol.pvalues <= 0.05].index)


df_sig_train, df_sig_test, y_train, y_test = train_test_split(df_pol[significant_features], y, test_size = 0.25, random_state = 450)

print('Train Data Count: {}'.format(df_sig_train.shape[0]))
print('Test Data Count: {}'.format(df_sig_test.shape[0]))

df_sig_train = sm.add_constant(df_sig_train)
results_sig = sm.OLS(y_train, df_sig_train).fit()
results_sig.summary()
# Model graph to see predictions


df_sig_test = sm.add_constant(df_sig_test)

y_preds = results_sig.predict(df_sig_test)
sns.set(color_codes=True)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Actual ltv")
plt.ylabel("Estimated ltv" )
plt.title("Actual vs Estimated Customer LTV-Polynomial Features with significant variables")
plt.show()
print("Mean Absolute Error (MAE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("Mean Sq. Error (MSE)          : {}".format(mse(y_test, y_preds)))
print("Root Mean Sq. Error (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("Mean Abs. Perc. Error (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
exp_ypreds = np.exp(y_preds)
exp_ytest = np.exp(y_test)

all_score.append((results_sig.rsquared,
                  mean_absolute_error(exp_ytest, exp_ypreds),
                 mse(exp_ytest, exp_ypreds),rmse(exp_ytest, exp_ypreds),
                 np.mean(np.abs((exp_ytest - exp_ypreds) / exp_ytest)) * 100))
df_allscore = pd.DataFrame(all_score)
df_allscore.index = ['Standard','Log with outliers','Without Outliers','Log without outliers',
                       'Polynomial Features',
                       'Polynomial with significant features']

df_allscore.columns = ['R2', 'MAE', 'MSE','RMSE','MAPE']


df_allscore
lrm = LinearRegression()
lrm.fit(df_pol_train, y_train)

y_train_predict = lrm.predict(df_pol_train)
y_test_predict = lrm.predict(df_pol_test)

print("Train observation number  : {}".format(df_pol_train.shape[0]))
print("Test observation number   : {}".format(df_pol_test.shape[0]), "\n")

print("Train R-Square  : {}".format(lrm.score(df_pol_train, y_train)))
print("-----Test Scores---")
print("Test R-Square   : {}".format(lrm.score(df_pol_test, y_test)))
print("Mean_absolute_error (MAE)             : {}".format(mean_absolute_error(y_test, y_test_predict)))
print("Mean squared error (MSE)              : {}".format(mse(y_test, y_test_predict)))
print("Root mean squared error(RMSE)         : {}".format(rmse(y_test, y_test_predict)))
print("Mean absolute percentage error (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_predict) / y_test)) * 100))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import PredictionError


# Create the train and test data
df_pol_train, df_pol_test, y_train, y_test = train_test_split(df_pol, y, test_size = 0.25, random_state = 450)

# Instantiate the linear model and visualizer
model = Lasso()
visualizer = PredictionError(model)

visualizer.fit(df_pol_train, y_train)  # Fit the training data to the visualizer
visualizer.score(df_pol_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot


# Instantiate the linear model and visualizer
Model = Ridge()
visualizer_residual = ResidualsPlot(Model)

visualizer_residual.fit(df_pol_train, y_train)  # Fit the training data to the visualizer
visualizer_residual.score(df_pol_test, y_test)  # Evaluate the model on the test data
visualizer_residual.show()                 # Finaliz