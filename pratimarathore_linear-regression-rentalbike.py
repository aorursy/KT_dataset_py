# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
#Importing dataset 
bike = pd.read_csv("day.csv")
bike.head()
#Checking whether if there is any missing value.
bike.isnull().sum()
bike = bike.drop(['instant','dteday','casual','registered'],axis = 1)
g=sns.catplot(x = 'season', y = 'cnt', hue = 'weathersit', data = bike , kind='box',height=5, aspect=2)
g.set(xticklabels=['spring', 'summer','fall','winter'])
g=sns.catplot(x = 'weekday', y = 'cnt', hue = 'holiday', data = bike,kind='violin',height=5, aspect=2)
g.set(xticklabels=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
g=sns.catplot(x="mnth", y="cnt",
                hue="workingday", col="yr",
                data=bike, kind="bar",palette=['pink','purple'],edgecolor=(0,0,0),
                  linewidth=2, height=10, aspect=1.2)
sns.set(rc={"font.style":"normal",
            'axes.labelsize':20,
            'xtick.labelsize':20,
            'font.size':20,
            'ytick.labelsize':20})

plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.scatterplot(x = 'temp', y = 'cnt', hue='holiday' ,data = bike, palette=['green','red'])
plt.subplot(2,3,2)
sns.scatterplot(x = 'hum', y = 'cnt', hue='holiday',data = bike, palette=['brown','dodgerblue'])
plt.subplot(2,3,3)
sns.scatterplot(x = 'windspeed', y = 'cnt',hue='holiday', data = bike)

sns.pairplot(bike)
plt.show()
plt.figure(figsize = (20, 10))
sns.heatmap(bike.corr(), annot = True, cmap="BuPu")
plt.show()
#Storing the original data 
original_data=bike[['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed','cnt']]
original_data.head()

print(bike.columns)
bike.dtypes
bike['season']= bike['season'].astype(object)
#bike['yr']= bike['yr'].astype(object)
bike['mnth']= bike['mnth'].astype(object)
#bike['holiday']= bike['holiday'].astype(object)
bike['weekday']= bike['weekday'].astype(object)
#bike['workingday']= bike['workingday'].astype(object)
bike['weathersit']= bike['weathersit'].astype(object)
bike.dtypes

X=bike[['season', 'mnth', 'weekday','weathersit']]

# Let's drop the first column to remove redundancy
cat_vars = pd.get_dummies(X,drop_first=True )

# Drop the columns that as we have created the dummies for 
bike.drop(['season', 'mnth', 'weekday','weathersit'], axis = 1, inplace = True)

bike = pd.concat([cat_vars, bike], axis = 1)
bike.columns
# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(bike, train_size = 0.7, test_size = 0.3, random_state = 100)

#Min-Max scaling
scaler = MinMaxScaler()

# Apply scaler() to all the columns except the 'dummy' variables.
num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
y_train = df_train.pop('cnt')
X_train = df_train
cor = original_data.corr()
#Correlation with output variable
cor_target = abs(cor['cnt'])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>abs(0.2) ]
relevant_features.sort_values(ascending=False)
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)            
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Columns supported by RFE
col = X_train.columns[rfe.support_]
col
# Columns not supported by RFE
X_train.columns[~rfe.support_]
# Creating X_train dataframe with RFE selected variables
X_train_rfe = X_train[col]
#Renaming the columns for better understanding
X_train_rfe.rename(columns = {'season_2':'summer', 'season_3':'fall', 
                              'season_4':'winter','mnth_8':'August','mnth_9':'September','mnth_10':'October',
                            'weekday_6':'Saturday','weathersit_2':'Mist','weathersit_3':'Light_Snow'
                             }, inplace = True) 
# Adding a constant variable  
X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model
lm= sm.OLS(y_train,X_train_rfe).fit()   
#Let's see the summary of our linear model
print(lm.summary())

#Dropping constant to check the VIF values
X=X_train_rfe.drop(['const'],axis = 1)

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_rfe.drop(['fall'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new = sm.add_constant(X_train_new)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new).fit()  

#Let's see the summary of our linear model
print(lm.summary())
#Dropping constant to check the VIF values
X=X_train_new.drop('const', 1)

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new2 = X_train_new.drop(['hum'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new2 = sm.add_constant(X_train_new2)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new2).fit()  

#Let's see the summary of our linear model
print(lm.summary())
#Dropping constant to check the VIF values
X = X_train_new2.drop(['const'],axis = 1)

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new3 = X_train_new2.drop(['temp'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new3 = sm.add_constant(X_train_new3)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new3).fit()  

#Let's see the summary of our linear model
print(lm.summary())
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new2 = sm.add_constant(X_train_new2)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new2).fit()  

#Let's see the summary of our linear model
print(lm.summary())
#Dropping constant to check the VIF values
X = X_train_new2.drop(['const'],axis = 1)

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new4 = X_train_new2.drop(['holiday'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new4 = sm.add_constant(X_train_new4)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new4).fit()  

#Let's see the summary of our linear model
print(lm.summary())
#Dropping constant to check the VIF values
X = X_train_new4.drop(['const'],axis = 1)

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#Correlation with output variable
A=round(X_train_new4.corrwith(y_train, axis = 0),3).sort_values(ascending = False)
A
#Saturday seems to be least correlated. Lets try removing that and compare with the model 
X_train_new5 = X_train_new4.drop(['Saturday'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new5 = sm.add_constant(X_train_new5)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new5).fit()  

#Let's see the summary of our linear model
print(lm.summary())
vif = pd.DataFrame()
X = X_train_new5.drop(['const'],axis = 1)
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new6 = X_train_new5.drop(['winter'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new6 = sm.add_constant(X_train_new6)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new6).fit()  

#Let's see the summary of our linear model
print(lm.summary())
# Lets try removing "October" that and compare with the model 
X_train_new7 = X_train_new5.drop(['October'],axis = 1)

# Adding a constant variable 
import statsmodels.api as sm  
X_train_new7 = sm.add_constant(X_train_new7)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new7).fit()  

#Let's see the summary of our linear model
print(lm.summary())
vif = pd.DataFrame()
X = X_train_new7.drop(['const'],axis = 1)
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new8 = X_train_new7.drop(['workingday'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new8 = sm.add_constant(X_train_new8)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new8).fit()  

#Let's see the summary of our linear model
print(lm.summary())
vif = pd.DataFrame()
X = X_train_new8.drop(['const'],axis = 1)
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new9 = X_train_new8.drop(['windspeed'],axis = 1)
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new9 = sm.add_constant(X_train_new9)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new9).fit()  

#Let's see the summary of our linear model
print(lm.summary())
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new8 = sm.add_constant(X_train_new8)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new8).fit()  

#Let's see the summary of our linear model
print(lm.summary())
vif = pd.DataFrame()
X = X_train_new8.drop(['const'],axis = 1)
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_cnt = lm.predict(X_train_new8)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
sns.scatterplot(x = 'temp', y = 'cnt', hue='holiday' ,data = bike, palette=['green','red'])
plt.subplot(2,2,2)
sns.scatterplot(x = 'windspeed', y = 'cnt',hue='holiday', data = bike)

vif = pd.DataFrame()
X = X_train_new8
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
res= (y_train - y_train_cnt)
fig= sm.qqplot(res,fit=True)
fig.suptitle(' Q-Q Plot of error terms', fontsize = 20)   
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle(' histogram of the error terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
# Adding a constant variable 
import statsmodels.api as sm  
X_train_new8 = sm.add_constant(X_train_new8)

 # Running the linear model
lm = sm.OLS(y_train,X_train_new8).fit()  

#Let's see the summary of our linear model
print(lm.summary())
# Plotting Fitted vs Residual Plot to understand Homoscedasticity

fig = plt.figure()
plt.scatter(y_train,y_train - y_train_cnt )
fig.suptitle('Fitted vs Residual Plot', fontsize = 20)              
plt.xlabel('Fitted value', fontsize = 18)                          
plt.ylabel('Residual', fontsize = 16)   
num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
y_test = df_test.pop('cnt')
X_test = df_test
X_test.rename(columns = {'season_2':'summer', 'season_3':'fall', 
                              'season_4':'winter','mnth_8':'August','mnth_9':'September','mnth_10':'October',
                            'weekday_6':'Saturday','weathersit_2':'Mist','weathersit_3':'Light_Snow'
                             }, inplace = True) 
# Now let's use our model to make predictions.

X_train_new8=X_train_new8.drop(['const'],axis = 1)

X_test_new = X_test[X_train_new8.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

 # Running the linear model
lm = sm.OLS(y_test,X_test_new).fit()  

#Let's see the summary of our linear model
print(lm.summary())
# Making predictions
y_pred = lm.predict(X_test_new)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
# Plotting y_test and y_pred 
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
