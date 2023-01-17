import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df= pd.read_csv("day.csv")
df.head()
df.shape
df.info()
df.describe()
df.info()
# Dropping Date ,casual,registered and instant columns as instant is indexing and date is not significant as we have month and years
# and cnt variable is addition of causal and registered
df.drop(['dteday','instant','casual','registered'], axis = 1, inplace = True)
df.head()
sns.pairplot(df, x_vars=['temp','atemp','hum','windspeed'], y_vars='cnt')
plt.show()
# Converting Season,month and Weathersit Categorical Variables into string
df['season']= df['season'].map({1 : 'spring', 2 :'summer', 3 :'fall', 4: 'winter'})
df['weathersit']= df['weathersit'].map({1 : 'Clear', 2 :'Mist', 3 :'Light Snow', 4:'Heavy Rain'})
# df['mnth'] = df['mnth'].map({1:'January', 2:'February', 3:'March',4:'April',5:'May', 6:'June',7:'July',8:'August',9:'September',10:'October',
#                               11:'November', 12:'December'})
# df['weekday'] = df['weekday'].map({0:'Sunday', 1:'Monday', 2:'Tuesday',3:'Wednesday',4:'Thursday', 5:'Friday',6:'Saturday'})
# df.head()
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
sns.boxplot(x = 'season', y = 'cnt', data = df)

plt.subplot(2,2,2)
sns.boxplot(x = 'weekday', y = 'cnt', data = df)
plt.subplot(2,2,3)
sns.boxplot(x = 'mnth', y = 'cnt', data = df)
plt.xticks(rotation=90)
plt.subplot(2,2,4)
plt.xticks(rotation=90)
sns.boxplot(x = 'weathersit', y = 'cnt', data = df)
plt.show()
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1)
plt.title('weather situation')
sns.boxplot(x = 'yr', y = 'cnt',hue = 'weathersit', data = df)

plt.subplot(2,2,2)
plt.title('season')
sns.boxplot(x = 'yr', y = 'cnt', hue = 'season',data = df)

plt.subplot(2,2,3)
plt.title('month')
sns.boxplot(x = 'yr', y = 'cnt',hue = 'mnth', data = df)

plt.subplot(2,2,4)
plt.title('weekday')
sns.boxplot(x = 'yr', y = 'cnt',hue = 'weekday', data = df)
plt.show()

df.corr()
# Removing atemp variable as atemp and temp are having high corelation
df.drop(['atemp'], axis = 1, inplace = True)
df.head()
 # Creating Dummy Variables from categorical variable
new_mnth=pd.get_dummies(df['mnth'],drop_first=True)
new_mnth=new_mnth.rename(columns={1:'jan', 2:'feb', 3:'mar', 4:'apr',
                                  5:'may', 6:'jun', 7:'jul', 8:'aug',
                                  9:'sep', 10:'oct', 11:'nov', 12:'dec'})
new_mnth
new_season = pd.get_dummies(df['season'],drop_first= True)
new_season
new_week = pd.get_dummies(df['weekday'],drop_first= True)
new_week=new_week.rename(columns={0:'sunday', 1:'monday', 2:'tuesday', 3:'wednesday',
                                  4:'thursday', 5:'friday', 6:'saturday'})
new_week
new_weather = pd.get_dummies(df['weathersit'],drop_first= True)
new_weather
df = pd.concat([df, new_mnth, new_season, new_week, new_weather], axis = 1)
df.drop(['mnth','season','weekday','weathersit'], axis = 1, inplace = True)
df.head()
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars =['temp','hum','windspeed']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
df_train.corr()
y_train = df_train.pop('cnt')
X_train = df_train
# Using RFE 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm,15)
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col= X_train.columns[rfe.support_]
col
X_train = X_train[col]
X_train_rfe = X_train[col]
import statsmodels.api as sm
X_train_rfe= sm.add_constant(X_train_rfe)
X_train_lm = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_lm).fit()
print(lm.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables and insignificant variables
# As hum is having infinite VIF dropping it
X_train_New =X_train.drop(['hum'],1)
X_train_lm = sm.add_constant(X_train_New)
lm = sm.OLS(y_train,X_train_lm).fit()
print(lm.summary())

vif = pd.DataFrame()
vif['Features'] = X_train_New.columns
vif['VIF'] = [variance_inflation_factor(X_train_New.values, i) for i in range(X_train_New.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_New1 =X_train_New.drop(['aug'],1)
X_train_lm1 = sm.add_constant(X_train_New1)
lm1 = sm.OLS(y_train,X_train_lm1).fit()
print(lm1.summary())
vif = pd.DataFrame()
vif['Features'] = X_train_New1.columns
vif['VIF'] = [variance_inflation_factor(X_train_New1.values, i) for i in range(X_train_New1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_New2 =X_train_New1.drop(['jun'],1)
X_train_lm2 = sm.add_constant(X_train_New2)
lm2 = sm.OLS(y_train,X_train_lm2).fit()
print(lm2.summary())
vif = pd.DataFrame()
vif['Features'] = X_train_New2.columns
vif['VIF'] = [variance_inflation_factor(X_train_New2.values, i) for i in range(X_train_New2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_New3 =X_train_New2.drop(['apr'],1)
X_train_lm3 = sm.add_constant(X_train_New3)
lm3 = sm.OLS(y_train,X_train_lm3).fit()
print(lm3.summary())
vif = pd.DataFrame()
vif['Features'] = X_train_New3.columns
vif['VIF'] = [variance_inflation_factor(X_train_New3.values, i) for i in range(X_train_New3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_New4 =X_train_New3.drop(['may'],1)
X_train_lm4 = sm.add_constant(X_train_New4)
lm4 = sm.OLS(y_train,X_train_lm4).fit()
print(lm4.summary())
vif = pd.DataFrame()
vif['Features'] = X_train_New4.columns
vif['VIF'] = [variance_inflation_factor(X_train_New4.values, i) for i in range(X_train_New4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Final Model
X_train_New5 =X_train_New4.drop(['mar'],1)
X_train_lm5 = sm.add_constant(X_train_New5)
lm_model = sm.OLS(y_train,X_train_lm5).fit()
print(lm_model.summary())
vif = pd.DataFrame()
vif['Features'] = X_train_New5.columns
vif['VIF'] = [variance_inflation_factor(X_train_New5.values, i) for i in range(X_train_New5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_pred = lm_model.predict(X_train_lm5)
res = y_train - y_train_pred
sns.distplot(res)
plt.title('Residula plot')
plt.show()
vif = pd.DataFrame()
vif['Features'] = X_train_New5.columns
vif['VIF'] = [variance_inflation_factor(X_train_New5.values, i) for i in range(X_train_New5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(X_train_New5.corr(), cmap="YlGnBu", annot = True, ax= ax)

plt.show()

#X_train_New5.corr()
# Validating Homoscedasticity
sns.scatterplot(y_train,res)
plt.title('Residual vs predicted Value', fontsize= 18)
plt.xlabel('y_train', fontsize = 16)                        
plt.ylabel('res', fontsize = 16) 
# Scaling on Test data
num_vars =['temp','hum','windspeed']
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.head()
df_test.describe()
y_test = df_test.pop('cnt')
X_test = df_test
# add constant to Test data
X_test_lm = sm.add_constant(X_test)
X_test_lm.head()
# Drop Variables all the variables not available in our model
X_test_lm = X_test_lm[['const','yr','holiday','temp','windspeed','sep','spring','winter','Light Snow','Mist']]

X_test_lm.head()
# ADjusted R2 for test data
test_lm_model = sm.OLS(y_test,X_test_lm).fit()
print(test_lm_model.summary())
# predict
y_test_pred = lm_model.predict(X_test_lm)
# Evaluation r2 score for Test data
from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
r_squared=r2_score(y_true= y_test,y_pred=y_test_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
# Evaluation r2 score for Train data
mse = mean_squared_error(y_train, y_train_pred)
r_squared=r2_score(y_true= y_train,y_pred=y_train_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)               
plt.xlabel('y_test', fontsize = 18)                           
plt.ylabel('y_test_pred', fontsize = 16)  