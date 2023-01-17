import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
%matplotlib inline
train = pd.read_csv('../input/Melbourne_housing_FULL.csv')
train.head()
train=train.dropna()
train.head()
plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['Price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['Price'], plot=plt)
plt.show()
train['Price'] = np.log1p(train['Price'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(train['Price'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['Price'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['Price'], plot=plt)
plt.show()
train.columns[train.isnull().any()]
train.isnull().sum()
train.dtypes
corr = train.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)
top_feature = corr.index[abs(corr['Price']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
train.Bedroom2.unique()
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['Price'], ascending=False, inplace=True)
corr.Price
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull())
train.columns
cols=('Suburb', 'Address', 'Type', 'Method', 'SellerG',
       'Date', 'Regionname', 'Propertycount')
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))
train.shape


#Take targate variable into y
y = train['Price']

train.head()
del train['CouncilArea']
del train['Price']
train.head()
X = train.values
y = y.values
print(y)
# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
#linear Regression
#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()
#Fit the model
model.fit(X_train, y_train)
#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))
y_predict = model.predict(X_test)
Actual_Price=y_test
out = pd.DataFrame({'Actual_Price': Actual_Price, 'predict_Price': y_predict,'Diff' :(Actual_Price-y_predict)})
out[['Actual_Price','predict_Price','Diff']].head(10)
print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
#Fit
model.fit(X_train, y_train)
#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)
#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
#Fit
GBR.fit(X_train, y_train)
print("Accuracy --> ", GBR.score(X_test, y_test)*100)
from sklearn.preprocessing import MinMaxScaler
dataset = MinMaxScaler().fit_transform(X)
X_trainn, X_testt, y_trainn, y_testt = train_test_split(dataset, y, test_size=0.3, random_state=40)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


reg2 = DecisionTreeRegressor()
reg3 = ExtraTreesRegressor()
reg4 = XGBRegressor()
reg5 = SVR()


reg2.fit( X_trainn,y_trainn )
reg3.fit( X_trainn,y_trainn )
reg4.fit( X_trainn,y_trainn )
reg5.fit( X_trainn,y_trainn )

label2 = reg2.predict( X_testt )
label3 = reg3.predict( X_testt )
label4 = reg4.predict( X_testt )
label5 = reg5.predict( X_testt )
# compare the loss of different models

from sklearn.metrics import mean_squared_error
print(label2)
print( 'the loss of DecisionTreeRegressor is ',mean_squared_error(y_testt,label2) )
print( 'the loss of ExtraTreesRegressor is ',mean_squared_error(y_testt,label3) )
print( 'the loss of XGBRegressor is ',mean_squared_error(y_testt,label4) )
print( 'the loss of SVR is ',mean_squared_error(y_testt,label5) )

print( '==='*10 )
# to compare the r^2 value of different regression models
# to chech the percentage of explained samples

from sklearn.metrics import r2_score
print( 'the r2 of DecisionTreeRegressor is ',r2_score(y_testt,label2) )
print( 'the r2 of ExtraTreesRegressor is ',r2_score(y_testt,label3) )
print( 'the r2 of XGBRegressor is ',r2_score(y_testt,label4) )
print( 'the r2 of SVR is ',r2_score(y_testt,label5) )

print( '++'*10 )
print( 'aparently, ExtraTreeRegressor performs the best in all the linear models with the loss presenting the least' )
print('---'*10)
#Score/Accuracy
print("Accuracy --> ", reg2.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg3.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg4.score(X_testt, y_testt)*100)
print("Accuracy --> ", reg5.score(X_testt, y_testt)*100)