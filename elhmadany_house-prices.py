# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output
#import using library
from sklearn.ensemble import RandomForestRegressor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
from scipy import stats 
from scipy.stats import norm, skew ,zscore#for some statistics
import matplotlib.pyplot as plt  # Matlab-style plotting
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#load our data train and test (test dat does not include the target feature)
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape,test.shape

#let know more info about features data
train.info()
#let's know the target feature by applying XOR Function between train and test
test.columns^train.columns

#let's select the numeric features 
numerc_fet=train.select_dtypes(include=np.number)

numerc_fet.head()

corr=numerc_fet.corr()
corr['SalePrice'].sort_values(ascending=False)[:9]

sns.heatmap(corr)
#Check if there any outliers
sns.boxplot(x=train['OverallQual'])
sns.boxplot(x=train['GarageCars'])
sns.boxplot(x=train['TotRmsAbvGrd'])
y=train['SalePrice']

six_cols = ['GrLivArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'LotArea', 'SalePrice']
sns.pairplot(train[six_cols]) 
plt.show()
totalData=pd.concat([train,test])

#Create Feature TotalSF
totalData['TotalSF'] = totalData['TotalBsmtSF'] + totalData['1stFlrSF'] + totalData['2ndFlrSF']
x=len(y)
train_fea=totalData.iloc[:x,:]

numerc_fet2=train_fea.select_dtypes(include=np.number)
#Measure the correlation between those numeric feaures regarding to the target
corr2=numerc_fet2.corr()
corr2['SalePrice'].sort_values(ascending=False)[:9]
totalData.shape
#Xtrain=totalData.drop('SalePrice',axis=1)
ytrain=train['SalePrice']
#Compute the missing data
missing=totalData.isnull().sum().sort_values(ascending=False)
missing=missing[missing>0]
missing
#Dealing With Missing data
#for catergorical variables, we replece missing data with None
Miss_cat=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 
          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in Miss_cat:
    totalData[col].fillna('None',inplace=True)
# for numerical variables, we replace missing value with 0
Miss_num=['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
          'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'] 
for col in Miss_num:
    totalData[col].fillna(0, inplace=True)
rest_val=['MSZoning','Functional','Utilities','Exterior1st', 'SaleType','Electrical', 'Exterior2nd','KitchenQual']
for col in rest_val:
    totalData[col].fillna(totalData[col].mode()[0],inplace=True)  #fill with most frequency data
totalData['LotFrontage']=totalData.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))#fill with median
totalData=totalData.drop('Id',axis=1)  #not important feature

totalData.head()
#convert the numeric values into string becuse there are many repetition 
totalData['YrSold'] = totalData['YrSold'].astype(str)
totalData['MoSold'] = totalData['MoSold'].astype(str)
totalData['MSSubClass'] = totalData['MSSubClass'].astype(str)
totalData['OverallCond'] = totalData['OverallCond'].astype(str)
totalData.head()
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(totalData[c].values)) 
    totalData[c] = lbl.transform(list(totalData[c].values))
# shape        
print('Shape totalData: {}'.format(totalData.shape))
totalData.head()
numeric_feats = totalData.dtypes[totalData.dtypes != "object"].index
string_feats=totalData.dtypes[totalData.dtypes == "object"].index
string_feats
#dealing with string_feats
dumies = pd.get_dummies(totalData[string_feats])
print(dumies.shape)
totalData=pd.concat([totalData,dumies],axis='columns')

totalData.shape

totalData=totalData.drop(string_feats,axis=1)

totalData.shape

#Dealing with out liers
len(totalData)   # number of rows befor remove the outliers
x=len(ytrain)

train_feature=totalData.iloc[:x,:]
test_feature=totalData.iloc[x:,:]
train_feature.head()
#Here we will not apply scalling for data features because there are many str features and by compare the result with and without scalling

#sc_X = MinMaxScaler()
#all_data_train_normalized = sc_X.fit_transform(train_feature)
#all_data_test_normalized = sc_X.transform(test_feature)
all_data_train_normalized=train_feature
all_data_test_normalized=test_feature
all_data_train_normalized.head()
sns.distplot(ytrain , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(ytrain)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(ytrain, plot=plt)
plt.show()
ytrain.skew()

ytrain.head()
ytrain=np.log(ytrain)

ytrain.skew()
ytrain=pd.DataFrame(ytrain)

sns.distplot(ytrain , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(ytrain)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
ytrain.head()

len(ytrain),len(all_data_train_normalized)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(all_data_train_normalized,ytrain,test_size=0.2,random_state=42)
model1= LinearRegression()
model1.fit(X_train,Y_train)
ypre1=model1.predict(X_test)

mean = mean_squared_error(y_pred=ypre1,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre1,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre1,y_true=Y_test)
print(mean,r2_scor,absloute)
model1.score(X_test,Y_test)
#predicting on the test set
predictions = model1.predict(X_test)
actual_values = Y_test
plt.scatter(predictions, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('Linear Regression Model')
plt.show()
#Try more Models

# Test Options and Evaluation Metrics
num_folds = 5
scoring = "neg_mean_squared_error"
# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('RFR', RandomForestRegressor()))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=0)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,    scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),   cv_results.std())
    print(msg)

RFR=RandomForestRegressor()
RFR.fit(X_train,Y_train)
ypreRFR=RFR.predict(X_test)

mean = mean_squared_error(y_pred=ypreRFR,y_true=Y_test)
r2_scor = r2_score(y_pred=ypreRFR,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypreRFR,y_true=Y_test)
print(mean,r2_scor,absloute)
RFR.score(X_test,Y_test)

from sklearn import ensemble
# Fit regression model
params = {'n_estimators': 1000, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model2 = ensemble.GradientBoostingRegressor(**params)

model2.fit(X_train, Y_train)
ypre2=model2.predict(X_test)

mean = mean_squared_error(y_pred=ypre2,y_true=Y_test)
r2_scor = r2_score(y_pred=ypre2,y_true=Y_test)
absloute = mean_absolute_error(y_pred=ypre2,y_true=Y_test)
print(mean,r2_scor,absloute)
model2.score(X_test,Y_test)
predictions2 = model2.predict(X_test)
actual_values = Y_test
plt.scatter(predictions2, actual_values, alpha= 0.75, color = 'b')

plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.title('gradient boosting Model')
plt.show()

for i in range(-2, 3):
    alpha = 10**i
    rm = Ridge(alpha = alpha)
    ridge_model = rm.fit(X_train, Y_train)
    preds_ridge = ridge_model.predict(X_test)
    
    plt.scatter(preds_ridge,Y_test, alpha= 0.75, c= 'b')
    plt.xlabel('Predicted price')
    plt.ylabel('Actual price')
    plt.title('Ridge redularization with alpha {}'.format(alpha))
    overlay = 'R square: {} \nMSE: {}'.format(ridge_model.score(X_test, Y_test), mean_squared_error(Y_test, preds_ridge))
    plt.annotate(s = overlay, xy = (12.1, 10.6), size = 'x-large')
    plt.show()

alphas = np.linspace(0.0002, 100, num=50)
scores = [
     np.sqrt(-cross_val_score(Ridge(alpha), X_train,Y_train, 
       scoring="neg_mean_squared_error")).mean()
     for alpha in alphas
]
scores = pd.Series(scores, index=alphas)
scores.plot(title = "Alphas vs error (Lowest error is best)")
rm = Ridge(alpha = 18)
ridge_model = rm.fit(X_train, Y_train)
preds_ridge = ridge_model.predict(X_test)
mean = mean_squared_error(y_pred=preds_ridge,y_true=Y_test)
r2_scor = r2_score(y_pred=preds_ridge,y_true=Y_test)
absloute = mean_absolute_error(y_pred=preds_ridge,y_true=Y_test)
print(mean,r2_scor,absloute)
model2.get_params

from keras.models import Sequential

from keras.layers import Dense
#Build MLP Model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(.2))   # Add Dropout layer
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(.2)) # Add another Dropout layer
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


from keras.callbacks import ModelCheckpoint

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
MLPR_model=NN_model.fit(X_train,Y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

test_loss = NN_model.evaluate(X_test, Y_test)
test_loss[0]

