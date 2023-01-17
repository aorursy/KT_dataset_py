import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/house-price/innercity.csv')

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
data.head()
data.info()
print('No of Rows: ',data.shape[0],'\nNo of columns: ',data.shape[1]) 

data.describe()
data.isnull().sum()
# Lets check the details of data 

def details(dataFrame):

    df = pd.DataFrame()

    df['Null_Values'] = dataFrame.isnull().sum()

    df['Data Type'] = dataFrame.dtypes

    df['Unique Values'] = dataFrame.nunique()

    return df

details(data)
lst = []

for i in data.lat:

    if i<47.255900:

        lst.append('ES')

    elif i>47.255900 and i<47.405900:

        lst.append('MS')

    elif i>47.405900 and i<47.555900:

        lst.append('MN')

    else:

        lst.append('EN')

data['SN_region'] = lst
lst = []

for i in abs(data.long):

    if i<122.105000:

        lst.append('EE')

    elif i>122.105000 and i<122.205000:

        lst.append('ME')

    elif i>122.205000 and i<122.328000:

        lst.append('MW')

    else:

        lst.append('EW')

data['EW_region'] = lst
sns.countplot(data.SN_region)

plt.title('South and North Region')

plt.show()

sns.countplot(data.EW_region)

plt.title('East and West Region')

plt.show()
sns.barplot(data.EW_region,data.price)

plt.title('East-West Region and Price of House')

plt.savefig('EWRegion.jpg')

plt.show()

sns.barplot(data.SN_region,data.price)

plt.title('South-North Region and Price of House')

plt.savefig('NSRegion.jpg')

plt.show()

# Skewness and Kurtosis

skw = data.skew();sk1 = skw[skw>3]

krt = data.kurt();kr = krt[krt>3]

sk = pd.DataFrame({'Skewness':sk1,'Kurtosis': kr})

sk
df=data.copy()
df['yr_sold'] = df['dayhours'].str.extract('(\d\d\d\d)',expand=True)

df['yr_sold'] = df['yr_sold'].astype(int)



df = df.drop('dayhours',axis=1)
# Checking total quality

df['Total_home_quality'] = df.quality + df.condition

df['Total_home_quality']  = df['Total_home_quality'].astype(int)
# Taking total area for 2015 remesurement

df['total_area15'] = df.living_measure15 + df.lot_measure15
# How much house is old

df['Age_house'] = df['yr_sold'] - df['yr_built']
# Age column has negative values so we are simply droping that row

df = df.drop(df[df['Age_house']==-1].index)

df[df['Age_house']==-1]
# Fetch all numeric features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32',

'float64']

numeric = []

for i in df.columns:

    if df[i].dtype in numeric_dtypes:

        numeric.append(i)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# Create new column for renovation done or not

lst = []

for i in df.yr_renovated:

    if  i >0:        

        

        lst.append(1)

    else:

        

        lst.append(0)



df['is_renovated']=lst
# House has basement or not

df['Have_basement'] = df['basement'].apply(lambda x: 1 if x>1 else 0)
df.room_bed = df.room_bed.replace(to_replace=33, value=df.room_bed.mode()[0])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

#from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

import os
df.head()
df.info()
#df5 = df5.drop(['SN_region','EW_region'],axis=1)
#df_dum = pd.get_dummies(df)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['SN_region'] = label_encoder.fit_transform(df['SN_region']).astype('float64')

df['EW_region'] = label_encoder.fit_transform(df['EW_region']).astype('float64')
df = df.drop('cid',axis=1)
corr = df.corr()

corr.head()
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False
selected_columns = df.columns[columns]

selected_columns.shape
data1 = df[selected_columns]
data1.head()
import statsmodels.api as sm

def backwardElimination(x, Y, sl, columns):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(endog=Y, exog= x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j, 1)

                    columns = np.delete(columns, j)

                    

    regressor_OLS.summary()

    return x, columns



SL = 0.05

data_modeled, selected_columns1 = backwardElimination(data1.iloc[:,1:].values, data1.iloc[:,0].values, SL, selected_columns)
selected_columns.size

df.shape
data1.iloc[:,0].head()
data1.head()
result = pd.DataFrame()

result['diagnosis'] = data1.iloc[:,0]

result.head()
data2 = data1[selected_columns1]

print(data2.shape)

data2.head()
data2.columns
df1 = data2.copy()
df1.info()
# Import module to Scale down the data into central scale

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x = df1.drop(['price'],axis=1)

y=df1.price
import statsmodels.api as sm

X = sm.add_constant(x)

model = sm.OLS(y,X).fit()

mod_pred = model.predict(X)

residual = model.resid

model.summary()
from   statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()



vif['VIF Factor'] =[variance_inflation_factor(df1.values,i) for i in range(df1.shape[1])]

vif['Feature'] = df1.columns

X1 =pd.DataFrame( ss.fit_transform(x),columns=x.columns)

X1.head()
#Fitting the PCA algorithm with our Data

pca = PCA().fit(X1)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('House Price Dataset Explained Variance')

plt.savefig('PCA')

plt.show()
pca=PCA(0.95)



pca.fit_transform(X1)



compo=pca.n_components_

compo
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(X1,y, test_size=0.3, random_state  = 14)



print('Training Data Shape:', xtrain1.shape)

print('Testing Data Shape:', xtest1.shape)

model_pca = PCA(n_components=compo,svd_solver='full')



new_train = model_pca.fit_transform(xtrain1)

new_test  = model_pca.fit_transform(xtest1)



print('\nTraining model with {} dimensions.'.format(new_train.shape[1]))



# Baseline is mean of training label 

baseline = np.mean(ytrain1)

base_error = np.mean(abs(baseline - ytest1))



print('Baseline Error: {:0.4f}.'.format(base_error))

# create object of model

model_new = LinearRegression()



# fit the model with the training data

model_new.fit(new_train,ytrain1)

# predict the target on the new train dataset

predict_train_pca = model_new.predict(new_train)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain1,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %model_new.score(new_train, ytrain1))



# predict the target on the new test dataset

predict_test_pca = model_new.predict(new_test)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest1,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %model_new.score(new_test, ytest1))
from sklearn.model_selection import cross_val_score

scoresdt = cross_val_score(model_new, new_train, ytrain1, cv=10)

scoresdt
np.mean(scoresdt)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA

import pandas as pd

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')



sc=StandardScaler()

pca=PCA()

poly=PolynomialFeatures(degree=1)

X=df1.drop('price',axis=1)

y=df1.price

trainR2=[]

testR2=[]

trainR2PCA=[]

testR2PCA=[]

for i in range(100):

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size= .20,random_state=i)

    Xtrain=poly.fit_transform(Xtrain)

    Xtest=poly.transform(Xtest)

    scaledXtrain=sc.fit_transform(Xtrain)

    scaledXtest=sc.transform(Xtest)

    pcascaledXtrain=pca.fit_transform(scaledXtrain)

    pcascaledXtest=pca.transform(scaledXtest)

    lr=LinearRegression()

    lr.fit(scaledXtrain,ytrain)

    lrpca=LinearRegression()

    lrpca.fit(pcascaledXtrain,ytrain)

    trainR2.append(lr.score(scaledXtrain,ytrain))

    testR2.append(lr.score(scaledXtest,ytest))

    trainR2PCA.append(lrpca.score(pcascaledXtrain,ytrain))

    testR2PCA.append(lrpca.score(pcascaledXtest,ytest))

print("Without PCA")

print("Testing R2")

print(np.mean(testR2))

print("Training R2")

print(np.mean(trainR2))

print("")

print("With PCA")

print("Testing R2")

print(np.mean(testR2PCA))

print("Training R2")

print(np.mean(trainR2PCA))


sns.set(style='whitegrid')

# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

plt.title('RandomForestRegressor')

plt.savefig('RandomForestRegressor.jpg')

gbr = GradientBoostingRegressor(n_estimators=6000,

                                                    learning_rate=0.01,

                                                    max_depth=4,

                                                    max_features='sqrt',

                                                    min_samples_leaf=15,

                                                    min_samples_split=10,

                                                    loss='huber',

                                                    random_state=42)



gbr.fit(pcascaledXtrain,ytrain)







# predict the target on the new train dataset

predict_train_pca = gbr.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %gbr.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = gbr.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %gbr.score(pcascaledXtest, ytest))
from sklearn.model_selection import cross_val_score

scoresdt = cross_val_score(gbr, pcascaledXtrain, ytrain, cv=10)

scoresdt
np.mean(scoresdt)



sns.set(style='whitegrid')

# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

plt.title('RandomForestRegressor')

plt.savefig('RandomForestRegressor.jpg')
# plot between predictions and Y_test

x_axis = np.array(range(0, predict_test_pca.shape[0]))

plt.figure(figsize=(20,10))

plt.plot(x_axis, predict_test_pca, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")

plt.plot(x_axis, ytest, linestyle="--", marker="o", alpha=0.7, color='g', label=" Actual(Y_test)")

plt.xlabel('Row number')

plt.ylabel('PRICE')

plt.title('Predictions vs Actual(Y_test)')

plt.legend(loc='lower right')

plt.savefig('RandomFor.jpg')
rf_reg = RandomForestRegressor(n_estimators=200, n_jobs=-1)

rf_reg.fit(pcascaledXtrain,ytrain)







# predict the target on the new train dataset

predict_train_pca = rf_reg.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %rf_reg.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = rf_reg.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %rf_reg.score(pcascaledXtest, ytest))
from sklearn.model_selection import cross_val_score

scoresdt = cross_val_score(rf_reg, pcascaledXtrain, ytrain, cv=10)

scoresdt
np.mean(scoresdt)
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

plt.title('RandomForestRegressor')

plt.savefig('RandomForestRegressor.jpg')

# plot between predictions and Y_test

x_axis = np.array(range(0, predict_test_pca.shape[0]))

plt.figure(figsize=(20,10))

plt.plot(x_axis, predict_test_pca, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")

plt.plot(x_axis, ytest, linestyle="--", marker="o", alpha=0.7, color='g', label=" Actual(Y_test)")

plt.xlabel('Row number')

plt.ylabel('PRICE')

plt.title('Predictions vs Actual(Y_test)')

plt.legend(loc='lower right')

plt.savefig('RandomFor.jpg')
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeRegressor



# Fitting the decision tree with default hyperparameters, apart from

# max_depth which is 5 so that we can plot and read the tree.

dt_default = DecisionTreeRegressor(max_depth=5)

dt_default.fit(pcascaledXtrain,ytrain)





# predict the target on the new train dataset

predict_train_pca = dt_default.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %dt_default.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = dt_default.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %dt_default.score(pcascaledXtest, ytest))
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

#plt.title('DecisionTreeRegressor')

plt.savefig('DecisionTreeRegressor.jpg')
# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                                    learning_rate=0.01,

                                                    max_depth=4,

                                                    max_features='sqrt',

                                                    min_samples_leaf=15,

                                                    min_samples_split=10,

                                                    loss='huber',

                                                    random_state=42)



gbr.fit(new_train,ytrain1)







# predict the target on the new train dataset

predict_train_pca = gbr.predict(new_train)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain1,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %gbr.score(new_train, ytrain1))



# predict the target on the new test dataset

predict_test_pca = gbr.predict(new_test)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest1,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %gbr.score(new_test, ytest1))
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

#plt.title('DecisionTreeRegressor')

plt.savefig('GradientBoostingRegressor.jpg')
# plot between predictions and Y_test

#x_axis = np.array(range(0, rf_reg_predict.shape[0]))

#plt.figure(figsize=(20,10))

#plt.plot(x_axis, rf_reg_pred, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")

#plt.plot(x_axis, ytest1, linestyle="--", marker="o", alpha=0.7, color='g', label=" Actual(Y_test)")

#plt.xlabel('Row number')

#plt.ylabel('PRICE')

#plt.title('Predictions vs Actual(Y_test)')

#plt.legend(loc='lower right')

#plt.savefig('GradientBoostingRegressor1.jpg')
# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor()



lightgbm.fit(pcascaledXtrain,ytrain)







# predict the target on the new train dataset

predict_train_pca = lightgbm.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %lightgbm.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = lightgbm.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %lightgbm.score(pcascaledXtest, ytest))
# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                                            n_estimators=6000,

                                            max_depth=4,

                                            min_child_weight=0,

                                            gamma=0.6,

                                            subsample=0.7,

                                            colsample_bytree=0.7,

                                            objective='reg:linear',

                                            nthread=-1,

                                            scale_pos_weight=1,

                                            seed=27,

                                            reg_alpha=0.00006,

                                            random_state=42)



xgboost.fit(pcascaledXtrain,ytrain)







# predict the target on the new train dataset

predict_train_pca = xgboost.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %xgboost.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = xgboost.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %xgboost.score(pcascaledXtest, ytest))
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

#plt.title('DecisionTreeRegressor')

plt.savefig('DecisionTreeRegressor.jpg')
!pip install tpot
from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=2, population_size=50, verbosity=2)

tpot.fit(pcascaledXtrain,ytrain)

print(tpot.score(pcascaledXtest,ytest))
#predict the target on the new train dataset

predict_train_pca = tpot.predict(pcascaledXtrain)



# Accuray Score on train dataset

rmse_train_pca = mean_squared_error(ytrain,predict_train_pca)**(0.5)

print('\nRMSE on new train dataset : ', rmse_train_pca)



print('R square is %1.3f' %tpot.score(pcascaledXtrain, ytrain))



# predict the target on the new test dataset

predict_test_pca = tpot.predict(pcascaledXtest)



# Accuracy Score on test dataset

rmse_test_pca = mean_squared_error(ytest,predict_test_pca)**(0.5)

print('\nRMSE on new test dataset : ', rmse_test_pca)



print('R square is %1.3f' %tpot.score(pcascaledXtest, ytest))
# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize

figsize(14, 8)



# Plot predictions

ax = plt.subplot(121)

ax.hist(predict_test_pca, bins = 100)

ax.set_xlabel('Score'); ax.set_ylabel('Count'); ax.set_title('Predicted Distribution')



# Plot true values

ax2 = plt.subplot(122)

ax2.hist(ytest, bins = 100)

ax2.set_xlabel('Score'); ax2.set_ylabel('Count'); ax2.set_title('Actual Distribution');

#plt.title('DecisionTreeRegressor')

plt.savefig('Genetic Algo.jpg')
# plot between predictions and Y_test

x_axis = np.array(range(0, predict_test_pca.shape[0]))

plt.figure(figsize=(20,10))

plt.plot(x_axis, predict_test_pca, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")

plt.plot(x_axis, ytest, linestyle="--", marker="o", alpha=0.7, color='g', label=" Actual(Y_test)")

plt.xlabel('Row number')

plt.ylabel('PRICE')

plt.title('Predictions vs Actual(Y_test)')

plt.legend(loc='lower right')

plt.savefig('Genetic Algo.jpg')
plt.style.use('fivethirtyeight')

figsize(8, 6)



# Dataframe to hold the results

model_comparison = pd.DataFrame({'model': ['Linear Regression', 'PCA LR',

                                           'Gradient Boosted','Random Forest', 

                                            'Decision Tree','Light GBM','XG Boost'],

                                 'mae': [lr, lrpca, gbr

                                        ,rf_reg,dt_default,lightgbm,xgboost]})



# Horizontal bar chart of test mae

model_comparison.sort_values('mae', ascending = False).plot(x = 'model', y = 'mae', kind = 'barh',

                                                           color = 'red', edgecolor = 'black')
# Loss function to be optimized

loss = ['ls', 'lad', 'huber']



# Number of trees used in the boosting process

n_estimators = [100, 500, 900, 1100, 1500]



# Maximum depth of each tree

max_depth = [2, 3, 5, 10, 15]



# Minimum number of samples per leaf

min_samples_leaf = [1, 2, 4, 6, 8]



# Minimum number of samples to split a node

min_samples_split = [2, 4, 6, 10]



# Maximum number of features to consider for making splits

max_features = ['auto', 'sqrt', 'log2', None]



# Define the grid of hyperparameters to search

hyperparameter_grid = {'loss': loss,

                       'n_estimators': n_estimators,

                       'max_depth': max_depth,

                       'min_samples_leaf': min_samples_leaf,

                       'min_samples_split': min_samples_split,

                       'max_features': max_features}
import sklearn
# Create the model to use for hyperparameter tuning

model = GradientBoostingRegressor(random_state = 42)



# Set up the random search with 4-fold cross validation

random_cv =  sklearn.model_selection.RandomizedSearchCV(estimator=model,

                               param_distributions=hyperparameter_grid,

                               cv=4, n_iter=25, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)
# Fit on the training data

random_cv.fit(pcascaledXtrain, ytrain)
# Get all of the cv results and sort by the test performance

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)



random_results.head(10)
final_model=random_cv.best_estimator_

final_model
%%timeit -n 1 -r 5

final_model.fit(pcascaledXtrain,ytrain)
from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score
final_pred = final_model.predict(pcascaledXtest)

print('Final model performance on the test set:   MAE = %0.4f.' % mae(ytest, final_pred))
print('\nr2_score for DecisionTree: ',r2_score(ytest,final_pred))