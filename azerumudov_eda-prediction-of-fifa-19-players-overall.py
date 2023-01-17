import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings  

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/data.csv')

data.dtypes
data = data.drop(['ID','Unnamed: 0','Photo','Potential','Preferred Foot','Real Face','Loaned From','Joined','Jersey Number','Contract Valid Until','Release Clause','Flag','Nationality','Club','Club Logo','Wage','Value','Special'],axis=1)

data = data.drop(data.iloc[:,11:37].columns,axis=1)

data.dtypes
data.iloc[:,0:20].head(10)
data.iloc[:,20:40].head(10)
data.iloc[:,40:45].head(10)
data.info()
data.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean',missing_values = np.nan)



data.iloc[:,3:4] = imputer.fit_transform(data.iloc[:,3:4])

data.iloc[:,4:5] = imputer.fit_transform(data.iloc[:,4:5])

data.iloc[:,5:6] = imputer.fit_transform(data.iloc[:,5:6])

data.iloc[:,11:12] = imputer.fit_transform(data.iloc[:,11:12])

data.iloc[:,12:13] = imputer.fit_transform(data.iloc[:,12:13])

data.iloc[:,13:14] = imputer.fit_transform(data.iloc[:,13:14])

data.iloc[:,14:15] = imputer.fit_transform(data.iloc[:,14:15])

data.iloc[:,15:16] = imputer.fit_transform(data.iloc[:,15:16])

data.iloc[:,16:17] = imputer.fit_transform(data.iloc[:,16:17])

data.iloc[:,17:18] = imputer.fit_transform(data.iloc[:,17:18])

data.iloc[:,18:19] = imputer.fit_transform(data.iloc[:,18:19])

data.iloc[:,19:20] = imputer.fit_transform(data.iloc[:,19:20])

data.iloc[:,20:21] = imputer.fit_transform(data.iloc[:,20:21])

data.iloc[:,21:22] = imputer.fit_transform(data.iloc[:,21:22])

data.iloc[:,22:23] = imputer.fit_transform(data.iloc[:,22:23])

data.iloc[:,23:24] = imputer.fit_transform(data.iloc[:,23:24])

data.iloc[:,24:25] = imputer.fit_transform(data.iloc[:,24:25])

data.iloc[:,25:26] = imputer.fit_transform(data.iloc[:,25:26])

data.iloc[:,26:27] = imputer.fit_transform(data.iloc[:,26:27])

data.iloc[:,27:28] = imputer.fit_transform(data.iloc[:,27:28])

data.iloc[:,28:29] = imputer.fit_transform(data.iloc[:,28:29])

data.iloc[:,29:30] = imputer.fit_transform(data.iloc[:,29:30])

data.iloc[:,30:31] = imputer.fit_transform(data.iloc[:,30:31])

data.iloc[:,31:32] = imputer.fit_transform(data.iloc[:,31:32])

data.iloc[:,32:33] = imputer.fit_transform(data.iloc[:,32:33])

data.iloc[:,33:34] = imputer.fit_transform(data.iloc[:,33:34])

data.iloc[:,34:35] = imputer.fit_transform(data.iloc[:,34:35])

data.iloc[:,35:36] = imputer.fit_transform(data.iloc[:,35:36])

data.iloc[:,36:37] = imputer.fit_transform(data.iloc[:,36:37])

data.iloc[:,37:38] = imputer.fit_transform(data.iloc[:,37:38])

data.iloc[:,38:39] = imputer.fit_transform(data.iloc[:,38:39])

data.iloc[:,39:40] = imputer.fit_transform(data.iloc[:,39:40])

data.iloc[:,40:41] = imputer.fit_transform(data.iloc[:,40:41])

data.iloc[:,41:42] = imputer.fit_transform(data.iloc[:,41:42])

data.iloc[:,42:43] = imputer.fit_transform(data.iloc[:,42:43])

data.iloc[:,43:44] = imputer.fit_transform(data.iloc[:,43:44])

data.iloc[:,44:45] = imputer.fit_transform(data.iloc[:,44:45])
data.isnull().sum()
sns.catplot(y='Body Type',data=data,kind='count')
data['Body Type'].replace(['Messi','C. Ronaldo','Neymar','Courtois','PLAYER_BODY_TYPE_25','Shaqiri','Akinfenwa'],['Lean','Lean','Lean','Normal','Normal','Stocky','Stocky'],inplace=True)

sns.catplot(y='Body Type',data=data,kind='count')
data = data.dropna(subset=['Height'])

categorical_imputer = SimpleImputer(strategy='constant',missing_values=np.nan)

data.iloc[:,8:9] = categorical_imputer.fit_transform(data.iloc[:,8:9])
data.isnull().sum()
plt.figure(figsize=(20,12))

sns.heatmap(data.corr(),annot=True)
data = data.drop(['Position','SlidingTackle','StandingTackle','BallControl','Balance','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'],axis=1)
sns.catplot(y='Work Rate',x='Overall', data=data,kind='box')
height_cm = []

for value in data['Height'].values :

    feet = float(value.split('\'')[0])

    inch = float(value.split('\'')[1])

    cm = feet * 30.48 + inch * 2.54

    height_cm.append(round(cm, 2))

    

data['Height_class'] = data['Height']

data.drop(['Height'],axis=1)

data['Height'] = height_cm
data['Height_class'][data['Height'] <= 173] ='Short'

data['Height_class'][(data['Height'] > 173) & (data['Height'] <= 186)] ='Middle'

data['Height_class'][data['Height'] > 186] ='Tall'

data = data.drop(['Height'],axis=1)
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.countplot(x='Height_class',data=data,ax=ax[0])

sns.stripplot(x='Height_class',y='Overall',data=data,jitter=0.4,ax=ax[1])
import re

weight_kg = []

for value in data['Weight'].values:

    lbs = float(re.findall(r'\d+',value)[0])

    kg = lbs * 0.45

    weight_kg.append(round(kg, 2))

 

data['Weight_class'] = data['Weight']

data.drop(['Weight'],axis=1)

data['Weight'] = weight_kg
data['Weight_class'][data['Weight'] <= 67] ='Thin'

data['Weight_class'][(data['Weight'] > 67) & (data['Weight'] <= 81)] ='Normal'

data['Weight_class'][data['Weight'] > 81] ='Fat'

data = data.drop(['Weight'],axis=1)
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.boxenplot(x='Weight_class',y='Overall',data=data,ax=ax[0])

sns.countplot('Weight_class',data=data,palette="ch:.25",ax=ax[1])
sns.distplot(data['Age'])
sns.relplot(x='Age',y='Overall',size='International Reputation',sizes=(20,200) ,col='Body Type',data=data)
sns.relplot(x='International Reputation',y='Overall',data=data,kind='line')
work_rates = pd.get_dummies(data['Work Rate'])

body_types = pd.get_dummies(data['Body Type'])

height_class = pd.get_dummies(data['Height_class'])

weight_class = pd.get_dummies(data['Weight_class'])
data = data.drop(['Work Rate','Body Type','Height_class','Weight_class'],axis=1)

data = pd.concat([data,work_rates],axis=1)

data = pd.concat([data,body_types],axis=1)

data = pd.concat([data,height_class],axis=1)

data = pd.concat([data,weight_class],axis=1)
plt.figure(figsize=(20,14))

sns.heatmap(data.corr(),annot=True)
data.columns

data = data.drop(['Name','Weak Foot','Acceleration','SprintSpeed','Agility','Jumping','Marking','High/ High','High/ Low','High/ Medium','Low/ High','Low/ Low','Low/ Medium','Medium/ High','Medium/ Low','Medium/ Medium','Lean','Normal','Stocky','Middle','Short','Tall','Fat','Normal','Thin'],axis=1)

plt.figure(figsize=(20,8))

sns.heatmap(data.corr(),annot=True)
from sklearn.model_selection import train_test_split

Y = data.iloc[:,1:2]

X = pd.concat([data.iloc[:,0:1],data.iloc[:,2:]],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
import statsmodels.regression.linear_model as sm

X_train = np.append(arr = np.ones((12166,1)).astype(int), values = X_train, axis = 1)

X_train_opt = X_train[:,[0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,6,7, 8,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,6,7,8,10,12,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,6,7,8,10,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,6,7,8,13,14,15,16,17,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,6,7,8,13,14,15,16,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train_opt = X_train[:,[0,1,2,3,4,5,7,8,13,14,15,16,18,19,20,21,22,23]]

regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()

regressor_OLS.summary()
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,[0,1,2,3,4,6,7,12,13,14,15,17,18,19,20,21,22]], Y, test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVR,SVR

from sklearn.kernel_approximation import Nystroem

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV
lr = LinearRegression()

lr.fit(X_train,y_train)

linear_predict = lr.predict(X_test)

linear_mae = mean_absolute_error(y_test,linear_predict)

linear_mse = mean_squared_error(y_test,linear_predict)

linear_r2 = r2_score(y_test,linear_predict)

print("Mean absolute error : ", linear_mae)

print("Mean squared error : ", linear_mse)

print("R2 score : ", linear_r2)
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.fit_transform(X_test)

poly_reg = LinearRegression()

poly_reg.fit(X_train_poly,y_train)

poly_predict = poly_reg.predict(X_test_poly)

poly_mae = mean_absolute_error(y_test,poly_predict)

poly_mse = mean_squared_error(y_test,poly_predict)

poly_r2 = r2_score(y_test,poly_predict)

print("Mean absolute error : ", poly_mae)

print("Mean squared error : ", poly_mse)

print("R2 score : ", poly_r2)
dtree = DecisionTreeRegressor(random_state=0)

dtree.fit(X_train,y_train)

dtree_predict = dtree.predict(X_test)

dtree_mae = mean_absolute_error(y_test,dtree_predict)

dtree_mse = mean_squared_error(y_test,dtree_predict)

dtree_r2 = r2_score(y_test,dtree_predict)

print("Mean absolute error : ", dtree_mae)

print("Mean squared error : ", dtree_mse)

print("R2 score : ", dtree_r2)
rf = RandomForestRegressor(n_estimators=10,random_state=0)

rf.fit(X_train,y_train)

rf_predict = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test,rf_predict)

rf_mse = mean_squared_error(y_test,rf_predict)

rf_r2 =  r2_score(y_test,rf_predict)

print("Mean absolute error : ", rf_mae)

print("Mean squared error : ", rf_mse)

print("R2 score : ", rf_r2)
svr_linear = LinearSVR()

svr_linear.fit(X_train,y_train)

svr_linear_predict = svr_linear.predict(X_test)

svr_linear_mae = mean_absolute_error(y_test,svr_linear_predict)

svr_linear_mse = mean_squared_error(y_test,svr_linear_predict)

svr_linear_r2 = r2_score(y_test,svr_linear_predict)

print("Mean absolute error : ",svr_linear_mae)

print("Mean squared error : ", svr_linear_mse)

print("R2 score : ", svr_linear_r2)
r2_scores = [linear_r2,poly_r2,dtree_r2,rf_r2,svr_linear_r2]

mean_absolute_errors = [linear_mae,poly_mae,dtree_mae,rf_mae,svr_linear_mae]

mean_squared_errors = [linear_mse,poly_mse,dtree_mse,rf_mse,svr_linear_mse]

models = ['Linear','Polynomial','Decision Tree','Random Forest','Linear SVR']

model_accuracy = pd.DataFrame({'model' : models,

                               'R2 score' : r2_scores,

                              'Mean absolute error' : mean_absolute_errors,

                               'Mean squared error' : mean_squared_errors})

model_accuracy
f,ax = plt.subplots(1,2,figsize=(18,6))

plt.subplots_adjust(bottom=0.15, wspace=0.3)

g = sns.lineplot(x=model_accuracy.index,y='R2 score',data=model_accuracy,ax=ax[0])

ax[0].set_yticks(np.arange(0.75,0.95,0.01))

ax[0].set_xticks(model_accuracy.index)

ax[0].set_xticklabels(model_accuracy.model)

sns.barplot(x='Mean absolute error',y='model',data=model_accuracy,ax=ax[1])

plt.xticks(np.arange(0,5,0.5))

sns.catplot(x='Mean squared error',y='model',data=model_accuracy,kind='bar')
def PolynomialRegression(degree=2, **kwargs):

        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
params = [{'polynomialfeatures__degree': np.arange(3), 'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]}]

gscv = GridSearchCV(estimator=PolynomialRegression(),param_grid=params,scoring='r2',cv=5)

success = gscv.fit(X_train,y_train)

print(success.best_score_)

print(success.best_params_)
poly_last = PolynomialFeatures(degree=2)

X_train_poly = poly_last.fit_transform(X_train)

X_test_poly = poly_last.fit_transform(X_test)

last_model = LinearRegression(fit_intercept=False,normalize=True)

last_model.fit(X_train_poly,y_train)

last_model_predict = last_model.predict(X_test_poly)
predict = pd.DataFrame(last_model_predict)

predict.columns = ['Overall']

for i,value in predict.stack().iteritems():

    predict.loc[i[0],'Overall'] = int(round(value))

predict

f,ax = plt.subplots(1,2,figsize=(18,8))

ax[0].grid(True)

ax[1].grid(True)

sns.scatterplot(x=y_test.index,y='Overall',data=y_test,ax=ax[0])

sns.scatterplot(x=y_test.index, y='Overall',data=predict,ax=ax[1])