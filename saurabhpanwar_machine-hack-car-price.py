# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 

import datetime

import random

# data processing, CSV file I/O (e.g. pd.read_csv)



#importing visualization lib.

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import matplotlib.pyplot as plt

import plotly





#stas and model

from scipy import stats

from scipy.special import boxcox1p

from scipy.stats import norm,skew

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_excel('../input/Data_Train.xlsx')

test = pd.read_excel('../input/Data_Test.xlsx')



train.shape, test.shape
train.head()
sns.set_style("white")

sns.set_color_codes(palette="deep")

f,ax= plt.subplots(figsize = (8,7))

sns.distplot(train['Price'])

ax.set(xlabel='Price')

ax.set(ylabel='Frequency')

ax.set(title = 'Price Distribution')

plt.show()
print('Skewness = %f' %train['Price'].skew())

print('Kurtiosis = %f' %train['Price'].kurt())
train['Price'].describe()
log_train = np.log1p(train['Price'])

log_train.describe()
train.head()
fig,ax = plt.subplots()

ax.scatter(x = train['Kilometers_Driven'], y = train['Price'])

plt.ylabel('Price of the car')

plt.xlabel('Kilometers Driven')

plt.show()
#deleting outliers 

train = train.drop(train[(train['Kilometers_Driven']>5000000) & (train['Price']<100)].index)

train = train.drop(train[(train['Kilometers_Driven']<200000) & (train['Price']>120)].index)

fig,ax = plt.subplots()

ax.scatter(x = train['Kilometers_Driven'], y = train['Price'])

plt.ylabel('Price of the car')

plt.xlabel('Kilometers Driven')

plt.show()
fig,ax = plt.subplots()

ax.scatter(x = train['Year'], y = train['Price'])

plt.ylabel('Price of the car')

plt.xlabel('Kilometers Driven')

plt.show()
sns.distplot(train['Price'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['Price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['Price'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["Price"] = np.log1p(train["Price"])



#Check the new distribution 

sns.distplot(train['Price'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['Price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['Price'], plot=plt)

plt.show()
y_train = train['Price'].values

all_data= pd.concat((train,test)).reset_index(drop = True)

all_data.drop(['Price'], axis = 1, inplace = True)
all_data.shape
all_data_na = (all_data.isnull().sum()/len(all_data))*100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({"Missing ratio ": all_data_na})

missing_data.head(20)
f,ax = plt.subplots(figsize =(8,10))

plt.xticks(rotation = '90')

sns.barplot(x = all_data_na.index, y = all_data_na)

plt.xlabel("Features")

plt.ylabel("Missing percentages")
all_data.drop(['New_Price'], axis = 1, inplace = True)
all_data.head()
all_data.dtypes
all_data['Engine'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

all_data['Power'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

all_data['Mileage'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
all_data.head()
all_data['Fuel_Type'] = all_data['Fuel_Type'].apply(str)

all_data['Engine'] = pd.to_numeric(all_data['Engine'])

all_data['Mileage'] = pd.to_numeric(all_data['Mileage'])

all_data['Power'] = pd.to_numeric(all_data['Power'])

all_data['Seats'] = all_data['Seats'].apply(str)

all_data['Year'] = all_data['Year'].apply(str)

vlist = ['CNG', 'LPG']

CNG = all_data[all_data['Fuel_Type'].isin(vlist)]

CNG['Mileage'] = CNG['Mileage']*0.719

CNG.head()
all_data.update(CNG)

all_data.head()
all_data["Engine"] = all_data["Engine"].transform(

    lambda x: x.fillna(x.median()))

all_data["Power"] = all_data["Power"].transform(

    lambda x: x.fillna(x.median()))

all_data["Seats"] = all_data["Seats"].transform(

    lambda x: x.fillna(x.mode()))

all_data["Mileage"] = all_data["Mileage"].transform(

    lambda x: x.fillna(x.median()))

all_data_na = (all_data.isnull().sum()/len(all_data))*100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({"Missing ratio ": all_data_na})

missing_data.head(20)
from sklearn.preprocessing import LabelEncoder



cols = ('Owner_Type', 'Seats', 'Year', 'Transmission')

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(all_data[c].values))

    all_data[c] = lbl.transform(list(all_data[c].values))





all_data.drop(['Name'], axis = 1, inplace = True)

all_data.head()

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[skewness >2]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)
final_data = pd.get_dummies(data=all_data, columns=['Fuel_Type', 'Location'])

#final_data = pd.get_dummies(all_data)

final_data.head()
ntrain = train.shape[0]

ntest = test.shape[0]

train = final_data[:ntrain]

test = final_data[ntrain:]
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 10



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='linear', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(learning_rate=0.05, max_depth=3, 

                             n_estimators=3000

                            )
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.02, n_estimators=3000)
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, model_lgb,model_xgb))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.10 +

               xgb_train_pred*0.80 + lgb_train_pred*0.10 ))
ensemble = stacked_pred*0.15 + xgb_pred*0.70 + lgb_pred*0.15
sub = pd.DataFrame()

sub['Price'] = ensemble

sub.to_csv('submission.csv',index=False)