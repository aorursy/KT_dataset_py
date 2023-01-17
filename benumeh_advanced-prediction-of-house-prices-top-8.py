#Import necessary libraries

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from scipy.stats import norm, skew

from sklearn import preprocessing

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

from scipy import stats

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import scipy

import numpy

import json

import sys

import csv

import os
print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
sns.set(style='white', context='notebook', palette='deep')

pylab.rcParams['figure.figsize'] = 12,8

warnings.filterwarnings('ignore')

mpl.style.use('ggplot')

sns.set_style('white')

%matplotlib inline
# import the ames housing dataset



#train = pd.read_csv('https://raw.githubusercontent.com/hjhuney/Data/master/AmesHousing/train.csv')

#test= pd.read_csv('https://raw.githubusercontent.com/hjhuney/Data/master/AmesHousing/test.csv')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#preview the train dataset

train.head()
#preview the test dataset

test.head()
# Get the shape of the train and test data sets

train_shape = train.shape

print("The shape of the train dataset is {}".format(train_shape))



test_shape = test.shape

print("The shape of the test dataset is {}".format(test_shape))
#Save the "Id" columns of the train and test datasets separately

id_train = train[["Id"]]

id_test = test["Id"]
#Save the target feature column "SalePrice" separately, drop it from the train dataframe and Concatenate the new train and test data to get the full data in one dataframe

target_feat = train[['SalePrice']]

train_mod = train.drop(['SalePrice'], axis=1, inplace=False)

full_data = pd.concat([train_mod, test]).reset_index(drop=True)
#dfrop the "Id" column from the full_data and print the first five rows to confirm that it is in the right state

full_data.drop(['Id'], axis=1, inplace=True)

full_data.head()
#print the first five of the target feature to confirm that it was well saved

target_feat.head()
#Confirm that the concatenation was well done by printing the shape of the full_data

print("full_data shape is : {}".format(full_data.shape))
#Print the dataset column labels

full_data.columns
import pandas_profiling as pp



#Explore the data using pandas_profiling

profile = pp.ProfileReport(full_data)

profile
#Plot the distribution of the target feature "SalePrice" 

sns.distplot(target_feat , fit=norm);



# Get the distribution parameters

(mu, sigma) = norm.fit(target_feat)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Plot the distribution

plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice Distribution')
#Plot a heatmap to visualise the correlation among the top 10 most correlated feature 



k = 10 #number of variables for heatmap

corr_mat = train.corr()

cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index

crm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)



sns.heatmap(crm, cbar=True,annot=True,square=True, fmt='.2f', cmap='RdYlGn',linewidths=0.2,annot_kws={'size': 20}, yticklabels=cols.values, xticklabels=cols.values) 

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
#Get the 10 features that are least correlated with "SalesPrice"

train_id_less = train.drop(['Id'], axis=1, inplace=False) #Drop the Id column from the train dataset

corr_table = train_id_less.corr()[['SalePrice']]

corr_table = abs(corr_table)

corr_table.sort_values(by=["SalePrice"], ascending=False, inplace=True)

corr_table[-10:] 

#Generate a table showing the skewness of the features sorted in descending order and select the top 10 skewed features

numeric_feats = full_data.dtypes[full_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = full_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness_table = pd.DataFrame({'skewness' :skewed_feats})

skewness_table.head(10)
#Do a pairplot of the 10 most correlated features

sns.set()

sns.pairplot(train[cols], size = 2.5)

plt.show()

#Identify the duplicate rows

dub_full = full_data.duplicated()

dub_full = dub_full.loc[dub_full == True]

dub_full
# Let us isolate the individual scatter plots of the features with outliers (i.e GrLivArea, TotalBsmtSF and 1stFlrSF) 

#to explore the exact values of the outliers

out_feat = {"GrLivArea", "TotalBsmtSF", "1stFlrSF"}

fig.suptitle('Features with Outliers')

for feature in (out_feat):

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(x=train[feature],y=train["SalePrice"])#, ax=ax)#(x = train['GrLivArea'], y = train['SalePrice'])

    plt.ylabel('SalePrice', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    plt.show()
#Drop the outliers from the features in question



#make a copy of the train dataset

train_copy = train.copy()

for ft in out_feat:

    new_train = train_copy.drop(train_copy[(train_copy[ft]>4000) & (train_copy['SalePrice']<200000)].index)

    train_copy = new_train



#Plot new scatterplots for the old "train" and "new_train" datasets to confirm the dropping of the outlier observations

for feature in (out_feat):

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(x=new_train[feature],y=new_train["SalePrice"])#, ax=ax)#(x = train['GrLivArea'], y = train['SalePrice'])

    plt.ylabel('SalePrice', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    plt.show()

#Drop the "SalePrice" column from "new_train" dataset and save it as "new_SalePrice", concatenate "test" dataset to it and assign to "new_full_data". 

#Then drop the "Id" column from  "new_full_data".



new_SalePrice = new_train[['SalePrice']]

new_train_mod = new_train.drop(['SalePrice'], axis=1, inplace=False)

new_full_data = pd.concat([new_train_mod, test]).reset_index(drop=True)

new_full_data = new_full_data.drop(["Id"], axis=1, inplace=False)

new_full_data.head()

#Transform the "SalePrice" using np.log1p

new_SalePrice = np.log1p(new_SalePrice)



#Plot the new distribution to see if it's normal

sns.distplot(new_SalePrice, fit=norm)

# Get the fitted parameters used by the function

#(mu, sigma) = norm.fit(new_SalePrice)

#print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal Distn. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('Normalized SalePrice distribution')

fig = plt.figure()

plt.show()



print("The transformed SalePrice distribution is obviously much normal now")
#Now let us separate this list of features with missing values into numerical and categorical features to 

#enable us process them differently.



#First, we make lists of the various features types from the classification we did at the beginning of the Jupyter notebook

nom_cat = ["MSSubClass","MSZoning","Street","Alley","LotShape","Neighborhood","RoofStyle","RoofMatl","Exterior1st",

                "Exterior2nd","MasVnrType","Foundation","Heating","CentralAir","GarageType","MiscFeature","MoSold",

                "SaleType","SaleCondition"]



ord_cat = ["LandContour","Utilities","LotConfig","LandSlope","Condition1","Condition2","BldgType","HouseStyle",

            "OverallQual","OverallCond","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",

            "BsmtFinType2","HeatingQC","Electrical","Functional","FireplaceQu","GarageFinish","GarageQual","GarageCond",

            "PavedDrive","PoolQC","Fence","KitchenQual"]



discr_num = ["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",

            "GarageYrBlt","GarageCars","YrSold","YearBuilt","YearRemodAdd"]



cont_num = ["2ndFlrSF","LowQualFinSF","GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch",

            "ScreenPorch","PoolArea","MiscVal","LotFrontage","LotArea","MasVnrArea","BsmtFinSF1",

            "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF"]



#Then we combine the nominal and ordinal categorical features into a single list; 

#and the discrete and continuous numerical features into anoyther list

cat_feat = nom_cat + ord_cat

num_feat = discr_num + cont_num

#Handling missing values 

#Create list of features with missing values

feat_miss = new_full_data.columns[new_full_data.isnull().any()].tolist() 



#Now we separate the list of features with missing values into categorical and numeric features

miss_cat = []

miss_num = []

for fm in feat_miss:

    if fm in cat_feat:

        miss_cat.append(fm)

    else:

        miss_num.append(fm)

    

print("The numerical features with missing values are:{}".format(miss_num))

print()

print("The categorical features with missing values are:{}".format(miss_cat))
#Now let's impute "none" for the missing values of the categorical features

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value = "none")

imputer = imputer.fit(new_full_data[miss_cat])

new_full_data[miss_cat] = imputer.transform(new_full_data[miss_cat])



#Now let's impute 0 for the missing values of the numerical features

new_full_data[miss_num] = new_full_data[miss_num].fillna(0)

new_full_data.head()
#Let's get the list of the highly skewed features from the skewness_table generated earlier

skew_list = list(skewness_table.index.values)[:23]

 

#Box Cox Transform these skewed explanatory continuous numerical features

from scipy.special import boxcox1p



lam = 0.15

for snf in skew_list:

    new_full_data[snf] = boxcox1p(new_full_data[snf], lam)

new_full_data.head()
#initiallize dataframe of numerical features

num_df = new_full_data[num_feat]

#Normalize the numerical features in the new_full_data

new_full_data[num_feat] = (num_df-num_df.mean())/num_df.std()

new_full_data[num_feat].head()
#Get the lists of unique values of the ordinal categorical features 

#new_ord_cat = [x for x in ord_cat if x not in ["Fence", "MiscFeature", "Utilities"]]

uniq_obs = []

#for od in new_ord_cat:

for od in ord_cat:

    vals = list(sorted(new_full_data[od].unique()))

    uniq_obs.append(vals)

    

#Print the list of the lists of unique values of the ordinal categorical features 

print(uniq_obs)

len(uniq_obs)
from sklearn.preprocessing import OrdinalEncoder

#Generate the list of correctly ordered lists of unique ordinal features values in the sequence that they appear in "ord_cat" earlier defined

ordered_feat_value = [['Low','Lvl','Bnk', 'HLS'],['ELO','NoSeWa','NoSewr',"AllPub"],['Inside','Corner', 'CulDSac', 'FR2', 'FR3'],

                     ['Sev', 'Mod','Gtl'],['Artery', 'Feedr', 'Norm','RRNn','RRAn', 'PosN','PosA','RRNe','RRAe'],

                     ['Artery', 'Feedr', 'Norm','RRNn','RRAn', 'PosN','PosA','RRNe','RRAe'],['1Fam','2fmCon','Twnhs','TwnhsE','Duplex'],

                     ['1Story','1.5Unf','1.5Fin','SFoyer','SLvl','2Story','2.5Unf', '2.5Fin'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

                     [1, 2, 3, 4, 5, 6, 7, 8, 9],['Fa','TA','Gd','Ex'],['Po','Fa','TA','Gd','Ex'],['none','Fa','TA','Gd','Ex'],

                     ['none','Po','Fa', 'TA','Gd'],['none','No','Mn', 'Av', 'Gd'],['none','Unf','LwQ','Rec', 'BLQ','ALQ','GLQ'],

                     ['none','Unf','LwQ','Rec', 'BLQ','ALQ','GLQ'],['Po','Fa','TA','Gd','Ex'],['none','Mix','FuseP','FuseF','FuseA','SBrkr'],

                     ['none','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],['none','Po', 'Fa','TA','Gd','Ex'],['none','Unf','RFn','Fin'],

                     ['none','Po','Fa','TA','Gd','Ex'],['none','Po','Fa','TA','Gd','Ex'],['N', 'P', 'Y'],['none','Fa','Gd','Ex'],

                     ['none','MnWw','GdWo','MnPrv','GdPrv'],['none','Fa','TA','Gd','Ex']]



for ftr,ordv in zip(ord_cat,ordered_feat_value):

    cat = pd.Categorical(new_full_data[ftr],categories = ordv, ordered = True)

    labels, unique = pd.factorize(cat, sort=True)

    new_full_data[ftr] = labels
#We adopt the pd.get_dummies to transform the nominal categorical features, while inserting the drop_first=True arguement

#to handle the multicollinearity that may result

#new_nom_cat = [x for x in nom_cat if x not in ["Fence", "MiscFeature", "Utilities"]]

encoded_nom = pd.get_dummies(new_full_data[nom_cat],drop_first=True)



#Drop the former norminal categorical features from new_full_data and concatenate the encoded norminal categorical features

new_full_data = new_full_data.drop(nom_cat, axis=1, inplace=False)

new_full_data = pd.concat([new_full_data, encoded_nom], axis=1)

len(new_full_data.columns.values)
#Import necesaary libraries and modules

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LassoCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedKFold , KFold

from sklearn.metrics import make_scorer



#get the fully and finally processed training and test data

train_final = new_full_data[:len(new_train_mod)]

test_final = new_full_data[len(new_train_mod):]



#Now here's the model selection function

def mod_sel_cv(model):

    kf = KFold(n_splits=10, shuffle=True, random_state=25).get_n_splits(train_final.values)

    scorer = np.sqrt(-cross_val_score(model, train_final.values, new_SalePrice.values,scoring="neg_mean_squared_error", cv=kf))

    return scorer
#We use the sklearn's Robustscaler() method to make the model more robust on outliers

lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0002,max_iter=501,random_state=1))

score1 = mod_sel_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))
#We use the sklearn's Robustscaler() method to make the model more robust on outliers

KRR = make_pipeline(RobustScaler(),KernelRidge(alpha=0.1, kernel="polynomial", degree=1, coef0=4.5))

score2 = mod_sel_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))
#We use the sklearn's Robustscaler() method to make the model more robust on outliers

RFR = RandomForestRegressor(max_depth=8, random_state=0,n_estimators=2200)

score3 = mod_sel_cv(RFR)

print("Random Forest Regression score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))
xg_reg = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=3.5, n_estimators=2200,

                             reg_alpha=0.2500, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



score4 = mod_sel_cv(xg_reg)

print("XGBoost score: {:.4f} ({:.4f})\n".format(score4.mean(), score4.std()))
#We use the sklearn's Robustscaler() method to make the model more robust on outliers

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00025, l1_ratio= 0.90, random_state=1))

score5 = mod_sel_cv(ENet)

print("Elastic Net Regression score: {:.4f} ({:.4f})\n".format(score5.mean(), score5.std()))
#We use the huber loss to make the model more robust to outliers

GBR = GradientBoostingRegressor(n_estimators=2200, learning_rate=0.01,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', subsample=0.85, random_state =5)

score7 = mod_sel_cv(GBR)

print("Gradient Boosting Regression score: {:.4f} ({:.4f})\n".format(score7.mean(), score7.std()))
import lightgbm as lgb

ml_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,

                              learning_rate=0.05, n_estimators=500,

                              max_bin = 100, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 6)



score_lgb = mod_sel_cv(ml_lgb)

print("Light Gradient Boosting Regression score: {:.4f} ({:.4f})\n".format(score_lgb.mean(), score_lgb.std()))

# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt



# Keras specific

import keras

from keras.models import Sequential

from keras.layers import Dense





X = train_final.values #df[predictors].values

y = new_SalePrice.values 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

print(X_train.shape); print(X_test.shape)





# Define model

keras_model = Sequential()

keras_model.add(Dense(500, input_dim=178, activation= "relu"))

keras_model.add(Dense(100, activation= "relu"))

keras_model.add(Dense(100, activation= "relu"))

keras_model.add(Dense(50, activation= "relu"))

keras_model.add(Dense(1))

#model.summary() #Print model Summary



keras_model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

keras_model.fit(X_train, y_train, epochs=20)



pred_train= keras_model.predict(X_train)

print(np.sqrt(mean_squared_error(y_train,pred_train)))
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor





def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(500, input_dim=178, kernel_initializer='normal', activation='relu'))

    model.add(Dense(100, activation= "relu"))

    model.add(Dense(100, activation= "relu"))

    model.add(Dense(100, activation= "relu"))

    model.add(Dense(50, activation= "relu"))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)

# evaluate model with standardized dataset

keras_estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

score_keras = mod_sel_cv(keras_estimator)

print("Keras Regression score: {:.4f} ({:.4f})\n".format(score_keras.mean(), score_keras.std()))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin





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



    

averaged_models = AveragingModels(models = (lasso,ENet,KRR,GBR,ml_lgb,xg_reg))



score = mod_sel_cv(averaged_models)

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

                #define out_of_fold_predictions[holdout_index, i] as 1D array for it to work in my code

                oofp = np.zeros(out_of_fold_predictions[holdout_index, i].shape[0]) 

                oofp = y_pred



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

    

    



        

stacked_averaged_models = StackingAveragedModels(base_models = (lasso,ENet,KRR,GBR,ml_lgb),meta_model = xg_reg)



score_st = mod_sel_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score_st.mean(), score_st.std()))
from mlxtend.regressor import StackingRegressor

from sklearn.model_selection import cross_validate



stack1 = StackingRegressor(regressors=[ENet,KRR,GBR,lasso,ml_lgb], meta_regressor=xg_reg, verbose=0)

stack_scores = mod_sel_cv(stack1)#cross_validate(stack1, X, y, cv=10)

print("StackingRegressor score: {:.4f} ({:.4f})\n".format(stack_scores.mean(), stack_scores.std()))
averaged_models.fit(train_final.values, new_SalePrice.values)

model_train_pred = averaged_models.predict(train_final.values)

model_test_pred = np.expm1(averaged_models.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, model_train_pred)))
model_lgb = lgb.LGBMRegressor(objective='regression',task = "predict",num_leaves=8,

                              learning_rate=0.05, n_estimators=500,

                              max_bin = 100, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 6)



model_lgb.fit(train_final.values, new_SalePrice.values)

model_lgb_train_pred = model_lgb.predict(train_final.values)

model_lgb_test_pred = np.expm1(model_lgb.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, model_lgb_train_pred)))
GBR.fit(train_final.values, new_SalePrice.values)

GBR_train_pred = GBR.predict(train_final.values)

GBR_test_pred = np.expm1(GBR.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, GBR_train_pred)))
xg_reg.fit(train_final.values, new_SalePrice.values)

xg_reg_train_pred = xg_reg.predict(train_final.values)

xg_reg_test_pred = np.expm1(xg_reg.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, xg_reg_train_pred)))
stack1.fit(train_final.values, new_SalePrice.values)

stack1_train_pred = stack1.predict(train_final.values)

stack1_test_pred = np.expm1(stack1.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, stack1_train_pred)))
keras_model.fit(train_final.values, new_SalePrice.values)

keras_train_pred = keras_model.predict(train_final.values)

keras_test_pred = np.expm1(keras_model.predict(test_final.values))

print(np.sqrt(mean_squared_error(new_SalePrice.values, keras_train_pred)))
ensemble_pred = stack1_test_pred*0.50 + model_test_pred*0.50
subm = pd.DataFrame()

subm['Id'] = id_test

subm['SalePrice'] = ensemble_pred * 1.01

subm.to_csv('submission.csv',index=False)