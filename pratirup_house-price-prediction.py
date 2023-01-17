import numpy as np

import pandas as pa

import seaborn as sn

import matplotlib.pyplot as plt

import matplotlib as mpl

import matplotlib.patches as matp

%matplotlib inline

import statsmodels.api as sm

import random



from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV,learning_curve

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.preprocessing import LabelEncoder,RobustScaler,Imputer



from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



from sklearn import metrics

from scipy import stats

from scipy.special import boxcox1p

color = sn.color_palette()

sn.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')


sample_submission = pa.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pa.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pa.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



print("Shape of house data {}".format(train.shape))

print("Length of house data {}\n".format(train.shape[0]))



test_id = test.Id



print("Shape of train data {}".format(train.shape))

print("Length of house data {}\n".format(train.shape[0]))



print("Shape of test data {}".format(test.shape))

print("Length of test data {}\n".format(test.shape[0]))



# Here we can see that our train dataset contains 81 features and length is 1000 



#Y_actual_test = pa.read_csv('Y_test.csv')
features = list(train.columns)

print(features)

print("\nTotal Number Of Features {}".format(len(features)))

train.head()
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(16,5))



sn.scatterplot(x='GrLivArea',y='SalePrice',data=train,ax=axis1).set_title("With Outliers ")





train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index)

sn.scatterplot(x='GrLivArea',y='SalePrice',data=train,ax=axis2).set_title("Without Outliers")

fig, (axis1,axis2,axis3,axis4) = plt.subplots(1,4,figsize=(20,4))



sn.lineplot(x='OverallQual',y='SalePrice',data=train,ax=axis1)

sn.barplot(x='OverallQual',y='SalePrice',data=train,ax=axis2)

sn.lineplot(x='OverallCond',y='SalePrice',data=train,ax=axis3)

sn.barplot(x='OverallCond',y='SalePrice',data=train,ax=axis4)



plt.figure(figsize=(20,6))

sn.lineplot(x='YearBuilt',y='SalePrice',data=train)

plt.show()
plt.figure(figsize=(20,6))

sn.lineplot(x='TotalBsmtSF',y='SalePrice',data=train)



plt.show()

plt.figure(figsize=(20,6))



sn.scatterplot(x='TotalBsmtSF',y='SalePrice',data=train)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,8))

sn.scatterplot(x='1stFlrSF',y='SalePrice',data=train,ax=axis1)

sn.scatterplot(x='2ndFlrSF',y='SalePrice',data=train,ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(16,6))

sn.boxplot(x='FullBath',y='SalePrice',data=train,ax=axis1)

#sn.barplot(x='HalfBath',y='SalePrice',data=train,ax=axis2)

sn.boxplot(x='HalfBath',y='SalePrice',data=train,ax=axis2)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,6))

sn.lineplot(x='KitchenAbvGr',y='SalePrice',data=train,ax=axis1)

sn.lineplot(x='BedroomAbvGr',y='SalePrice',data=train,ax=axis2)

sn.lineplot(x='TotRmsAbvGrd',y='SalePrice',data=train,ax=axis3)
area = ['LotFrontage','GarageArea','OpenPorchSF','EnclosedPorch','ScreenPorch','SalePrice']

sn.pairplot(train[area],size=2)
plt.figure(figsize=(20,6))

sn.barplot(x='Neighborhood',y='SalePrice',data=train)

plt.xticks(rotation=45)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(20,5))



sn.barplot(x='PoolQC',y='SalePrice',data=train,ax=axis1)

sn.barplot(x='PavedDrive',y='SalePrice',data=train,ax=axis2)

sn.barplot(x='RoofStyle',y='SalePrice',data=train,ax=axis3)



plt.figure(figsize=(20,6))

## Here we will create a waffle chart for roofstyle to check the proportion of each roof style 



roof_dataframe = pa.DataFrame(train['RoofStyle'].value_counts())

roof_dataframe.rename(columns={'RoofStyle':'Total'},inplace=True)

total_values = sum(roof_dataframe['Total'])



category_proportions = [(float(value) / total_values) for value in roof_dataframe['Total']]



width = 90

height = 20

total_number_tiles = width * height



# compute the number of tiles for each catagory

tiles_per_category = [round(proportion * total_number_tiles) for proportion in category_proportions]



# initialize the waffle chart as an empty matrix

waffle_chart = np.zeros((height, width))



# define indices to loop through waffle chart

category_index = 0

tile_index = 0



# populate the waffle chart

for col in range(width):

    for row in range(height):

        tile_index += 1



        # if the number of tiles populated for the current category is equal to its corresponding allocated tiles...

        if tile_index > sum(tiles_per_category[0:category_index]):

            

            category_index += 1       

            

        # set the class value to an integer, which increases with class

        waffle_chart[row, col] = category_index

        

print ('Waffle chart populated!')



plt.figure(figsize=(20,6))



colormap = plt.cm.coolwarm

plt.matshow(waffle_chart,cmap=colormap)

plt.colorbar()



ax = plt.gca()



ax.set_xticks(np.arange(-.5,(width),1),minor=True)

ax.set_yticks(np.arange(-.5,(height),1),minor=True)



ax.grid(which='minor',color='w',linestyle='-',linewidth=2)

plt.xticks([])



plt.yticks([])



values_cumsum = np.cumsum(roof_dataframe['Total'])

total_values = values_cumsum[len(values_cumsum) - 1]



# create legend

legend_handles = []

for i, category in enumerate(roof_dataframe.index.values):

    label_str = category + ' (' + str(roof_dataframe['Total'][i]) + ')'

    color_val = colormap(float(values_cumsum[i])/total_values)

    legend_handles.append(matp.Patch(color=color_val, label=label_str))



# add legend to chart

plt.legend(handles=legend_handles,

           loc='lower center', 

           ncol=len(roof_dataframe.index.values),

           bbox_to_anchor=(0., -0.2, 0.95, .1)

          )
fig, (axis1) = plt.subplots(1,1,figsize=(16,5))





plt.figure(figsize=(16,5))

sn.distplot(train['SalePrice'],color='k',label='Skewness : %.2f'%train['SalePrice'].skew(),ax=axis1).set_title("Density Plot of SalePrice")

plt.ylabel("Frequency")

#plt.legend(loc='best')



stats.probplot(train.SalePrice,plot=plt)

plt.title("Probability plot of SalePrice")

plt.show()
# Here we can see that it need to be normalize 

### 3 types of normal check whch perfomr better means which better normalize the value

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,5))

sn.distplot(train['SalePrice'],fit=stats.johnsonsu,ax=axis1,label='Skewness :%.2f'%train['SalePrice'].skew()).set_title("JhonsonSU Transformation")



sn.distplot(train['SalePrice'],fit=stats.lognorm,ax=axis2,label='Skewness :%.2f'%train['SalePrice'].skew()).set_title("Log Transformation")



sn.distplot(train['SalePrice'],fit=stats.norm,ax=axis3,label='Skewness :%.2f'%train['SalePrice'].skew()).set_title("Normal Transformation")



print("From here we can see that log transformation performs much better than others so we will use Log Transformation to normalize\n SalePrice")
train['SalePrice'] = np.log1p(train['SalePrice'])

plt.figure(figsize=(16,5))

sn.distplot(train['SalePrice'],color='k',label='Skewness : %.2f'%train['SalePrice'].skew()).set_title("Density Plot of Sale Price(Normalized)")

plt.ylabel("Frequency")

plt.legend(loc='best')

plt.show()

plt.figure(figsize=(16,5))



stats.probplot(train.SalePrice,plot=plt)

plt.show()



house_data = pa.concat([train,test])

house_data.drop('SalePrice',axis=1,inplace=True)

house_data.shape
missing_data = (house_data.isnull().sum() / len(house_data)) * 100

missing_data = missing_data.drop(missing_data[missing_data == 0].index).sort_values(ascending=False)

missing_data_ratio = pa.DataFrame({'Missing Ratio':missing_data})



plt.figure(figsize=(20,10))

g = sn.barplot(x=missing_data_ratio.index,y='Missing Ratio',data=missing_data_ratio).set_title("Misssing Data Ratio")

plt.xticks(rotation=45)
corrmat = train.corr()

plt.figure(figsize=(20,10))

sn.heatmap(corrmat, vmax=0.9, square=True)
house_data["PoolQC"] = house_data["PoolQC"].fillna("None")

house_data["MiscFeature"] = house_data["MiscFeature"].fillna("None")

house_data["Alley"] = house_data["Alley"].fillna("None")

house_data["Fence"] = house_data["Fence"].fillna("None")

house_data["FireplaceQu"] = house_data["FireplaceQu"].fillna("None")



house_data['LotFrontage'] = house_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    house_data[col] = house_data[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    house_data[col] = house_data[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 

            'BsmtFullBath', 'BsmtHalfBath'):

    house_data[col] = house_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

            'BsmtFinType2'):

    house_data[col] = house_data[col].fillna('None')

    

house_data["MasVnrType"] = house_data["MasVnrType"].fillna("None")

house_data["MasVnrArea"] = house_data["MasVnrArea"].fillna(0)

house_data['MSZoning'] = house_data['MSZoning'].fillna(house_data['MSZoning'].mode()[0])



house_data = house_data.drop('Utilities',axis=1)



house_data.Functional.value_counts()

house_data["Functional"] = house_data["Functional"].fillna("Typ")



house_data.Electrical.value_counts()

house_data['Electrical'] = house_data['Electrical'].fillna(house_data['Electrical'].mode()[0])



house_data['KitchenQual'] = house_data['KitchenQual'].fillna(house_data['KitchenQual'].mode()[0])

house_data['KitchenQual'] = house_data['KitchenQual'].fillna(house_data['KitchenQual'].mode()[0])

house_data['Exterior1st'] = house_data['Exterior1st'].fillna(house_data['Exterior1st'].mode()[0])

house_data['Exterior2nd'] = house_data['Exterior2nd'].fillna(house_data['Exterior2nd'].mode()[0])



house_data['SaleType'] = house_data['SaleType'].fillna(house_data['SaleType'].mode()[0])

house_data['MSSubClass'] = house_data['MSSubClass'].fillna("None")
#MSSubClass=The building class

house_data['MSSubClass'] = house_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

house_data['OverallCond'] = house_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

house_data['YrSold'] = house_data['YrSold'].astype(str)

house_data['MoSold'] = house_data['MoSold'].astype(str)
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(house_data[c].values)) 

    house_data[c] = lbl.transform(list(house_data[c].values))



# shape        

print('Shape all_data: {}'.format(house_data.shape))
house_data['TotalSF'] = house_data['TotalBsmtSF'] + house_data['1stFlrSF'] + house_data['2ndFlrSF']



house_data['Total_Bathrooms'] = (house_data['FullBath'] + (0.5 * house_data['HalfBath']) +

                               house_data['BsmtFullBath'] + (0.5 * house_data['BsmtHalfBath']))



house_data['Total_porch_sf'] = (house_data['OpenPorchSF'] + house_data['3SsnPorch'] +

                              house_data['EnclosedPorch'] + house_data['ScreenPorch'] +

                              house_data['WoodDeckSF'])
house_data['haspool'] = house_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

house_data['has2ndfloor'] = house_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

house_data['hasgarage'] = house_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

house_data['hasbsmt'] = house_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

house_data['hasfireplace'] = house_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
numercic_features = house_data.dtypes[house_data.dtypes != 'object'].index

skewed_fetures = house_data[numercic_features].apply(lambda x:x.skew()).sort_values(ascending=False)

skewness = pa.DataFrame({'Skew' :skewed_fetures})

skewness.info()
skewness = skewness.drop(['haspool','has2ndfloor','hasgarage','hasbsmt','hasfireplace'],axis=0)

plt.figure(figsize=(20,6))

skewed_feaures_more = skewness[skewness['Skew'] > 0.75].index

skewd_data = skewness.Skew[skewness['Skew'] > 0.75]

sn.barplot(skewed_feaures_more,skewd_data)

plt.xticks(rotation=90)
skewness = skewness[abs(skewness.Skew) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    house_data[feat] = boxcox1p(house_data[feat], lam)
house_data = pa.get_dummies(house_data)

print("After converting the remaining catagorical features into dummy variables we get {}".format(house_data.shape))
y_train =train.SalePrice.values

train = house_data[:len(train)]

test = house_data[len(train):]
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#linear = make_pipeline(RobustScaler(),LinearRegression())

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
from sklearn.svm import SVR

svr = make_pipeline(RobustScaler(),

                      SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(svr)

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
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

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
stacked_averaged_models = StackingAveragedModels(base_models = (KRR, lasso, ENet, GBoost, model_xgb, model_lgb),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
lasso.fit(train,y_train)

lasso_train_pred = lasso.predict(train)

lasso_pred = np.expm1(lasso.predict(test))

print(rmsle(y_train,lasso_train_pred))
svr.fit(train,y_train)

svr_train_pred = svr.predict(train)

svr_pred = np.expm1(svr.predict(test.values))

print(rmsle(y_train,svr_train_pred))
KRR.fit(train,y_train)

KRR_train_pred = KRR.predict(train)

KRR_pred = np.expm1(KRR.predict(test.values))

print(rmsle(y_train,KRR_train_pred)) 
ENet.fit(train,y_train)

ENet_train_pred = ENet.predict(train)

ENet_pred = np.expm1(ENet.predict(test.values))

print(rmsle(y_train,ENet_train_pred))
GBoost.fit(train,y_train)

GBoost_train_pred = GBoost.predict(train)

GBoost_pred = np.expm1(GBoost.predict(test.values))

print(rmsle(y_train,GBoost_train_pred)) 
averaged_models.fit(train,y_train)

averaged_models_train_pred = averaged_models.predict(train)

averaged_models_pred =np.expm1(averaged_models.predict(test))

print(rmsle(y_train,averaged_models_train_pred)) 
print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.25 +

               xgb_train_pred*0.15 + lgb_train_pred*0.10+GBoost_train_pred*0.15 + svr_train_pred*0.15 +

               lasso_train_pred*0.05+KRR_train_pred*0.15))
ensemble = stacked_pred*0.25 + xgb_pred*0.15 + lgb_pred*0.10 + GBoost_pred*0.15 + svr_pred*0.15 + lasso_pred*0.05+KRR_pred*0.15
Y_test = pa.DataFrame(ensemble + 0.11658)
sub = pa.DataFrame()

sub['Id'] = test_id

sub['SalePrice'] = ensemble

sub.to_csv('final_sub.csv',index=False)