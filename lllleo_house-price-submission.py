import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
%matplotlib inline

# data input


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
test_id = test['Id']

print ('Train data set\'s info',train.shape)
print ('Test data set\'s info',test.shape)
# 'SalePrice'distribution

print (train['SalePrice'].describe())

sns.distplot(train['SalePrice'],fit=norm)
#  dataset's information

total = pd.concat([train,test],ignore_index=True)
print (total.shape)

train_row = train.shape[0]
test_row=test.shape[0]
print ('obezave\'s num',train_row,'\n test\'s num',test_row)


print (total.info())
print (total.tail())
# missing values


total.loc[:,total.isnull().sum()>0].isnull().sum().sort_values(ascending=True)
# filling missing values according to different conditions
# for catagories data, if NaN means no installation, they were replaced by N
#  for numerial data, if NaN means have not, they were replaced by 0.
#  others would choosed replaced by mode or mean value of each catagory.

total['PoolQC'] = total['PoolQC'].fillna("N")
total['MiscFeature'] = total['MiscFeature'].fillna("N")
total['Alley'] = total['Alley'].fillna("N")
total['Fence'] = total['Fence'].fillna("N")
total['FireplaceQu'] = total['FireplaceQu'].fillna("N")
total['GarageQual'] = total['GarageQual'].fillna("N")
total['GarageCond'] = total['GarageCond'].fillna("N")
total['GarageFinish'] = total['GarageFinish'].fillna("N")
total['GarageType'] = total['GarageType'].fillna("N")
total['BsmtExposure'] = total['BsmtExposure'].fillna("N")
total['BsmtCond'] = total['BsmtCond'].fillna("N")
total['BsmtQual'] = total['BsmtQual'].fillna('N')
total['BsmtFinType2'] = total['BsmtFinType2'].fillna("N")
total['BsmtFinType1'] = total['BsmtFinType1'].fillna("N")
total['MasVnrType'] = total['MasVnrType'].fillna("N")


total['MSZoning'] = total['MSZoning'].fillna(total['MSZoning'].mode()[0])
total['BsmtFullBath'] = total['BsmtFullBath'].fillna(total['BsmtFullBath'].mode()[0])
total['BsmtHalfBath'] = total['BsmtHalfBath'].fillna(total['BsmtHalfBath'].mode()[0])
total['Utilities'] = total['Utilities'].fillna(total['Utilities'].mode()[0])
total['Functional'] = total['Functional'].fillna(total['Functional'].mode()[0])
total['Electrical'] = total['Electrical'].fillna(total['Electrical'].mode()[0])
total['Exterior1st'] = total['Exterior1st'].fillna(total['Exterior1st'].mode()[0])
total['Exterior2nd'] = total['Exterior2nd'].fillna(total['Exterior2nd'].mode()[0])
total['GarageCars'] = total['GarageCars'].fillna(total['GarageCars'].mode()[0])
total['KitchenQual'] = total['KitchenQual'].fillna(total['KitchenQual'].mode()[0])
total['SaleType'] = total['SaleType'].fillna(total['SaleType'].mode()[0])


total['BsmtUnfSF'] = total['BsmtUnfSF'].fillna(total['BsmtUnfSF'].median())


total['TotalBsmtSF'] = total['TotalBsmtSF'].fillna(total['TotalBsmtSF'].mean())
total['BsmtFinSF2'] = total['BsmtFinSF2'].fillna(total['BsmtFinSF2'].mean())
total['BsmtFinSF1'] = total['BsmtFinSF1'].fillna(total['BsmtFinSF1'].mean())
total['GarageArea'] = total['GarageArea'].fillna(total['GarageArea'].mean())

total.loc[total["MasVnrType"] == "N","MasVnrArea"] = 0

### LotFrontage and GarageYrBlt missed many values, deleting them

total = total.drop(['LotFrontage','GarageYrBlt'],axis=1)

total.shape
# outlier


plt.scatter(train['GrLivArea'],train['SalePrice'])
# removing outlier according to suggestion in doc

total = total.drop(total[(total['GrLivArea']>4000) & (total['SalePrice']<300000)].index)

total.shape
# transfer numerial data which refer to catagory to catagory type

total = total.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30',40: 'SubClass_40',
45: 'SubClass_45',50: 'SubClass_50',60: 'SubClass_60',70: 'SubClass_70',
75: 'SubClass_75',80: 'SubClass_80',85: 'SubClass_85',90: 'SubClass_90',
120: 'SubClass_120',150: 'SubClass_150',160: 'SubClass_160',180: 'SubClass_180',
190: 'SubClass_190'}})

total = total.replace({'MoSold': {1: 'Jan', 2: 'Feb',3: 'Mar',
4: 'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sep',10: 'Oct',
11: 'Nov',12: 'Dec'}})

total['YrSold'] = total['YrSold'].astype(str)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# coding the data

cata_list = total.dtypes[total.dtypes=='object'].index.tolist()
num_list = total.dtypes[total.dtypes!='object'].index.tolist()

nominal_cata_list = ['MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',
                    'Condition1', 'Condition2', 'BldgType','HouseStyle','RoofStyle','RoofMatl',
                     'Exterior1st', 'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
                    'GarageType','MiscFeature', 'SaleCondition', 'SaleType','MSSubClass','MoSold','YrSold'] 

ordinal_cata_list = [ 'LotShape','Utilities',  'LandSlope','ExterQual', 'ExterCond','BsmtCond',
                     'BsmtQual','BsmtExposure', 'BsmtFinType1','BsmtFinType2','HeatingQC',
                    'Electrical', 'KitchenQual','Functional','FireplaceQu','GarageFinish',
                     'GarageQual', 'GarageCond','PavedDrive','PoolQC','Fence']


# for ordinal

for i in ordinal_cata_list:
    encoder = LabelEncoder()
    encoder.fit(list(total[i].values))
    total[i] = encoder.transform(list(total[i].values))

# for nominal

total = pd.get_dummies(total,columns=nominal_cata_list,drop_first=True)
total.shape
# drop 'Id' column

pd.set_option('display.max_columns',300)
total = total.drop(['Id'],axis=1)

total.shape
# check skewness

choice_num_list = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1',
                    'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF',
                   'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea',
                   'GarageCars', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotShape',
                    'LowQualFinSF', 'MiscVal', 'OpenPorchSF', 'PoolArea',
                'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF']


skew_cols = total.iloc[:,skew(total[choice_num_list])>0.75].columns
print (skew_cols)

for i in skew_cols:
    total[i]=np.log1p(total[i])

total.shape
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

# scaler data

y_saleprice = total['SalePrice']
total = total.drop(['SalePrice'],axis=1)

rob = RobustScaler()
rob.fit(total)
total = rob.transform(total)
# split train and test set
y_cv = np.log1p(y_saleprice[:train_row-2])
print (y_cv)

train_cv = total[:train_row-2]
print (train_cv.shape)

test_new = total[train_row-2:]
print (test_new.shape)


from sklearn.cross_validation import cross_val_score,KFold

# validation RMSE
kf = KFold(10,n_folds=10,shuffle= True, random_state=42)

def rmse_cv(model,train_cv=train_cv,y_cv=y_cv):
    rmse=np.sqrt(-cross_val_score(model,train_cv,y_cv,scoring='neg_mean_squared_error',cv=kf))
    return (rmse.mean(),rmse.std())
# base models

from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV,LinearRegression
import matplotlib.pyplot as plt
%matplotlib inline

# for RidgeCV

alphas_rd = [0.001,0.01,0.1,1.0,8.0,9.0,10.0,11.0, 100.0]
scores=[]

for i in alphas_rd:
    score = rmse_cv(RidgeCV(alphas=[i]))
    scores.append(score[0])
    print ('alpha:',i,'scoring:',score)

plt.plot(alphas_rd,scores)



ridge_cv = RidgeCV(alphas = alphas_rd)
# for LassoCV model

alphas_la = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,0.0006, 0.0007, 0.0008]

lasso_cv = LassoCV(eps=0.001,alphas=alphas_la,max_iter=1e6, random_state=42).fit(train_cv,y_cv)
score = lasso_cv.mse_path_
print ('alpha:',lasso_cv.alpha_)

plt.plot(alphas_la,score)

lasso_cv = LassoCV(alphas=alphas_la,max_iter=1e6,random_state = 42)
# for Elasticnet Model

l1_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
alphas_el = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

els_cv = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas_el, max_iter=1e6, random_state=42)
els_cv.fit(train_cv,y_cv)
score=els_cv.mse_path_

print ('l1_ratio:',els_cv.l1_ratio_,'alpha:',els_cv.alpha_)

els_cv = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas_el, max_iter=1e6, random_state=42)
# for linearregression model

linear_reg = LinearRegression()

# rmse values of base models

print (rmse_cv(linear_reg))
print (rmse_cv(els_cv))
print (rmse_cv(lasso_cv))
print (rmse_cv(ridge_cv))
# advanced model

# simplely average the base models

linear_reg.fit(train_cv,y_cv)
els_cv.fit(train_cv,y_cv)
lasso_cv.fit(train_cv,y_cv)
ridge_cv.fit(train_cv,y_cv)

a1= linear_reg.predict(test_new)
a2= els_cv.predict(test_new)
a3= lasso_cv.predict(test_new)
a4= ridge_cv.predict(test_new)

pred_simple_avg = pd.DataFrame({'a':a1,'b':a2,'c':a3,'d':a4})
pred_simple_avg_result = np.mean(pred_simple_avg,axis=1)   
from mlxtend.regressor import StackingCVRegressor

#  mlxtend package

stack = StackingCVRegressor(regressors=(linear_reg,els_cv,lasso_cv,ridge_cv),
                           meta_regressor=ridge_cv, use_features_in_secondary=True)

rmse_cv(stack,train_cv,y_cv.values)
# adjested average-models

stack.fit(train_cv,y_cv.values)

a5 = stack.predict(test_new)


stacking_pred = (0.26*a1) + (0.245*a2) + (0.235*a3) + (0.26*a4)
from sklearn.base import BaseEstimator, RegressorMixin,clone

# ensembling model
# code from https://www.kaggle.com/liyenhsu/feature-selection-and-ensemble-of-5-models

las = LassoCV(alphas=alphas_la,max_iter=1e6,random_state = 42)

class Ensemble(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None,meta=las):
        self.regressors = regressors
        self.meta=meta
        
    def level0_to_level1(self, X):
        self.predictions_ = []

        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X).reshape(X.shape[0],1))

        return np.concatenate(self.predictions_, axis=1)
    
    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)
            
            
        self.new_features = self.level0_to_level1(X)
        
        # using a large L2 regularization to prevent the ensemble from biasing toward 
        # one particular base model
        self.combine = self.meta   
        self.combine.fit(self.new_features, y)

#        self.coef_ = self.combine.coef_

    def predict(self, X):
        self.new_features = self.level0_to_level1(X)
            
        return self.combine.predict(self.new_features).reshape(X.shape[0])
model = Ensemble(regressors=[linear_reg,els_cv,lasso_cv,ridge_cv])
model.fit(train_cv, y_cv)
ensemble_pred = model.predict(test_new)

rmse_cv(model)
from sklearn.base import BaseEstimator, RegressorMixin,clone, TransformerMixin
from sklearn.model_selection import KFold

# stacking model2
# code from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

class stacking2(BaseEstimator, RegressorMixin,TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models=base_models
        self.meta_model =meta_model
        self.n_folds = n_folds

        
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds,shuffle=True, random_state=156)        
        
        out_of_fold_preidictions = np.zeros((X.shape[0], len(self.base_models)))
                                 
        for i, model in enumerate(self.base_models):
            for train_index,holdout_index in kfold.split(X,y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index],y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_preidictions[holdout_index,i]=y_pred
                              
        self.meta_model_.fit(out_of_fold_preidictions,y)
        return self
    
    def predict(self,X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                                                         for base_models in self.base_models_])

       
        return self.meta_model_.predict(meta_features) 
stacked2 = stacking2(base_models =[els_cv,lasso_cv,ridge_cv], meta_model = linear_reg)
stacked2.fit(train_cv,y_cv.values)
stacked2_pred = stacked2.predict(test_new)

# base_models =[els_cv,lasso_cv,ridge_cv], meta_model = ridge_cv (lb RMSE 0.11879)
rmse_cv(stacked2, y_cv=y_cv.values)
# simple average model

pred_average = np.expm1(pred_simple_avg_result.values)

pred_df_63_1 = pd.DataFrame({'SalePrice':pred_average},index=test_id)

pred_df_63_1.to_csv('pred_df_63_1.csv')
# StackCVRegressor model

pred_sr = np.expm1(a5)

pred_df_63_2 = pd.DataFrame({'SalePrice':pred_sr},index=test_id)

pred_df_63_2.to_csv('pred_df_63_2.csv')
# adjusted model

pred_stacking = np.expm1(stacking_pred)

pred_df_63_3 = pd.DataFrame({'SalePrice':pred_stacking},index=test_id)

pred_df_63_3.to_csv('pred_df_63_3.csv')
# stacked2 model 

pred_stacked2 = np.expm1(stacked2_pred)

pred_df_63_5 = pd.DataFrame({'SalePrice':pred_stacked2},index=test_id)

pred_df_63_5.to_csv('pred_df_63_4.csv')
