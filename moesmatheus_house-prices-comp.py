import seaborn as sns

import pandas as pd

import pandas_profiling

from matplotlib import pyplot as plt

import numpy as np



from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax
# Load Train Data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 0)

train['train'] = True

print('Train Data Shape: ', train.shape)

# Load Test Data

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col = 0)

test['train'] = False

print('Test Data Shape: ', test.shape)



# Join frames

full = train.append(test, sort=False)



# Drop 

full.drop(['Utilities','LandSlope', 'Alley'], axis = 1,inplace = True)



full['SalePrice'] = np.log(full['SalePrice'])

train.head()
#train.loc[:,train.dtypes == 'object'].columns



numerical = [

    'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'LotArea',

    'YearBuilt','YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 

    ]



categorical = [

    'MSSubClass', 'MSZoning', 'Street',  'LotShape', 'LandContour','LotConfig',

    'Neighborhood', 'BldgType', 'Condition1', 'Condition2', 'HouseStyle', 'RoofStyle',

    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',  'Electrical',

    'Heating', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'

    #'Alley'

    ]



ordinal = [

    'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 

    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC','KitchenQual','Functional',

    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',

    'Fence', 'CentralAir']

d1 = {'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}

d2 = {np.NAN:0,'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

d3 = {np.NAN:0,'No':1, 'Mn':2, 'Av':3, 'Gd':4}

d4 = {np.NAN:0,'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5,'GLQ':6}

d5 = {np.NAN:7, 'Sal':0,'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5,'Min1':6, 'Typ':7}

d6 = {np.NAN:0,'Unf':1, 'RFn':2, 'Fin':3}

d7 = {np.NAN:0,'MnWw':1, 'GdWo':2, 'MnPrv':3,'GdPrv':4}

d8 = {'N':0, 'P':1, 'Y':2}

d9 = {np.NAN:0, 'MnWw':1, 'GdWo':2, 'MnPrv':3,'GdPrv':4}

d10 = {np.NAN:0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}

d11 = {'N':0, 'Y':1}

d12 = {np.NAN:2, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4} # to correct the missing value on kitchenqual



ordinal_dics = {

    'ExterQual':d1, 'ExterCond':d1, 'BsmtQual':d2, 'BsmtCond':d2, 

    'BsmtExposure':d3, 'BsmtFinType1':d4, 'BsmtFinType2':d4, 'HeatingQC':d1,'KitchenQual':d12,

    'Functional':d5, 'FireplaceQu':d2, 'GarageFinish':d6, 'GarageQual':d2, 'GarageCond':d2,

    'PavedDrive':d8,  'PoolQC':d10,'Fence':d9  , 'CentralAir':d11

}



full_ord = full.copy()



# Correct categorical that was stored as numeric

full_ord['MSSubClass'] = full_ord['MSSubClass'].astype('str')



for a in ordinal_dics:

    full_ord[a] = full[a].map(ordinal_dics[a])
missing = full_ord.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(ascending = False,inplace=True)

missing.plot.bar()

plt.show()
full_na_corr = full_ord.copy()



#Replace what should be 0

l = ['LotFrontage', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'GarageCars',

'GarageArea','TotalBsmtSF', 'BsmtFinSF2', 'BsmtUnfSF']

for a in l:

    full_na_corr[a].fillna(0, inplace = True)



# Replace with mode

l = ['Electrical', 'Exterior1st', 'SaleType', 'MSZoning', 'MasVnrType']

for a in l:

    full_na_corr[a].fillna(full_na_corr[a].mode()[0], inplace = True)



# Fill with minimum value

full_na_corr['GarageYrBlt'].fillna(full_na_corr['GarageYrBlt'].min(), inplace = True)



full_na_corr['Functional'].fillna('Typ', inplace = True)



#full_na_corr['KitchenQual'].fillna('Ta', inplace = True)



full_na_corr['Exterior2nd'].fillna(full_na_corr['Exterior1st'], inplace = True)



# Fill categorical with 'None' category 

l = ['MiscFeature', 'GarageType' ]

for a in l:

    full_na_corr[a].fillna('None', inplace = True)
full_na_corr.isnull().sum().sort_values(ascending = False).head(5)
full_eng = full_na_corr.copy()



# Different entry conditions

full_eng['diff cond'] = np.where(full_eng['Condition1'] == full_eng['Condition2'],1,0)



# Different exteriors in the house

full_eng['diff ext'] = np.where(full_eng['Exterior1st'] == full_eng['Exterior2nd'],1,0)



# No garage in the house

full_eng['No Garage'] = np.where(full_eng['GarageType'].isnull,1,0)



#full_eng['RoomSqft'] = full_eng['GrLivArea']/full_eng['BedroomAbvGr']



#full_eng['Remodeled'] = np.where(full_eng['YearBuilt'] == full_eng['YearRemodAdd'],1,0)

 

full_eng['Total Area'] = full_eng['GrLivArea'] + full_eng['TotalBsmtSF']



#full_eng['Total_Bathrooms'] = full_eng['FullBath'] + (0.5 * full_eng['HalfBath'])

full_eng['Total_Bathrooms'] = (full_eng['FullBath'] + (0.5 * full_eng['HalfBath']) +

                               full_eng['BsmtFullBath'] + (0.5 * full_eng['BsmtHalfBath']))



#full_eng['has2ndfloor'] = full_eng['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)





numerical_new = numerical + ['diff cond', 'diff ext', 'No Garage'] 

categorical_new = categorical

ordinal_new = ordinal
full_corr_skew = full_eng.copy()



skew_features = full_eng[numerical].apply(lambda x: skew(x)).sort_values(ascending=False)



skew_features.head(30)
print('Features adjusted: \n', skew_features[skew_features > 0.5] )

for a in skew_features[skew_features > 0.5].index:

    #full_corr_skew[a] = np.log(full_corr_skew[a] + 1)

    full_corr_skew[a] = boxcox1p(full_corr_skew[a], boxcox_normmax(full_corr_skew[a] + 1))
full_corr_skew[numerical].apply(lambda x: skew(x)).sort_values(ascending=False).head()
full_corr_out = full_corr_skew.copy()



full_corr_out = full_corr_out[

    ~((full_corr_out['GrLivArea'] > np.log(4000+1)) & (full_corr_out['train'] == True))]
corr = full_corr_out.corr()

plt.subplots(figsize=(15,12))

sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.base import clone



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



import warnings

warnings.filterwarnings("ignore")
fr = full_corr_out.copy()

p = fr[categorical]

fr.drop(categorical,axis = 1, inplace = True)

fr = fr.join(pd.get_dummies(p))

print(fr.shape)



train_set = fr[fr['train']].drop('train', axis = 1)

eval_set = fr[~fr['train']].drop(['train', 'SalePrice'], axis = 1)

print('Shapes: ',train_set.shape, eval_set.shape)



x = train_set.drop(['SalePrice'], axis = 1)

y = train_set['SalePrice']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
cat = CatBoostRegressor(iterations=2000,loss_function='RMSE', random_state = 1, learning_rate=0.01,

                     depth=5, one_hot_max_size=10, boosting_type='Plain',l2_leaf_reg=0.01)



xg = XGBRegressor(learning_rate=0.1,  n_estimators=2000, max_depth=8,min_child_weight=0,

                       gamma=0.06,subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror',

                       nthread=-1, scale_pos_weight=1,seed=1,reg_alpha=0.006,# booster='gblinear',

                       random_state=42)



rf = RandomForestRegressor(n_estimators= 2000, max_depth = 8, random_state = 1)



lm = LinearRegression()







kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]





# Kernel Ridge Regression : made robust to outliers

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



# LASSO Regression : made robust to outliers

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                    alphas=alphas2,random_state=42, cv=kfolds))



# Elastic Net Regression : made robust to outliers

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))



svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))



gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,

                                max_features='sqrt', min_samples_leaf=15, min_samples_split=10,

                                loss='huber', random_state =42) 



def rmse(x,y):

    return np.sqrt(mean_squared_error(x,y))
class stack_model():

    def __init__(self,base_models, meta_model, n_folds = 10):

        '''

        base_models as {'name':model, 'name2':model2}

        '''

        self.n_folds = n_folds

        self.base_models = base_models

        self.meta_model = clone(meta_model)

        self.kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

    def fit_base_models(self,x,y, score_engine = mean_squared_error, fit_param = {},

                       meta_model_param = None):

        # Dictionqry to hold models

        self.model_collection = {}

        # Hold scores from models

        self.model_scores = {}

        # Hold Out of fold predictions

        self.out_of_fold_pred = pd.DataFrame(

                index = np.arange(x.shape[0]),columns = self.base_models.keys())

        for i in self.base_models:

            print('Model: ',i)

            n = 1

            scores = []

            self.model_collection[i] = []

            for a,b in self.kfold.split(x,y):

                print('Fold: %i' %n#, end='\r'

                     )

                # Clone model to new version

                prov_model = clone(self.base_models[i])

                # Fit parameters if there is extra keys or not

                if i in fit_param.keys():

                    prov_model.fit(x.iloc[a],y.iloc[a], **fit_param[i])

                else:

                    prov_model.fit(x.iloc[a],y.iloc[a])

                # Append model to collection

                self.model_collection[i].append(prov_model)

                # Make out of fold predictions

                prediction = prov_model.predict(x.iloc[b])

                #Calculate model score

                score = score_engine(prediction, y.iloc[b])

                scores.append(score)

                self.out_of_fold_pred[i][b] = prediction

                n+=1

                

            print('Scores: ', scores)

            print('AVG: ', np.mean(scores))

            self.model_scores[i] = np.mean(scores)

        if meta_model_param is None:

            self.meta_model.fit(self.out_of_fold_pred, y)

        else:

            self.meta_model.fit(self.out_of_fold_pred, y, **meta_model_param)



    def return_predict_base(self, x_eval):

        p = pd.DataFrame(

            {i:np.mean(

                [a.predict(x_eval) for a in m.model_collection[i]], axis = 0

                )

             for i in self.base_models

            }

        )

        return p

    

    def predict(self, x_eval):

        return self.meta_model.predict( self.return_predict_base(x_eval) )

    

    def predict_all(self, x_eval):

        p = self.return_predict_base(x_eval)

        p['Stack'] = self.meta_model.predict(p)

        return p

        

            

        

                

            
m = stack_model({

    'xg':xg,'rf':rf,'cat':cat, 'lm':lm,

    'ridge':ridge,'lasso':lasso,'elasticnet':elasticnet, 'svr':svr, 'gb':gbr

},

    meta_model=cat)

m.fit_base_models(x,y, score_engine = rmse, fit_param = {'cat':{'verbose':0}},

                  meta_model_param = {'verbose':0}

                 )
# Blend Models

p = np.average(m.predict_all(eval_set), axis = 1, weights=[0.5,0.5,1,1,1,1,1,1,0.5,5])

#Create Dataframe

out = pd.DataFrame(np.exp(p),index = eval_set.index, columns = ['SalePrice'])

#Create submission

out.to_csv('submission.csv')