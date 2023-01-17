import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.drop('Id', axis = 1)
print(train.shape)
print(test.shape)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train = train[train.GrLivArea < 4000]
train.shape
missing = train.isnull().sum().sort_values(ascending = False)
missing_pct = missing /len(train)
features_to_discard = list(missing_pct[missing_pct > 0.8].index)

print('features discared:', features_to_discard)
train.drop(features_to_discard, axis=1, inplace = True)
nominal_features = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                   'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                   'GarageType', 'SaleType', 'SaleCondition']
ordinal_features = ['Street', 'CentralAir', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterCond', 'ExterQual', 
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
                   'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
#the propotion of bedroom, bathroom, kitchen compare to total rooms maybe a good feature
discrete_features = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'YrSold', ]
#the proportion of living area compare to total area maybe a good feature
continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                      'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                      'PoolArea', 'MiscVal', ]

misc_features = ['MoSold', ]
from sklearn.base import BaseEstimator, TransformerMixin

#Encode ordinal features with order infomation
oridnal_map = {
    #qual, cond ect.
    'No':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 
    #yes or no
    'N':0, 'Y': 2, 
    # fence feature
    'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4,
    #LotShape
    'IR3':1, 'IR2':2, 'IR1':3, 'Reg':4,
    #Utilities
    'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4,
    #Land Slope
    'Sev':1, 'Mod':2, 'Gtl':3,
    #BsmtFin Type 1/2
    'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6,
    #Electrical
    'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5,
    #Functional
    'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8,
    #Garage Finish
    'Unf':1, 'RFn':2, 'Fin':3,
    #Paved Drive
    'N':0, 'P':1, 'Y':2,
    #Street
    'Grvl':1, 'Pave':2,
    #Basement exposure
    'Mn':2, 'Av':3, 'Gd':4
}


# no fit is needed, function transformer maybe enough
class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        df_x = pd.DataFrame(X, columns=ordinal_features)
        df_x = df_x.fillna('No')
        df_x =  df_x.applymap(lambda x:oridnal_map.get(x,x))
        return df_x
ordinal_transformer = OrdinalTransformer()
x =ordinal_transformer.fit_transform(train[ordinal_features])
x
train[discrete_features].isnull().sum()
from sklearn.preprocessing import Imputer

discrete_imputer = Imputer(strategy='most_frequent')
X = discrete_imputer.fit_transform(train[discrete_features])
df_x = pd.DataFrame(X, columns=discrete_features)
df_x.isnull().sum()
df_x
train[continuous_features].isnull().sum()
continuous_inputer = Imputer(strategy='mean')
X = continuous_inputer.fit_transform(train[continuous_features])
df_x = pd.DataFrame(X, columns=continuous_features)
print(df_x.isnull().sum())
df_x
continuous_inputer.statistics_
from scipy.stats import skew
df_x.apply(lambda x:skew(x))
def hist_plot(df, features_of_interest):
    _ = plt.subplots(figsize = (18,18))
    num_plots_per_row = 4
    all_plots = len(features_of_interest)
    for i, feature in enumerate(features_of_interest):
        ax = plt.subplot(all_plots/num_plots_per_row + 1, num_plots_per_row, i+1)
        ax.hist(x = df[feature], bins = 50)
        ax.set_xlabel(feature)
hist_plot(df_x, continuous_features)
from sklearn.preprocessing import StandardScaler

std_transformer = StandardScaler()
X = std_transformer.fit_transform(df_x)
df_x = pd.DataFrame(X, columns=continuous_features)
df_x
hist_plot(df_x, continuous_features)
class CatImputer(BaseEstimator, TransformerMixin):
    """ Impute categorical features, using most frequent strategy """
    
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

nominal_pipeline = Pipeline(steps=[
    ('imputer', CatImputer()),
    ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

train[nominal_features].isnull().sum()
nominal_X = nominal_pipeline.fit_transform(train[nominal_features])
nominal_X.shape

continous_pipeline = Pipeline(steps=[
    ('imputer', Imputer(strategy='mean')),
    ('log1p', FunctionTransformer(np.log1p)),
    ('std_scaler', StandardScaler()),
])
ordinal_pipeline = Pipeline(steps=[
    ('transformer', OrdinalTransformer()),
    # std scaler here really dont change the result much
    #('std_scaler', StandardScaler()),
])
discrete_pipeline = Pipeline(steps=[
    ('imputer', Imputer(strategy='most_frequent')),
    # std scaler here really dont change the result much
   # ('std_scaler', StandardScaler()),
])

nominal_pipeline = Pipeline(steps=[
    ('imputer', CatImputer()),
    ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

prepare_pipeline = ColumnTransformer([
    ('ordinal', ordinal_pipeline, ordinal_features),
    ('discrete', discrete_pipeline, discrete_features),
    ('continuous', continous_pipeline, continuous_features),
    ('nominal', nominal_pipeline, nominal_features),
])
train_prepared = prepare_pipeline.fit_transform(train)

train_prepared.shape
encoder = prepare_pipeline.named_transformers_.nominal.steps[1][1]


print('Before preprocess, number of nominal features: ', len(nominal_features))
print('After preprocess, number of categories:', len(encoder.categories_))

tmp = zip(nominal_features, encoder.categories_)

all_cats = [feature + '_' + str(suffix) for feature, cat in tmp for suffix in cat]

transformed_features = ordinal_features + discrete_features + continuous_features + all_cats

len(transformed_features)
df = pd.DataFrame(train_prepared, columns=transformed_features)
df['Target'] = train.SalePrice
bins = 15

year_cat = pd.cut(df.YearBuilt, bins = bins)
df['year_cat'] = year_cat

_= plt.subplots(figsize = (12,12))
sns.boxplot(x = 'year_cat', y = 'Target', data = df)
plt.xticks(rotation = 90)


x1 = df.YearBuilt > 1890
x2 = df.YearBuilt < 1900

df[x1 & x2].loc[:, ['YearBuilt', 'Target']]

fig,ax = plt.subplots(figsize = (20,20))
df[continuous_features].hist(ax = ax)
corr_matrix = df.corr()

corr_matrix.Target.sort_values(ascending=False)
_ = plt.subplots(figsize = (20,20))
sns.heatmap(corr_matrix)
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(train, test_size = 0.2, shuffle= True, random_state = 42)

print('train set:', train_set.shape)
print('test set:', test_set.shape)
step1_y = np.log(train_set.SalePrice.copy())
step1_X = prepare_pipeline.fit_transform(train_set)
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

lasso_reg = Lasso()

lasso_reg.fit(step1_X, step1_y)
pred = lasso_reg.predict(step1_X)
print(np.sqrt(mean_squared_error(step1_y, pred)))

from sklearn.linear_model import LassoCV
eps = 0.0005
alphas = np.logspace(-5 , 1, 100)

lasso_cv = LassoCV(alphas=alphas, cv = 5, verbose=1, random_state=42)
lasso_cv.fit(step1_X,step1_y)

best_alpha = lasso_cv.alpha_
print('best alpha', best_alpha)

alphas = np.arange(0.95, 1.05, 0.001) * best_alpha
lasso_cv = LassoCV(alphas= alphas, cv = 10, verbose=1, random_state=42)
lasso_cv.fit(step1_X,step1_y)

best_alpha = lasso_cv.alpha_
print('refined best alpha', best_alpha)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

scorer = make_scorer(mean_squared_error, greater_is_better = False)

scores = cross_val_score(Lasso(alpha=best_alpha), step1_X, step1_y, scoring=scorer, cv = 10, verbose=2 )

scores = np.sqrt(-scores)
print('mean:', scores.mean(), '\t sdv:', scores.std())
test_X = prepare_pipeline.transform(test_set)
test_y = np.log(test_set.SalePrice.copy())

print(step1_X.shape)
print(test_X.shape)

lasso_reg = Lasso(alpha=best_alpha)
lasso_reg.fit(step1_X,step1_y)
pred = lasso_reg.predict(test_X)
np.sqrt(mean_squared_error(test_y, pred))
all_X = prepare_pipeline.fit_transform(train)
all_y = np.log(train.SalePrice.copy())

lasso_reg.fit(all_X, all_y)
final_X = prepare_pipeline.transform(test)
pred = np.exp(lasso_reg.predict(final_X))
result = pd.DataFrame()
result['Id'] = test.Id
result['SalePrice'] = pred
result.to_csv('it2_result.csv', index=False)
coefs = pd.Series(lasso_reg.coef_, index = transformed_features)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
_ = plt.subplots(figsize = (12,12))
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")