#libraries we need

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

from datetime import date

pd.options.mode.chained_assignment = None

import h2o

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt





#libraries we need

# !pip install h2o



from scipy.special import expit



from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from h2o.grid.grid_search import H2OGridSearch



from sklearn.model_selection import train_test_split

from h2o.estimators import H2OGradientBoostingEstimator

SEED  = 1111   # global random seed for better reproducibility



from sklearn.tree import export_graphviz

# from sklearn.externals.six import StringIO  

from IPython.display import Image  

# import pydotplus



h2o.init(max_mem_size='24G', nthreads=4) # start h2o with plenty of memory and threads

h2o.remove_all()                         # clears h2o memory

h2o.no_progress() 
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Drop the id column from both test and training data

train.drop(['Id'],axis=1, inplace=True)

test.drop(['Id'],axis=1, inplace=True)



print('The shape of train data is {}'.format(train.shape))

print('The shape of test data is {}'.format(test.shape))



#concat both the datasets for easier cleaning 

full = train.append(test, ignore_index=True)

print('The shape of full data is {}'.format(full.shape))
pd.DataFrame(full.isna().sum()*100/full.shape[0]).plot.bar(figsize=(20,5))
#NA already existing category

full.update(full[['BsmtCond','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtQual',

                  'GarageType','GarageQual','GarageFinish','GarageCond','FireplaceQu',

                  'MiscFeature','Fence','PoolQC','Alley','Electrical','MasVnrType']].fillna('None'))



#nan with zero as constant

full.update(full[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtHalfBath',

                  'BsmtFullBath','GarageArea','GarageCars','MasVnrArea','TotalBsmtSF']].fillna(0)) 



# Replacing the missing values with mode for the list of variables ['Exterior1st','Exterior2nd','Functional','KitchenQual','MSZoning','SaleType','Utilities']

full['Exterior1st']=full['Exterior1st'].fillna(full.Exterior1st.value_counts().index[0])

full['Exterior2nd']=full['Exterior2nd'].fillna(full.Exterior2nd.value_counts().index[0])

full['Functional']=full['Functional'].fillna(full.Functional.value_counts().index[0])

full['KitchenQual']=full['KitchenQual'].fillna(full.KitchenQual.value_counts().index[0])

full['MSZoning']=full['MSZoning'].fillna(full.MSZoning.value_counts().index[0])

full['SaleType']=full['SaleType'].fillna(full.SaleType.value_counts().index[0])

full['Utilities']=full['Utilities'].fillna(full.Utilities.value_counts().index[0])



#Dropping irrelavent columns from the whole dataset based on the EDA on the training dataset

#GarageQual is repetitive, which has the same meaning as Garage Cond

#PoolQC is mostly NA and won't provide much info, and we've already have PoolArea

#MSSubClass is a combination of dweiing and year

full= full.drop(['MoSold','GarageQual','PoolQC','MSSubClass'],axis=1)



#filled missing garage years

#It makes no sense to fill year with 0, so we assume the garage was built when the house was built

full['GarageYrBlt'] = full['GarageYrBlt'].fillna(full['YearBuilt'])



#Create new features to make them more comprehensive to common sense

#converting years into age 

currentYear = datetime.now().year

full['Age_House']=currentYear-full['YearBuilt']

full['Age_Renovation']=currentYear-full['YearRemodAdd']

full['Garage_age']=currentYear-full['GarageYrBlt']

full = full.drop(['YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1)



#Changing OverallCond into a categorical variable, they will be label encoded afterwards

#These're ordinal variables

full['OverallCond'] = full['OverallCond'].astype(str)

full['YrSold'] = full['YrSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageCond', 'ExterQual', 

        'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'OverallCond', 

        'YrSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lb = LabelEncoder() 

    lb.fit(list(full[c].values)) 

    full[c] = lb.transform(list(full[c].values))

    

    

    

# Adding total sqfootage feature 

full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']
pd.DataFrame(full.isna().sum()*100/full.shape[0]).plot.bar(figsize=(20,5))
#spillitng the data again

train = full[full['SalePrice'].notnull()]

test = full[full['SalePrice'].isnull()]

train_y = train['SalePrice']

train_x = train.drop(['SalePrice'],axis=1)

test_x = test.drop(['SalePrice'],axis=1)
# Get train LotFrontage dummy variables

LotFrontage_Dummies_df = pd.get_dummies(train_x[['LotFrontage', 'MSZoning', 'LotArea', 'LotConfig', 'Neighborhood']])



# Get full dummy variables

# Split the data into LotFrontage known and LotFrontage unknown

LotFrontageKnown = LotFrontage_Dummies_df[LotFrontage_Dummies_df["LotFrontage"].notnull()]

LotFrontageUnknown = LotFrontage_Dummies_df[LotFrontage_Dummies_df["LotFrontage"].isnull()]



# Training data knowing LotFrontage

LotFrontage_Known_X = LotFrontageKnown.drop(["LotFrontage"], axis = 1)

LotFrontage_Known_y = LotFrontageKnown["LotFrontage"]

# Training data unknown LotFrontage

LotFrontage_Unknown_X = LotFrontageUnknown.drop(["LotFrontage"], axis = 1)

# Build model using random forest

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=1,n_estimators=500,n_jobs=-1)

rfr.fit(LotFrontage_Known_X, LotFrontage_Known_y)

rfr.score(LotFrontage_Known_X, LotFrontage_Known_y)
# Predict training data unknown LotFrontage

LotFrontage_Unknown_y = rfr.predict(LotFrontage_Unknown_X)

train_x.loc[train_x["LotFrontage"].isnull(), "LotFrontage"] = LotFrontage_Unknown_y
# Repeat same process for test data

# Get train LotFrontage dummy variables

LotFrontage_Dummies_df = pd.get_dummies(test_x[['LotFrontage', 'MSZoning', 'LotArea', 'LotConfig', 'Neighborhood']])



# Get full dummy variables

# Split the data into LotFrontage known and LotFrontage unknown

LotFrontageKnown = LotFrontage_Dummies_df[LotFrontage_Dummies_df["LotFrontage"].notnull()]

LotFrontageUnknown = LotFrontage_Dummies_df[LotFrontage_Dummies_df["LotFrontage"].isnull()]



# Testing data knowing LotFrontage

LotFrontage_Known_X = LotFrontageKnown.drop(["LotFrontage"], axis = 1)

LotFrontage_Known_y = LotFrontageKnown["LotFrontage"]

# Testing data unknown LotFrontage

LotFrontage_Unknown_X = LotFrontageUnknown.drop(["LotFrontage"], axis = 1)

# Build model using random forest

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=1,n_estimators=500,n_jobs=-1)

rfr.fit(LotFrontage_Known_X, LotFrontage_Known_y)

rfr.score(LotFrontage_Known_X, LotFrontage_Known_y)
# Predict testing data unknown LotFrontage

LotFrontage_Unknown_y = rfr.predict(LotFrontage_Unknown_X)

test_x.loc[test_x["LotFrontage"].isnull(), "LotFrontage"] = LotFrontage_Unknown_y
train['LotFrontage'] = train_x['LotFrontage']

test['LotFrontage'] = test_x['LotFrontage']
sns.distplot(train['LotFrontage'])
train.plot.scatter(x='Age_House', y='SalePrice', ylim=(0,800000))
#box plot overallqual/saleprice

var = 'MSZoning'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
result = pd.concat([train_x, train_y], axis=1)

Corr = result.corr().iloc[:-1,-1]



fig, ax_ = plt.subplots(figsize=(8, 10))

_ =  Corr.plot(kind='barh', ax=ax_, colormap='gnuplot')

_ = ax_.set_xlabel('Pearson Correlation for continuous variables')
train['SalePrice'] = np.log(train['SalePrice'])

test['SalePrice'] = np.log(test['SalePrice'])



train_y = train['SalePrice']

train_x = train.drop(['SalePrice'],axis=1)



test_y = test['SalePrice']

test_x = test.drop(['SalePrice'],axis=1)
train_df = pd.get_dummies(train)

test_df = pd.get_dummies(test)
train_y_df = train_df['SalePrice']

train_x_df = train_df.drop('SalePrice', axis = 1)
r = 'SalePrice'

x = list(train_x_df.columns.values)
hf=h2o.H2OFrame(train_df)

gf=h2o.H2OFrame(test_df)
hyper_params = {'alpha': [0, .25, .5, .75, 1]

                ,'lambda':[1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]

               }



glm = H2OGeneralizedLinearEstimator(family = 'gaussian',standardize = True,lambda_search = True)



# build grid search with previously made GLM and hyperparameters

grid = H2OGridSearch(model = glm, hyper_params = hyper_params,

                     search_criteria = {'strategy': "Cartesian"})





grid.train(x = x, y = r, training_frame = hf,nfolds=5,seed=1)
sorted_grid = grid.get_grid(sort_by='RMSLE', decreasing=False)

best_model = sorted_grid.models[0]

best_model.cross_validation_metrics_summary()
pred_glm_tr =  best_model.predict(h2o.H2OFrame(train_x_df))

pred_glm_tr = pred_glm_tr.as_data_frame()

co = best_model.coef()
cc = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64']]

cc.remove('SalePrice')
cont_coef = pd.DataFrame.from_dict(dict((k, co[k]) for k in cc),orient='index')
cont_coef = cont_coef.rename(columns={ 0: "Beta"})
cont_coef.plot.barh(figsize=(20, 20),color='orange')
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.30, random_state=1111)
X_train = pd.concat([X_train, y_train], axis=1)

X_valid = pd.concat([X_valid, y_valid], axis=1)

X_train_hf = h2o.H2OFrame(X_train)

X_valid_hf = h2o.H2OFrame(X_valid)



SEED  = 1111   # global random seed for better reproducibility
y_name = 'SalePrice'

x_names = list(train.columns.drop('SalePrice'))



predictors = x_names

response = "SalePrice"
params = {'learn_rate': [0.01, 0.05, 0.1], 

          'max_depth': list(range(2,13,2)),

          'ntrees': [20, 50, 80, 110, 140, 170, 200],

          'sample_rate': [0.5,0.6,0.7,0.9,1], 

          'col_sample_rate': [0.2,0.4,0.5,0.6,0.8,1]

          }





# Prepare the grid object

grid = H2OGridSearch(model=H2OGradientBoostingEstimator,   # Model to be trained

                     grid_id='gbm_grid1',

                     hyper_params=params,              # Dictionary of parameters

                     search_criteria={"strategy": "RandomDiscrete", "max_models": 500}   # RandomDiscrete

                     )



# Train the Model

grid.train(x=predictors,y=response, 

           training_frame=X_train_hf, 

           validation_frame=X_valid_hf,

           seed = SEED) # Grid Search ID
# Identify the best model generated with least error

sorted_final_grid = grid.get_grid(sort_by='rmsle',decreasing = False)
best_model_id = sorted_final_grid.model_ids[0]

best_gbm_from_grid = h2o.get_model(best_model_id)

best_gbm_from_grid.summary()
preds_train = best_gbm_from_grid.predict(X_train_hf).exp().as_data_frame()
best_gbm_from_grid.model_performance(X_valid_hf)
X_test_hf = h2o.H2OFrame(test_x)

preds = best_gbm_from_grid.predict(X_test_hf)

final_preds = preds.exp()

final_preds = final_preds.as_data_frame()

pred_pandas=final_preds
raw_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

raw_id = raw_test['Id']

output = pd.concat([raw_id, final_preds], axis=1)

output = output.rename(columns={'exp(predict)': "SalePrice"})
best_gbm_from_grid.varimp_plot()
contributions = best_gbm_from_grid.predict_contributions(X_test_hf)

#contributions.head(5)
import shap

shap.initjs()

contributions_matrix = contributions.as_data_frame().iloc[:,:].values



X = list(train.columns)

X.remove('SalePrice')

len(X)
shap_values = contributions_matrix[:,:76]

shap_values.shape
expected_value = contributions_matrix[:,:76].min()

expected_value
shap.force_plot(expected_value, shap_values, X)
shap.force_plot(expected_value, shap_values[0,:], X)
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")
Continuous = [key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64']]
dd = ['TotalSF','OverallQual','1stFlrSF']



for i in dd:

    print(best_gbm_from_grid.partial_plot(data = X_train_hf, cols = [i], server=True, plot = True))
from sklearn.tree import DecisionTreeRegressor,tree

dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.04,

random_state=SEED)

pred_pandas = h2o.as_list(preds)

test_x_dummies = pd.get_dummies(test_x)
dt = dt.fit(test_x_dummies,np.exp(pred_pandas))
dt.score(test_x_dummies,np.exp(pred_pandas))
# feature_cols = list(test_x_dummies.columns.values)



# dot_data = StringIO()

# export_graphviz(dt, out_file=dot_data,  

#                 filled=True, rounded=True,

#                 special_characters=True,feature_names = feature_cols)

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# Image(graph.create_png())
residual = np.exp(train['SalePrice']).sub(preds_train['exp(predict)'], axis = 0).abs()
residual = pd.DataFrame(residual,columns=['Residual'])
residual['SalePrice']= np.exp(train['SalePrice'])
residual = residual.fillna(0)
df = pd.concat([residual,train_x],axis=1)
residual.mean()
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(20, 10))

plt.scatter(residual['SalePrice'],residual['Residual'],color='r')

plt.xlabel('SalePrice')

plt.ylabel('Residual')

plt.show()
import seaborn as sns

sns.set(font_scale=0.9)                                         

sns.set_style('whitegrid') 



groups = df.groupby(x_names)



sorted_ = df.sort_values(by='Neighborhood') 



g=sns.FacetGrid(df, col="Neighborhood",col_wrap=5)

g= (g.map(plt.scatter, "SalePrice", "Residual").add_legend())
sns.set(font_scale=0.9)                                         

sns.set_style('whitegrid') 



groups = df.groupby(x_names)



sorted_ = df.sort_values(by='OverallCond') 



g=sns.FacetGrid(df, col="OverallCond",col_wrap=3)

g= (g.map(plt.scatter, "SalePrice", "Residual").add_legend())
fig, ax = plt.subplots(figsize=(20, 10)) 

plt.plot(df['SalePrice'])

plt.plot(np.exp(pred_pandas['predict']),color='orange')

plt.plot(np.exp(pred_glm_tr['predict']),color='deeppink')

_ = ax.set_xlabel('Ranked Row Index')
fig, ax = plt.subplots(figsize=(20, 10)) 

plt.plot(df['SalePrice'],color='deeppink')

plt.plot(np.exp(pred_pandas['predict']),color='orange')
fig, ax = plt.subplots(figsize=(20, 10)) 

plt.plot(df['SalePrice'],color='deeppink')

plt.plot(np.exp(pred_glm_tr['predict']))