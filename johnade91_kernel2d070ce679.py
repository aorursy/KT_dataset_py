# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
Test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
Train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

print(Test_df.shape)
print(Train_df.shape)

values = {'Alley': 'None','BsmtQual': 'None','BsmtCond': 'None','BsmtExposure': 'None','BsmtFinType1': 'None','BsmtFinType2': 'None','FireplaceQu': 'None','GarageType': 'None','GarageQual': 'None','GarageCond': 'None','PoolQC': 'None','Fence': 'None', 'MiscFeature': 'None'}

Test_df.fillna(value=values, inplace=True)

#Next we will look at GarageYrBlt and GarageFinish, which have 159 missing variables each. 
#The missing data in these fields is due to the property having no garage. We can replace the GarageYrBlt and GarageFinish with 0. 
#This will ensure the model treats properties with and without garages separately given that we already have a fetaure GarageType 
#that identifies whether or not the property has a garage.

Test_df.replace({'GarageYrBlt':np.nan,'GarageFinish':np.nan},0, inplace=True)

# Fills the blank values in Exterior1st, Exterior2nd, MasVnrType with group modes
Test_df.replace({'Exterior1st':np.nan},pd.Series.mode(Test_df['Exterior1st'])[0], inplace=True)
Test_df.replace({'Exterior2nd':np.nan},pd.Series.mode(Test_df['Exterior2nd'])[0], inplace=True)
Test_df.replace({'MasVnrType':np.nan},pd.Series.mode(Test_df['MasVnrType'])[0], inplace=True)

#Blank entries in MasVnrArea occur where there is no MasVnrType there MasVnrArea should be set to 0
Test_df.replace({'MasVnrArea':np.nan},0, inplace=True)

#Fill the blank entries in MSZoning with  mode
Test_df.replace({'MSZoning':np.nan},pd.Series.mode(Test_df['MSZoning'])[0], inplace=True)

#Fill utilities with  modes 
Test_df.replace({'Utilities':np.nan},pd.Series.mode(Test_df['Utilities'])[0], inplace=True)

#Fill blank entry  with 0 corresponding to no basement
Test_df.replace({'BsmtFinSF1':np.nan},0, inplace=True)
Test_df.replace({'BsmtFinSF2':np.nan},0, inplace=True)
Test_df.replace({'BsmtUnfSF':np.nan},0, inplace=True)
Test_df.replace({'TotalBsmtSF':np.nan},0, inplace=True)
Test_df.replace({'BsmtFullBath':np.nan},0, inplace=True)
Test_df.replace({'BsmtHalfBath':np.nan},0, inplace=True)

#Fill blank entry for Electrical with mode
Test_df.replace({'Electrical':np.nan},pd.Series.mode(Test_df['Electrical'])[0], inplace=True)

#Fill blank entry for KitchenQual with mode
Test_df.replace({'KitchenQual':np.nan},pd.Series.mode(Test_df['KitchenQual'])[0], inplace=True)

#Fill blank entry for Functional with mode
Test_df.replace({'Functional':np.nan},pd.Series.mode(Test_df['Functional'])[0], inplace=True)

#Fill blank entry for GarageCars and GarageArea with mode and median respectively
Test_df.replace({'GarageCars':np.nan},pd.Series.mode(Test_df['GarageCars'])[0], inplace=True)
Test_df.replace({'GarageArea':np.nan},0, inplace=True)


#Fill blank entry for SaleType with mode
Test_df.replace({'SaleType':np.nan},pd.Series.mode(Test_df['SaleType'])[0], inplace=True)

#Lot Frontage 
Test_df.replace({'LotFrontage':np.nan},Test_df.LotFrontage.median(), inplace = True)


values = {'Alley': 'None','BsmtQual': 'None','BsmtCond': 'None','BsmtExposure': 'None','BsmtFinType1': 'None','BsmtFinType2': 'None','FireplaceQu': 'None','GarageType': 'None','GarageQual': 'None','GarageCond': 'None','PoolQC': 'None','Fence': 'None', 'MiscFeature': 'None'}

Train_df.fillna(value=values, inplace=True)


Train_df.replace({'GarageYrBlt':np.nan,'GarageFinish':np.nan},0, inplace=True)

# Fills the blank values in Exterior1st, Exterior2nd, MasVnrType with group modes
Train_df.replace({'Exterior1st':np.nan},pd.Series.mode(Train_df['Exterior1st'])[0], inplace=True)
Train_df.replace({'Exterior2nd':np.nan},pd.Series.mode(Train_df['Exterior2nd'])[0], inplace=True)
Train_df.replace({'MasVnrType':np.nan},pd.Series.mode(Train_df['MasVnrType'])[0], inplace=True)

#Blank entries in MasVnrArea occur where there is no MasVnrType there MasVnrArea should be set to 0
Train_df.replace({'MasVnrArea':np.nan},0, inplace=True)

#Fill the blank entries in MSZoning with  mode
Train_df.replace({'MSZoning':np.nan},pd.Series.mode(Train_df['MSZoning'])[0], inplace=True)

#Fill utilities with  modes 
Train_df.replace({'Utilities':np.nan},pd.Series.mode(Train_df['Utilities'])[0], inplace=True)

#Fill blank entry  with 0 corresponding to no basement
Train_df.replace({'BsmtFinSF1':np.nan},0, inplace=True)
Train_df.replace({'BsmtFinSF2':np.nan},0, inplace=True)
Train_df.replace({'BsmtUnfSF':np.nan},0, inplace=True)
Train_df.replace({'TotalBsmtSF':np.nan},0, inplace=True)
Train_df.replace({'BsmtFullBath':np.nan},0, inplace=True)
Train_df.replace({'BsmtHalfBath':np.nan},0, inplace=True)

#Fill blank entry for Electrical with mode
Train_df.replace({'Electrical':np.nan},pd.Series.mode(Train_df['Electrical'])[0], inplace=True)

#Fill blank entry for KitchenQual with mode
Train_df.replace({'KitchenQual':np.nan},pd.Series.mode(Train_df['KitchenQual'])[0], inplace=True)

#Fill blank entry for Functional with mode
Train_df.replace({'Functional':np.nan},pd.Series.mode(Train_df['Functional'])[0], inplace=True)

#Fill blank entry for GarageCars and GarageArea with mode and median respectively
Train_df.replace({'GarageCars':np.nan},pd.Series.mode(Train_df['GarageCars'])[0], inplace=True)
Train_df.replace({'GarageArea':np.nan},0, inplace=True)


#Fill blank entry for SaleType with mode
Train_df.replace({'SaleType':np.nan},pd.Series.mode(Train_df['SaleType'])[0], inplace=True)

#Lot Frontage 
Train_df.replace({'LotFrontage':np.nan},Train_df.LotFrontage.median(), inplace = True)


import seaborn as sns
from matplotlib import pyplot as plt

fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)

d = Train_df['SalePrice']
sns.set_style('darkgrid')
sns.distplot(d)
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)


sns.set_style('darkgrid')
sns.distplot(np.log(Train_df['SalePrice']))

fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)

plt.scatter(Train_df['GrLivArea'],Train_df['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel(('Above ground living area'))
#Strip out anomalous data points
Train_df = Train_df.loc[Train_df['GrLivArea']<4000]

#Repeat scatter plot with anamalous results omitted
fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)

plt.scatter(Train_df['GrLivArea'],Train_df['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel(('Above ground living area'))
Full_df = pd.concat([Train_df,Test_df])
print(Full_df.shape)
print(Test_df.shape)
print(Train_df.shape)
#create a function to generate dummies
def dum_df(feature,total_df):
    dummy_feature = pd.get_dummies(total_df[feature],prefix=feature)
    return dummy_feature
#Convert all categorical variables to numeric variables so data set can be analysed

features = ['MSZoning',
'Street',
'Alley',
'LotShape',
'LandContour',
'Utilities',
'LotConfig',
'LandSlope',
'Neighborhood',
'Condition1',
'Condition2',
'BldgType',
'HouseStyle',
'RoofStyle',
'RoofMatl',
'Exterior1st',
'Exterior2nd',
'MasVnrType',
'ExterQual',
'ExterCond',
'Foundation',
'BsmtQual',
'BsmtCond',
'BsmtExposure',
'BsmtFinType1',
'BsmtFinType2',
'Heating',
'HeatingQC',
'CentralAir',
'Electrical',
'KitchenQual',
'Functional',
'FireplaceQu',
'GarageType',
'GarageFinish',
'GarageQual',
'GarageCond',
'PavedDrive',
'PoolQC',
'Fence',
'MiscFeature',
'SaleType',
'SaleCondition',
'MSSubClass',
'OverallQual',
'OverallCond'
]


df_initial = pd.DataFrame()

for f in features:
    df_initial =  pd.concat([df_initial,dum_df(f,Full_df)], axis=1)
    
Full_dfDummies = pd.concat([df_initial,Full_df.drop(columns=features)], axis=1)

# don't forget to apply log transformation on sales price 
Full_dfDummies['logSalePrice'] = np.log(Full_dfDummies['SalePrice'])

#from sklearn import preprocessing
# Get list of numeric fetaures

names = ['LotFrontage',
'LotArea',
'YearBuilt',
'YearRemodAdd',
'MasVnrArea',
'BsmtFinSF1',
'BsmtFinSF2',
'BsmtUnfSF',
'TotalBsmtSF',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'BsmtFullBath',
'BsmtHalfBath',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'GarageYrBlt',
'GarageCars',
'GarageArea',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold',
]




Full_dfDummies.shape
Train_df2 = Full_dfDummies.loc[Full_dfDummies['SalePrice'].notnull()].drop(columns='Id')
print(Train_df2.isna().sum())
Train_df2.head()
Train_df2
print(Train_df2.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr


print(Train_df2.shape)

#create a function that returns tuple of pearsons correlation, 

def multivariate_pearsonr(X,y):
    scores,pvalues = [],[]
    for column in range (X.shape[1]):
        cur_score,cur_p = pearsonr(X[:,column],y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return(np.array(scores),np.array(pvalues))
    

#Run a persons correlation against SalesPrice
K=len(names)

X = Train_df2[names]
y = Train_df2['logSalePrice']
#apply SelectKBest class to extract top k best features
bestfeatures = SelectKBest(score_func=multivariate_pearsonr, k=K)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
dfpvals = pd.DataFrame(fit.pvalues_)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores,dfpvals],axis=1)
featureScores.columns = ['Specs','Score','Pvalue']  #naming the dataframe columns
#print(featureScores.nlargest(K,'Score'))  #print k best features

#Numeric_Feat = featureScores.nlargest(K,'Score')[['Specs','Score']]

#set parameters 
corrval = 0.3
pvalues = 0.05

Numeric_Feat2 = featureScores.loc[featureScores['Pvalue']<pvalues]
Numeric_Feat3 = Numeric_Feat2.loc[np.absolute(Numeric_Feat2['Score'])>corrval]


Features_1 = list(Numeric_Feat3['Specs'])
Features_1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr

print(Train_df2.shape)

X = Train_df2.drop(columns=names).drop(columns=['logSalePrice','SalePrice'])
y = Train_df2['SalePrice']

K= len(X.columns)

#apply SelectKBest class to extract top k best features
bestfeatures = SelectKBest(score_func=chi2, k=K)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
dfpvals = pd.DataFrame(fit.pvalues_)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores,dfpvals],axis=1)
featureScores.columns = ['Specs','Score','Pvalue']  #naming the dataframe columns

# Record top 10 features by predictive power 
chival = 10
pvalues = 0.05

#cat_features = featureScores.nlargest(K,'Score')[['Specs','Score']]

#eliminate insignificant results 
cat_features = featureScores.loc[(featureScores['Pvalue']<pvalues)]

#Keep results over threshold
cat_features2 = cat_features.loc[(cat_features['Score']>chival)]



Features_2 = list(cat_features2['Specs'])
Features_2


Predictive_Features = Features_1+Features_2
Predictive_Features
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

X = Train_df2[Predictive_Features]
VIF_tab = calc_vif(X)
print(VIF_tab)



remove = ['YearRemodAdd','GarageYrBlt','Exterior1st_CemntBd', 'BsmtQual_None',
'BsmtCond_None',
'BsmtCond_Po',
'BsmtFinType1_None',
'Electrical_Mix',
'OverallCond_1',
'1stFlrSF'

]


Predictive_Features2= list(set(Predictive_Features).difference(set(remove)))



print(len(Predictive_Features2))
len(Predictive_Features)

#Look at VIF for shortlist of features
X = Train_df2[Predictive_Features2]
VIF_tab = calc_vif(X)
print(VIF_tab)


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()

Train_normalised_a = Train_df2.drop(columns=['SalePrice','logSalePrice'])

Train_normalised = Scaler.fit_transform(Train_normalised_a)


Train_normalised = pd.DataFrame(Train_normalised, columns=Train_normalised_a.columns)


X = Train_normalised
y = Train_df2[['logSalePrice','SalePrice']]


Train_df3 = pd.concat([X.reset_index(drop=True),y.reset_index(drop=True)], axis=1)
Train_df3
from sklearn.model_selection import train_test_split

X = Train_df3[Predictive_Features2]
y = Train_df3['logSalePrice']
meancrossvalscore = []
testscore = []
alphavalues = [1,10,50,100,1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


for a in alphavalues:
    RidgeModel2 = Ridge(solver='saga', alpha=a)
    Input=[('scale',StandardScaler()), ('model',RidgeModel2)]
    Pipe_RR = Pipeline(Input)
    Pipe_RR.fit(X_train,y_train)
    crossvalscore = cross_val_score(Pipe_RR,X_train,y_train,scoring ='r2',cv=5).mean()
    testing_score = Pipe_RR.score(X_test,y_test)
    meancrossvalscore.append(crossvalscore)
    testscore.append(testing_score)


print(meancrossvalscore)
print(testscore)
print(alphavalues)
#plt.plot(alphavalues,meancrossvalscore)
plt.plot(alphavalues,testscore , label = 'Test Score')
plt.plot(alphavalues,meancrossvalscore , label= 'Mean Cross Val')

plt.ylabel('R^2')
plt.xlabel('Alpha')
plt.legend(loc="upper left")
plt.show()

print('Regularisation param that maximises testscore:' ,alphavalues[testscore.index(max(testscore))])
print('Regularisation param that maximises Mean cross val score:' ,alphavalues[meancrossvalscore.index(max(meancrossvalscore))])

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

X = (Train_df2.drop(columns=['logSalePrice', 'SalePrice']))
#X = Train_df2[Predictive_Features]
y = Train_df2['logSalePrice']


MeanR2CrossVal = []
TestScore = []
maxdepth  = [15,20,25,30]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


for i in maxdepth:
    rf = RandomForestRegressor(criterion = 'mse', bootstrap=True, n_jobs =-1, max_features=len(Predictive_Features), n_estimators=1000, max_depth =i, random_state=42)
    rf.fit(X_train,y_train)
    meanscore = cross_val_score(rf,X_train,y_train,scoring='r2',cv=5).mean()
    testing_Score  = rf.score(X_test,y_test) 
    MeanR2CrossVal.append(meanscore)
    TestScore.append(testing_Score)
    

print(maxdepth)
print(MeanR2CrossVal)
print(TestScore)


plt.plot(maxdepth,TestScore , label = 'Test Score')
plt.plot(maxdepth,MeanR2CrossVal, label= 'Mean Cross Val')
#plt.legend(handles, labels)

plt.ylabel('R^2')
plt.xlabel('Max Depth')
plt.legend(loc="upper left")
plt.show()


#added gradient boosted regressor to improve predictive performance

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

#Parameters for gridsearch
param_grid = {'model__learning_rate':[0.1,1,2],'model__loss':['lad'], 'model__n_estimators' :[1000]}


#Pipeline will standardise input then fit model. 
gbr = GradientBoostingRegressor()
Input=[('scale',StandardScaler()), ('model',gbr)]
gbr2 = Pipeline(Input)

#optimise with gridsearchCV
mod_out = GridSearchCV(gbr2, param_grid = param_grid, scoring ='r2')

#gbr2.get_params().keys()

mod_out.fit(X_train,y_train)


#Train xgboost model with parameters 

# cross validation on total train data
print('Test score:', mod_out.score(X_test,y_test))


#run cross val with best grid search params
params = mod_out.best_params_

#rename dictionary keys to correct model inputs
params['learning_rate'] = params.pop('model__learning_rate')
params['loss'] = params.pop('model__loss')
params['n_estimators'] = params.pop('model__n_estimators')


#run cross validation on entire training set with paraemeters set to results of best gridsearch 
gbr3 = GradientBoostingRegressor(**params)
cross_valscores = cross_val_score(gbr3,X,y,scoring='r2')
                                    





 
plt.plot([1,2,3,4,5],cross_valscores, label= 'Cross Val')
#plt.legend(handles, labels)

plt.ylabel('R^2')
plt.xlabel('Sample')
plt.legend(loc="upper left")
plt.show()


#fit model with optmised params to final train set
gbr3.fit(X,y)

Test_df2 = Full_dfDummies.loc[Full_dfDummies['SalePrice'].isnull()]
X = Test_df2.drop(columns=['SalePrice','logSalePrice', 'Id'])

Test_df2['SalePrice'] = np.exp(gbr3.predict(X))
Final_Out = Test_df2[['Id','SalePrice']]



# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = Final_Out

# create a link to download the dataframe
create_download_link(df)
