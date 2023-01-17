# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df =  pd.read_csv('../input/train.csv', low_memory=False ) #parse_dates = ['pickup_datetime']
test_df =  pd.read_csv('../input/test.csv')

#set_rf_samples(1000)  set the number of samples to make the process run quicker.
train_df_aside = train_df.copy() #set aside train_df for use later on
train_df.info()   #looking closer at the train_df 
# build decision tree as fast as possible and then create feature importances. fastai methodology
#1) drop id's
dropped_feat = ['Id']
train_df.drop(dropped_feat, axis=1, inplace=True)
#this is code if you need to convert a pandas column to log before the algorithm process it
##creating a new df that contains log of SalePrice
#train_df['log_SalePrice']=np.log(train_df['SalePrice']+1)
#saleprices=train_df[['SalePrice','log_SalePrice']]

#train_df = train_df.drop('SalePrice', axis=1)  #drop sales prices from train_df_log & just keep the log of sales prices.

#saleprices.head(5)  #compare sales price and log of sales price
#y_train = saleprices['log_SalePrice']

#fastai libraries for split_vals, onehot encoding, proc_df
from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *
from fastai.structured import *
train_cats(train_df) #convert objects (strings) to pandas categories
df, y, nas = proc_df(train_df, 'SalePrice') #run proc_df from fastai library to prepare the dataframe easily for the random forest.  this encodes all x categorical values, seperates dependent to y, fills na's with medium
df.head() #review df
#split the data to train/test/split randomly
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42) 
#fit random forest to the data
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=500, max_depth = 20, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

%time rf_fit = rf.fit(X_train, y_train)
rf_predict = rf_fit.predict(X_test) #predictions

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_rf_predict=np.log(rf_predict+1)
log_y_test=np.log(y_test+1)

#calculate Root Mean Squared Error for predictions 'rf_predict' against actuals 'y_test'
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(log_y_test, log_rf_predict)
rmsqe = np.sqrt(mse)
print("RMSQE: ", rmsqe)
#finding feature importances
feat_labels = X_train.columns
importance = rf_fit.feature_importances_  #determing feature importance from the algorithm"
#print(feat_labels)
#print(importance)
#creating new dataframe of feature importances
df_feat = pd.DataFrame(importance, index=feat_labels)
df_feat = df_feat.reset_index()
df_feat = df_feat.rename(index=str, columns={0: "importance", 'index': 'feature'})
df_feat = df_feat.sort_values('importance', ascending=False, inplace=False, axis=0)
df_feat.head()
#plotting feature importances
def plot_fi(df): 
  return df.plot('feature','importance','barh', figsize=(12,7), legend=False)
plot_fi(df_feat[:30]);
#note that we can do a similar thing with correlation analysis, which is quicker but not as robost
# find the most correlated features
corrmat = train_df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#creating new dataframe 'df_keep' with best features only
to_keep = df_feat[df_feat.importance>0.005].feature; len(to_keep)
df_keep = df[to_keep].copy()  #df is the original df where we did proc_df

X_train, X_test, y_train, y_test = train_test_split(df_keep, y, test_size=0.2, random_state=42)

df_keep.info()
#fitting new RF on df_keep which just has the best features to check if RMSE stays similar
rf_best = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
rf_best_fit = rf_best.fit(X_train, y_train)
rf_bfp_predict = rf_best_fit.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_rf_bfp_predict=np.log(rf_bfp_predict+1)
log_y_test=np.log(y_test+1)

mse = mean_squared_error(log_y_test, log_rf_bfp_predict)
rmsqe = np.sqrt(mse)
print("RMSQE rf_best: ", rmsqe)
#finding feature importances round 2 on the new best fit RF - we do this to reduce collinearity. ie removing redundent features, since they are basically measuring the same thing.
feat_labels = X_train.columns
importance = rf_best_fit.feature_importances_  #determing feature importance from the algorithm"

df_feat = pd.DataFrame(importance, index=feat_labels)
df_feat = df_feat.reset_index()
df_feat = df_feat.rename(index=str, columns={0: "importance", 'index': 'feature'})
df_feat = df_feat.sort_values('importance', ascending=False, inplace=False, axis=0)
df_feat.head()
plot_fi(df_feat[:30]);
#cluster analysis to further find variables that are too similar using correlation between variables.
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, 
      orientation='left', leaf_font_size=16)
plt.show()
#things  closer to the right like fireplaces are variables are similar and need to be investigatedfurther
#next step to create a function (out of bag score (oob_score)) #remember oob is the score on the 3rd of trees not included in the bagging in RF.
def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, 
           max_features=0.6, n_jobs=-1, oob_score=True)
    #x, _ = split_vals(df, n_trn)
    m.fit(X_train, y_train)
    return m.oob_score_

get_oob(df_keep)
#compare oob for entire dataframe with oob if we were to drop one of these variables, 

for c in ('Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageCars', 
          '1stFlrSF', 'TotalBsmtSF', 'GarageYrBlt', 'YearBuilt' ):
    print(c, get_oob(df_keep.drop(c, axis=1)))
    
#Following are scores if that variable is removed.  
#If oob_score doesn't go down, on removing a variable its safe to say I can get rid of it, but only on variable in the pair!
#lets try get rid of 1 option in some of the above pairs
to_drop = ['Fireplaces', 'GarageCars', 'TotalBsmtSF', 'GarageYrBlt']
get_oob(df_keep.drop(to_drop, axis=1))
#and the oob score goes up!  so now ive dropped variables and my model is simpler still
#run the whole thing again now, with less variables: 

df_keep.drop(to_drop, axis=1, inplace=True)
#X_train, X_valid = split_vals(df_keep, n_trn)
X_train, X_test, y_train, y_test = train_test_split(df_keep, y, test_size=0.2, random_state=42)
#np.save('tmp/keep_cols.npy', np.array(df_keep.columns))
#keep_cols = np.load('tmp/keep_cols.npy')
#df_keep = df_trn[keep_cols]
df_keep.head()
reset_rf_samples()
rf_refined = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
rff = rf_refined.fit(X_train, y_train)

rff_predict = rff.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_rff_predict=np.log(rff_predict+1)
log_y_test=np.log(y_test+1)

mse = mean_squared_error(log_y_test, log_rff_predict)
rmsqe = np.sqrt(mse)
print("RMSQE rf_best: ", rmsqe)
print("oob_score: ", rff.oob_score_)
#ok,  now we have refined the dataset, just to features we really need, the features that we need to dig into!
#but before that, let's review some other models as well, to see scores: 

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)

lr_predict = lr_fit.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_lr_predict=np.log(lr_predict+1)
log_y_test=np.log(y_test+1)

mse = mean_squared_error(log_y_test, log_lr_predict)
rmsqe = np.sqrt(mse)
print("RMSQE lr: ", rmsqe)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet 

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

l2Regr = Ridge(alpha=9.0, fit_intercept = True)
l2Regr.fit(X_train, y_train)
pred_test_l2 = l2Regr.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_l2Regr_predict=np.log(pred_test_l2+1)
log_y_test=np.log(y_test+1)

mse = mean_squared_error(log_y_test, log_l2Regr_predict)
rmsqe = np.sqrt(mse)
print("Ridge l2r: ", rmsqe)
from sklearn.ensemble import GradientBoostingRegressor

myGBR = GradientBoostingRegressor(n_estimators=400, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state = 5) 
    
myGBR.fit(X_train,y_train)
pred_GBR = myGBR.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
log_GBR_predict=np.log(pred_GBR+1)
log_y_test=np.log(y_test+1)

mse = mean_squared_error(log_y_test, log_GBR_predict)
rmsqe = np.sqrt(mse)
print("GBR rmse: ", rmsqe)    
#from sklearn.svm import SVR

#SVR = SVR(kernel='poly', degree=3)
#SVR_fit = SVR.fit(X_train, y_train)

#SVR_predict = SVR_fit.predict(X_test)

#convert predictions and actuals to log scale, since that is how we are being evaluated. 
#log_SVR_predict=np.log(SVR_predict+1)
#log_y_test=np.log(y_test+1)

#mse = mean_squared_error(log_y_test, log_SVR_predict)
#rmsqe = np.sqrt(mse)
#print("RMSQE SVR: ", rmsqe)
#first lets see these features again
feat_labels = X_train.columns
importance = rff.feature_importances_  #determing feature importance from the algorithm"

df_feat = pd.DataFrame(importance, index=feat_labels)
df_feat = df_feat.reset_index()
df_feat = df_feat.rename(index=str, columns={0: "importance", 'index': 'feature'})
df_feat = df_feat.sort_values('importance', ascending=False, inplace=False, axis=0)
#df_feat.head()
plot_fi(df_feat[:30]);
#let's start again with the train_df, but using just the geatures we want (ie start processing these features again without the categorisation done before)
#note: train_df_aside: this is the train_df I put aside originally
train_df_top_feat = train_df_aside[to_keep]
train_df_top_feat.head()
train_df_top_feat.drop(to_drop, axis=1, inplace=True) #dropping the similar features with colinearity
train_df_top_feat.head()
#looking at null data
null_data = pd.DataFrame(train_df_top_feat.isnull().sum().sort_values(ascending=False))

null_data.columns = ['Null Count']
null_data.index.name = 'Feature'
null_data

(null_data/len(train_df_top_feat)) * 100  # change it to %'s'
# Visualising missing data
f, ax = plt.subplots(figsize=(20, 7));
plt.xticks(rotation='90');
sns.barplot(x=null_data.index, y=null_data['Null Count']);
plt.xlabel('Features', fontsize=15);
plt.ylabel('Percent of missing values', fontsize=15);
plt.title('Percent missing data by feature', fontsize=15);
#train_df_top_feat['GarageType'].value_counts() #digging down into some features
#in this case replace Nans with 0 since if they are missing they simply dont have it. 
train_df_top_feat['FireplaceQu'] = train_df_top_feat['FireplaceQu'].fillna(0)
train_df_top_feat['LotFrontage'] = train_df_top_feat['LotFrontage'].fillna(0)
train_df_top_feat['GarageType'] = train_df_top_feat['GarageType'].fillna(0)
train_df_top_feat['BsmtQual'] = train_df_top_feat['BsmtQual'].fillna(0)

train_df_top_feat['SalePrice'] = train_df['SalePrice']  #add sale price back in for analysis
train_df_top_feat.info()
#first let's review the top 5 features against sale price.  Then can dig down into those graphs
from pandas.plotting import scatter_matrix
attributes = ['SalePrice','OverallQual', 'GrLivArea', 'YearBuilt', 'GarageArea', '1stFlrSF']
scatter_matrix(train_df_top_feat[attributes], figsize=(20,12))
#GrLivArea looks to have some outliers, let's drill down.  
train_df_top_feat.plot('1stFlrSF', 'SalePrice', 'scatter', alpha=0.2, figsize=(20,8));  #not using the log saleprice, because in the scaled diagram hard to visualise outliers.
#can fidn quite a few outliers eg (year built vs sale price)

##another good way to plot: 
#plt.figure(figsize = (10,7))
#sns.regplot('GarageArea','SalePrice',data=train_df,color = 'red');
#removing outliers from visual analysis
train_df_top_feat = train_df_top_feat.drop(train_df_top_feat[(train_df_top_feat['1stFlrSF']>2000) & (train_df_top_feat['SalePrice']>700000)].index)
train_df_top_feat = train_df_top_feat.drop(train_df_top_feat[(train_df_top_feat['1stFlrSF']>3000) & (train_df_top_feat['SalePrice']<300000)].index)
train_df_top_feat.plot('1stFlrSF', 'SalePrice', 'scatter', alpha=0.2, figsize=(20,8));
#---
##looking at distribution of the training set vs test set
from pandas.plotting import scatter_matrix
attributes = ['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageArea', '1stFlrSF']
scatter_matrix(X_test[attributes], figsize=(20,12))

#looking at distribution of the test set to make sure its similar
from pandas.plotting import scatter_matrix
attributes = ['OverallQual', 'GrLivArea', 'YearBuilt', 'GarageArea', '1stFlrSF']
scatter_matrix(test_df[attributes], figsize=(20,12))
#now lets get ready for Algorithm again
##my mistake before using log rather then actual
#train_df_top_feat['log_SalePrice']=np.log(train_df_top_feat['SalePrice']+1) #calculate log of sale prices
#train_df_top_feat = train_df_top_feat.drop('SalePrice', axis=1) #drop sale price

train_cats(train_df_top_feat) #convert string objects to pandas categories
df_trn, y, nas = proc_df(train_df_top_feat, 'SalePrice') #, max_n_cat=7
df_trn.head()
#run algorithm - sue entire training set df_trn
#use RF-->
reset_rf_samples()
rf_nf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, max_depth = 30, max_features=0.5, n_jobs=-1, oob_score=True)
rf_nff = rf_nf.fit(df_trn, y)

#useGBR -->
myGBR = GradientBoostingRegressor(n_estimators=400, learning_rate=0.02,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=50,
                                      loss='huber', random_state = 5) 
    
GBR_final = myGBR.fit(df_trn, y)


#ignore---
#rff_p1 = rf_nff.predict(X_test)

#mse = mean_squared_error(y_test, rff_p1)
#rmsqe = np.sqrt(mse)
#print("RMSQE rf_best: ", rmsqe)
#print("oob_score: ", rff.oob_score_)
#print_score(m)

#----- predict the test set
test_df.head()
key = test_df['Id']
dropped_feat = ['Id']
test_df.drop(dropped_feat, axis=1, inplace=True)
test_df = test_df[to_keep]
test_df.drop(to_drop, axis=1, inplace=True)
test_df.info()
apply_cats(test_df, train_df_top_feat) #applying the categories we learnt on the training set
#df, y, nas = proc_df(train_df, 'log_SalePrice')
#df_test, = proc_df(test_df)
df_test, _, _ = proc_df(test_df) #, max_n_cat=7

df_test.info()
cols_to_drop =  ['GarageArea_na', 'BsmtFinSF1_na', 'LotFrontage_na']  #these were created on the set, so need to remove them
df_test = df_test.drop(cols_to_drop, axis=1)
df_test.info()
predictions = GBR_final.predict(df_test)
predictions
#prepare submission file
submission = pd.DataFrame({'Id':key,'SalePrice':predictions})

filename = 'house_prices_mark3.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
#next try playing with the hyper_parameters
