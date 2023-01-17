import pandas as pd

import numpy as np

#importing data 

train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

#creating a combined dataset 

test['SalePrice']=999

train['SalePrice']=np.log(train.SalePrice)

train['train']=1

test['train']=0

combined=pd.concat((train,test),axis=0)
# checking for columns with missing values

missing_bool=combined.isnull().any()

missing_cols=missing_bool[missing_bool==True]



#collecting data type of columns with missing values 

dtypes=train[missing_cols.index.tolist()].dtypes

#collecting % of missing values for missing columns 

missing_per=train[missing_cols.index.tolist()].isnull().sum()/train.shape[0]

#merging both

missing_table=pd.concat((dtypes,missing_per),axis=1)

missing_table
#replacing with median

float_missing=dtypes[dtypes=='float64'].index.tolist()

for col in float_missing:

    median_val=train[col].median()

    combined[col]=combined[col].fillna(median_val)



#replacing with mode and creating indicator for missing rows

obj_missing=dtypes[dtypes=='object'].index.tolist()

for col in obj_missing:

    mode_val=train[col].mode()[0]

    combined[col+'_missingind']=np.where(combined[col].isnull(),1,0)

    combined[col]=combined[col].fillna(mode_val)

#converting character variabels to category codes   

dtypes_all=combined.dtypes[combined.dtypes=='object'].index.tolist()

for col in dtypes_all:

    combined[col]=combined[col].astype('category').cat.codes
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

model=RandomForestRegressor(n_estimators=500,max_samples=0.8,min_samples_leaf=30,max_features='sqrt',oob_score=True)

dep_var=[col for col in combined.columns.tolist() if col not in ['Id','SalePrice','train']]

params={'n_estimators':[100,300,400],'max_samples':[0.5,0.8],'max_features':['sqrt'],'min_samples_leaf':[30,60],'max_depth':[5,8,10]}

#scoring function

import math

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

from sklearn.metrics import make_scorer

negative_rmse=make_scorer(rmse,greater_is_better=False)



grid=GridSearchCV(model,param_grid=params,scoring='r2')

grid.fit(combined[combined.train==1][dep_var],combined[combined.train==1]['SalePrice'])
#variable reduction using feature importance

grid.best_params_
# variable reduction using only feature importance and dropping one element every step

base_var=dep_var.copy()

performance={}

max_perf=-1

while(len(base_var)>=5):

    model=RandomForestRegressor(max_depth= 10,max_features='sqrt',max_samples=0.8,min_samples_leaf=30,n_estimators=400,oob_score=True)

    model.fit(combined[combined.train==1][base_var],combined[combined.train==1]['SalePrice'])

    sorted_idx=np.argsort(-model.feature_importances_)

    #storing oob score for diagonstic purposes

    performance[len(base_var)]=model.oob_score_

    if model.oob_score_>max_perf:

        best_model=model

        max_perf=model.oob_score_

        best_var=base_var.copy()

    #removing list important variable from the predictors list

    base_var.pop(sorted_idx[-1])
from matplotlib import pyplot as plt

fig=plt.figure()

fig.canvas.draw()

# first plot -dropping one variable at a time

ax1=fig.add_subplot(111, label="1")

xvalues1=[i for i in range(0,98)]

yvalues1=[performance[102-i] for i in range(0,98)]

ax1.plot(xvalues1,yvalues1)

ax1.set_xlabel("#variables in model")

ax1.set_ylabel("R-square on OOB sample")

new_labels = [103-(i-1)*20 for i in range(6)]



ax1.set_xticklabels(labels=new_labels)

plt.show()
base_model=RandomForestRegressor(max_depth= 10,max_features='sqrt',max_samples=0.8,min_samples_leaf=30,n_estimators=400,oob_score=True)

base_model.fit(combined[combined.train==1][dep_var],combined[combined.train==1]['SalePrice'])

importance=base_model.feature_importances_

sorted_idx=np.argsort(-importance)

importance_dict={}

for idx in sorted_idx:

    importance_dict[dep_var[idx]]=importance[idx]
#variable reduction using feature importance and correlation

#cycle1

curr_var=list(importance_dict.keys())

threshold_list=[0.8-i*0.1 for i in range(6)]

metrics=[]

for threshold in threshold_list:

    for idx,var1 in enumerate(curr_var):

        for var2 in curr_var[idx+1:]:

            corr=combined[combined.train==1][var1].corr(combined[combined.train==1][var2])

            if corr>threshold:

                curr_var.remove(var2)

    model=RandomForestRegressor(max_depth= 10,max_features='sqrt',max_samples=0.8,min_samples_leaf=30,n_estimators=400,oob_score=True)

    model.fit(combined[combined.train==1][curr_var],combined[combined.train==1]['SalePrice'])

    metrics.append( (threshold,len(curr_var),model.oob_score_) )
metrics
#dendrogram 

import scipy

from scipy.cluster import hierarchy as hc

corr = np.round(scipy.stats.spearmanr(combined[combined.train==1][best_var]).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,10))

dendrogram = hc.dendrogram(z, labels=best_var, 

      orientation='left', leaf_font_size=16)

plt.show()
def get_oob(drop_var):

    m = RandomForestRegressor(max_depth= 10,max_features='sqrt',

                              max_samples=0.8,min_samples_leaf=30,n_estimators=400,oob_score=True)

    pred_var=[var for var in best_var if var is not drop_var]

    m.fit(combined[combined.train==1][pred_var],combined[combined.train==1]['SalePrice'])

    return m.oob_score_



get_oob('KitchenQual')
from sklearn.decomposition import PCA



def pca_get_oob(n):

    pca=PCA(n_components=n)

    pca_data=pca.fit_transform(combined[combined.train==1][dep_var])

    pca_df=pd.DataFrame(data=pca_data,columns=['pca'+str(i) for i in range(n)])

    m = RandomForestRegressor(max_depth= 10,max_features='sqrt',

                              max_samples=0.8,min_samples_leaf=30,n_estimators=400,oob_score=True)

    m.fit(pca_df,combined[combined.train==1]['SalePrice'])

    return m.oob_score_



pca_rsquare=[]

for i in range(30,90):

    pca_rsquare.append(pca_get_oob(i))
from matplotlib import pyplot as plt

plt.plot(range(30,90),pca_rsquare)

plt.xlabel('n components')

plt.ylabel('Rsquare on OOB sample')

plt.show()