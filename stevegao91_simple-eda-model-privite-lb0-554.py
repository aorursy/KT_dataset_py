import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_file="../input/mercedes-benz-greener-manufacturing/train.csv.zip"

test_file="../input/mercedes-benz-greener-manufacturing/test.csv.zip" 

#Don't know why I need .zip to read data from kaggle environment
# read datasets

train = pd.read_csv(train_file)

test=pd.read_csv(test_file)

full=pd.concat([train,test])
train.shape,test.shape,full.shape
train.sample(10)
#check types of features

train.dtypes
train.dtypes.value_counts()
train.isna().sum().sum()
test.isna().sum().sum()
#data is clean!
#X0 to X8 have type object
full["X0"].value_counts()
#check details and the relationship to y for each nominal column using box plot
from IPython.core.pylabtools import figsize # import figsize

figsize(10, 15) 
fig,ax=plt.subplots(4,1)

sns.boxplot(x='X0',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[0])

ax[0].set_title('X0')

sns.boxplot(x='X1',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[1])

ax[1].set_title('X1')

sns.boxplot(x='X2',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[2])

ax[2].set_title('X2')

sns.boxplot(x='X3',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[3])

ax[3].set_title('X3')

fig.tight_layout()
fig,ax=plt.subplots(4,1)

sns.boxplot(x='X4',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[0])

ax[0].set_title('X4')

sns.boxplot(x='X5',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[1])

ax[1].set_title('X5')

sns.boxplot(x='X6',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[2])

ax[2].set_title('X6')

sns.boxplot(x='X8',y='y',data=train,width=0.5,fliersize=0.7,ax=ax[3])

ax[3].set_title('X8')

fig.tight_layout()
#Some features do have impact on y

#They will be dectected in feature selection
#check distribution of other columns

figsize(6,3)

full['X10'].hist()
#looks like all other columns have binary value 0,1 check if its true

df_int=full.select_dtypes(include=['int64'])

df_int.drop('ID',axis=1,inplace=True)
#Only exist 0 ,1 in dataframe

np.unique(df_int.values)
#plot with y. 

sns.violinplot(x='X10',y='y',data=train,width=0.5,fliersize=0.7)
#Check distribution of each columns

#Count 0s and 1s first

df_int_info=pd.DataFrame(columns=df_int.columns)

for (c_name,c) in df_int.iteritems():

#     print(c.value_counts())

    df_int_info.loc[0,c_name]=c.value_counts()[0]

    df_int_info.loc[1,c_name]=c.value_counts()[1]
#Check percentage

for (c_name,c) in df_int_info.iteritems():

    df_int_info.loc['percentage',c_name]=c[1]/8418
#plot

df_int_info.loc['percentage'].plot.bar()
#distribution. Most columns have huge

df_int_info.loc['percentage'].hist()
#The calculation can also be done only using mean() of each column cause they only have 1 and 0

plt.hist(df_int.mean())
full_obj=full[full.columns[full.dtypes=='object']]

full_obj.columns
#use pd_get_dummies

full_obj_dummies=pd.get_dummies(full_obj)

full_obj_dummies.shape
from sklearn.decomposition import PCA

pca=PCA()
#get df only with

pca_feautres=pca.fit_transform(np.array(df_int.values))
#plot explained variance

plt.plot(pca.explained_variance_ratio_)
#plot less

plt.plot(pca.explained_variance_ratio_[0:50])
#Consider only include 20 pca features in further steps

pca20=PCA(n_components=20)

pca20_features=pca20.fit_transform(df_int)

pca20_features.shape
plt.plot(pca20.explained_variance_ratio_)
#put into dataframe with name like pca1,pca2 etc

pca20_features_df=pd.DataFrame(pca20_features,columns=['pca'+str(x) for x in range(1,21)])

pca20_features_df.sample()
#put all features together

full_obj_dummies.shape,df_int.shape,pca20_features_df.shape
#index of pca20_features is not the same as others, do reindex to

#dataframes, the data will not change

full_obj_dummies.reset_index(drop=True,inplace=True)

df_int.reset_index(drop=True,inplace=True)
#concatenat

full_features=pd.concat([full_obj_dummies,df_int,pca20_features_df],axis=1)

full_features.shape


from sklearn.model_selection import train_test_split

X=full_features[:4209]

y=train.y

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#use random forest with default parameters

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor()
#fit data

rf.fit(X_train,y_train)
#check score

rf.score(X_train,y_train),rf.score(X_test,y_test)
#Fund overfitting. Use gridsearch to find better params

from sklearn.model_selection import GridSearchCV
# parameters

params={

    'max_depth':[2,3,4,5],

    'min_samples_leaf':[2,4,6]

}

#define grid search

grid=GridSearchCV(estimator=rf,param_grid=params)
#fit data

grid_result=grid.fit(X_train,y_train)
#result of grid search

# also use grid_result.cv_results_ to find more details

grid_result.best_score_,grid_result.best_params_
#build new model, fit data, and look feature importance

rf=RandomForestRegressor(max_depth=3,min_samples_leaf=2)

rf.fit(X_train,y_train)
#put into dataframe

feature_importance_df=pd.DataFrame(data=rf.feature_importances_.reshape(1,-1),columns=X_train.columns,index=['rf'])
#same for gradient boosting regressor

from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

#params

params={'learning_rate':[0.1,0.05,0.01],'min_samples_leaf':[3,6],'max_depth':[3,5]}

#grid search

gridCV=GridSearchCV(estimator=gbr,param_grid=params)

gridCV.fit(X_train,y_train)

#best params

gridCV.best_score_,gridCV.best_params_
#new model with best params

gbr=GradientBoostingRegressor(learning_rate=0.05,max_depth=3,min_samples_leaf=6)

gbr.fit(X_train,y_train)
#put features importance into dataframe

feature_importance_df.loc['gbr']=gbr.feature_importances_
# transpose ,sort

feature_importance_df_t=feature_importance_df.transpose()

feature_importance_df_t=feature_importance_df_t.sort_values(by='rf',ascending=False)
# and plot the top features

figsize(15,4)

feature_importance_df_t.iloc[:50].plot.bar()
#the feature importance from rf and gbr are consistent

#Top features include X314,X315.. from binary features

#and pca6 pca7 

#and X1_f, X8_a etc from nominal features

#considering only using part of features to modeling
#part of features list

f30=feature_importance_df_t.index[0:30]

f50=feature_importance_df_t.index[0:50]

f100=feature_importance_df_t.index[0:100]

f300=feature_importance_df_t.index[0:300]

flist=[f30,f50,f100,f300]

print(f30)
#all datasets with different numbers of features

subdataset_list=[]

for f in flist:

    X_train_subset=X_train[f]

    subdataset_list.append(X_train_subset)
#models

rf=RandomForestRegressor(max_depth=3,min_samples_leaf=2)

lr=LinearRegression()

dt=DecisionTreeRegressor()

gbr=GradientBoostingRegressor(learning_rate=0.05,max_depth=3,min_samples_leaf=6)

models=[rf,lr,dt,gbr]

models_name=['randomforest','linear','decisiontree','gradientboostingregressor']
#cross validation for mutiple data (feature) sets * models

score_df=pd.DataFrame(columns=['data','model','mean','scores'])

index=0

for subsetname,X_sub in zip(['f30','f50','f100','f300'],subdataset_list):

    for model,model_name in zip(models,models_name):

        scores=cross_val_score(model,X,y,cv=5)

        means=scores.mean()

        print('data',subsetname,model_name,'mean:',means,'cv:',scores)

        score_df.loc[str(index)]=[subsetname,model_name,means,scores]

        index+=1
#score of all methods

#score doesn't change with increasing number of features, consider only using 30 features

score_df
#plot

score_df['mean'].plot.bar()

plt.ylim((0,1))
#use sns for better plot

sns.barplot(y='mean',x='data',hue='model',data=score_df[score_df['mean']>=0])
#final model

gbr=GradientBoostingRegressor(learning_rate=0.05,max_depth=3,min_samples_leaf=6)
#using all data (4209 rows),30 features as training set.

gbr.fit(X[f30],y)

gbr.score(X[f30],y)
#predict

X_pred=full_features[4209:]

pred=gbr.predict(X_pred[f30])
#save as dataframe with original ID

pred_df=pd.DataFrame(data=pred,index=test.index,columns=['y'])
pred_df.sample(10)
#save as

pred_df.to_csv('pred.csv')