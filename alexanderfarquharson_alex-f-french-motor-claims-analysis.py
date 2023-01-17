from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np 

import os 

import pandas as pd

import seaborn as sns
nRowsRead = None

df = pd.read_csv('../input/freMTPL2freq.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'french_claims_data'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')



df['LogDensity'] = np.log(df['Density']).round(6)

df['Freq']=df['ClaimNb']/df['Exposure']
df.head()
df.describe()
#any null values

all(df.notnull())
sns.heatmap(data=df.corr(), cmap = 'viridis', annot=True)
#claims as func of age

#gives sum of exposure per category

EVY= df.groupby('DrivAge',as_index=False).agg({'Exposure': 'sum'})

#gives no.claims(weighted by exposure) per category

Freq= df.groupby('DrivAge',as_index=False).agg({'Freq': 'mean'})



plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(data=pd.pivot_table(index='DrivAge', values='Freq', data=df))

#sns.distplot(Freq)

plt.subplot(1,2,2)

sns.barplot(x='DrivAge', y='Exposure',data=df.groupby('DrivAge',as_index=False).agg({'Exposure': 'sum'}))
print('min driv age', df.DrivAge.min())

print('mean driv age',df.DrivAge.mean())

df[(df.DrivAge <20) | (df.DrivAge >90)]['DrivAge'].value_counts()
#claims as func of BousMalus, removed values of BonusMalus with low exposure

df=df.assign(Freq_BonusMalus=df.groupby('BonusMalus')['BonusMalus'].transform('count')/df['Exposure'])

by_bm = df.groupby('BonusMalus')[['Freq','Freq_BonusMalus']].mean()

by_bm=by_bm.sort_values(by=['Freq_BonusMalus'],ascending=False).drop('Freq_BonusMalus',axis=1).iloc[:56]



plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(data=by_bm)

#sns.distplot(Freq)

plt.subplot(1,2,2)

sns.barplot(x='BonusMalus', y='Exposure',data=df.groupby('BonusMalus',as_index=False).agg({'Exposure': 'sum'}))
#veh power exposure and claims

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.barplot(x='VehPower', y='Freq',data= df.groupby('VehPower',as_index=False).agg({'Freq': 'mean'}))

#sns.distplot(Freq)

plt.subplot(1,2,2)

sns.barplot(x='VehPower', y='Exposure',data=df.groupby('VehPower',as_index=False).agg({'Exposure': 'sum'}))
#veh age

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(data=pd.pivot_table(index='VehAge', values='Freq', data=df))

#sns.distplot(Freq)

plt.subplot(1,2,2)

sns.barplot(x='VehAge', y='Exposure',data=df.groupby('VehAge',as_index=False).agg({'Exposure': 'sum'}))
#area

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.barplot(x='Area', y='Freq',data= df.groupby('Area',as_index=False).agg({'Freq': 'mean'}))



plt.subplot(1,2,2)

sns.barplot(x='Area', y='Exposure',data=df.groupby('Area',as_index=False).agg({'Exposure': 'sum'}))
#Veh Brand

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.barplot(x='VehBrand', y='Freq',data= df.groupby('VehBrand',as_index=False).agg({'Freq': 'mean'}))



plt.subplot(1,2,2)

sns.barplot(x='VehBrand', y='Exposure',data=df.groupby('VehBrand',as_index=False).agg({'Exposure': 'sum'}))
#Veh Gas

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.barplot(x='VehGas', y='Freq',data= df.groupby('VehGas',as_index=False).agg({'Freq': 'mean'}))



plt.subplot(1,2,2)

sns.barplot(x='VehGas', y='Exposure',data=df.groupby('VehGas',as_index=False).agg({'Exposure': 'sum'}))
#Density

plt.figure(figsize=(20,8))



plt.subplot(1,4,1)

sns.scatterplot(data=pd.pivot_table(index='Density', values='Freq', data=df))

plt.title('Density')

plt.subplot(1,4,2)

sns.distplot(df.Density,bins=30)





plt.subplot(1,4,3)

sns.scatterplot(data=pd.pivot_table(index='LogDensity', values='Freq', data=df))

plt.title('LogDensity')



plt.subplot(1,4,4)

sns.distplot(df.LogDensity,bins=30)
#Veh REGION

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

sns.barplot(x='Region', y='Freq',data= df.groupby('Region',as_index=False).agg({'Freq': 'mean'}))



plt.subplot(1,2,2)

sns.barplot(x='Region', y='Exposure',data=df.groupby('Region',as_index=False).agg({'Exposure': 'sum'}))
df.drop(['Freq_BonusMalus', 'Density'],axis=1,inplace=True)
#unproportionally high number of claims at v. low exposure

print('low exposure, mean freq {}'.format(df[df['Exposure']<0.01]['Freq'].mean()))

print('high exposure, mean freq {}'.format(df[df['Exposure']>0.9]['Freq'].mean()))

print('overall, mean freq {}'.format(df['Freq'].mean()))

print('low exposure, mean claims {}'.format(df[df['Exposure']<0.01]['ClaimNb'].mean()))

print('high exposure, mean claims {}'.format(df[df['Exposure']>0.9]['ClaimNb'].mean()))

print('overall, mean claims {}'.format(df['ClaimNb'].mean()))

df['Exposure']=df['Exposure'].round(4)

df.drop('IDpol',axis=1).duplicated().value_counts()

df_dup = df[df.drop(['IDpol','Freq','ClaimNb'],axis=1).duplicated(keep=False)]

print(df[df.ClaimNb >5])

print(df_dup[df_dup.ClaimNb >5])
#predict freq of claims, so drop claimsnb and use regressor algorithm

#remove IDpol and Freq and make dummy variabls of categories

#use exposure column as weight

#use ClaimNb as target

df_modelling = df

df_modelling.head()
#sort out categorical variables into binary

Area_dummies = pd.get_dummies(df_modelling['Area'], drop_first=True)

VehBrand_dummies = pd.get_dummies(df_modelling['VehBrand'], drop_first=True)

VehGas_dummies = pd.get_dummies(df_modelling['VehGas'], drop_first=True)

Region_dummies = pd.get_dummies(df_modelling['Region'], drop_first=True)

df_modelling = pd.concat([df_modelling.drop('Area',axis=1),Area_dummies],axis=1)

df_modelling = pd.concat([df_modelling.drop('VehBrand',axis=1),VehBrand_dummies],axis=1)

df_modelling = pd.concat([df_modelling.drop('VehGas',axis=1),VehGas_dummies],axis=1)

df_modelling = pd.concat([df_modelling.drop('Region',axis=1),Region_dummies],axis=1)
df_modelling.head(0)
#splitting the data (AB)

from sklearn.model_selection import train_test_split



# Get index sorted with ascending IDpol, just in case it is out or order, also remove IDpol

df_all = df_modelling.sort_values('IDpol').reset_index(drop=True)



# Split out training data

df_train, df_not_train = train_test_split(df_all, test_size=0.3, random_state=51, shuffle=True)



# Split remaining data between validation and holdout

df_validation, df_holdout = train_test_split(df_not_train, test_size=0.5, random_state=13, shuffle=True)



X_train = df_train.drop(['Freq','ClaimNb','Exposure','IDpol'],axis=1)

y_train = df_train['ClaimNb']

X_validation = df_validation.drop(['Freq','ClaimNb','Exposure','IDpol'],axis=1)

y_validation = df_validation['ClaimNb']
#use of neg_mean_square_error in CV to score model

import sklearn

sorted(sklearn.metrics.SCORERS.keys())
#simple tree, no bagging, no boosting

#cv scored on mean squared error

#hyperoarameters tuned

#paper suggests min_leaf_sample(weighted) = 10000, max_leaf_nodes=12

#weighted by exposure, exposure removed as a feature

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

#instantiate model

dt= DecisionTreeRegressor()

#hypertuning parameters

params = {'max_leaf_nodes':[8,12,16,

                            20,24],

          'min_weight_fraction_leaf':[0.005

                            ,0.01,0.015,0.02,0.05

                            ]}

grid_dt = GridSearchCV(estimator=dt,param_grid=params,scoring='neg_mean_squared_error',cv=5,verbose=2)

grid_dt.fit(X_train,y_train,sample_weight=df_train['Exposure'].values)
#getting best hyperparameters

#Best params and model chosen to predict off validation set

best_hyperparams = grid_dt.best_params_

print(best_hyperparams)

best_CV_score = grid_dt.best_score_

print('best_CV_score',best_CV_score)

best_model = grid_dt.best_estimator_

y_pred=best_model.predict(X_validation)
#evaluation of model

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error as mse

print('rmse from validation data:',np.sqrt(mse(y_validation,y_pred, sample_weight=df_validation['Exposure'].values)))

#how do you weight on cv?

print('rmse from cv of train data (5 folds) unweighted:',np.sqrt(best_CV_score*-1))
importances=pd.Series(data=best_model.feature_importances_,

                      index=X_train.columns).sort_values(ascending=False).iloc[:8]

importances.plot(kind='bar')
#random forest model withoout grid search

#include boostrapping, no. features, estimators, using tuned hyperparameters from paper

from sklearn.ensemble import RandomForestRegressor

grid_rf=RandomForestRegressor(bootstrap = False, max_features='sqrt',max_leaf_nodes=20, min_weight_fraction_leaf=0.005, n_estimators=250)

grid_rf.fit(X_train,y_train,sample_weight=df_train['Exposure'].values)

best_model_rf = grid_rf

y_pred_rf=best_model_rf.predict(X_validation)
print('Random Forest rmse from validation data:',np.sqrt(mse(y_validation,y_pred_rf,sample_weight=df_validation['Exposure'].values)))

print('Single Tree rmse from validation data:',np.sqrt(mse(y_validation,y_pred, sample_weight=df_validation['Exposure'].values)))
claimnb_validation = pd.concat((pd.Series(y_pred,index=y_validation.index), pd.Series(y_pred_rf,index=y_validation.index), y_validation,df_validation['Exposure']),axis=1)

claimnb_validation.sort_index()

def weighted_avg(column,weights=claimnb_validation['Exposure']):

    column = pd.to_numeric(column)

    for x in column:

        weighted_avg = x*weights/weights.sum()

        return weighted_avg.sort_index()



claimnb_validation['y_pred weighted'] = weighted_avg(claimnb_validation[0])

claimnb_validation['y_pred_rf weighted'] = weighted_avg(claimnb_validation[1])

claimnb_validation['y_validation weighted'] = weighted_avg(pd.to_numeric(claimnb_validation['ClaimNb']))
df_validation1 = df_validation[['Exposure','ClaimNb']]

df_validation1.index = df_validation['IDpol']
#for random forest, is X_validation, best_model_rf, y_pred_rf 

rf_df = pd.DataFrame(index=df_validation['IDpol'], data = y_pred_rf,columns = ['Random Forest Predictions'])

rf_df = pd.concat((rf_df,df_validation1),axis=1).sort_index()

pd.DataFrame.to_pickle(rf_df,'Alex_Farquharson_rf_dataframe.gzip')
rf_df['weighted rf predictions']=rf_df['Random Forest Predictions']*rf_df['Exposure']

rf_df['weighted actual values']=rf_df['ClaimNb']*rf_df['Exposure']



print('weighted predicted result over weighted actual result',np.sum(rf_df['weighted rf predictions'])/np.sum(rf_df['weighted actual values']))

rf_df.sum()
importances_rf=pd.Series(data=best_model_rf.feature_importances_,

                      index=X_train.columns).sort_values(ascending=False).iloc[:20]

importances_rf.plot(kind='bar')

plt.title('influence of each feature on the model')
#plot of predicted freq and actual freq for the two models ( validation data set)

fig, _ = plt.subplots(2,4, figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(x=y_pred,y=y_validation)

plt.title('Single tree')

plt.subplot(1,2,2)

sns.scatterplot(x=y_pred_rf,y=y_validation)

plt.title('Random forest')
#plot of predicted freq and actual freq for the two models ( train data set)

fig, _ = plt.subplots(2,4, figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(x=best_model.predict(X_train),y=y_train)

plt.title('Single tree')

plt.subplot(1,2,2)

sns.scatterplot(x=best_model_rf.predict(X_train),y=y_train)

plt.title('Random forest')
def plot_two_histograms(data1, data_source1, data2, data_source2,bins1=20,bins2=20):

    fig, _ = plt.subplots(2, 2, figsize=(12, 4))

    plt.subplot(1,2,1)

    sns.distplot(data1,kde=False,bins =bins1)

    plt.title('Frequency of values from {}'.format(data_source1))

    plt.subplot(1,2,2)

    sns.distplot(data2,kde=False,bins=bins2)

    plt.title('Frequency of claims from {}'.format(data_source2))



plot_two_histograms(y_pred,'tree model',y_pred_rf, 'random forest model',bins1=100, bins2=100)
#removal of claims >  certain number

sns.distplot(df[df['ClaimNb']>4]['ClaimNb'],kde=False,bins=20)

df_no_fraud = df_modelling

df_no_fraud = df_no_fraud[df_no_fraud['ClaimNb']<=4]

df[df['ClaimNb']>4]
#re-doing with df_no_fraud data

# Get index sorted with ascending IDpol, just in case it is out or order, also remove IDpol

df_all_no_fraud = df_no_fraud.sort_values('IDpol').reset_index(drop=True)

df_all_no_fraud.drop('IDpol',axis=1,inplace=True)



# Split out training data

df_train_no_fraud, df_not_train_no_fraud = train_test_split(df_all, test_size=0.3, random_state=51, shuffle=True)



# Split remaining data between validation and holdout

df_validation_no_fraud, df_holdout_no_fraud = train_test_split(df_not_train, test_size=0.5, random_state=13, shuffle=True)



X_train_no_fraud = df_train_no_fraud.drop(['Freq','ClaimNb','Exposure'],axis=1)

y_train_no_fraud  = df_train_no_fraud['ClaimNb']

X_validation_no_fraud  = df_validation_no_fraud.drop(['Freq','ClaimNb','Exposure'],axis=1)

y_validation_no_fraud  = df_validation_no_fraud['ClaimNb']
#random forest model

#include boostrapping, no. features, estimators, using tuned hyperparameters from paper

from sklearn.ensemble import RandomForestRegressor

rf_no_fraud=RandomForestRegressor(min_weight_fraction_leaf=0.01,max_leaf_nodes=12, bootstrap = False, n_estimators = 100, max_features ='sqrt')

rf_no_fraud.fit(X_train_no_fraud,y_train_no_fraud,sample_weight=df_train_no_fraud['Exposure'].values)
y_pred_no_fraud=rf_no_fraud.predict(X_validation_no_fraud)

print(np.sqrt(mse(y_validation_no_fraud,y_pred_no_fraud,sample_weight=df_validation_no_fraud['Exposure'].values)))

print(y_pred.sum())

print(y_validation_no_fraud.sum())
fig, _ = plt.subplots(2,4, figsize=(15,8))

plt.subplot(1,2,1)

sns.scatterplot(x=y_pred_no_fraud,y=y_validation_no_fraud)

plt.subplot(1,2,2)

sns.scatterplot(x=rf_no_fraud.predict(X_train_no_fraud),y=y_train_no_fraud)
#quantile lift (best model is best_model_rf)

#1 sort validation data by predicted loss

#1.1 get df with predicted claimnb, actual claimnb id

predicted_claimnb = best_model_rf.predict(X_validation)

predicted_claimnb= pd.Series(predicted_claimnb)

predicted_claimnb.index=X_validation.index

df_lift = pd.concat((predicted_claimnb, y_validation, df_validation['Exposure']),axis=1)

df_lift.rename(columns={0:'Predicted ClaimNb'},inplace=True)

assert len(predicted_claimnb)==len(df_lift)==len(y_validation)

print('yes')

df_lift=df_lift.sort_values(by='Predicted ClaimNb',ascending=False)

df_lift.index=np.arange(1,101703)
#2bucket into equally weighted 

#2.1 cum exposure column

#2.2 make function tos plit by weights

#2.3 check its done

df_lift['Cum Exposure'] = df_lift['Exposure'].cumsum()



from pandas._libs.lib import is_integer



def weighted_qcut(values, weights, q,**kwargs):

    #Return weighted quantile cuts from a given series

    if is_integer(q):

        quantiles = np.linspace(0, 1, q + 1)

    else:

        quantiles = q

    order = weights.iloc[values.argsort()].cumsum() #makes series of cumulative exposure (sorted by values)

    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs) #cuts into q quantiles along order (cumulative exposure) column

    return bins.sort_index() #makes column in line with index of original dataframe





#add weighted column to dataframe

df_lift['weighted_cut'] = weighted_qcut(df_lift['Predicted ClaimNb'],df_lift['Exposure'],10,labels=False)

#check it worked

#assert df_lift[df_lift['weighted_cut']== 9]['Exposure'].sum().round(0) == df_lift[df_lift['weighted_cut']== 0]['Exposure'].sum().round(0)

#print('split by weights correctly')
#3 calculate means on predicted and real for each bucket

predicted_mean_values = []

actual_mean_values= []

for x in np.arange(10):

    Predicted_mean = df_lift[df_lift['weighted_cut']==x]['Predicted ClaimNb'].mean()

    predicted_mean_values.append(Predicted_mean)

    Actual_mean = df_lift[df_lift['weighted_cut']==x]['ClaimNb'].mean()

    actual_mean_values.append(Actual_mean)



colnames=['predicted']

means=pd.DataFrame(columns= colnames,data=predicted_mean_values)

means['actual'] = actual_mean_values

means['index']=np.arange(1,11)

means
sns.set_style('darkgrid')

sns.scatterplot(data=means,x='actual',y='actual')

sns.scatterplot(data=means, x='actual',y='predicted')

plt.ylim(0.02,0.12)

plt.ylabel('predicted and actual')

plt.title('Predicted (orange) and actual (blue) ClaimsNb')

print('model differentitation between top and bottom decile',means.iloc[9]['predicted'] / means.iloc[0]['predicted'])

print('actual differentitation between top and bottom decile',means.iloc[9]['actual'] / means.iloc[0]['actual'])
#attempt at double lift

#make dataframe

predicted_claimnb_rf = pd.Series(best_model_rf.predict(X_validation))

predicted_claimnb_tree = pd.Series(best_model.predict(X_validation))

predicted_claimnb_rf.index=X_validation.index

predicted_claimnb_tree.index=X_validation.index

df_d_lift = pd.concat((predicted_claimnb_rf,predicted_claimnb_tree, y_validation, df_validation['Exposure']),axis=1)

df_d_lift.rename(columns={0:'Predicted ClaimNb rf',1:'Predicted ClaimNb tree'},inplace=True)

assert len(predicted_claimnb_rf)==len(df_d_lift)==len(y_validation)==len(predicted_claimnb_tree)

print('yes')

df_d_lift['ratio']=df_d_lift['Predicted ClaimNb rf']/df_d_lift['Predicted ClaimNb tree']

df_d_lift=df_d_lift.sort_values(by='ratio',ascending=False)

df_d_lift.index=np.arange(1,101703)

df_d_lift



#split into quartiles

df_d_lift['weighted_cut'] = weighted_qcut(df_d_lift['ratio'],df_d_lift['Exposure'],10,labels=False)



#check it worked

#assert df_d_lift[df_d_lift['weighted_cut']== 9]['Exposure'].sum().round(0) == df_d_lift[df_d_lift['weighted_cut']== 0]['Exposure'].sum().round(0)

#print('split by weights correctly')



#calculate means on predicted and real for each bucket

predicted_rf_mean_values = []

predicted_tree_mean_values = []

actual_mean_values = []

for x in np.arange(10):

    Predicted_rf_mean = df_d_lift[df_d_lift['weighted_cut']==x]['Predicted ClaimNb rf'].mean()

    predicted_rf_mean_values.append(Predicted_rf_mean)

    Predicted_tree_mean = df_d_lift[df_d_lift['weighted_cut']==x]['Predicted ClaimNb tree'].mean()

    predicted_tree_mean_values.append(Predicted_tree_mean)

    Actual_mean = df_d_lift[df_d_lift['weighted_cut']==x]['ClaimNb'].mean()

    actual_mean_values.append(Actual_mean)



colnames=['predicted rf']

means=pd.DataFrame(columns= colnames,data=predicted_rf_mean_values)

means['predicted tree'] = predicted_tree_mean_values

means['actual'] = actual_mean_values

means['index']=np.arange(1,11)



#plot it

sns.scatterplot(data=means,x='actual',y='actual', color ='blue')

sns.scatterplot(data=means, x='actual',y='predicted rf',color ='orange')

sns.scatterplot(data = means, x='actual', y='predicted tree',color = "darkred")



plt.ylabel('predicted and actual')

plt.title('Predicted rf (orange), predicted tree (red) and actual (blue) ClaimsNb')

print('rf model differentitation between top and bottom decile',means.iloc[0]['predicted rf'] / means.iloc[9]['predicted rf'])

print('tree model differentitation between top and bottom decile',means.iloc[0]['predicted tree'] / means.iloc[9]['predicted tree'])

print('model differentitation between top and bottom decile',means.iloc[0]['actual'] / means.iloc[9]['actual'])
#for random forest, is X_validation, best_model_rf, y_pred_rf 

rf_df = pd.DataFrame(index=X_validation.index, data = y_pred_rf,columns = ['Random Forest Predictions'])

rf_df = pd.concat((rf_df,y_validation, df_validation['Exposure']),axis=1).sort_index()

pd.DataFrame.to_pickle(rf_df,'Alex_Farquharson_rf_dataframe.gzip')



pd.DataFrame.to_pickle(X_train,'Alex_Farquharson_X_train_dataframe.gzip')

pd.DataFrame.to_pickle(y_train,'Alex_Farquharson_y_train_dataframe.gzip')

pd.DataFrame.to_pickle(X_validation,'Alex_Farquharson_X_validation_dataframe.gzip')

pd.DataFrame.to_pickle(y_validation,'Alex_Farquharson_y_validation_dataframe.gzip')
import pickle

Alex_Farquharson_rf_model = pickle.dumps(best_model_rf)

best_model_rf_2 = pickle.loads(Alex_Farquharson_rf_model)
from joblib import dump,load

dump(best_model_rf,'rf_model.gzip')

best_model_rf_2 = load('rf_model.gzip')
#dataframe of models

models_list = [best_model, best_model_rf, rf_no_fraud]

model_name = ['best_model', 'best_model_rf', 'rf_no_fraud']

weighted = ['Yes','Yes','Yes']

trained_on = ['normal dataset', 'normal dataset', 'fraud omitted dataset']

target_variable = ['claimNb','claimNb','claimNb']

models = pd.DataFrame()

models['model']=models_list

models['model_name']=model_name

models['weighted']=weighted

models['trained_on']=trained_on

models['target_variable']=target_variable

models