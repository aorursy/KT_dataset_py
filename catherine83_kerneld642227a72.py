import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import os

print(os.listdir("../input"))
##Ames, Iowa ... prediction prix des maisons

df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')

#concatener les 2 avant de faire les dummies

df_merged= pd.concat([df1, df2], axis=0, sort=False)
#total 81 columns et 1460 lignes

df_merged.info()

pd.isnull(df_merged).sum()

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]
na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]
def recap_nan(df_merged):

    

    feature_understanding = pd.DataFrame(columns=['feature_name','value_type','unique_value_number','unique_value_list','nan_percentage','nan_nbr']) 



    cols = list(df_merged.columns)

    for col in cols:

        #nan_nbr

        nb_nan = df_merged[col].isna().sum()

        #nan_percentage

        total = len(df_merged)

        nan_perc = nb_nan / total *100

        #value_type

        type_val = []

        #unique_value_list

        un_val_list = list(df_merged[col].unique()) 

        for val in un_val_list:

            if type(val) not in type_val:

                type_val.append(type(val))

        #unique_value_number        

        un_val_nb = len(un_val_list)

        line = { 'feature_name' : col, 'value_type' : type_val, 'unique_value_number' : un_val_nb, 

                'unique_value_list' : un_val_list, 'nan_percentage' : nan_perc,'nan_nbr':nb_nan}

        #append in feature_understanding df

        feature_understanding = feature_understanding.append(line,ignore_index=True)

    return feature_understanding[feature_understanding['nan_percentage'] > 0]



recap_nan(df_merged)
na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]            
#Explorer les variables ci-dessous pour remplir les nan

df_merged['MSZoning'].unique()

pd.isnull(df_merged.MSZoning).sum()

df_merged['MSZoning'].value_counts().plot(kind='bar')

values = {'MSZoning': 'RL'}

df_merged.fillna(value=values,inplace=True)
#MSZoning                

df_merged['MSZoning'].unique()

pd.isnull(df_merged.MSZoning).sum()

df_merged['MSZoning'].value_counts().plot(kind='bar')

values = {'MSZoning': 'RL'}

df_merged.fillna(value=values,inplace=True)



#LotFrontage           

pd.isnull(df_merged.LotFrontage).sum()                

df_merged.hist(column='LotFrontage')

df_merged['LotFrontage'].fillna((df_merged['LotFrontage'].mean()), inplace=True)



#Alley       

df_merged['Alley'].value_counts().plot(kind='bar')

pd.isnull(df_merged.Alley).sum()                

df_merged['Alley'].unique

values = {'Alley': 'NA'}

df_merged.fillna(value=values,inplace=True)



#Utilities  

df_merged['Utilities'].value_counts().plot(kind='bar')

pd.isnull(df_merged.Utilities).sum()                

values = {'Utilities': 'AllPub'}

df_merged.fillna(value=values,inplace=True)





#MasVnrType     

df_merged['MasVnrType'].value_counts().plot(kind='bar')

values = {'MasVnrType': 'None'}

df_merged.fillna(value=values,inplace=True)



#MasVnrArea    

pd.isnull(df_merged.MasVnrArea).sum()                

df_merged.hist(column='MasVnrArea')

median=df1['MasVnrArea'].median(skipna=True, numeric_only=None)

df_merged['MasVnrArea'].fillna((df_merged['MasVnrArea'].median()), inplace=True)



#BsmtQual  TO INTEGER

pd.isnull(df_merged.BsmtQual).sum()                

df_merged['BsmtQual'].value_counts().plot(kind='bar')

values = {'BsmtQual': 'NA'} #NA is String

df_merged.fillna(value=values,inplace=True)



#BsmtCond    TO INTEGER

pd.isnull(df_merged.BsmtCond).sum()

values = {'BsmtCond': 'NA'}  #NA is String   

df_merged.fillna(value=values,inplace=True)



#BsmtExposure

pd.isnull(df_merged.BsmtExposure).sum()

df_merged['BsmtExposure'].value_counts().plot(kind='bar')

values = {'BsmtExposure': 'NA'}

df_merged.fillna(value=values,inplace=True)



#BsmtFinType1    

pd.isnull(df_merged.BsmtFinType1).sum()

df_merged['BsmtFinType1'].value_counts().plot(kind='bar')

df_merged['BsmtFinType1'].unique()

values = {'BsmtFinType1': 'NA'}

df_merged.fillna(value=values,inplace=True)



#BsmtFinType2   

pd.isnull(df_merged.BsmtFinType2).sum()

df_merged['BsmtFinType2'].value_counts().plot(kind='bar')

df_merged['BsmtFinType2'].unique()

values = {'BsmtFinType2': 'NA'}

df_merged.fillna(value=values,inplace=True)



#BsmtFullBath   

pd.isnull(df_merged.BsmtFullBath).sum()

df_merged.hist(column='BsmtFullBath')

values = {'BsmtFullBath': 0}

df_merged.fillna(value=values,inplace=True)



#BsmtHalfBath   

pd.isnull(df_merged.BsmtHalfBath).sum()

df_merged.hist(column='BsmtHalfBath')

values = {'BsmtHalfBath': 0.}

df_merged.fillna(value=values,inplace=True)



#Functional  

pd.isnull(df_merged.Functional).sum()

df_merged['Functional'].unique()

df_merged['Functional'].value_counts().plot(kind='bar')

values = {'Functional': 'Typ'}

df_merged.fillna(value=values,inplace=True)



#FireplaceQu   .... TO INTEGER  

df_merged['FireplaceQu'].value_counts().plot(kind='bar')

df_merged['FireplaceQu'].unique()

values = {'FireplaceQu': 'NA'} #NA is String

df_merged.fillna(value=values,inplace=True)



#GarageType     

df_merged['GarageType'].value_counts().plot(kind='bar')

df_merged['GarageType'].unique()

values = {'GarageType': 'NA'}

df_merged.fillna(value=values,inplace=True)



#GarageYrBlt  

pd.isnull(df_merged.Functional).sum()



df1['GarageYrBlt'].unique()

df1.hist(column='GarageYrBlt')

values = {'GarageYrBlt': '0'} 

df_merged.fillna(value=values,inplace=True) 



#GarageFinish   

#Interior finish of the garage

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]

df_merged['GarageFinish'].value_counts().plot(kind='bar')

values = {'GarageFinish': 'NA'}

df_merged.fillna(value=values,inplace=True)



#GarageQual  ... TO INTEGER

#Garage quality

df_merged['GarageQual'].value_counts().plot(kind='bar')

df1['GarageQual'].unique()

values = {'GarageQual': 'NA'} #NA is string

df_merged.fillna(value=values,inplace=True)



#GarageCond    ... TO  INTEGER

#Garage condition

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]

df_merged['GarageCond'].value_counts().plot(kind='bar')

values = {'GarageCond': 'NA'}#NA is string

df_merged.fillna(value=values,inplace=True)



#PoolQC ... TO INTEGER

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]

df_merged['PoolQC'].value_counts().plot(kind='bar')

values = {'PoolQC': 'NA'} #NA is string

df_merged.fillna(value=values,inplace=True)



#Fence        2348  

#Fence quality

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]

df_merged['Fence'].value_counts().plot(kind='bar')

values = {'Fence': 'NA'}

df_merged.fillna(value=values,inplace=True)



#MiscFeature  2814  

na_df = pd.DataFrame(pd.isnull(df1).sum())

na_df[na_df[0] >= 1]

df_merged['MiscFeature'].value_counts().plot(kind='bar')

values = {'MiscFeature': 'NA'}

df_merged.fillna(value=values,inplace=True)



    #rechercher string 'nan'

pd.isnull(df_merged).sum() 

na_df = pd.DataFrame(pd.isnull(df_merged).sum()) #ne prend pas somme = 1

na_df[na_df[0] == 1]



#Electrical

df_merged['Electrical'].unique()

df_merged['Electrical'].value_counts().plot(kind='bar')

values = {'Electrical': 'SBrkr'}

df_merged.fillna(value=values,inplace=True)



#Exterior1st

pd.isnull(df_merged.Exterior1st).sum() 

df_merged['Exterior1st'].value_counts().plot(kind='bar')

values = {'Exterior1st': 'VinylSd'}

df_merged.fillna(value=values,inplace=True)



#Exterior2nd

pd.isnull(df_merged.Exterior2nd).sum() 

df_merged['Exterior2nd'].value_counts().plot(kind='bar')

values = {'Exterior2nd': 'VinylSd'}

df_merged.fillna(value=values,inplace=True)



#BsmtFinSF1

pd.isnull(df_merged.BsmtFinSF1).sum() 

df1.hist(column='BsmtFinSF1')

median=df1['BsmtFinSF1'].median(skipna=True, numeric_only=None)

df_merged['BsmtFinSF1'].fillna((df_merged['BsmtFinSF1'].median()), inplace=True)

temp = df_merged['BsmtFinSF1']



#BsmtFinSF2

pd.isnull(df_merged.BsmtFinSF2).sum() 

df1.hist(column='BsmtFinSF2')

median=df1['BsmtFinSF2'].median(skipna=True, numeric_only=None)

df_merged['BsmtFinSF2'].fillna((df_merged['BsmtFinSF2'].median()), inplace=True)

temp = df_merged['BsmtFinSF2']



#BsmtUnfSF

pd.isnull(df_merged.BsmtUnfSF).sum() 

df1.hist(column='BsmtUnfSF')



median=df1['BsmtUnfSF'].median(skipna=True, numeric_only=None)

df_merged['BsmtUnfSF'].fillna((df_merged['BsmtUnfSF'].median()), inplace=True)



#TotalBsmtSF

pd.isnull(df_merged.TotalBsmtSF).sum() 

df1.hist(column='TotalBsmtSF')

mean=df1['TotalBsmtSF'].mean(skipna=True, numeric_only=None)

df_merged['TotalBsmtSF'].fillna((df_merged['TotalBsmtSF'].mean()), inplace=True)



#KitchenQual 

pd.isnull(df_merged.KitchenQual).sum() 

df_merged['KitchenQual'].value_counts().plot(kind='bar')

values = {'KitchenQual': 'TA'}

df_merged.fillna(value=values,inplace=True)



#GarageCars

df_merged['GarageCars'].unique()

pd.isnull(df_merged.GarageCars).sum() 

df1.hist(column='GarageCars')

values = {'GarageCars': 0}

df_merged.fillna(value=values,inplace=True)



#GarageArea

df_merged['GarageArea'].unique()

pd.isnull(df_merged.GarageCars).sum() 

df1.hist(column='GarageArea')

values = {'GarageArea': 0}

df_merged.fillna(value=values,inplace=True)



#SaleType

df_merged['SaleType'].value_counts().plot(kind='bar')

values = {'SaleType': 'WD'}

df_merged.fillna(value=values,inplace=True)

df_merged['SaleType'].unique()

pd.isnull(df_merged.SaleType).sum() 
#index = Id

df_merged.set_index('Id', inplace=True)
#rassembler les variables numériques

df_numeriques =df_merged[['LotFrontage','LotArea','OverallQual',

                    'OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','ExterQual',

                    'ExterCond','BsmtQual','BsmtCond',

                    'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                    'TotalBsmtSF','HeatingQC','1stFlrSF',

                    '2ndFlrSF','LowQualFinSF','GrLivArea',

                    'BsmtFullBath','BsmtHalfBath','FullBath',

                    'HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual',

                    'TotRmsAbvGrd','Fireplaces','FireplaceQu',

                    'GarageCars','GarageArea','GarageQual',

                    'GarageCond','WoodDeckSF','OpenPorchSF',

                    'EnclosedPorch','3SsnPorch','ScreenPorch',

                    'PoolArea','PoolQC','MiscVal','MoSold',

                    'YrSold','SalePrice','GarageYrBlt']] 
#transormer 'GarageYrBlt' type integer

df_merged['GarageYrBlt'] = df_merged['GarageYrBlt'].astype(int)
#replace string by an integer 'notes d'évaluation'

mapping = {'Ex': 10, 'Gd': 8, 'TA':6 , 'Fa':4 ,'Po': 2, 'NA':0}

df_numeriques.replace({'ExterQual': mapping, 'ExterCond':mapping, 'BsmtQual':mapping,

                      'BsmtCond':mapping, 'HeatingQC':mapping,'KitchenQual':mapping, 

                      'FireplaceQu':mapping,

                      'GarageQual':mapping, 'GarageCond': mapping, 'PoolQC':mapping, 

                      }, inplace=True) #10 mappings #NE PAS RECLIQUER "df_numeriques =df_merged["
# convertir le type en integer

df_merged['MSSubClass'] = df_merged['MSSubClass'].astype(int).astype('str')

#rassembler les variables catégoriques

df_categoriques = df_merged[['MSSubClass','MSZoning','Street','Alley',

          'LotShape','LandContour','Utilities',

          'LotConfig','LandSlope','Neighborhood',

          'Condition1','Condition2','BldgType',

          'HouseStyle','RoofStyle','RoofMatl',

          'Exterior1st','Exterior2nd','MasVnrType',

          'Foundation','Heating','CentralAir',

          'Electrical','Functional','GarageType',

          'GarageFinish','PavedDrive','Fence','MiscFeature',

          'SaleType','SaleCondition','BsmtFinType2',

          'BsmtExposure','BsmtFinType1']] 
#créer les variables dummies

df_categoriques=pd.get_dummies(df_categoriques,drop_first=True)#5 valeurs, et 4 dummy OK !

#concaténer le tout

df_concat = pd.concat([df_numeriques, df_categoriques], axis=1, sort=False)    

#repérer ou spliter

house_price =df_concat['SalePrice']

#spliter en train et test df

dfs = np.split(df_concat, [1460], axis=0)

dfs[0] #train

dfs[1] #test



#déclarer les variables

#train df

train_features = dfs[0].drop('SalePrice', axis = 1) 

train_labels= dfs[0]['SalePrice'] 

#test df

test_features = dfs[1].drop('SalePrice', axis = 1) 

test_labels= dfs[1]['SalePrice'] 

    
#train df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_features, #on split le train

                                train_labels, test_size = 0.25, random_state = 42)



from sklearn.ensemble import  RandomForestRegressor

model = RandomForestRegressor(n_estimators = 1000, random_state = 42)



# Fit on training data

model.fit(X_train, y_train)

y_pred_train_set = model.predict(X_test) 

#score sur le train df

from sklearn.metrics import mean_squared_log_error

score =  np.sqrt(mean_squared_log_error(y_pred_train_set,y_test))

score



# R^2 

##Accuracy=TP+TNTP/(TN+FP+FN)

score = model.score(X_test,y_test)

score
#FEATURES SELECTION



fi = pd.DataFrame({'feature': list(train_features.columns),

                   'importance': model.feature_importances_}).sort_values('importance', ascending = False)



fi.head()

train_features = dfs[0][['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','BsmtFinSF1']] 

train_labels= dfs[0]['SalePrice'] 



test_features = dfs[1][['OverallQual','GrLivArea','TotalBsmtSF','2ndFlrSF','BsmtFinSF1']] 

test_labels= dfs[1]['SalePrice'] 



#sur le train df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels, test_size = 0.25, random_state = 42)
from sklearn.ensemble import RandomForestRegressor 

model = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_depth = None, 

                               min_samples_split = 2, min_samples_leaf = 1)



# Fit on training data

model.fit(X_train, y_train)

y_pred_train_set = model.predict(X_test)
from sklearn.metrics import mean_squared_log_error

score =  np.sqrt(mean_squared_log_error(y_pred_train_set,y_test))

score



# R^2 

score = model.score(X_test,y_test)

score

#Out[135]: 0.8721801649043771
from pprint import pprint



# Regarder les paramètre actuel  forest

print('Parameters currently in use:\n')

pprint(model.get_params())



from sklearn.model_selection import GridSearchCV



# créer le param_grid based 

#1 * 4 * 3 * 3 * 3 * 4 = 288 combinations of settings



param_grid = {

    'bootstrap': [True], 

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3, 5], 

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]} #nbre de simulations

# Create a based model

rf = RandomForestRegressor()



# Instancier le grid search model avec le param_grid

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



# Fit the grid search pour avoir les meilleurs paramètres du GridSearchCV

grid_search.fit(train_features, train_labels)

# IDEM

grid_search.best_params_

'''

Out[108]: 

{'bootstrap': True,

 'max_depth': 90,

 'max_features': 3,

 'min_samples_leaf': 3,

 'min_samples_split': 8,

 'n_estimators': 100}

'''



#OU BIEN avoir les meilleurs estimateurs du RandomForestRegressor

best_grid = grid_search.best_estimator_

best_grid

''''

Out[110]: 

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=90,

           max_features=3, max_leaf_nodes=None, min_impurity_decrease=0.0,

           min_impurity_split=None, min_samples_leaf=3,

           min_samples_split=8, min_weight_fraction_leaf=0.0,

           n_estimators=100, n_jobs=None, oob_score=False,

           random_state=None, verbose=0, warm_start=False)

'''
y_pred_train_set = grid_search.predict(X_test)
#Scoring avant  .predict(test_features) sur le test df

from sklearn.metrics import mean_squared_log_error

score =  np.sqrt(mean_squared_log_error(y_pred_train_set,y_test))

score

# R^2 

score = grid_search.score(X_test,y_test)

score



#We see that the accuracy was boosted to almost 100%.
#.predict sur le test df

y_pred_test_set = grid_search.predict(test_features)
# Index to df

df_y_pred_test_set = pd.DataFrame({'col':y_pred_test_set}) #float to df

list_index=dfs[1].index.tolist() #index to list

df_index=pd.DataFrame(list_index,columns=['Id']) #list to df

# y_pred_test_set to df

df_result= pd.concat([df_index,df_y_pred_test_set], axis=1)  

df_result.rename(index=str, columns={"col": "SalePrice"}, inplace=True)

df_result.to_csv("output5.csv",index=False)
