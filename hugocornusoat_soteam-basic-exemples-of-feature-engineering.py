# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



train = pd.read_csv("../input/train.csv")

# ON copie train pour ne pas l'altérer

data = train.copy()
data.head(10)
train.describe().transpose()
corr_matrix = data.corr()

# on s'interesse aux features qui ont une correlation proche de 1 ou de -1

(corr_matrix['SalePrice']**2).sort_values().tail(10)
missing_data = pd.DataFrame()

missing_data['Total'] = data.isnull().sum()

missing_data['Percent'] = (data.isnull().sum()/data.shape[0])*100

missing_data['type'] = data.dtypes

missing_data.sort_values(['Percent'], ascending=[False], inplace=True)



missing_data.head(20)
#On degage toutes les colonnes où il manque plus de 20% des données

data = data.drop((missing_data[missing_data['Percent'] > 20]).index,axis =1)



#Pour les valeurs numériques, LotFronatage,MasVnrArea

#On impute la valeur en prenant la moyenne

data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace=True)

data['MasVnrArea'].fillna(data['MasVnrArea'].mean(),inplace=True)



#Pour les GarageXXX, on voit que c'est les mêmes lignes qui posent problèmes, on les vire

data = data.drop(data[data['GarageYrBlt'].isnull()].index)



#Pour les BsmXXX, on va carrement dropper les colonnes parce qu'elle porte sur la qualité de la cave

data = data.drop(['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'],axis=1,errors='ignore')



#Pour MasVnrType, on impute la valeur la plus Fréquente 

data['MasVnrType'].fillna(data['MasVnrType'].mode()[0],inplace=True)



#Pour Electrical on vire juste la ligne manquante

data = data.drop(data[data['Electrical'].isnull()].index)





data.reset_index(drop=True, inplace=True)

data.isnull().sum().max()
data.hist(column=['SalePrice'],bins=20)

data['SalePrice'].describe()
extreme_price = data['SalePrice'].mean()+5*data['SalePrice'].std()

data = data.drop((data[data['SalePrice'] > extreme_price]).index)

data.reset_index(drop=True, inplace=True)
data.shape
ordinal_features = ['Street','LandContour','Utilities'] # On y mets aussi les binaires

categorical_features = ['SaleCondition','Foundation','Heating']

numeric_features = data.select_dtypes('number').columns
# On fait les ordianals

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

data[ordinal_features] = oe.fit_transform(data[ordinal_features])

data[ordinal_features].describe().transpose()
# On fait les non ordianals (Ils n'ont pas d'ordres)

# On utilse du one-hot encoding 

# Amélioration possible : le one n'est pas forcément idéal partout



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first',sparse=False)  # On met drop First pour virer une categorie par colone

encoded = ohe.fit_transform(data[categorical_features])

#data[categorical_features] = ohe.fit_transform(data[categorical_features])

#data[categorical_features].describe().transpose()

categories = []

feature_index = 0

for feature_categories in ohe.categories_ : 

    feature_categories_ = feature_categories[1:]  # On drop la first

    for category in feature_categories_ : 

        categories.append(categorical_features[feature_index]+'-'+category)

    feature_index = feature_index+1

e = pd.DataFrame(encoded, columns = categories)





data[categories] = e

data[categories].describe().transpose()


#data['YearBuilt2'] = data['YearBuilt']**(3)

#data['CentralAir2'] = data.apply((lambda x: 1 if x['CentralAir'] == 'Y' else 0),axis = 1)



#data['LotFrontage'] = data.apply((lambda x: 70 if  math.isnan(x['LotFrontage'])  else x['LotFrontage']),axis = 1)



#data['1Story'] = data.apply((lambda x: 1 if x['HouseStyle'] == '1Story' else 0),axis = 1)

#data['1.5Fin'] = data.apply((lambda x: 1 if x['HouseStyle'] == '1.5Fin' else 0),axis = 1)

#data['1.5Unf'] = data.apply((lambda x: 1 if x['HouseStyle'] == '1.5Unf' else 0),axis = 1)

#data['2Story'] = data.apply((lambda x: 1 if x['HouseStyle'] == '2Story' else 0),axis = 1)

#data['2.5Fin'] = data.apply((lambda x: 1 if x['HouseStyle'] == '2.5Fin' else 0),axis = 1)

#data['2.5Unf'] = data.apply((lambda x: 1 if x['HouseStyle'] == '2.5Unf' else 0),axis = 1)

#data['SFoyer'] = data.apply((lambda x: 1 if x['HouseStyle'] == 'SFoyer' else 0),axis = 1)



















#data.plot.scatter(x= 'YearBuilt', y= 'SalePrice') 











test = data[['HouseStyle','MSZoning','YearBuilt']]



train.select_dtypes('object').head(10)

from sklearn.linear_model import LinearRegression



X = data.select_dtypes('number').drop(columns=['SalePrice'])

Y = data[['SalePrice']]



reg = LinearRegression().fit(X, Y)

reg.score(X,Y)
# Methode bourrine (pas de feature engineering)



# Pour la regression lineaire on ne garde que les colonnes à valeurs continues 

onlyNumeric = train.select_dtypes('number')

# On drop les lignes qui ont des valeurs vides 

onlyNumericCleaned = onlyNumeric.dropna()



X = onlyNumericCleaned.drop(columns=['SalePrice'])

Y = onlyNumericCleaned[['SalePrice']]

reg = LinearRegression().fit(X, Y)

reg.score(X,Y)
