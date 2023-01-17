# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt
house_train = pd.read_csv("../input/train.csv")

house_test = pd.read_csv("../input/test.csv")

#this step puts the saleprice and ID at front of data

full = pd.concat([house_train,house_test])

cols = list(full)

cols.insert(0, cols.pop(cols.index('Id')))

cols.insert(1, cols.pop(cols.index('SalePrice')))

full = full.ix[:,cols]

#separating numeric and categorical features into two datasets

numeric_feats = full.select_dtypes(include=['int','int64','float','float64'])

cat_feats = full.select_dtypes(include=['object'])
#For looping to imputer similar missing categorical variables with less code

cols_list =  cat_feats.columns.tolist()

import itertools as it

for i in it.chain(cols_list[0:9],cols_list[11:12], cols_list[15:17],cols_list[19:25],cols_list[27:30],cols_list[32:39]):

    cat_feats[i]=np.where(cat_feats[i].isnull(),'NA',cat_feats[i])
#These variables below had so little missing values that we just imputed the most common level

cat_feats['Electrical']=np.where(cat_feats['Electrical'].isnull(), 'SBrkr', cat_feats['Electrical'])

cat_feats['Exterior1st']=np.where(cat_feats['Exterior1st'].isnull(), 'Other', cat_feats['Exterior1st'])

cat_feats['Exterior2nd']=np.where(cat_feats['Exterior2nd'].isnull(), 'Other', cat_feats['Exterior2nd'])

cat_feats['KitchenQual']=np.where(cat_feats['KitchenQual'].isnull(), 'TA', cat_feats['KitchenQual'])

cat_feats['Functional']=np.where(cat_feats['Functional'].isnull(), 'Typ', cat_feats['Functional'])

cat_feats['SaleType']=np.where(cat_feats['SaleType'].isnull(), 'Oth', cat_feats['SaleType'])

cat_feats['MSZoning']=np.where(cat_feats['MSZoning'].isnull(), 'RL', cat_feats['MSZoning'])
#Checking the missing values of the cat data

miss_check = cat_feats.isnull().sum()
#Now we can check the variables and see which ones are bad data 

freq = {}

for i in cols_list:

    freq[i] = cat_feats[i].value_counts()

#print(freq) 
#Utilities was horrible so we are dropping that from the data

cat_feats = cat_feats.drop('Utilities',axis=1)



#We will drop Street from the data too since only 2 levels exist and only 12 are gravel

cat_feats = cat_feats.drop('Street',axis=1)



#Dropping poolQC because it is all one value except like 6

cat_feats = cat_feats.drop('PoolQC',axis=1)
cat_feats['Alley']=np.where(cat_feats['Alley'] != 'NA', 'Has_alley','None')

cat_feats['BldgType']=np.where(cat_feats['BldgType'] != '1Fam', 'other', '1Fam')

cat_feats['BsmtCond']=np.where(cat_feats['BsmtCond'] != 'TA', 'other', 'TA')

cat_feats['BsmtExposure']=np.where(cat_feats['BsmtExposure'] != 'No', 'other', 'No')

cat_feats.ix[cat_feats.BsmtFinType1.isin(['ALQ','Rec','BLQ','LwQ','NA']), 'BsmtFinType1'] = 'other'

cat_feats['BsmtFinType2']=np.where(cat_feats['BsmtFinType2'] != 'Unf', 'other', 'Unf')

cat_feats.ix[cat_feats.BsmtQual.isin(['Ex','Fa','NA']), 'BsmtQual'] = 'other'

cat_feats['Condition1']=np.where(cat_feats['Condition1'] != 'Norm', 'other',cat_feats['Condition1'])

cat_feats['Condition2']=np.where(cat_feats['Condition2'] != 'Norm', 'other',cat_feats['Condition2'])

cat_feats['Electrical']=np.where(cat_feats['Electrical'] != 'SBrkr', 'other','SBrkr')

cat_feats['ExterCond']=np.where(cat_feats['ExterCond'] != 'TA', 'other', 'TA')

cat_feats['ExterQual']=np.where(cat_feats['ExterQual'] != 'TA', 'other', 'TA')

cat_feats.ix[cat_feats.Exterior1st.isin(['BrkFace','WdShing','AsbShng','Stucco','BrkComm','Stone','CBlock','AsphShn',\

'ImStucc','Other']), 'Exterior1st'] = 'Other'

cat_feats.ix[cat_feats.Exterior2nd.isin(['BrkFace','Wd Shng','AsbShng','Stucco','Brk Cmn','Stone','CBlock','AsphShn',\

'ImStucc','Other', 'Wd Sdng']), 'Exterior2nd'] = 'Other'

cat_feats['Fence']=np.where(cat_feats['Fence'] != 'NA', 'Has_Fence','None')

cat_feats['FireplaceQu']=np.where(cat_feats['FireplaceQu'] != 'NA', 'Has_fireplace','None')

cat_feats.ix[cat_feats.Foundation.isin(['BrkTil','Slab','Stone','Wood']), 'Foundation'] = 'other'

cat_feats['Functional']=np.where(cat_feats['Functional'] != 'Typ', 'other','Typ')

cat_feats['GarageCond']=np.where(cat_feats['GarageCond'] != 'TA', 'other', 'TA')

cat_feats['GarageQual']=np.where(cat_feats['GarageQual'] != 'TA', 'other', 'TA')

cat_feats.ix[cat_feats.GarageType.isin(['BuiltIn', 'NA', 'Basment','2Types','CarPort']), 'GarageType'] = 'other'

cat_feats['Heating']=np.where(cat_feats['Heating'] != 'GasA', 'other', 'GasA')

cat_feats['HeatingQC']=np.where(cat_feats['HeatingQC'] != 'Ex', 'other', 'Ex')

cat_feats.ix[cat_feats.HouseStyle.isin(['1.5Fin','SLvl','SFoyer','2.5Unf','1.5Unf','2.5Fin']), 'HouseStyle'] = 'other'

cat_feats.ix[cat_feats.KitchenQual.isin(['Ex','Fa']), 'KitchenQual'] = 'other'

cat_feats['LandContour']=np.where(cat_feats['LandContour'] != 'Lvl', 'other', 'Lvl')

cat_feats['LotConfig']=np.where(cat_feats['LotConfig'] != 'Inside', 'other', 'Inside')

cat_feats['LotShape']=np.where(cat_feats['LotShape'] != 'Reg', 'other', 'Reg')

cat_feats['MSZoning']=np.where(cat_feats['MSZoning'] != 'RL', 'other', 'RL')

cat_feats['MasVnrType']=np.where(cat_feats['MasVnrType'] != 'None', 'Has_it', 'None')

cat_feats['MiscFeature']=np.where(cat_feats['MiscFeature'] != 'NA', 'Has_Feature','None')

cat_feats['RoofMatl']=np.where(cat_feats['RoofMatl'] != 'CompShg', 'other','CompShg')

cat_feats['RoofStyle']=np.where(cat_feats['RoofStyle'] != 'Gable', 'other','Gable')

cat_feats['SaleCondition']=np.where(cat_feats['SaleCondition'] != 'Normal', 'other','Normal')

cat_feats['SaleType']=np.where(cat_feats['SaleType'] != 'WD', 'other','WD')
#Getting the dummy variables

cat_feats = pd.get_dummies(cat_feats)
#Dropping reference levels I need to figure out how to make a program to drop levels

cat_feats = cat_feats.drop(['Alley_None','BldgType_1Fam','BsmtCond_TA','BsmtExposure_No',\

'BsmtFinType1_other','BsmtFinType2_Unf','BsmtQual_TA','CentralAir_Y','Condition1_Norm',\

'Condition2_Norm','Electrical_SBrkr','ExterCond_TA','ExterQual_TA','Exterior1st_VinylSd',\

'Exterior2nd_VinylSd','Fence_None','FireplaceQu_None','Foundation_PConc','Functional_Typ',\

'GarageCond_TA','GarageFinish_Unf','GarageQual_TA','GarageType_Attchd','Heating_GasA',\

'HeatingQC_Ex','HouseStyle_1Story','KitchenQual_TA','LandContour_Lvl','LandSlope_Gtl',\

'LotConfig_Inside','LotShape_Reg','MSZoning_RL','MasVnrType_None','MiscFeature_None',\

'Neighborhood_NAmes','PavedDrive_Y','RoofMatl_CompShg','RoofStyle_Gable','SaleCondition_Normal',\

'SaleType_WD'], axis=1)
#Now that we finished the categorical imputations we will do an easy median imputation for numerics

saleprice = numeric_feats.ix[:,[0,1]]

numeric_feats = numeric_feats.drop(['Id','SalePrice'], axis=1)

numeric_feats=numeric_feats.fillna(numeric_feats.median())

miss_check = numeric_feats.isnull().sum()
#Lets combine the numeric and categorical features together then split back to the test and training sets.

full = pd.concat([saleprice,numeric_feats,cat_feats], axis=1)

house_train = full[~full['SalePrice'].isnull()]

house_test = full[full['SalePrice'].isnull()]
#In this step we will transform the target

house_train['SalePrice'] = np.log1p(house_train['SalePrice'])
#Making xtrain and y values

id_values = full.ix[:,[0,1]]

full = full.drop(['Id','SalePrice'], axis=1)

x_train = full[:house_train.shape[0]]

x_test = full[house_train.shape[0]:]

y_train = house_train.SalePrice
#Below will be steps for Lasso regression

from sklearn.linear_model import LassoCV

from sklearn.cross_validation import cross_val_score



def rmse_cv(model):

    rmse=np.sqrt(-cross_val_score(model, x_train, y_train, scoring='mean_squared_error', cv=5))

    return(rmse)



model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(x_train,y_train)

rmse_cv(model_lasso).mean()
#checking the residuals

matplotlib.rcParams['figure.figsize'] = (10.0,10.0)

preds = pd.DataFrame({'preds':model_lasso.predict(x_train),'true':y_train})

preds['residuals']=preds['true']-preds['preds']

preds.plot(x = 'preds', y='residuals',kind='scatter')
#using the model on the test dataset

lasso_preds=np.expm1(model_lasso.predict(x_test))



final_product = pd.DataFrame({'SalePrice':lasso_preds,'id':house_test.Id })

final_product.to_csv('Lasso_1.csv', index=False)