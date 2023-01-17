#handy libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#get the data loaded into dataframes

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print ('data loaded')

print (str(len(train_df))+" rows for training set")

print (str(len(test_df))+" rows for test set")
#take a peek at the data

train_df.head(3)
#get all numeric features here

num_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1',

'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'

,'BsmtHalfBath','GarageYrBlt','GarageArea','WoodDeckSF','OpenPorchSF',

                'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','YrSold']



#for years I engineered a feature of YrSold - YearBuilt, to give age of the home

train_df['age'] = train_df['YrSold']-train_df['YearBuilt']

train_df['age_since_remod'] = train_df['YrSold']-train_df['YearRemodAdd']



#adding a dummy variable for if there IS a 2nd floor

def val_dummy(val):

    if val > 0:

        return 1

    else:

        return 0



#2nd floor dummy

train_df['second_floor'] = train_df['2ndFlrSF'].apply(val_dummy)

#wood deck dummy

train_df['wood_deck'] = train_df['WoodDeckSF'].apply(val_dummy)
#check out numeric feature distributions 1 by 1, look for which seem less informative

train_df[num_features[12]].hist(bins=25,figsize=(5,3))
#so far, informative features (high enough count and variance after removing outliers) seem to be:

num_keepers = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF',

              '1stFlrSF','GrLivArea','GarageArea','OpenPorchSF']



#those I am throwing out

num_exclude = [obj for obj in num_features if obj not in num_keepers]



#drop the num_excludes

for z in range(len(num_exclude)):

    

     #get variable name

    var_name = num_exclude[z]

    

    #delete variable from training df

    train_df = train_df.drop(var_name, 1)   



#some other fields need to be kepts but not transformed later



#create dummy for MoSold to control for seasonality

#create the dummies

dummy_df = pd.get_dummies(train_df['MoSold'])    

#concat the new columns to the training data

train_df = pd.concat([train_df,dummy_df],axis=1, join_axes=[train_df.index]) 

#get rid of old column

train_df = train_df.drop('MoSold',1)



#other fields, such as integers, that should be in another list

other_num_keepers = ['BsmtFullBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr'

                     ,'TotRmsAbvGrd','Fireplaces','GarageCars']
#look at how some of the informative numeric variables relate to price

g = sns.PairGrid(train_df[['LotArea','BsmtFinSF1','SalePrice']])

g.map_offdiag(plt.scatter)

g.map_diag(plt.hist)
#going to go through each num_keeper 1 by 1 to see if there is a relationship

#looking last at lot size, since I think that matters

#look at how some of the informative numeric variables relate to price

g = sns.PairGrid(train_df[['BsmtFinSF1','SalePrice']])

g.map_offdiag(plt.scatter)

g.map_diag(plt.hist)
#looking at lot size by itself, since it pertains to most records

#very skewed, long tail, going to limit the max size of this feature for better detail

train_df["LotArea"][train_df["LotArea"]<30000].hist(bins=10,figsize=(6,3))

#should prolly plot this vs. price too
#trying regplot after removing skew from LotArea via logarithmic transform:

train_df['LotArea2'] = np.log1p(train_df["LotArea"])



print (train_df[['LotArea2','SalePrice']].corr())



#now do a regplot

ax2 = sns.regplot(x=train_df["LotArea2"],

                  y=train_df["SalePrice"], color="g")



#relationship seem to get a lil stronger when I transform SalePrice too

train_df['SalePrice2'] = np.log1p(train_df["SalePrice"])

print (train_df[['LotArea2','SalePrice2']].corr())



#now drop that new column

train_df = train_df.drop('LotArea2',1)
#going to do logistic transformation on all num_keepers to deal with skew

for z in range(len(num_keepers)):

    

    train_df[num_keepers[z]] = np.log1p(train_df[num_keepers[z]])
#could plot all ints or other_num_keepers to see if there is a relationship with price
#categorical variables present:

cat_features = ['MSSubClass','MSZoning','Street','Alley','LotShape','Utilities','LotConfig','LandSlope'

                ,'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',

                'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',

                'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish',

                'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType',

               'SaleCondition']



#visualize ALL of these puppies:

#in case I want to save these: sns_plot.savefig("output.png")
sns.countplot(x=cat_features[0],data=train_df)
#I was looking for categories that did not only have a HUGE majority in one class - unbalanced

#and also had a significant amount of counts

#I.e., I was looking for potential information in a variable,

#not sure if there are significant relationships with price yet though



#Variables I am going to keep after eye-balling include: 

cat_keepers = ['MSSubClass','MSZoning','Alley','LotShape','LotConfig','Neighborhood','BldgType',

              'HouseStyle','RoofStyle','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

              'Foundation','BsmtQual','BsmtExposure','BsmtFinType1','HeatingQC','CentralAir'

              ,'KitchenQual','FireplaceQu','GarageType','GarageFinish','Fence','SaleCondition']



#those I am throwing out

cat_exclude = [obj for obj in cat_features if obj not in cat_keepers]



#unfortunately, one-hot encoding may render some of these less-useful,

#it will create more dimensions and I need less, more-valuable D
#now I will check price levels across the category values, want to find variance in price

#first create the one-hot encoding for cat_keepers



#loop through each category and create dummies

for z in range(len(cat_keepers)):

    

    #get variable name

    var_name = cat_keepers[z]



    #create the dummies

    dummy_df = pd.get_dummies(train_df[var_name])    

    

    #delete original variable from training df

    train_df = train_df.drop(var_name, 1)

    

    #concat the new columns to the training data

    train_df = pd.concat([train_df,dummy_df],axis=1, join_axes=[train_df.index]) 
#drop the cat_excludes

for z in range(len(cat_exclude)):

    

     #get variable name

    var_name = cat_exclude[z]

    

    #delete variable from training df

    train_df = train_df.drop(var_name, 1)   
#peak at new train_df

train_df.head(3)
#ordinal variables include ratings: OveralQual, OveralCond,

#check out ordinal distributions

area_attr = pd.DataFrame({"Overall Qual":train_df["OverallQual"],

                             "Overall Condition":train_df['OverallCond']

                            })

area_attr.hist(bins=10,figsize=(10,4))
#looks like I need not transform 10-scale variables into categorical or dummies

#I will look at correlations

print (train_df[['OverallQual','SalePrice2']].corr())



#now do a regplot

ax2 = sns.regplot(x=train_df["OverallQual"],

                  y=train_df["SalePrice2"], color="g")
#looks like I need not transform 10-scale variables into categorical or dummies

#I will look at correlations

print (train_df[['OverallCond','SalePrice2']].corr())



#now do a regplot

ax2 = sns.regplot(x=train_df["OverallCond"],

                  y=train_df["SalePrice2"], color="g")
#throwing out the 2nd variable

train_df = train_df.drop('OverallCond', 1)   
#going to run all informative attribute through Lasso to see what sticks...

from sklearn.linear_model import LassoCV



predictor_list = num_keepers+other_num_keepers+['second_floor','wood_deck']



#add the cat dummies next

''',

                     20,30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190, 'C (all)',

 'FV', 'RH', 'RL', 'RM', 'Grvl', 'Pave', 'IR1', 'IR2', 'IR3', 'Reg', 'Corner', 'CulDSac', 'FR2',

 'FR3', 'Inside', 'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor',

 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge',

 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker',

 '1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE', '1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf',

 '2Story', 'SFoyer', 'SLvl', 'Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed', 'AsbShng',

 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Plywood',

 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace',

 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc', 'MetalSd', 'Other', 'Plywood', 'Stone', 'Stucco',

 'VinylSd', 'Wd Sdng', 'Wd Shng', 'BrkCmn', 'BrkFace', 'None', 'Stone', 'Ex', 'Fa', 'Gd', 'TA', 'BrkTil',

 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood', 'Ex', 'Fa', 'Gd', 'TA', 'Av', 'Gd', 'Mn', 'No', 'ALQ',

 'BLQ', 'GLQ', 'LwQ', 'Rec', 'Unf', 'Ex', 'Fa', 'Gd', 'Po', 'TA', 'N', 'Y', 'Ex', 'Fa', 'Gd', 'TA',

 'Ex', 'Fa', 'Gd', 'Po', 'TA', '2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd',

 'Fin', 'RFn', 'Unf', 'GdPrv', 'GdWo', 'MnPrv', 'MnWw', 'Abnorml', 'AdjLand', 'Alloca', 'Family',

 'Normal', 'Partial', 'OverallCond', 'age','age_since_remod']

'''

y = train_df['SalePrice2']

X = train_df[predictor_list]

#model = LassoCV().fit(X, y)

#handle blanks & stuff, put rules of imputation here
#At the end, to transform predicted prices back to normal, if I transform price,

#will need to use np.expm1() method