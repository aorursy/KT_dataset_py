# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt



train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')



train.fillna(value='No',inplace='True')

test.fillna(value='No',inplace='True')



#print "AAAA"

#print test.head()

col= np.array(train.columns)

#print col



scaler=MinMaxScaler()



rescaled_lot_area_train=scaler.fit_transform(train['LotArea'])

rescaled_lot_area_test=scaler.fit_transform(test['LotArea'])

#print rescaled_lot_area_train

rescaled_gr_area_train=scaler.fit_transform(train['GrLivArea'])

rescaled_gr_area_test=scaler.fit_transform(test['GrLivArea'])



#plt.scatter(rescaled_lot_area,train['SalePrice'],c='blue',marker='s')

#plt.title('Looking for outliers')

#plt.xlabel('Lot Area')

#plt.ylabel('Sale Price')

#plt.show()



#plt.scatter(rescaled_gr_area,train['SalePrice'],c='blue',marker='s')

#plt.title('Looking for outliers')

#plt.xlabel('GR Area')

#plt.ylabel('Sale Price')

#plt.show()



#plt.scatter(rescaled_gr_area+rescaled_lot_area,train['SalePrice'],c='blue',marker='s')

#plt.title('Looking for outliers')

#plt.xlabel('GR Area')

#plt.ylabel('Sale Price')

#plt.show()



train['Re_gr_lot']=rescaled_lot_area_train+rescaled_gr_area_train

test['Re_gr_lot']=rescaled_lot_area_test+rescaled_gr_area_test

train=train[train['Re_gr_lot']<0.8]

#print train.head()



#plt.scatter(train['Re_gr_lot'],train['SalePrice'],c='blue',marker='s')

#plt.title('Looking for outliers')

#plt.xlabel('GR Area')

#plt.ylabel('Sale Price')

#plt.show()





#print train['MSZoning'].unique()



train['MSZoning']=train['MSZoning'].replace({'No':0,'A':1,'C (all)':2,'FV':3,'I':4,'RH':5,'RL':6,'RP':7,'RM':8})

test['MSZoning']=test['MSZoning'].replace({'No':0,'A':1,'C (all)':2,'FV':3,'I':4,'RH':5,'RL':6,'RP':7,'RM':8})



#print train['Street'].unique()

train['Street']=train['Street'].replace({'No':0,'Grvl':1,'Pave':2})

test['Street']=test['Street'].replace({'No':0,'Grvl':1,'Pave':2})



#print train['Alley'].unique()

test['Alley']=test['Alley'].replace({'No':0,'Grvl':1,'Pave':2})

train['Alley']=train['Alley'].replace({'No':0,'Grvl':1,'Pave':2})



#print train['LotShape'].unique()

train['LotShape']=train['LotShape'].replace({'No':0,'Reg':1,'IR1':2,'IR2':3,'IR3':4})

test['LotShape']=test['LotShape'].replace({'No':0,'Reg':1,'IR1':2,'IR2':3,'IR3':4})



train['LandContour']=train['LandContour'].replace({'No':0,'Lvl':1,'Bnk':2,'HLS':3,'Low':4})

test['LandContour']=test['LandContour'].replace({'No':0,'Lvl':1,'Bnk':2,'HLS':3,'Low':4})



train['Utilities']=train['Utilities'].replace({'No':0,'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':4})

test['Utilities']=test['Utilities'].replace({'No':0,'AllPub':1,'NoSewr':2,'NoSeWa':3,'ELO':4})



train['LotConfig']=train['LotConfig'].replace({'No':0,'Inside':1,'Corner':2,'CulDSac':3,'FR2':4,'FR3':5})

test['LotConfig']=test['LotConfig'].replace({'No':0,'Inside':1,'Corner':2,'CulDSac':3,'FR2':4,'FR3':5})



train['LandSlope']=train['LandSlope'].replace({'No':0,'Gtl':1,'Mod':2,'Sev':3})

test['LandSlope']=test['LandSlope'].replace({'No':0,'Gtl':1,'Mod':2,'Sev':3})



train['Neighborhood']=train['Neighborhood'].replace({'No':0,'Blmngtn':1,'Blueste':2,'BrDale':3,'BrkSide':4,'ClearCr':5,'CollgCr':6,'Crawfor':7,'Edwards':8,'Gilbert':9,'IDOTRR':10,'MeadowV':11,'Mitchel':12,'NAmes':13,'NoRidge':14,'NPkVill':15,'NridgHt':16,'NWAmes':17,'OldTown':18,'SWISU':19,'Sawyer':20,'SawyerW':21,'Somerst':22,'StoneBr':23,'Timber':24,'Veenker':25})

test['Neighborhood']=test['Neighborhood'].replace({'No':0,'Blmngtn':1,'Blueste':2,'BrDale':3,'BrkSide':4,'ClearCr':5,'CollgCr':6,'Crawfor':7,'Edwards':8,'Gilbert':9,'IDOTRR':10,'MeadowV':11,'Mitchel':12,'NAmes':13,'NoRidge':14,'NPkVill':15,'NridgHt':16,'NWAmes':17,'OldTown':18,'SWISU':19,'Sawyer':20,'SawyerW':21,'Somerst':22,'StoneBr':23,'Timber':24,'Veenker':25})



train['Condition1']=train['Condition1'].replace({'No':0,'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9})

test['Condition1']=test['Condition1'].replace({'No':0,'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9})



train['Condition2']=train['Condition2'].replace({'No':0,'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9})

test['Condition2']=test['Condition2'].replace({'No':0,'Artery':1,'Feedr':2,'Norm':3,'RRNn':4,'RRAn':5,'PosN':6,'PosA':7,'RRNe':8,'RRAe':9})



train['BldgType']=train['BldgType'].replace({'No':0,'1Fam':1,'2fmCon':2,'Duplex':3,'TwnhsE':4,'TwnhsI':5,'Twnhs':6})

test['BldgType']=test['BldgType'].replace({'No':0,'1Fam':1,'2fmCon':2,'Duplex':3,'TwnhsE':4,'TwnhsI':5,'Twnhs':6})



train['HouseStyle']=train['HouseStyle'].replace({'No':0,'1Story':1,'1.5Fin':2,'1.5Unf':3,'2Story':4,'2.5Fin':5,'2.5Unf':6,'SFoyer':7,'SLvl':8})

test['HouseStyle']=test['HouseStyle'].replace({'No':0,'1Story':1,'1.5Fin':2,'1.5Unf':3,'2Story':4,'2.5Fin':5,'2.5Unf':6,'SFoyer':7,'SLvl':8})



train['RoofStyle']=train['RoofStyle'].replace({'No':0,'Flat':1,'Gable':2,'Gambrel':3,'Hip':4,'Mansard':5,'Shed':6})

test['RoofStyle']=test['RoofStyle'].replace({'No':0,'Flat':1,'Gable':2,'Gambrel':3,'Hip':4,'Mansard':5,'Shed':6})



train['RoofMatl']=train['RoofMatl'].replace({'No':0,'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4,'Roll':5,'Tar&Grv':6,'WdShake':7,'WdShngl':8})

test['RoofMatl']=test['RoofMatl'].replace({'No':0,'ClyTile':1,'CompShg':2,'Membran':3,'Metal':4,'Roll':5,'Tar&Grv':6,'WdShake':7,'WdShngl':8})



train['Exterior1st']=train['Exterior1st'].replace({'No':0,'AsbShng':1,'AsphShn':1,'BrkComm':2,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'CemntBd':5,'HdBoard':6,'ImStucc':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'PreCast':13,'Stone':14,'Stucco':15,'VinylSd':16,'Wd Shng':17,'WdShing':18,'Wd Sdng':19})

test['Exterior1st']=test['Exterior1st'].replace({'No':0,'AsbShng':1,'AsphShn':1,'BrkComm':2,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'CemntBd':5,'HdBoard':6,'ImStucc':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'PreCast':13,'Stone':14,'Stucco':15,'VinylSd':16,'Wd Shng':17,'WdShing':18,'Wd Sdng':19})



train['Exterior2nd']=train['Exterior2nd'].replace({'No':0,'AsbShng':1,'AsphShn':1,'BrkComm':2,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'PreCast':13,'Stone':14,'Stucco':15,'VinylSd':16,'Wd Shng':17,'WdShing':18,'Wd Sdng':19})

test['Exterior2nd']=test['Exterior2nd'].replace({'No':0,'AsbShng':1,'AsphShn':1,'BrkComm':2,'Brk Cmn':2,'BrkFace':3,'CBlock':4,'CmentBd':5,'HdBoard':6,'ImStucc':7,'ImStucc':8,'MetalSd':9,'Other':10,'Plywood':11,'PreCast':12,'PreCast':13,'Stone':14,'Stucco':15,'VinylSd':16,'Wd Shng':17,'WdShing':18,'Wd Sdng':19})



train['MasVnrType']=train['MasVnrType'].replace({'No':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'None':4,'Stone':5})

test['MasVnrType']=test['MasVnrType'].replace({'No':0,'BrkCmn':1,'BrkFace':2,'CBlock':3,'None':4,'Stone':5})





train['ExterQual']=train['ExterQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['ExterQual']=test['ExterQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['ExterCond']=train['ExterCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['ExterCond']=test['ExterCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['BsmtCond']=train['BsmtCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['BsmtCond']=test['BsmtCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['BsmtQual']=train['BsmtQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['BsmtQual']=test['BsmtQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['Foundation']=train['Foundation'].replace({'No':0,'BrkTil':1,'CBlock':2,'PConc':3,'Slab':4,'Stone':5,'Wood':6})

test['Foundation']=test['Foundation'].replace({'No':0,'BrkTil':1,'CBlock':2,'PConc':3,'Slab':4,'Stone':5,'Wood':6})



train['BsmtExposure']=train['BsmtExposure'].replace({'Gd':5,'Av':4,'Mn':3,'No':2})

test['BsmtExposure']=test['BsmtExposure'].replace({'Gd':5,'Av':4,'Mn':3,'No':2})



train['BsmtFinType1']=train['BsmtFinType1'].replace({'No':0,'GLQ':1,'ALQ':2,'BLQ':3,'Rec':4,'LwQ':5,'Unf':6})

test['BsmtFinType1']=test['BsmtFinType1'].replace({'No':0,'GLQ':1,'ALQ':2,'BLQ':3,'Rec':4,'LwQ':5,'Unf':6})



train['BsmtFinType2']=train['BsmtFinType2'].replace({'No':0,'GLQ':1,'ALQ':2,'BLQ':3,'Rec':4,'LwQ':5,'Unf':6})

test['BsmtFinType2']=test['BsmtFinType2'].replace({'No':0,'GLQ':1,'ALQ':2,'BLQ':3,'Rec':4,'LwQ':5,'Unf':6})



train['HeatingQC']=train['HeatingQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['HeatingQC']=test['HeatingQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['Heating']=train['Heating'].replace({'No':0,'Floor':1,'GasA':2,'GasW':3,'Grav':4,'OthW':5,'Wall':6})

test['Heating']=test['Heating'].replace({'No':0,'Floor':1,'GasA':2,'GasW':3,'Grav':4,'OthW':5,'Wall':6})



train['CentralAir']=train['CentralAir'].replace({'N':0,'Y':1,'No':0})

test['CentralAir']=test['CentralAir'].replace({'N':0,'Y':1,'No':0})



train['Electrical']=train['Electrical'].replace({'SBrkr':1,'FuseA':2,'FuseF':3,'FuseP':4,'Mix':5,'No':0})

test['Electrical']=test['Electrical'].replace({'SBrkr':1,'FuseA':2,'FuseF':3,'FuseP':4,'Mix':5,'No':0})



train['KitchenQual']=train['KitchenQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['KitchenQual']=test['KitchenQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['FireplaceQu']=train['FireplaceQu'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['FireplaceQu']=test['FireplaceQu'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['Functional']=train['Functional'].replace({'No':0,'Typ':1,'Min1':2,'Min2':3,'Mod':4,'Maj1':5,'Maj2':6,'Sev':7,'Sal':8})

test['Functional']=test['Functional'].replace({'No':0,'Typ':1,'Min1':2,'Min2':3,'Mod':4,'Maj1':5,'Maj2':6,'Sev':7,'Sal':8})



train['GarageType']=train['GarageType'].replace({'No':0,'2Types':1,'Attchd':2,'Basment':3,'BuiltIn':4,'CarPort':5,'Detchd':6})

test['GarageType']=test['GarageType'].replace({'No':0,'2Types':1,'Attchd':2,'Basment':3,'BuiltIn':4,'CarPort':5,'Detchd':6})



train['GarageFinish']=train['GarageFinish'].replace({'No':0,'Fin':1,'RFn':2,'Unf':3})

test['GarageFinish']=test['GarageFinish'].replace({'No':0,'Fin':1,'RFn':2,'Unf':3})



train['GarageQual']=train['GarageQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['GarageQual']=test['GarageQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['GarageCond']=train['GarageCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['GarageCond']=test['GarageCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['PavedDrive']=train['PavedDrive'].replace({'No':0,'Y':1,'P':2,'N':0})

test['PavedDrive']=test['PavedDrive'].replace({'No':0,'Y':1,'P':2,'N':0})



train['PoolQC']=train['PoolQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})

test['PoolQC']=test['PoolQC'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No':0})



train['Fence']=train['Fence'].replace({'No':0,'GdPrv':1,'MnPrv':2,'GdWo':3,'MnWw':4})

test['Fence']=test['Fence'].replace({'No':0,'GdPrv':1,'MnPrv':2,'GdWo':3,'MnWw':4})



train['MiscFeature']=train['MiscFeature'].replace({'No':0,'Elev':1,'Gar2':2,'Othr':3,'Shed':4,'TenC':5,'Tenc':5})

test['MiscFeature']=test['MiscFeature'].replace({'No':0,'Elev':1,'Gar2':2,'Othr':3,'Shed':4,'TenC':5,'Tenc':5})



train['SaleType']=train['SaleType'].replace({'No':0,'WD':1,'CWD':2,'VWD':3,'New':4,'COD':5,'Con':6,'ConLw':7,'ConLI':8,'ConLD':9,'Oth':10})

test['SaleType']=test['SaleType'].replace({'No':0,'WD':1,'CWD':2,'VWD':3,'New':4,'COD':5,'Con':6,'ConLw':7,'ConLI':8,'ConLD':9,'Oth':10})



train['SaleCondition']=train['SaleCondition'].replace({'No':0,'Normal':1,'Abnorml':2,'AdjLand':3,'Alloca':4,'Family':5,'Partial':6})

test['SaleCondition']=test['SaleCondition'].replace({'No':0,'Normal':1,'Abnorml':2,'AdjLand':3,'Alloca':4,'Family':5,'Partial':6})





#print train.head()

#print test.head()



corr = train.corr()

corr.sort_values(['SalePrice'],ascending=False,inplace=True)

#print corr['SalePrice']





categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

#print("Numerical features : " + str(len(numerical_features)))

#print("Categorical features : " + str(len(categorical_features)))

train_num = train[numerical_features]

train_cat = train[categorical_features]





num_features=test.select_dtypes(include=['object']).columns

cat_features=test.select_dtypes(exclude=['object']).columns



test_num=test[num_features]

test_cat=test[cat_features]



#print train.shape



#print cat_features



#print train_num

#print train_cat



#print train['MSZoning'].unique()

#print "AAAAAAAAA"

#print train['LotFrontage'].unique()

#print "AAAAAAAAA"

#print train['Exterior1st'].unique()

#print "AAAAAAAAA"

#print train['Exterior2nd'].unique()

#print "AAAAAAAAA"

#print train['MasVnrArea'].unique()

#print "AAAAAAAAA"

#print train['GarageYrBlt'].unique()

#print "AAAAAAAAA"

#print train['MiscFeature'].unique()

#print "AAAAAAAAA"



X = pd.concat([train_num, train_cat], axis = 1)

y = train['SalePrice']



X_test_set=pd.concat([test_num,test_cat], axis = 1)



#print X_test_set

#print test_cat

#print X

#print y



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)





from sklearn.linear_model import LinearRegression,Ridge,Lasso



rid=Ridge(alpha=100)

las=Lasso(alpha=100)

lr=LinearRegression()



rid.fit(X_train,y_train)

lr.fit(X_train,y_train)

las.fit(X_train,y_train)



y_rid=rid.predict(X_test)

y_lr=lr.predict(X_test)

y_las=las.predict(X_test)



from sklearn.metrics import r2_score



acc=r2_score((y_test),y_rid)

#print 'rid', acc



acc=r2_score((y_test),y_lr)

#print 'lr', acc



acc=r2_score((y_test),y_las)

#print 'las', acc







las.fit(X,y)

y_pred=las.predict(X_test_set)

#print y_pred.shape



dictionary={'Id':test['Id'],'SalePrice':y_pred}

final_df=pd.DataFrame(dictionary)

final_df.to_csv('ans.csv',index=False)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.