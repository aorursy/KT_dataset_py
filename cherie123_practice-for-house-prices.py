# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import norm

from sklearn import linear_model

from sklearn.model_selection import cross_val_score



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



train_data.head()
train_data.columns

train_data['SalePrice'].describe()
sns.distplot(train_data['SalePrice'])
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)



data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'GarageArea'

data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)



data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
var = 'OverallQual'

data= pd.concat([train_data['SalePrice'],train_data[var]],axis=1)



fig = sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ymin=0,ymax=800000);
var = 'YearBuilt'

data = pd.concat([train_data[var],train_data['SalePrice']],axis =1)



f, ax = plt.subplots(figsize=(16, 8))

sns.boxplot(x=var, y = 'SalePrice',data=data)

plt.xticks(rotation=90);
corrMat = train_data.corr()

f,ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrMat,vmax=0.8,square=True);
# saleprice correlation matrix

k = 10

cols = corrMat.nlargest(k,'SalePrice')['SalePrice'].index



corrM = np.corrcoef(train_data[cols].values.T)

histM = sns.heatmap(corrM, cbar=True, annot = True, fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
#scatterplot

sns.set()

cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(train_data[cols],size = 2.5);

#missing data

sample_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],test_data.loc[:,'MSSubClass':'SaleCondition']])





total = sample_data.isnull().sum().sort_values(ascending=False)

percent = (sample_data.isnull().sum()/sample_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['total','percent'])

missing_data.head(40)



#total = train_data.isnull().sum().sort_values(ascending=False)

#percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)

#missing_data = pd.concat([total,percent],axis=1,keys=['total','percent'])

#missing_data.head(20)
train_data = train_data.drop((missing_data[missing_data['total']>1]).index,1)

train_data = train_data.drop((train_data[train_data['Electrical'].isnull()]).index)





test_data = test_data.drop((missing_data[missing_data['total']>1]).index,1)

#test_data = test_data.fillna(test_data.median(axis=1))

var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'],train_data[var]],1)



data.plot.scatter(x=var,y='SalePrice')
train_data=train_data.drop((train_data[train_data['GrLivArea']>4500]).index)





var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'],train_data[var]],1)



data.plot.scatter(x=var,y='SalePrice')
sns.distplot(train_data['SalePrice'],fit=norm)

fig = plt.figure()

res = stats.probplot(train_data['SalePrice'], plot=plt)

stats.probplot(np.log1p(train_data['SalePrice']),plot=plt)
x_train = train_data.loc[:,'MSSubClass':'SaleCondition']

y_train = train_data.SalePrice



x_train=x_train.replace({"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                         "Street" : {"Grvl" : 1, "Pave" : 2},

                         "LandContour":{"Lvl":1,"Bnk":2,"Low":3,"HLS":4},

                         "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                         "LotConfig":{"Inside":1,"FR2":2,"Corner":3,"CulDSac":4,"FR3":5},

                         "Neighborhood":{"CollgCr":1,"Veenker":2,"Crawfor":3,"NoRidge":4,"Mitchel":5,

                                         "Somerst":6,"NWAmes":7,"OldTown":8,"BrkSide":9,"Sawyer":10,

                                         "NridgHt":11,"NAmes":12,"SawyerW":13,"IDOTRR":14,"MeadowV":15,

                                         "Edwards":16,"Timber":17,"Blmngtn":18,"SWISU":19,"StoneBr":20,

                                         "BrDale":21,"Blueste":22,"ClearCr":23,"NPkVill":24,"Gilbert":25

                                        },

                         "Condition1":{"Norm":1,"Feedr":2,"RRNn":3,"Artery":4,"PosN":5,"RRAe":6,"RRNe":7,

                                      "PosA":8,"RRAn":9},

                         "Condition2":{"Norm":1,"Feedr":2,"RRNn":3,"Artery":4,"PosN":5,"RRAe":6,"RRNe":7,

                                      "PosA":8,"RRAn":9},

                         "BldgType":{"1Fam":1,"2fmCon":2,"Duplex":3,"TwnhsE":4,"Twnhs":5},

                         "HouseStyle":{"SLvl":1,"1Story":2,"1.5Fin":3,"2Story":4,"2.5Unf":5,"2.5Fin":6,

                                      "SFoyer":7,"1.5Unf":8},

                         "RoofStyle":{"Gable":1,"Hip":2,"Gambrel":3,"Flat":4,"Shed":5,"Mansard":6},

                         "RoofMatl":{"CompShg":1,"Membran":2, "WdShake":3, "Metal":4, "WdShngl":5, "Roll":6, 

                                     "Tar&Grv":7},

                         "Exterior1st":{"HdBoard":1,"Wd Shng":2,"MetalSd":3,"VinylSd":4,"Plywood":5,"CemntBd":6,

                                        "BrkFace":7,"AsbShng":8,"Stucco":9,"AsphShn":10,"Stone":11,

                                        "CBlock":12,"ImStucc":13,"WdShing":14,"BrkComm":15,"Wd Sdng":16},



                         "Exterior2nd": {"HdBoard":1,"Wd Shng":2,"MetalSd":3,"VinylSd":4,"Plywood":5,"CmentBd":6,

                                        "BrkFace":7,"AsbShng":8,"Stucco":9,"AsphShn":10,"Stone":11,

                                        "CBlock":12,"ImStucc":13,"Wd Sdng":14,"Brk Cmn":15,"Other":16},

                         "ExterQual":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "ExterCond":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "Foundation":{"PConc":1,"CBlock":2,"BrkTil":3,"Wood":4,"Slab":5,"Stone":6},

                         "Heating":{"GasW":1,"GasA":2, "Wall":3,"Grav":4,"Floor":5,"OthW":6},

                         "HeatingQC":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "CentralAir":{"N":1,"Y":2},

                         "Electrical":{"FuseA":1,"FuseP":2,"Mix":3,"SBrkr":4,"FuseF":5},

                         "KitchenQual":{"Fa":0,"TA":1,"Gd":2,"Ex":3},

                         "PavedDrive":{"N":1,"P":2,"Y":3},

                         "SaleType":{"New":1, "COD":2,"ConLD":3,"WD":4,"Con":5,"CWD":6,"Oth":7,

                                      "ConLw":8,"ConLI":9},

                         "SaleCondition":{"Abnorml":1,"Partial":2,"AdjLand":3,"Normal":4,"Family":5,"Alloca":6}    

})



x_test = test_data.loc[:,'MSSubClass':'SaleCondition']

x_test=x_test.replace({"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                         "Street" : {"Grvl" : 1, "Pave" : 2},

                         "LandContour":{"Lvl":1,"Bnk":2,"Low":3,"HLS":4},

                         "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                         "LotConfig":{"Inside":1,"FR2":2,"Corner":3,"CulDSac":4,"FR3":5},

                         "Neighborhood":{"CollgCr":1,"Veenker":2,"Crawfor":3,"NoRidge":4,"Mitchel":5,

                                         "Somerst":6,"NWAmes":7,"OldTown":8,"BrkSide":9,"Sawyer":10,

                                         "NridgHt":11,"NAmes":12,"SawyerW":13,"IDOTRR":14,"MeadowV":15,

                                         "Edwards":16,"Timber":17,"Blmngtn":18,"SWISU":19,"StoneBr":20,

                                         "BrDale":21,"Blueste":22,"ClearCr":23,"NPkVill":24,"Gilbert":25

                                        },

                         "Condition1":{"Norm":1,"Feedr":2,"RRNn":3,"Artery":4,"PosN":5,"RRAe":6,"RRNe":7,

                                      "PosA":8,"RRAn":9},

                         "Condition2":{"Norm":1,"Feedr":2,"Artery":3,"PosN":4,"PosA":5},

                         "BldgType":{"1Fam":1,"2fmCon":2,"Duplex":3,"TwnhsE":4,"Twnhs":5},

                         "HouseStyle":{"SLvl":1,"1Story":2,"1.5Fin":3,"2Story":4,"2.5Unf":5,"2.5Fin":6,

                                      "SFoyer":7,"1.5Unf":8},

                         "RoofStyle":{"Gable":1,"Hip":2,"Gambrel":3,"Flat":4,"Shed":5,"Mansard":6},

                         "RoofMatl":{"CompShg":1,"Membran":2, "WdShake":3, "Metal":4, "WdShngl":5, "Roll":6, 

                                     "Tar&Grv":7},

                         "Exterior1st":{"HdBoard":1,"Wd Shng":2,"MetalSd":3,"VinylSd":4,"Plywood":5,"CemntBd":6,

                                        "BrkFace":7,"AsbShng":8,"Stucco":9,"AsphShn":10,"Stone":11,

                                        "CBlock":12,"ImStucc":13,"WdShing":14,"BrkComm":15,"Wd Sdng":16},



                         "Exterior2nd": {"HdBoard":1,"Wd Shng":2,"MetalSd":3,"VinylSd":4,"Plywood":5,"CmentBd":6,

                                        "BrkFace":7,"AsbShng":8,"Stucco":9,"AsphShn":10,"Stone":11,

                                        "CBlock":12,"ImStucc":13,"Wd Sdng":14,"Brk Cmn":15,"Other":16},

                         "ExterQual":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "ExterCond":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "Foundation":{"PConc":1,"CBlock":2,"BrkTil":3,"Wood":4,"Slab":5,"Stone":6},

                         "Heating":{"GasW":1,"GasA":2, "Wall":3,"Grav":4,"Floor":5,"OthW":6},

                         "HeatingQC":{"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5},

                         "CentralAir":{"N":1,"Y":2},

                         "Electrical":{"FuseA":1,"FuseP":2,"Mix":3,"SBrkr":4,"FuseF":5},

                         "KitchenQual":{"Fa":0,"TA":1,"Gd":2,"Ex":3},

                         "PavedDrive":{"N":1,"P":2,"Y":3},

                         "SaleType":{"New":1, "COD":2,"ConLD":3,"WD":4,"Con":5,"CWD":6,"Oth":7,

                                      "ConLw":8,"ConLI":9},

                         "SaleCondition":{"Abnorml":1,"Partial":2,"AdjLand":3,"Normal":4,"Family":5,"Alloca":6}    

})







for cols in x_test.columns:

    temp = x_test[cols]

    if temp.isnull().sum()==1 :

        index = temp.isnull().index

        temp[index]=temp.median()

        x_test[cols]=temp





x_test.isnull().sum()
alphas = [0.05,0.1,0.3,1,3,5,10,15,30,50,75] 

y_train=np.log1p(y_train)

mean_RMSE=[]

for alpha in alphas:

    model_ridge = linear_model.Ridge(alpha)

    RMSE= np.sqrt(-cross_val_score(model_ridge,x_train,y_train,scoring="neg_mean_squared_error",cv=5))

    mean_RMSE = np.append(mean_RMSE,RMSE.mean())



plt.figure(1)

plt.subplot(111)

plt.plot(alphas,mean_RMSE)

plt.show()

best_alpha = 30

ridge = linear_model.Ridge(alpha=best_alpha)

ridge.fit(x_train,y_train)

ridge_preds = ridge.predict(x_test)

ridge_preds = np.expm1(ridge_preds)-1

sns.distplot(ridge_preds)