# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in `



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
df =pd.read_csv("../input/train.csv")

tf = pd.read_csv("../input/test.csv")

df.head()
df.dtypes
#Drop the columns who has more than 50% missing values



cols=df.columns[df.isnull().mean()>0.47]

print(cols)

df=df.drop(cols,axis=1)

df.head()
df_n = df.isnull().sum()

df_n = df_n[df_n>0]

df_n.sort_values(ascending=False)

df.SalePrice.describe()
# Checking the most frequently occuring values



miss = df.isnull().sum().sort_values(ascending=False) # finds total number of null values and sorts in descending order

miss = miss[miss > 0] # all null values which are greater than zero

column_name = miss.index # names of column of null values

for var in column_name:

    print(str(var))

    print(df[var].value_counts())

    print("--"*10)
#Display the rows and columns with missing values

df[df.isnull().any(axis=1)]

df.head()
#percentage and total number of missing values



total = df.isnull().sum().sort_values(ascending=False)

percent = df.isnull().mean().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(14)

#dropping the columns that are not required



drop_var = ["GarageType","GarageYrBlt","GarageFinish","GarageCond","GarageQual","BsmtExposure","BsmtFinType1","BsmtCond"]

for var in drop_var:

    df= df.drop(var,axis =1)

df.head()
df.isnull().sum().sort_values(ascending = False).head(6)
#filling missing values



missing_var = ['LotFrontage','Electrical',"MasVnrArea","MasVnrType","BsmtFinType2","BsmtQual"]

for var in missing_var:

    df= df.fillna(df[var].value_counts().index[0])
df.isnull().sum().sum()
df.dtypes
#

#

#

# New code



numerical = df.select_dtypes(exclude=["object"])

corr = numerical.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:8], '\n')

print (corr['SalePrice'].sort_values(ascending=False)[-5:])

df.OverallQual.unique()
quality_pivot = df.pivot_table(index = "OverallQual",values = "SalePrice", aggfunc = np.median)

quality_pivot

                  
quality_pivot.plot(kind = "bar" , color = "blue")

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x = df["GrLivArea"],y = df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
df= df[df["GrLivArea"]<4500]

plt.scatter(x = df["GrLivArea"],y = df.SalePrice)

plt.xlim(0,6000)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=df['OverallQual'], y=df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=df['GarageArea'], y=df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
df = df[df["GarageArea"]<1200]

plt.xlim(-100,1500)

plt.scatter(x=df['GarageArea'], y=df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()

plt.scatter(x=df['TotalBsmtSF'], y=df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
df = df[df['TotalBsmtSF']<3000]

plt.xlim(-100,4000)

plt.scatter(x=df['TotalBsmtSF'], y=df.SalePrice)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
# categorical data to numerical data conversion



from sklearn.preprocessing import LabelEncoder

model = LabelEncoder()

labl = ["Street","CentralAir"]

for var in labl:

    df[var]= model.fit_transform(df[var])

df.head()
# categorical data to numerical data conversion



replace_Lots = {"LotShape":{"Reg":3,"IR1":2,"IR2":1,"IR3":0}}

df.replace(replace_Lots,inplace=True)



replace_land_s = {"LandSlope":{"Gtl":2,"Mod":1,"Sev":0}}

df.replace(replace_land_s,inplace = True)



data_h = ["ExterQual","ExterCond","HeatingQC","KitchenQual"]

for var in data_h:

    replace_data_h = {var:{"Ex":4,"Gd":3,"TA":2,"Fa":1,"Po":0}}

    df.replace(replace_data_h,inplace = True)



replace_bsmtqual = {"BsmtQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0}}

df.replace(replace_bsmtqual,inplace = True)



replace_pave_d = {"PavedDrive":{"Y":2,"P":1,"N":0}}

df.replace(replace_pave_d,inplace = True)



df.head()
numercial = df.select_dtypes(exclude=["object"])
# categorical data to numerical data conversion



catego = df.select_dtypes(include=["object"])

catego_copy = catego.copy()

catego = pd.get_dummies(catego)

catego.head()

df.head()
df =pd.concat([df,catego],axis =1)

df.head()
catego_copy.head()
#getting names of columns that can be dropped



x = catego_copy.columns

print(x)

for var in x:

    highest = df[var].value_counts().index[0]

    print(highest)

    for num in highest:

        name = var+"_"+highest

        print(name)

        break

        
df.drop(["MSZoning_RL","LandContour_Lvl","Utilities_AllPub","LotConfig_Inside","Neighborhood_NAmes","Condition1_Norm","Condition2_Norm",

        "BldgType_1Fam","HouseStyle_1Story","RoofStyle_Gable","RoofMatl_CompShg","Exterior1st_VinylSd","Exterior2nd_VinylSd",

        "MasVnrType_None","Foundation_PConc","BsmtFinType2_Unf","Heating_GasA","Electrical_SBrkr","Functional_Typ","SaleType_WD",

       "SaleCondition_Normal"],axis=1,inplace= True)
df.head()
x = catego_copy.columns

for var in x:

    df= df.drop (var,axis=1)

    

df = df.drop("Id",axis = 1)

df.head()
list(df.columns)



df.columns.get_loc("SalePrice")
df.dtypes
df = df.set_index('SalePrice').reset_index()

df.head()
fm = df.iloc[:,1:]

tv = df.iloc[:,0]
print(fm.shape) 

print(fm.ndim)

print(tv.shape)

print(tv.ndim)
from sklearn.model_selection import train_test_split

fm_train,fm_test,tv_train,tv_test = train_test_split(fm,tv,test_size = 0.2,random_state=123)



print(fm_train.shape)

print(fm_test.shape)

print(tv_train.shape)

print(tv_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X=fm_train, y= tv_train)

print(model.score(X=fm_train, y=tv_train))

print(model.score(X=fm_test, y=tv_test))
for i in fm_train.columns : 

    print(i)

    print("--" * 10)

    for j in np.arange(1, 4, 1) : 

        print("Order : " + str(j))

        print("--" * 10)

        model = LinearRegression()

        model.fit(X=fm_train[[i]] ** j, y=tv_train)



        print("Train : " , model.score(X=fm_train[[i]] ** j, y=tv_train))

        print("Test  : " , model.score(X=fm_test[[i]] ** j, y=tv_test))

        print("-" * 10)
#LotShape,OverallCond2,CentralAir,2ndFlrSF2,HalfBath,BedroomAbvGr3,KitchenAbvGr,PavedDrive2,WoodDeckSF,OpenPorchSF,Neighborhood_Edwards,Neighborhood_OldTown,Exterior1st_CemntBd,Exterior2nd_CmentBd,MasVnrType_BrkFace,MasVnrType_Stone,Foundation_BrkTil,Electrical_FuseA,SaleType_New,SaleCondition_Partial        



#OverallQual,YearBuilt,YearRemodAdd,ExterQual3,HeatingQC3,FullBath,KitchenQual(1 or 2 ),GarageCars2,GarageArea,MSZoning_RM,Neighborhood_NridgHt,Foundation_CBlock                  





#MasVnrArea,TotRmsAbvGrd,Fireplaces
fm_train["OverallCond2"] = fm_train["OverallCond"] ** 2

fm_test["OverallCond2"] = fm_test["OverallCond"] ** 2



fm_train["2ndFlrSF2"] = fm_train["2ndFlrSF"] ** 2

fm_test["2ndFlrSF2"] = fm_test["2ndFlrSF"] ** 2



fm_train["BedroomAbvGr3"] = fm_train["BedroomAbvGr"] ** 3

fm_test["BedroomAbvGr3"] = fm_test["BedroomAbvGr"] ** 3



fm_train["PavedDrive2"] = fm_train["PavedDrive"] ** 2

fm_test["PavedDrive2"] = fm_test["PavedDrive"] ** 2



fm_train["ExterQual3"] = fm_train["ExterQual"] ** 3

fm_test["ExterQual3"] = fm_test["ExterQual"] ** 3



fm_train["HeatingQC3"] = fm_train["HeatingQC"] ** 3

fm_test["HeatingQC3"] = fm_test["HeatingQC"] ** 3



fm_train["GarageCars2"] = fm_train["GarageCars"] ** 2

fm_test["GarageCars2"] = fm_test["GarageCars"] ** 2
fm_train = fm_train.drop(["GarageCars","HeatingQC","ExterQual","PavedDrive","BedroomAbvGr","2ndFlrSF","OverallCond"],axis = 1)

fm_test =  fm_test.drop(["GarageCars","HeatingQC","ExterQual","PavedDrive","BedroomAbvGr","2ndFlrSF","OverallCond"],axis = 1)
fm_train.head()
fm_train.shape
figures_per_time = 4

count = 0

vars = ["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock"]

for var in vars:

    x = fm_train[var]

    plt.figure(count//figures_per_time,figsize=(25,5))

    plt.subplot(1,figures_per_time,np.mod(count,4)+1)

    plt.scatter(x, tv_train)

    plt.title('f model: T= {}'.format(var))

    count+=1
model= LinearRegression()

model.fit(X=fm_train,y=tv_train)



print(model.score(X=fm_train, y=tv_train))

print(model.score(X=fm_test, y=tv_test))

#fm_train_new = fm_train[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock"]]

#fm_test_new = fm_test[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock"]]



#model = LinearRegression()

#model.fit(X=fm_train_new,y=tv_train)



#print(model.score(X=fm_train_new, y=tv_train))

#print(model.score(X=fm_test_new, y=tv_test))
#fm_train_new = fm_train[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock","LotShape",

#                         "OverallCond2","CentralAir","2ndFlrSF2","HalfBath","BedroomAbvGr3","KitchenAbvGr","PavedDrive2",

#                         "WoodDeckSF","OpenPorchSF","Neighborhood_Edwards","Neighborhood_OldTown","Exterior1st_CemntBd",

#                         "Exterior2nd_CmentBd","MasVnrType_BrkFace","MasVnrType_Stone","Foundation_BrkTil",

#                         "Electrical_FuseA","SaleType_New","SaleCondition_Partial"]]

#fm_test_new = fm_test[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock","LotShape",

#                         "OverallCond2","CentralAir","2ndFlrSF2","HalfBath","BedroomAbvGr3","KitchenAbvGr","PavedDrive2",

#                         "WoodDeckSF","OpenPorchSF","Neighborhood_Edwards","Neighborhood_OldTown","Exterior1st_CemntBd",

#                         "Exterior2nd_CmentBd","MasVnrType_BrkFace","MasVnrType_Stone","Foundation_BrkTil",

#                         "Electrical_FuseA","SaleType_New","SaleCondition_Partial"]]



#model = LinearRegression()

#model.fit(X=fm_train_new,y=tv_train)



#print(model.score(X=fm_train_new, y=tv_train))

#print(model.score(X=fm_test_new, y=tv_test))
#fm_train_new = fm_train[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock","LotShape",

#                         "OverallCond2","CentralAir","2ndFlrSF2","HalfBath","BedroomAbvGr3","KitchenAbvGr","PavedDrive2",

#                         "WoodDeckSF","OpenPorchSF","Neighborhood_Edwards","Neighborhood_OldTown","Exterior1st_CemntBd",

#                         "Exterior2nd_CmentBd","MasVnrType_BrkFace","MasVnrType_Stone","Foundation_BrkTil",

#                        "Electrical_FuseA","SaleType_New","SaleCondition_Partial","MasVnrArea","TotRmsAbvGrd","Fireplaces"]]

#fm_test_new = fm_test[["OverallQual","YearBuilt","YearRemodAdd","ExterQual3","HeatingQC3","FullBath","KitchenQual",

#                        "GarageCars2","GarageArea","MSZoning_RM","Neighborhood_NridgHt","Foundation_CBlock","LotShape",

#                         "OverallCond2","CentralAir","2ndFlrSF2","HalfBath","BedroomAbvGr3","KitchenAbvGr","PavedDrive2",

#                         "WoodDeckSF","OpenPorchSF","Neighborhood_Edwards","Neighborhood_OldTown","Exterior1st_CemntBd",

#                         "Exterior2nd_CmentBd","MasVnrType_BrkFace","MasVnrType_Stone","Foundation_BrkTil",

#                         "Electrical_FuseA","SaleType_New","SaleCondition_Partial","MasVnrArea","TotRmsAbvGrd","Fireplaces"]] '''



#model = LinearRegression()

#model.fit(X=fm_train_new,y=tv_train)



#print(model.score(X=fm_train_new, y=tv_train))

#print(model.score(X=fm_test_new, y=tv_test))
tf.head()
cols=tf.columns[tf.isnull().mean()>0.47]

print(cols)

tf=tf.drop(cols,axis=1)

tf.head()
drop_var = ["GarageType","GarageYrBlt","GarageFinish","GarageCond","GarageQual","BsmtExposure","BsmtFinType1","BsmtCond"]

for var in drop_var:

    tf= tf.drop(var,axis =1)

tf.head()
#filling missing values



missing_var = ['LotFrontage','Electrical',"MasVnrArea","MasVnrType","BsmtFinType2","BsmtQual"]

for var in missing_var:

    tf= tf.fillna(tf[var].value_counts().index[0])



# categorical data to numerical data conversion

#Label encoding

from sklearn.preprocessing import LabelEncoder

model = LabelEncoder()

labl = ["Street","CentralAir"]

for var in labl:

    tf[var]= model.fit_transform(tf[var])



#Categorical data with some pattern

replace_Lots = {"LotShape":{"Reg":3,"IR1":2,"IR2":1,"IR3":0}}

tf.replace(replace_Lots,inplace=True)



replace_land_s = {"LandSlope":{"Gtl":2,"Mod":1,"Sev":0}}

tf.replace(replace_land_s,inplace = True)



data_h = ["ExterQual","ExterCond","HeatingQC","KitchenQual"]

for var in data_h:

    replace_data_h = {var:{"Ex":4,"Gd":3,"TA":2,"Fa":1,"Po":0}}

    tf.replace(replace_data_h,inplace = True)



replace_bsmtqual = {"BsmtQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"NA":0}}

tf.replace(replace_bsmtqual,inplace = True)



replace_pave_d = {"PavedDrive":{"Y":2,"P":1,"N":0}}

tf.replace(replace_pave_d,inplace = True)



tf.head()

#one hot encoding using get_dummies

catego_tf = tf.select_dtypes(include=["object"])

catego_tf_copy = catego_tf.copy()

catego_tf = pd.get_dummies(catego_tf)

catego_tf.head()
#combine the data

tf =pd.concat([tf,catego_tf],axis =1)

tf.head()
x = catego_tf_copy.columns

print(x)

for var in x:

    highest = tf[var].value_counts().index[0]

    print(highest)

    for num in highest:

        name = var+"_"+highest

        print(name)
#dropping unwanted columns

tf.drop(["MSZoning_RL","LandContour_Lvl","Utilities_AllPub","LotConfig_Inside","Neighborhood_NAmes","Condition1_Norm","Condition2_Norm",

        "BldgType_1Fam","HouseStyle_1Story","RoofStyle_Gable","RoofMatl_CompShg","Exterior1st_VinylSd","Exterior2nd_VinylSd",

        "MasVnrType_None","Foundation_PConc","BsmtFinType2_Unf","Heating_GasA","Electrical_SBrkr","Functional_Typ","SaleType_WD",

       "SaleCondition_Normal"],axis=1,inplace= True)



x = catego_tf_copy.columns

for var in x:

    tf= tf.drop (var,axis=1)

tf.head()
tf["OverallCond2"] = tf["OverallCond"] ** 2



tf["2ndFlrSF2"] = tf["2ndFlrSF"] ** 2



tf["BedroomAbvGr3"] = tf["BedroomAbvGr"] ** 3



tf["PavedDrive2"] = tf["PavedDrive"] ** 2



tf["ExterQual3"] = tf["ExterQual"] ** 3



tf["HeatingQC3"] = tf["HeatingQC"] ** 3



tf["GarageCars2"] = tf["GarageCars"] ** 2
tf = tf.drop(["GarageCars","HeatingQC","ExterQual","PavedDrive","BedroomAbvGr","2ndFlrSF","OverallCond"],axis = 1)

tf.head()
fm_train.head()
tf.select_dtypes(include=["object"]).sum().sum()
tf.shape
missing_cols = set( fm_train.columns ) - set( tf.columns )

missing_cols
# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    tf[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

tf = tf[fm_train.columns]
model= LinearRegression()

model.fit(X=fm_train,y=tv_train)



print(model.score(X=fm_train, y=tv_train))

print(model.score(X=fm_test, y=tv_test))
Pred_P = model.predict(tf)

Pred_P[:5]
submission = pd.DataFrame()

submission['SalePrice'] = Pred_P

submission.head()