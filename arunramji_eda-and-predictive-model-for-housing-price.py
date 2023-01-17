#Let's import required libraries

import pandas as pd

pd.options.display.max_rows=999

pd.options.display.max_columns =999

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
#read input file

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()
 #Let's see basic stats of entire dataset

df.describe()
#Basic stats for target variable 

df['SalePrice'].describe()
fig , ax = plt.subplots(figsize=(12,6))



sns.distplot(df['SalePrice'])

plt.show()
#df[df['SalePrice']==df['SalePrice'].max()]

df['SalePrice'].nlargest(10)
df.drop([691,1182,1169,898,803,1046,440,769,178,798],inplace=True)

df_1 = df.drop(['Id'],axis=1)
df_1.head(10)
#correlation Heatmap

corr = df_1.corr()

corr = pd.DataFrame(corr.loc['SalePrice'])

corr = corr[corr['SalePrice']>0.5]

corr

data = df_1[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF'

         ,'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea'

         ,'SalePrice']]



fig , ax = plt.subplots(figsize=(12,6))

sns.heatmap(data.corr(),annot=True)

corr = df_1.corr()

corr = pd.DataFrame(corr.loc['SalePrice'])

corr = corr[corr['SalePrice']<0]

corr
fig , ax = plt.subplots(1,2,figsize=(18,5))



ax[0].plot(df_1['SalePrice'],df_1['KitchenAbvGr'],'rx')

ax[1].plot(df_1['SalePrice'],df_1['YrSold'],'gx')
fig , ax = plt.subplots(1,2,figsize=(18,5))



ax[0].plot(df_1['SalePrice'],df_1['OverallQual'],'rx')

ax[0].set_title('Overall Quality')

ax[1].plot(df_1['SalePrice'],df_1['GrLivArea'],'gx')

ax[1].set_title('Living Area Square Feet')

plt.show()
sns.pairplot(df_1[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF'

         ,'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea'

         ,'SalePrice']])
Null_Cols = pd.DataFrame(df.select_dtypes(include='object').isnull().sum(),columns=['Null_count'])
Null_Cols[Null_Cols.Null_count>0]
Null_Cols[(Null_Cols.Null_count/len(df_1))>0.8]
df_1 = df_1.drop(['Alley','PoolQC','Fence','MiscFeature','MSSubClass','OverallCond',

                 'BsmtFinSF2','LowQualFinSF','BsmtHalfBath','KitchenAbvGr','EnclosedPorch'

                 ,'MiscVal','YrSold'],axis=1)

df_1.head()
dataset = df_1.select_dtypes(include='float')

pd.DataFrame(dataset.isnull().sum(),columns=['Null_count'])
df_1['LotFrontage'].fillna(np.mean(df_1['LotFrontage']),inplace=True)

df_1['MasVnrArea'].fillna(np.mean(df_1['MasVnrArea']),inplace=True)

df_1['GarageYrBlt'].fillna(np.round(np.mean(df_1['GarageYrBlt'])),inplace=True)  #rounding of the value as it is year value

#np.round(np.mean(df_1['GarageYrBlt']))



df_1.sample(10)



#Categorical Variable

df_1.select_dtypes(include='object').count()
df_1['MSZoning'].unique()
data= pd.get_dummies(df_1['MSZoning']).head(100)

data

#data.sort_values(by=['Pave'],ascending=False).head()
df_1 = pd.get_dummies(df_1,columns=['MSZoning','Street','Utilities','LotConfig','Neighborhood','Condition1'

                                    ,'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st'

                                   ,'Exterior2nd','MasVnrType','Foundation','Heating','CentralAir'

                                   ,'GarageType','SaleType','SaleCondition','MasVnrType','LandContour'],drop_first=True)



#drop_first tells to drop one of the encoded variable as it may cause "dummy variabel trap" .
#Let's visualise some radom sample from data after encoding

df_1.sample(10)
df_1['LandSlope'].unique()
data = df_1.select_dtypes(include=object).isna().sum()

data = pd.DataFrame(data,columns=['Count'])

data[data['Count']>0]

cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

    'BsmtFinType2','Electrical','FireplaceQu','FireplaceQu'

    ,'GarageFinish','GarageQual','GarageCond']
df_1[cols] = df_1[cols].replace({np.nan:'Unknown'}) #Replacing missing values with 'Unknown'
#Ordinal categorical variable from dataset

data
dict_BsmtQual = {"BsmtQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_BsmtCond = {"BsmtCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_BsmtExposure = {"BsmtExposure":{"Gd":5,"Av":4,"Mn":3,"No":2,"Unknown":0}}

dict_BsmtFinType1 = {"BsmtFinType1":{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"Unknown":0}}

dict_BsmtFinType2 = {"BsmtFinType2":{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"Unknown":0}}

dict_Electrical = {"Electrical":{"SBrkr":6,"FuseA":5,"FuseF":4,"FuseP":3,"Mix":0,"Unknown":0}}

dict_FireplaceQu = {"FireplaceQu":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_GarageFinish = {"GarageFinish":{"Fin":6,"RFn":5,"Unf":4,"Unknown":0}}

dict_GarageQual = {"GarageQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_GarageCond = {"GarageCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_LotShape = {"LotShape":{"Reg":5,"IR1":4,"IR2":3,"IR3":2}}

dict_LandSlope = {"LandSlope":{"Gtl":5,"Mod":4,"Sev":3}}

dict_ExterQual = {"ExterQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_ExterCond = {"ExterCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_HeatingQC = {"HeatingQC":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_KitchenQual = {"KitchenQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"Unknown":0}}

dict_Functional = {"Functional":{"Typ":5,"Min1":4,"Min2":3,"Mod":2,"Maj1":1,"Maj2":0,

                                "Sev":0}}

dict_PavedDrive = {"PavedDrive":{"Y":3,"P":2,"N":1}}
for i in [dict_BsmtQual,dict_BsmtCond,dict_BsmtExposure,dict_BsmtFinType1,dict_BsmtFinType2,dict_Electrical

         ,dict_FireplaceQu,dict_GarageFinish,dict_GarageQual,dict_GarageCond,dict_LotShape,dict_LandSlope

         ,dict_ExterQual,dict_ExterCond,dict_HeatingQC,dict_KitchenQual,dict_Functional,dict_PavedDrive] :

    #print(type(i))

    df_1.replace(i,inplace=True)
df_1.sample(10)
data = df_1.drop(columns='SalePrice')
#Variable Assignment

X = data.values

y = df_1['SalePrice']
y[0:10]
#split train and test set

from sklearn.model_selection import train_test_split

#tuple unpacking of train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) #20% data splitted as test set 
#train the model for training set

#from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

#poly = PolynomialFeatures(degree=2)

#X_poly = poly.fit_transform(X_train)

lm = LinearRegression()  #instantiating linear regression object

lm.fit(X_train,y_train)
#Intercept calculated by model

print('Intercept found by Linear regression Model :\n',lm.intercept_)
#Coefficient for each features

cdf = pd.DataFrame(lm.coef_,df_1.columns[df_1.columns!='SalePrice'],

                   columns=['Parameter'])

cdf
#prediction

prediction = lm.predict(X_test)

prediction
pred = pd.DataFrame({'Actual':y_test,'Predicted':prediction})

pred
fig , ax = plt.subplots(figsize=(12,6))

plt.scatter(y_test,prediction)
data = pred.head(25)

data.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#distribution of residuals 

fig , ax = plt.subplots(figsize=(12,6))



sns.distplot(y_test-prediction)
#Model Evaluation

from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,prediction))

print('MSE:',metrics.mean_squared_error(y_test,prediction))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))

#print('MSLE:',metrics.mean_squared_log_error(y_test,prediction))
metrics.r2_score(y_test,prediction)