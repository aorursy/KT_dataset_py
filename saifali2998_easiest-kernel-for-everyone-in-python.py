import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



data1=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')





data1= pd.DataFrame(data=data1)

total_miss = data1.isnull().sum()

perc_miss = total_miss/data1.isnull().count()*100



missing_data = pd.DataFrame({'Total missing':total_miss,

                            '% missing':perc_miss})



missing_data.sort_values(by='Total missing',

                         ascending=False).head(26)
print(data1.shape)

data1.sort_values('Id', inplace =True) 



data1.drop(['PoolQC','MiscFeature'	,'Alley'	,'Fence'	,'FireplaceQu'	,'LotFrontage'	,'GarageYrBlt'	,'GarageQual'	,'GarageFinish','GarageCond'	,'GarageType'	,'BsmtCond'	,'BsmtQual'	,'BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType','MasVnrArea'	,'MSZoning','BsmtHalfBath','Utilities','Functional','BsmtFullBath','BsmtFinSF1'	,'BsmtFinSF2'	,'BsmtUnfSF'	,'KitchenQual'	,'TotalBsmtSF'	,'Exterior2nd'	,'GarageCars'	,'Exterior1st'	,'GarageArea'	,'SaleType'	],axis=1,inplace=True)



print(data1.shape)

data1=data1.dropna()

print(data1.shape)



#data1.to_csv("clean_data1.csv",index=False, encoding='utf8')

data1.info()

data1.describe()
import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



data2=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')





data2= pd.DataFrame(data=data2)

total_miss = data2.isnull().sum()

perc_miss = total_miss/data2.isnull().count()*100



missing_data = pd.DataFrame({'Total missing':total_miss,

                            '% missing':perc_miss})



missing_data.sort_values(by='Total missing',

                         ascending=False).head(35)
print(data2.shape)

data2.sort_values('Id', inplace =True) 



data2.drop(['PoolQC','MiscFeature'	,'Alley'	,'Fence'	,'FireplaceQu'	,'LotFrontage'	,'GarageYrBlt'	,'GarageQual'	,'GarageFinish','GarageCond'	,'GarageType'	,'BsmtCond'	,'BsmtQual'	,'BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType','MasVnrArea'	,'MSZoning','BsmtHalfBath','Utilities','Functional','BsmtFullBath','BsmtFinSF1'	,'BsmtFinSF2'	,'BsmtUnfSF'	,'KitchenQual'	,'TotalBsmtSF'	,'Exterior2nd'	,'GarageCars'	,'Exterior1st'	,'GarageArea'	,'SaleType'	],axis=1,inplace=True)



print(data2.shape)









#data2.to_csv("clean_data2.csv",index=False, encoding='utf8')

data2.info()

data2.describe()
data1.info()
category_column =['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl' ,'ExterQual','ExterCond','Foundation', 'Heating','HeatingQC', 'CentralAir','Electrical','PavedDrive','SaleCondition'] 

for x in category_column:

    print (x)

    print (data1[x].value_counts())

 
for col in category_column:

    b, c = np.unique(data1[col], return_inverse=True) 

    data1[col] = c



data1.head()
category_column =['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl' ,'ExterQual','ExterCond','Foundation', 'Heating','HeatingQC', 'CentralAir','Electrical','PavedDrive','SaleCondition'] 

for x in category_column:

    print (x)

    print (data2[x].value_counts())
for col in category_column:

    b, c = np.unique(data2[col], return_inverse=True) 

    data2[col] = c



data2.head()
from sklearn.tree import DecisionTreeRegressor

# Get all the columns from the dataFrame

columns = data1.columns.tolist()

columns2 = data2.columns.tolist()



# Filter the columns to remove data we do not want

train_pr = [c for c in columns if c not in ['SalePrice']]

test_pr = [c for c in columns2  if c not in ['SalePrice']]







model = DecisionTreeRegressor()



#set prediction data to factors that will predict, and set target to SalePrice

train_data = data1[train_pr]

test_data = data2[test_pr]

target = data1['SalePrice']



#fitting model with prediction data and telling it my target

model.fit(train_data, target)

prediction=model.predict(test_data)

print(prediction)
prediction = prediction.reshape(len(prediction), 1)



dataTest = np.concatenate((data2, prediction), axis = 1)

print(dataTest)

data2['SalePrice'] = prediction

data2.sort_values('Id', inplace =True) 



data2.head()
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn import metrics





#X=row_concat[['country of birth self','major occupation code','age','tax filer status']].values

X=data1[['SalePrice']].values





y= data2[['SalePrice']].values



X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=10)



clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

predn=clf.predict(X_test)

print('The accuracy of the model using decision tree is',metrics.accuracy_score(predn,y_test))
data2.drop(['MSSubClass','LotArea'        ,'Street'          ,'LotShape'      ,'LandContour'  ,'LotConfig'        ,'LandSlope'      ,'Neighborhood' ,'Condition1'   ,'Condition2'     ,'BldgType'     ,'HouseStyle' ,'OverallQual'   ,'OverallCond'   ,'YearBuilt'    ,'YearRemodAdd'  ,'RoofStyle' ,'RoofMatl'  ,'ExterQual'   ,'ExterCond' ,'Foundation' ,'Heating'  ,'HeatingQC' ,'CentralAir' ,'Electrical' ,'1stFlrSF','2ndFlrSF'     ,'LowQualFinSF' ,'GrLivArea'  ,'FullBath','HalfBath' ,'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch', 'ScreenPorch','PoolArea'   ,'MiscVal','MoSold','YrSold','SaleCondition'],axis=1,inplace=True)
data2.to_csv("submission_output.csv",index=False, encoding='utf8')
data2.head()
import matplotlib.pyplot as plt

import seaborn as sns



data1.hist(figsize = (20, 20))

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



data2.hist(figsize = (20, 20))

plt.show()