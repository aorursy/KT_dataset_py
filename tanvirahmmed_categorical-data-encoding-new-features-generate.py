import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
df = pd.read_csv('/kaggle/input/house-prices-data/train.csv')
dt = pd.read_csv('/kaggle/input/house-prices-data/test.csv')
df.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
dt.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
y = df['SalePrice']
df.drop(['SalePrice'], axis = 1, inplace = True)
data = pd.concat([df,dt], axis = 0)
data.shape
year_all = ['YearBuilt', 'YearRemodAdd','YrSold','MoSold','GarageYrBlt']
for i in data:
    if data[i].dtypes == object or i in year_all:
        data[i] = data[i].fillna(data[i].mode()[0])
    else:
        data[i] = data[i].fillna(data[i].mean())
(((data.isnull().sum())*100)/len(data)).sort_values(
            ascending = False, kind = 'mergesort').head(5)
data_copy = data.copy()

# Encode Ordinal Data
qual_listt = ['HeatingQC','OverallQual','ExterQual','BsmtQual','KitchenQual','FireplaceQu','GarageQual']
cond_listt = ['OverallCond','ExterCond','BsmtCond','GarageCond']
dic = {'NA':.5,'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 3.5, 'Ex': 5}
for i in (qual_listt+cond_listt):
  if data_copy[i].dtype == object:
    data_copy[i] = data_copy[i].map(dic)

house_style = {'1.5Unf':1,'SFoyer':2, '1.5Fin': 3, '2.5Unf': 4, 'SLvl': 5, '1Story': 6, '2Story': 7, '2.5Fin': 8}
utilities = {'NoSeWa':1,'AllPub':2}
roof_matl = {'Roll':1,'ClyTile':2, 'CompShg': 3, 'Metal': 4, 'Tar&Grv': 5, 'WdShake': 6, 'Membran': 7, 'WdShngl': 8}
heating = {'Floor':1,'Grav':2, 'Wall': 3, 'OthW': 4, 'GasW': 5, 'GasA': 6}
electrical = {'Mix':1,'FuseP':2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}

data_copy['Utilities'] = data_copy['Utilities'].map(utilities)
data_copy['HouseStyle'] = data_copy['HouseStyle'].map(house_style)
data_copy['RoofMatl'] = data_copy['RoofMatl'].map(roof_matl)
data_copy['Heating'] = data_copy['Heating'].map(heating)
data_copy['Electrical'] = data_copy['Electrical'].map(electrical)
data_copy.head()
# Count Encoding
data_copy = data.copy()
import operator
def count_encoding(data,i):
  if data[i].dtype == object or i in year_all:
    Mean_encoded_subject = data[i].value_counts().to_dict() 
    sorted_d = dict(sorted(Mean_encoded_subject.items(), key=operator.itemgetter(1)))
    li = list(sorted_d)
    new_dict = {}
    for j in li:
      new_dict[j] = li.index(j)+1
    data[i] =  data[i].map(new_dict) 
  return data[i]
for i in data_copy:
  if data_copy[i].dtype == object or i in year_all:
      data_copy[i] = count_encoding(data_copy,i)
data_copy.head()
# Mean Encoding
tx = data.iloc[:len(y), :]
ty = data.iloc[len(tx):, :]
tx['SalePrice'] = y.copy()
tx_copy = tx.copy()
ty_copy = ty.copy()
def mean_encoding(data,data1):
  for i in data:
    if data[i].dtypes == object or i in year_all:
      Mean_encoded_subject = data.groupby([i])['SalePrice'].mean().to_dict() 
      sorted_d = dict(sorted(Mean_encoded_subject.items(), key=operator.itemgetter(1)))
      li = list(sorted_d)
      new_dict = {}
      for j in li:
        new_dict[j] = li.index(j)+1
      data[i] =  data[i].map(new_dict) 
      data1[i] =  data1[i].map(new_dict) 
  return data, data1  
tx_copy,ty_copy = mean_encoding(tx_copy,ty_copy)
tx_copy.head()
data_copy = pd.get_dummies(data, drop_first=True)
data_copy.head()
data_copy = data.copy()
data_copy['AgeOfHouse'] = abs(data_copy['YrSold'] - data_copy['YearBuilt'])
data_copy['TotalExtraArea'] = data_copy['WoodDeckSF'] + data_copy['OpenPorchSF'] + data_copy['EnclosedPorch'] + data_copy['3SsnPorch']+ data_copy['PoolArea']
data_copy['GarageAreaPerCar'] = (data_copy['GarageArea']+1) / (data_copy['GarageCars'] +1)
data_copy['TotalBath'] = data_copy['BsmtFullBath'] + data_copy['BsmtHalfBath'] + data_copy['FullBath'] + data_copy['HalfBath']
data_copy['CompletedFloorSF'] = data_copy['1stFlrSF'] + data_copy['2ndFlrSF']
data_copy['CompletedBstmSf'] = data_copy['TotalBsmtSF']- data_copy['BsmtUnfSF']

data_copy.loc[data_copy['Exterior1st'] == data_copy['Exterior2nd'],'Exterior'] = 1
data_copy.loc[data_copy['Exterior1st'] != data_copy['Exterior2nd'],'Exterior'] = 2

data_copy['CompletedBstmSf'] = data_copy['TotalBsmtSF'] + data_copy['BsmtUnfSF']


data_copy.loc[data_copy['Condition1'] == data_copy['Condition2'],'Condition'] = 1
data_copy.loc[data_copy['Condition1'] != data_copy['Condition2'],'Condition'] = 2


data_copy['RemodAdd'] = data_copy['YearBuilt']
for i in range(len(data_copy)):
  if data_copy['YearBuilt'].iloc[i] == data_copy['YearRemodAdd'].iloc[i]:
    data_copy['RemodAdd'].iloc[i] = 0
  else:
    data_copy['RemodAdd'].iloc[i] = abs(data_copy['YearBuilt'].iloc[i]- data_copy['YearRemodAdd'].iloc[i])

data_copy.head()
