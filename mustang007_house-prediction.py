# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')

sub.head(20)
train.tail()
train.info()
data = pd.concat([train,test], axis=0, ignore_index=True)
data
data['MSSubClass'].nunique()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(15,5))

ax =  sns.countplot(x='MSSubClass', data = data)
ax.set_xlabel('MSSubClass', fontsize=18)
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()/2
    ax.text(p.get_x()+width,height + 3,
           '{:1}'.format(height),
           ha = "center")
    
plt.show()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['MSSubClass'] = le.fit_transform(data['MSSubClass'])

data[['MSSubClass']].head()
data['MSZoning'].nunique()
plt.figure(figsize=(15,5))
ax = sns.countplot(data['MSZoning'])
ax.set_xlabel('MSZoning', fontsize=18)
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()/2
    ax.text(p.get_x()+width,height + 3,
           '{:1}'.format(height),
           ha = "center")
    
plt.show()



dum = pd.get_dummies(data['MSZoning'],prefix='MSZoning')
data = pd.concat([data,dum], axis=1)
data
data.drop(columns=['MSZoning'], inplace=True)

data['LotFrontage'].nunique()
data['LotFrontage'].isnull().sum()

# it has 486 null values
plt.figure(figsize=(15,5))
sns.distplot(data['LotFrontage'])
sns.boxplot(data['LotFrontage'])
from sklearn.impute import KNNImputer
imputer = KNNImputer( n_neighbors=10, metric='nan_euclidean',weights='uniform')
x = imputer.fit_transform(data[['LotFrontage']]) 
Xtrans = imputer.transform(x)
data['LotFrontage'] = Xtrans
data.head(10)
  
from scipy.stats import skew 
print('\nSkewness for data : ', skew(data['LotFrontage'])) 

# i = data['LotFrontage'].quantile(.98)
# data = data[data['LotFrontage']<i]
p = data['LotFrontage']
sns.distplot(p)
skew(p)
data['LotArea'].nunique()
data['LotArea'].isnull().sum()
plt.figure(figsize=(15,5))

sns.distplot(data['LotArea'])
skew(data['LotArea'])
data['LotArea'] = np.log1p(data['LotArea'])


print(skew(data['LotArea']))
plt.figure(figsize=(15,5))

sns.distplot(data['LotArea'])
data
data['Street'].nunique()
plt.figure(figsize=(6,5))
ax = sns.countplot(data['Street'])

ax.set_xlabel('Street', fontsize=18)
for p in ax.patches:
    height = p.get_height()
    width = p.get_width()/2
    ax.text(p.get_x()+width,height + 3,
           '{:1}'.format(height),
           ha = "center")
    
plt.show()
dum = pd.get_dummies(data['Street'], prefix='Street')

data = pd.concat([data,dum], axis=1)
data.drop(columns='Street', inplace=True)
data
data['Alley'].isnull().sum()
data.drop(columns='Alley', inplace=True)
data['LotShape'].nunique()
plt.figure(figsize=(15,5))
ax = sns.countplot(data['LotShape'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()   
    
dum = pd.get_dummies(data['LotShape'], prefix='LotShape')
data = pd.concat([data,dum], axis=1)
data.drop(columns='LotShape', inplace=True)
data
plt.figure(figsize=(15,5))
ax = sns.countplot(data['LandContour'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()   
    
dum = pd.get_dummies(data['LandContour'], prefix='LandContour')
dum
data = pd.concat([data,dum], axis=1)
data.drop(columns='LandContour', inplace=True)
data
plt.figure(figsize=(15,5))
ax = sns.countplot(data['Utilities'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()   
    
data['Utilities'].isnull().sum()
data['Utilities'].fillna('AllPub', inplace=True)
le = preprocessing.LabelEncoder()
data['Utilities'] = le.fit_transform(data['Utilities'])
data
plt.figure(figsize=(15,5))
ax = sns.countplot(data['LotConfig'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()  
data['LotConfig'].isnull().sum()
dum = pd.get_dummies(data['LotConfig'], prefix='LotConfig')
dum
data = pd.concat([data,dum], axis=1)
data.drop(columns='LotConfig', inplace=True)
data
print(data['LandSlope'].nunique())
print(data['LandSlope'].isnull().sum())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['LandSlope'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()  
dum = pd.get_dummies(data['LandSlope'], prefix='LandSlope')
dum
data = pd.concat([data,dum], axis=1)
data.drop(columns='LandSlope', inplace=True)
data
print(data['Neighborhood'].nunique())
print(data['Neighborhood'].isnull().sum())
plt.figure(figsize=(25,5))
ax = sns.countplot(data['Neighborhood'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show() 
le = preprocessing.LabelEncoder()
data['Neighborhood'] = le.fit_transform(data['Neighborhood'])
data
sns.countplot(data['Condition1'])
data['Condition1'] = data['Condition1'].map({'Norm':1,
                                            'Feedr':2,
                                            'PosN':3,
                                            'Artery':4,
                                            'RRAe':5,
                                            'RRNn':6,
                                            'RRAn':7,
                                            'PosA':8,
                                            'RRNe':9})
data['Condition2'] = data['Condition2'].map({'Norm':1,
                                            'Feedr':2,
                                            'PosN':3,
                                            'Artery':4,
                                            'RRAe':5,
                                            'RRNn':6,
                                            'RRAn':7,
                                            'PosA':8,})
                                            
print(data['BldgType'].isnull().sum())
print(data['BldgType'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['BldgType'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show() 
dum = pd.get_dummies(data['BldgType'], prefix='BldgType')
dum
data = pd.concat([data,dum], axis=1)
data.drop(columns='BldgType', inplace=True)
print(data['HouseStyle'].isnull().sum())
print(data['HouseStyle'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['HouseStyle'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show() 
le = preprocessing.LabelEncoder()
data['HouseStyle'] = le.fit_transform(data['HouseStyle'])
data
data['Quality'] =  data['OverallQual'] + data['OverallCond']
skew(data['Quality'])
data.drop(columns=['OverallQual','OverallCond'], inplace=True)
data['YearBuilt'].nunique()
data['Date_diff'] = data['YearRemodAdd'] - data['YearBuilt']

skew(data['Date_diff'])

data.drop(columns=['YearBuilt','YearRemodAdd'], inplace=True)
print(data['RoofStyle'].isnull().sum())
print(data['RoofStyle'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['RoofStyle'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show() 
le = preprocessing.LabelEncoder()
data['RoofStyle'] = le.fit_transform(data['RoofStyle'])
data
print(data['RoofMatl'].isnull().sum())
print(data['RoofMatl'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['RoofMatl'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
le = preprocessing.LabelEncoder()
data['RoofMatl'] = le.fit_transform(data['RoofMatl'])
data['RoofMatl']
print(data['Exterior1st'].isnull().sum())
print(data['Exterior1st'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['Exterior1st'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['Exterior1st'].fillna('VinylSd', inplace=True)
print(data['Exterior2nd'].isnull().sum())
print(data['Exterior2nd'].nunique())
data['Exterior2nd'].fillna('VinylSd', inplace=True)
plt.figure(figsize=(15,5))
ax = sns.countplot(data['Exterior2nd'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()


data['Exterior1st'] = data['Exterior1st'].map({'AsbShng':1,
                                              'AsphShn':2,
                                              'BrkComm':3,
                                              'BrkFace':4,
                                              'CBlock':5,
                                              'CemntBd':6,
                                              'HdBoard':7,
                                              'ImStucc':8,
                                              'MetalSd':9,
                                              'Plywood':10,
                                              'Stone':11,
                                              'Stucco':12,
                                              'VinylSd':13,
                                              'Wd Sdng':14,
                                              'WdShing':15})




data['Exterior2nd'] = data['Exterior2nd'].map({'Other' :0,
                                              'AsbShng':1,
                                              'AsphShn':2,
                                              'Brk Cmn':3,
                                              'BrkFace':4,
                                              'CBlock':5,
                                              'CmentBd':6,
                                              'HdBoard':7,
                                              'ImStucc':8,
                                              'MetalSd':9,
                                              'Plywood':10,
                                              'Stone':11,
                                              'Stucco':12,
                                              'VinylSd':13,
                                              'Wd Sdng':14,
                                              'Wd Shng':15})

data.iloc[:,9:].head(12)
print(data['MasVnrType'].isnull().sum())
print(data['MasVnrType'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['MasVnrType'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['MasVnrType'].fillna('None', inplace=True)

dum = pd.get_dummies(data['MasVnrType'], prefix='MasVnrType', drop_first=True)
data = pd.concat([data,dum], axis=1)
data
data.drop(columns='MasVnrType', inplace=True)

print(data['MasVnrArea'].isnull().sum())
print(data['MasVnrArea'].nunique())
from sklearn.impute import KNNImputer
imputer = KNNImputer( n_neighbors=10, metric='nan_euclidean',weights='uniform')
x = imputer.fit_transform(data[['MasVnrArea']]) 
Xtrans = imputer.transform(x)
data['MasVnrArea'] = Xtrans

plt.figure(figsize=(15,5))
sns.distplot(data['MasVnrArea'])
print(skew(data['MasVnrArea']))
a = np.log1p(data['MasVnrArea'])
skew(a)
data['MasVnrArea'] = np.log1p(data['MasVnrArea'])
skew(data['MasVnrArea'])
print(data['ExterQual'].isnull().sum())
print(data['ExterQual'].nunique())
print(data['ExterCond'].isnull().sum())
print(data['ExterCond'].nunique())
d = ['ExterQual','ExterCond']
for i in d:
    print(i)
    plt.figure(figsize=(15,5))
    ax = sns.countplot(data[i])
    for p in ax.patches:
        h = p.get_height()
        w = p.get_width()/2
        ax.text(p.get_x()+w, h+3,
                '{:1}'.format(h),
               ha="center")
    plt.show()

data['ExterCond'] = data['ExterCond'].map({   'Po':0,
                                              'TA':1,
                                              'Gd':2,
                                              'Fa':3,
                                              'Ex':4,})
data['ExterQual'] = data['ExterQual'].map({   
                                              'TA':1,
                                              'Gd':2,
                                              'Fa':3,
                                              'Ex':4,})

data.iloc[:,11:].head(12)
data['Exter_quality'] = data['ExterCond'] + data['ExterQual']
data.drop(columns=['ExterCond','ExterQual'],inplace=True)
print(data['Foundation'].isnull().sum())
print(data['Foundation'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['Foundation'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
le = preprocessing.LabelEncoder()
data['Foundation'] = le.fit_transform(data['Foundation'])
print(data['BsmtQual'].isnull().sum())
print(data['BsmtQual'].nunique())
data['BsmtQual'].fillna(0, inplace=True)
print(data['BsmtCond'].isnull().sum())
print(data['BsmtCond'].nunique())
data['BsmtCond'].fillna(0, inplace=True)
print(data['BsmtExposure'].isnull().sum())
print(data['BsmtExposure'].nunique())
data['BsmtExposure'].fillna(0, inplace=True)
d = ['BsmtQual','BsmtCond','BsmtExposure']
for i in d:
    print(i)
    plt.figure(figsize=(15,5))
    ax = sns.countplot(data[i])
    for p in ax.patches:
        h = p.get_height()
        w = p.get_width()/2
        ax.text(p.get_x()+w, h+3,
                '{:1}'.format(h),
               ha="center")
    plt.show()
data['BsmtExposure'] = data['BsmtExposure'].map({
                                         'Gd':4,
                                         'Av':3,
                                         'Mn':2,
                                         'No':1,
                                          0:0,
                                         })
data['BsmtCond'] = data['BsmtCond'].map({
                                         'Gd':4,
                                         'TA':3,
                                         'Fa':2,
                                         'Po':1,
                                          0:0,
                                         })
data['BsmtQual'] = data['BsmtQual'].map({'Ex':5,
                                         'Gd':4,
                                         'TA':3,
                                         'Fa':2,
                                          0:0,
                                         })

print(data['BsmtFinType1'].isnull().sum())
print(data['BsmtFinType1'].nunique())
data['BsmtFinType1'].fillna(0, inplace=True)
print(data['BsmtFinType2'].isnull().sum())
print(data['BsmtFinType2'].nunique())
data['BsmtFinType2'].fillna(0, inplace=True)
d = ['BsmtFinType1','BsmtFinType2']
for i in d:
    print(i)
    plt.figure(figsize=(15,5))
    ax = sns.countplot(data[i])
    for p in ax.patches:
        h = p.get_height()
        w = p.get_width()/2
        ax.text(p.get_x()+w, h+3,
                '{:1}'.format(h),
               ha="center")
    plt.show()
data['BsmtFinType1'] = data['BsmtFinType1'].map({
                        'GLQ':6,
                        'ALQ':5,
                        'BLQ':4,
                        'Rec':3,
                        'LwQ':2,
                        'Unf':1,
                        0:0
                            })

data['BsmtFinType2'] = data['BsmtFinType2'].map({
                        'GLQ':6,
                        'ALQ':5,
                        'BLQ':4,
                        'Rec':3,
                        'LwQ':2,
                        'Unf':1,
                        0:0
                            })
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF lets drop them
data.drop(columns=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'], inplace=True)
print(data['TotalBsmtSF'].isnull().sum())
print(data['TotalBsmtSF'].nunique())
plt.figure(figsize=(15,5))
sns.distplot(data['TotalBsmtSF'])
from sklearn.impute import KNNImputer
imputer = KNNImputer( n_neighbors=10, metric='nan_euclidean',weights='uniform')
x = imputer.fit_transform(data[['TotalBsmtSF']]) 
Xtrans = imputer.transform(x)
Xtrans
data['TotalBsmtSF'] = Xtrans


skew(data['TotalBsmtSF'])
print(data['Heating'].isnull().sum())
print(data['Heating'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['Heating'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
le = preprocessing.LabelEncoder()
data['Heating'] = le.fit_transform(data['Heating'])

print(data['HeatingQC'].isnull().sum())
print(data['HeatingQC'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['HeatingQC'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['HeatingQC'] = data['HeatingQC'].map({
                                        'Ex':5,
                                        'Gd':4,
                                        'TA':3,
                                        'Fa':2,
                                        'Po':1
    
})
print(data['CentralAir'].isnull().sum())
print(data['CentralAir'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['CentralAir'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
le = preprocessing.LabelEncoder()
data['CentralAir'] = le.fit_transform(data['CentralAir'])
print(data['Electrical'].isnull().sum())
print(data['Electrical'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['Electrical'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['Electrical'].fillna('SBrkr', inplace=True)
le = preprocessing.LabelEncoder()
data['Electrical'] = le.fit_transform(data['Electrical'])
print(data['GrLivArea'].isnull().sum())
print(data['GrLivArea'].nunique())
plt.figure(figsize=(15,5))
sns.distplot(data['GrLivArea'])
print(skew(data['GrLivArea']))
data['GrLivArea'] = np.log1p(data['GrLivArea'])
skew(data['GrLivArea'])
data.drop(columns=['1stFlrSF','2ndFlrSF','LowQualFinSF'], inplace=True)
print(data['KitchenQual'].isnull().sum())
print(data['KitchenQual'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['KitchenQual'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['KitchenQual'].fillna('TA', inplace=True)
dum = pd.get_dummies(data['KitchenQual'], prefix='KitchenQual')
dum
data = pd.concat([data,dum], axis=1)

data.drop(columns=['KitchenQual'], inplace=True)
print(data['Functional'].isnull().sum())
print(data['Functional'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['Functional'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['Functional'].fillna('Typ', inplace=True)
le = preprocessing.LabelEncoder()
data['Functional'] = le.fit_transform(data['Functional'])
print(data['FireplaceQu'].isnull().sum())
print(data['FireplaceQu'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['FireplaceQu'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['FireplaceQu'].fillna(0, inplace=True)
data['FireplaceQu'] = data['FireplaceQu'].map({
                                            'Ex':5,
                                            'Gd':4,
                                            'TA':3,
                                            'Fa':2,
                                            'Po':1,
                                             0:0
                                            
})
print(data['GarageType'].isnull().sum())
print(data['GarageType'].nunique())
data['GarageType'].fillna(0, inplace=True)

plt.figure(figsize=(15,5))
ax = sns.countplot(data['GarageType'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['GarageType'] = data['GarageType'].map({
                                            '2Types':6,
                                            'Attchd':5,
                                            'Basment':4,
                                            'BuiltIn':3,
                                            'CarPort':2,
                                            'Detchd':1,
                                                0:0
})
data.drop(columns='GarageYrBlt', inplace=True)
print(data['GarageFinish'].isnull().sum())
print(data['GarageFinish'].nunique())
data['GarageFinish'].fillna('0', inplace=True)

plt.figure(figsize=(15,5))
ax = sns.countplot(data['GarageFinish'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
dum = pd.get_dummies(data['GarageFinish'], prefix='GarageFinish')
dum
data = pd.concat([data,dum], axis=1)

data.drop(columns='GarageFinish', inplace=True)
print(data['GarageArea'].isnull().sum())
print(data['GarageArea'].nunique())
data['GarageArea'].fillna(0, inplace=True)
plt.figure(figsize=(15,5))
sns.distplot(data['GarageArea'])
print(skew(data['GarageArea']))
a = np.log1p(data['GarageArea'])
skew(a)
print(data['GarageQual'].isnull().sum())
print(data['GarageQual'].nunique())
data['GarageQual'].fillna(0, inplace=True)
print(data['GarageCond'].isnull().sum())
print(data['GarageCond'].nunique())
data['GarageCond'].fillna(0, inplace=True)
i = ['GarageQual','GarageCond']

for a in i:
    print(a)
    
    plt.figure(figsize=(15,5))
    ax = sns.countplot(data[a])
    for p in ax.patches:
        h = p.get_height()
        w = p.get_width()/2
        ax.text(p.get_x()+w, h+3,
                '{:1}'.format(h),
               ha="center")
    plt.show()

data['GarageQual'] = data['GarageQual'].map({
                                        'Ex':5,
                                        'Gd':4,
                                        'TA':3,
                                        'Fa':2,
                                        'Po':1,
                                        0:0
})

data['GarageCond'] = data['GarageCond'].map({
                                        'Ex':5,
                                        'Gd':4,
                                        'TA':3,
                                        'Fa':2,
                                        'Po':1,
                                        0:0
})
data['Garage_Quality'] = data['GarageQual'] + data['GarageCond']

data.drop(columns=['GarageQual','GarageCond'], inplace=True)
print(data['PavedDrive'].isnull().sum())
print(data['PavedDrive'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['PavedDrive'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()

data['PavedDrive'] = data['PavedDrive'].map({
                                        'Y':2,
                                        'P':1,
                                        'N':0
})
print(data['WoodDeckSF'].isnull().sum())
print(data['WoodDeckSF'].nunique())
plt.figure(figsize=(15,5))
sns.distplot(data['WoodDeckSF'])
print(skew(data['WoodDeckSF']))
a = np.log1p(data['WoodDeckSF'])
skew(a)
data['WoodDeckSF'] = np.log1p(data['WoodDeckSF'])
skew(data['WoodDeckSF'])
print(data['OpenPorchSF'].isnull().sum())
print(data['OpenPorchSF'].nunique())
plt.figure(figsize=(15,5))
sns.distplot(data['OpenPorchSF'])
print(skew(data['OpenPorchSF']))
a = np.log1p(data['OpenPorchSF'])
skew(a)
data['OpenPorchSF'] = np.log1p(data['OpenPorchSF'])
skew(data['OpenPorchSF'])
print(data['EnclosedPorch'].isnull().sum())
print(data['EnclosedPorch'].nunique())
data['EnclosedPorch'].plot()
print(skew(data['EnclosedPorch']))
a = np.log1p(data['EnclosedPorch'])
skew(a)
data['EnclosedPorch'] = np.log1p(data['EnclosedPorch'])
skew(data['EnclosedPorch'])
print(data['3SsnPorch'].isnull().sum())
print(data['3SsnPorch'].nunique())
data['3SsnPorch'].plot()
print(skew(data['3SsnPorch']))
a = np.log1p(data['3SsnPorch'])
# a = np.log(a)
skew(a)
data.drop(columns=['3SsnPorch','ScreenPorch','PoolArea'], inplace=True)
print(data['PoolQC'].isnull().sum())
print(data['PoolQC'].nunique())
data['PoolQC'].fillna(0, inplace=True)

plt.figure(figsize=(15,5))
ax = sns.countplot(data['PoolQC'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()

data['PoolQC'] = data['PoolQC'].map({
                                    'Ex':3,
                                    'Gd':2,
                                    'Fa':1,
                                        0:0
})
print(data['Fence'].isnull().sum())
print(data['Fence'].nunique())
data['Fence'].fillna(0, inplace=True)

plt.figure(figsize=(15,5))
ax = sns.countplot(data['Fence'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()

data['Fence'] = data['Fence'].map({
                                'GdPrv':4,
                                'MnPrv':3,
                                'GdWo':2,
                                'MnWw':1,
                                0:0
})
print(data['MiscFeature'].isnull().sum())
print(data['MiscFeature'].nunique())
data['MiscFeature'].fillna(0, inplace=True)

plt.figure(figsize=(15,5))
ax = sns.countplot(data['MiscFeature'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()

data['MiscFeature'] = data['MiscFeature'].map({
                                            'Gar2':1,
                                            'Othr':1,
                                            'Shed':1,
                                            'TenC':1,
                                                0:0
    
                                            
})
print(data['MiscVal'].isnull().sum())
print(data['MiscVal'].nunique())
data['MiscVal'].plot()
print(skew(data['MiscVal']))
a = np.log1p(data['MiscVal'])
skew(a)
data['MiscVal'] = np.log1p(data['MiscVal'])
skew(data['MiscVal'])
print(data['SaleType'].isnull().sum())
print(data['SaleType'].nunique())

plt.figure(figsize=(15,5))
ax = sns.countplot(data['SaleType'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()

data['SaleType'].fillna('WD', inplace=True)
le = preprocessing.LabelEncoder()
data['SaleType'] = le.fit_transform(data['SaleType'])
print(data['SaleCondition'].isnull().sum())
print(data['SaleCondition'].nunique())
plt.figure(figsize=(15,5))
ax = sns.countplot(data['SaleCondition'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
le = preprocessing.LabelEncoder()
data['SaleCondition'] = le.fit_transform(data['SaleCondition'])
print(data['BsmtFullBath'].isnull().sum())
print(data['BsmtFullBath'].nunique())


plt.figure(figsize=(15,5))
ax = sns.countplot(data['BsmtFullBath'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['BsmtFullBath'].fillna(0, inplace=True)
print(data['BsmtHalfBath'].isnull().sum())
print(data['BsmtHalfBath'].nunique())


plt.figure(figsize=(15,5))
ax = sns.countplot(data['BsmtHalfBath'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['BsmtHalfBath'].fillna(1, inplace=True)

print(data['GarageCars'].isnull().sum())
print(data['GarageCars'].nunique())


plt.figure(figsize=(15,5))
ax = sns.countplot(data['GarageCars'])
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:1}'.format(h),
           ha="center")
plt.show()
data['GarageCars'].fillna(2, inplace=True)
data
data.info()
  
train_data = data.iloc[:train.shape[0]]
train_data['SalePrice'].tail()
test_data = data.iloc[train.shape[0]:]
test_data['SalePrice'].head()
plt.figure(figsize=(15,5))
sns.distplot(train_data['SalePrice'])
print(skew(train_data['SalePrice']))
a = np.log1p(train_data['SalePrice'])
skew(a)
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
skew(train_data['SalePrice'])
train_data.to_csv('Train_data.csv' )
test_data.to_csv('Test_data.csv')
train_data

X=train_data.drop(['Id','SalePrice'],1)
y=train_data['SalePrice']
test_data=test_data.drop(['Id','SalePrice'],1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=7)

from xgboost import XGBRegressor
model_2 = XGBRegressor(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=10,
 gamma=0.1,
 colsample_bytree=0.8,
 seed=100,
 eval_metric='rmse'
 )
model_2.fit(X_train, y_train, eval_metric='rmse', 
          eval_set=[(X_test, y_test)], early_stopping_rounds=500, verbose=100)
xgb = XGBRegressor(
 learning_rate =0.01,
 n_estimators=3897,
 max_depth=10,
 gamma=0.1,
 colsample_bytree=0.8,
 seed=100,
 eval_metric='rmse'
 )

xgb_model=xgb.fit(X,y)
y_pred1=xgb.predict(test_data)
y_pred1=np.expm1(y_pred1)
y_pred1
from lightgbm import LGBMRegressor
lgb_fit_params={"early_stopping_rounds":500, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'verbose':100
           }

lgb_params = {'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'verbose': 0,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.001,
 'max_bin': 255,
 'max_depth': 9,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 31}
lgb_params
clf_lgb = LGBMRegressor(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, y_train, **lgb_fit_params)
clf_lgb.best_iteration_
clf_lgb=LGBMRegressor(n_estimators=int(clf_lgb.best_iteration_*1.2), **lgb_params)
lgb_model=clf_lgb.fit(X, y)

y_pred2=lgb_model.predict(test_data)
y_pred2=np.expm1(y_pred2)
y_pred2
y_pred=(0.6*y_pred1)+(y_pred2*0.4)
y_pred
sub.head()
sub['SalePrice'] = y_pred
sub.to_csv('submission.csv')
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmsle_cv(ENet)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from math import sqrt 
from sklearn.metrics import mean_squared_error, mean_squared_log_error

errrf = []
y_pred_totrf = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rf = RandomForestRegressor(random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    print("RMSLE: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))

    errrf.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))
    p = rf.predict(test_data)
    y_pred_totrf.append(p)
final = np.exp(np.mean(y_pred_totrf,0))

sub['SalePrice'] = y_pred
sub.to_csv('submission.csv')