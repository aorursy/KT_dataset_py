# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sample_sub  = pd.read_csv('../input/sample_submission.csv')

train_df  = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv')

# sample_sub.head()
train_df.head()
len(train_df.columns)   
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train_df.shape))

print("The test data size before dropping Id feature is : {} ".format(test_df.shape))



#Save the 'Id' column

train_ID = train_df['Id']

test_ID = test_df['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train_df.drop("Id", axis = 1, inplace = True)

test_df.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test_df.shape))
corr = train_df.corr()

fig , ax = plt.subplots(figsize = (7,7))

sns.heatmap(corr, vmax = 0.8, square = True)
#### Sales Price Correlation Matrix 



k = 10     # number of variables we are looking for.



cols = corr.nlargest(k,'SalePrice')['SalePrice'].index

# cols

cm = np.corrcoef(train_df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train_df.shape
ntrain = train_df.shape[0]

ntest = test_df.shape[0]

y_train = train_df.SalePrice.values
y_train
all_data = pd.concat((train_df, test_df), sort= False).reset_index(drop = True)

all_data.drop(['SalePrice'], axis=1,inplace=True)

print('The combined data variable size is {}:'.format(all_data.shape))

# all_data.head(5)

all_data_na = (all_data.isna().sum()/len(all_data))*100

all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending = False)

missing_data = pd.DataFrame({'Missing Data ':all_data_na})



# len(all_data_na)   # 34 



missing_data.head(10)
fig, ax = plt.subplots(figsize =(10,7))

plt.xticks(rotation='90')

sns.barplot(x = all_data_na.index, y = all_data_na)

plt.xlabel('Features', fontsize = 15)

plt.ylabel('Percentage of missing values', fontsize = 15)

plt.title('Percentage missing by Total values')
all_data.shape 

all_data = all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis =1 )

all_data.shape   #down to 75 variables now.
missing_data = missing_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis =0)
missing_data.head(10)

# len(missing_data)
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna("None")

# all_data['FireplaceQu']
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# all_data['LotFrontage']

all_data = all_data.drop(['GarageCars', 'GarageYrBlt', 'GarageQual'], axis =1)

all_data.shape  #The second dimension dropped from 75 to 72 as expected because we dropped 3 variables/features. 
for col in ('GarageType', 'GarageFinish', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data = all_data.drop(['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis = 1)

all_data.shape  #The second dimension dropped from 72 to 69 as expected. 
for col_n in ('BsmtQual', 'BsmtCond'):

    all_data[col_n] = all_data[col].fillna('None')

    

for col_0 in ('TotalBsmtSF','BsmtFinSF1' , 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF'):

    all_data[col_0] = all_data[col_0] .fillna(0)
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop('Utilities', axis =1 )
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond',  'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual',  

        'Functional', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))

# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# all_data.dtypes



# all_data['MSZoning']
# Let's create a list of the features that we are interested in and add sales price to that as well

hot_feats_list = ['GarageArea', 'GrLivArea', 'OverallQual', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd']

bsmt_feats = ['BsmtQual', 'BsmtCond', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath',

              'TotalSF']

misc_feats = ['Functional', 'KitchenQual']

# target_feat= ['SalePrice']
# all_data['BsmtCond']
all_data.shape
# (all_data['LowQualFinSF']!= 0).values.sum()



all_data.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'], axis = 1, inplace = True)
all_data.shape
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

test = all_data[ntrain:]

from sklearn.ensemble import RandomForestRegressor

rand_for_clf =RandomForestRegressor()

rand_for_clf.fit(train, y_train)
pred_price = rand_for_clf.predict(test)

print(pred_price)
my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': pred_price})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)