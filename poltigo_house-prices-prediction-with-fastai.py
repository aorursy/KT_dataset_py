# import libraries

import numpy as np 

import pandas as pd 

from scipy import stats

from scipy.stats import norm, skew

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')



from fastai.tabular import *

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) #Limiting floats output to 2 decimal points
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
sns.distplot(train['SalePrice']) ;



# Get the fitted parameters 

(mu, sigma) = norm.fit(train['SalePrice'])



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution');
# log-transform the price

train['SalePrice'] = train['SalePrice'].apply(np.log)



sns.distplot(train['SalePrice'])

plt.show();
# lets remove houses with living area <4000 as recommended by the author of the dataset

train = train[train['GrLivArea']<4000]
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values.copy()

all_data = pd.concat((train, test), sort = 'True').reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
missing_data = (all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)*100



missing_data = pd.DataFrame(missing_data)



plt.figure(figsize = (10,7))

missing_data.head(20).plot(kind = 'bar')

plt.title('Percent of missing data');
# we could drop the first four columns as they miss too much data

droplist = list(missing_data.head(4).index)
all_data.drop(droplist, axis = 1, inplace = True)
col = train.corr().nlargest(10, 'SalePrice')['SalePrice'].index

corr_matrix = np.corrcoef(train[col].values.T)
plt.figure(figsize = (10,8))

sns.heatmap(corr_matrix, cmap = 'coolwarm', annot = True, xticklabels= col.values, yticklabels= col.values);
#These are the candidates to drop if there are too little distinct values in the columns

train.nunique().sort_values(ascending = False).tail(7)
train.Utilities.value_counts()
train.CentralAir.value_counts()
train.Street.value_counts()
train.Alley.value_counts()
#we can drop Street and Utilities as they contain no unique useful information

all_data.drop(['Street', 'Utilities'], axis = 1, inplace = True)
#FireplaceQu : data description says NA means "no fireplace"

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
"""

LotFrontage : Since the area of each street connected to the house property most likely have a similar area

to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

"""

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')
# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# Functional : data description says NA means typical

all_data["Functional"] = all_data["Functional"].fillna("Typ")
# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
# SaleType : Fill in again with most frequent which is "WD"

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
# MSSubClass : Na most likely means No building class. We can replace missing values with None

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
missing_data = (all_data.isnull().sum() / len(all_data)).sort_values(ascending = False)*100



missing_data = pd.DataFrame(missing_data)

missing_data.head(1)
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data.head()
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
train = all_data[:ntrain]

test = all_data[ntrain:]
dep_var = 'SalePrice'
cat_names = list(train.select_dtypes(include = ['object', 'bool']).columns)
cont_names = list(train.select_dtypes(exclude = ['object', 'bool']).columns)
# Add back sale prices

train['SalePrice'] = y_train
# defining steps to process the input data

procs = [FillMissing, Categorify, Normalize]



# Test tabularlist

test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)



# Train data bunch - important to define label_cls = FloatList to ensure the model works with scalar values

data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .split_by_rand_pct(valid_pct = 0.2)

                        .label_from_df(cols = dep_var, label_cls = FloatList, log = False )

                        .add_test(test)

                        .databunch( bs = 64))
data.show_batch(rows=5, ds_type=DatasetType.Valid)
max_log_y = (np.max(train['SalePrice'])*1.2)

y_range = torch.tensor([0, max_log_y], device=defaults.device)
# create the model

learn = tabular_learner(data, layers=[600,300], ps=[0.001,0.01], y_range = y_range, emb_drop=0.04, metrics=exp_rmspe)



learn.model



# select the appropriate learning rate

learn.lr_find()



# we typically find the point where the slope is steepest

learn.recorder.plot()
# Fit the model based on selected learning rate

learn.fit_one_cycle(50, max_lr = 1e-02)
#Plotting The losses for training and validation

learn.recorder.plot_losses(skip_start = 500)
learn.fit_one_cycle(5, 3e-4)
# get predictions

preds, targets = learn.get_preds(DatasetType.Test)

labels = [np.exp(p[0].data.item()) for p in preds]



# create submission file to submit in Kaggle competition

submission = pd.DataFrame({'Id': test_ID, 'SalePrice': labels})

submission.to_csv('submission.csv', index=False)



submission.describe()
submission.head()