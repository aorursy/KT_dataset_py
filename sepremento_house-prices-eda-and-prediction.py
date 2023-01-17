import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats # we will need norm function to fit the seaborn.distplot



# models

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LassoCV

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# supplementary

from sklearn.model_selection import KFold, cross_val_score, train_test_split



from scipy.special import boxcox1p



import os

pd.set_option('display.max_columns', None)

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sns.distplot(df.SalePrice, fit=stats.norm, color='indigo');
# taking the logarithmic SalePrice

logSP = np.log(df.SalePrice)

logdf = df.drop(['SalePrice'], axis=1)

logdf['LogSalePrice']=logSP



# creating a correlation matrix

corrmat = df.corr() ** 2

plt.figure(figsize=(15,12))

sns.set(font_scale=0.75)

sns.heatmap(corrmat, vmax=.75, square = True, cmap="RdYlGn");
corrmat = df.corr()**2

corrmat_log = logdf.corr()**2



k=10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cols_log = corrmat_log.nlargest(k,'LogSalePrice')['LogSalePrice'].index

cm = np.corrcoef(df[cols].values.T)**2

cm_log = np.corrcoef(logdf[cols_log].values.T)**2

sns.set(font_scale=1.25)



plt.figure(figsize=(23,7))

plt.subplot(121)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':10}, yticklabels=cols.values,

                xticklabels=cols.values, cmap='viridis', vmax=0.7)

plt.subplot(122)

hm_log = sns.heatmap(cm_log, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':10}, yticklabels=cols_log.values,

                xticklabels=cols_log.values, cmap='viridis',vmax=0.7)

plt.show()
# select numerical features

num_feat = df.select_dtypes(include='number')

num_feat.describe()
yr_features = ['YearBuilt','YearRemodAdd', 'GarageYrBlt']

plt.figure(figsize=(18,30))

for i, feature in enumerate(yr_features):

    plt.subplot(3,1,i+1)

    sns.boxplot(x=feature,y='SalePrice', data=df)

    plt.xticks(rotation=90)

    plt.xticks(fontsize=8)
date_sale = ['MoSold', 'YrSold']

plt.figure(figsize=(12,10))

for i, feature in enumerate(date_sale):

    plt.subplot(2,2,2*i+1)

    sns.boxplot(x=feature,y='SalePrice', data=df)

    plt.subplot(2,2,2*(i+1))

    sns.countplot(x=feature, data=df)
baths = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']

plt.figure(figsize=(12,18))

for i, feature in enumerate(baths):

    plt.subplot(4,2,2*i+1)

    sns.boxplot(x=feature,y='SalePrice', data=df)

    plt.subplot(4,2,2*(i+1))

    sns.countplot(x=feature, data=df)
temp = df[['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','SalePrice']]

temp['NumOfBaths'] = temp['BsmtFullBath']+temp['BsmtHalfBath']+temp['FullBath']+temp['HalfBath']

plt.figure(figsize=(12,5))

plt.subplot(121)

sns.boxplot(x='NumOfBaths',y='SalePrice', data=temp);

plt.subplot(122)

sns.countplot(x='NumOfBaths', data=temp);
rooms_n_garage = ['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

plt.figure(figsize=(12,23))

for i, feature in enumerate(rooms_n_garage):

    plt.subplot(5,2,2*i+1)

    sns.boxplot(x=feature,y='SalePrice', data=df)

    plt.subplot(5,2,2*(i+1))

    sns.countplot(x=feature, data=df)
features = ['MSSubClass', 'OverallQual', 'OverallCond']

plt.figure(figsize=(12,14))

for i, feature in enumerate(features):

    plt.subplot(3,2,2*i+1)

    sns.boxplot(x=feature,y='SalePrice', data=df)

    plt.subplot(3,2,2*(i+1))

    sns.countplot(x=feature, data=df)
df.loc[df.BsmtFinSF2 > 0, 'HasBsmt2'] = 'Yes'

df.loc[df.BsmtFinSF2 ==0, 'HasBsmt2'] = 'No'

df = df.drop('BsmtFinSF2', axis=1)
df.loc[df.LowQualFinSF > 0, 'LowQualFin'] = 'Yes'

df.loc[df.LowQualFinSF ==0, 'LowQualFin'] = 'No'

df = df.drop('LowQualFin', axis=1)
bsmt_area = ['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF']

plt.figure(figsize=(25,18))

for i, feature in enumerate(bsmt_area):

    plt.subplot(3,3,3*i+1)

    plt.scatter(df[feature], df['SalePrice'], c='brown', alpha=0.5)

    plt.subplot(3,3,3*i+2)

    sns.distplot(df[feature], color='brown')

    plt.subplot(3,3,3*i+3)

    stats.probplot(df[feature], plot=plt);
areas = ['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea']

plt.figure(figsize=(25,24))

for i, feature in enumerate(areas):

    plt.subplot(4,3,3*i+1)

    plt.scatter(df[feature], df['SalePrice'], c='orange',alpha=0.5)

    plt.subplot(4,3,3*i+2)

    sns.distplot(df[feature], color='orange')

    plt.subplot(4,3,3*i+3)

    stats.probplot(df[feature], plot=plt);
lot_areas = ['LotArea', 'LotFrontage']

plt.figure(figsize=(25,10))

for i, feature in enumerate(lot_areas):

    plt.subplot(2,3,3*i+1)

    plt.scatter(df[feature], df['SalePrice'], c='darkgreen', alpha=0.5)

    plt.subplot(2,3,3*i+2)

    sns.distplot(df[feature], color='darkgreen')

    plt.subplot(2,3,3*i+3)

    stats.probplot(df[feature], plot=plt);
# Box-Cox transformation

GrLivArea_tr = boxcox1p(df.GrLivArea, stats.boxcox_normmax(df.GrLivArea +1))



plt.figure(figsize=(25,12))



# before box-cox transformation

plt.subplot(231)

plt.scatter(df.GrLivArea, df.SalePrice);

plt.subplot(232)

sns.distplot(df.GrLivArea);

plt.subplot(233)

stats.probplot(df.GrLivArea, plot=plt);



# after box-cox transformation

plt.subplot(234)

plt.scatter(GrLivArea_tr, df.SalePrice);

plt.subplot(235)

sns.distplot(GrLivArea_tr);

plt.subplot(236)

stats.probplot(GrLivArea_tr, plot=plt);
cat_feat = df.select_dtypes(include=['O'])

cat_feat.describe()
for i, feature in enumerate(cat_feat.columns):

    cat_df = df[['SalePrice', feature]].groupby(feature).mean().round(2)

    count_df = df[['SalePrice',feature]].groupby(feature).count().rename(columns={'SalePrice':'Count'})

    cat_df = cat_df.join(count_df).sort_values(by='SalePrice', ascending=False)

    print(cat_df)

    print('-'*40)
cat_msz = df[['SalePrice','MSZoning']].groupby('MSZoning').mean().round(2)

count_df = df[['SalePrice','MSZoning']].groupby('MSZoning').count().rename(columns={'SalePrice':'Count'})

cat_msz = cat_msz.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,8))

plt.subplot(131)

bbox=[0.2, 0.2, 0.6, 0.6]

mpl_table = plt.table(cellText = np.round(cat_msz.values,2), rowLabels = cat_msz.index, colLabels=cat_msz.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.MSZoning);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='MSZoning', y='SalePrice', data=df);
cat_lshp = df[['SalePrice','LotShape']].groupby('LotShape').mean().round(2)

count_df = df[['SalePrice','LotShape']].groupby('LotShape').count().rename(columns={'SalePrice':'Count'})

cat_lshp = cat_lshp.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,8))

plt.subplot(131)

bbox=[0.2, 0.2, 0.6, 0.6]

mpl_table = plt.table(cellText = np.round(cat_lshp.values,2), rowLabels = cat_lshp.index, colLabels=cat_lshp.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.LotShape);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='LotShape', y='SalePrice', data=df);
cat_lctr = df[['SalePrice','LandContour']].groupby('LandContour').mean().round(2)

count_df = df[['SalePrice','LandContour']].groupby('LandContour').count().rename(columns={'SalePrice':'Count'})

cat_lctr = cat_lctr.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,8))

plt.subplot(131)

bbox=[0.2, 0.2, 0.6, 0.6]

mpl_table = plt.table(cellText = np.round(cat_lctr.values,2), rowLabels = cat_lctr.index, colLabels=cat_lctr.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.LandContour);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='LandContour', y='SalePrice', data=df);
cat_hstl = df[['SalePrice','HouseStyle']].groupby('HouseStyle').mean().round(2)

count_df = df[['SalePrice','HouseStyle']].groupby('HouseStyle').count().rename(columns={'SalePrice':'Count'})

cat_hstl = cat_hstl.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,8))

plt.subplot(131)

bbox=[0.2, 0.2, 0.6, 0.6]

mpl_table = plt.table(cellText = np.round(cat_hstl.values,2), rowLabels = cat_hstl.index, colLabels=cat_hstl.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.HouseStyle);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='HouseStyle', y='SalePrice', data=df);
cat_nbh = df[['SalePrice','Neighborhood']].groupby('Neighborhood').mean().round(2)

count_df = df[['SalePrice','Neighborhood']].groupby('Neighborhood').count().rename(columns={'SalePrice':'Count'})

cat_nbh = cat_nbh.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,7))

plt.subplot(131)

bbox=[0, 0, 1, 1]

mpl_table = plt.table(cellText = np.round(cat_nbh.values,2), rowLabels = cat_nbh.index, colLabels=cat_nbh.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.Neighborhood);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='Neighborhood', y='SalePrice', data=df);
cat_ex1 = df[['SalePrice','Exterior1st']].groupby('Exterior1st').mean().round(2)

count_df = df[['SalePrice','Exterior1st']].groupby('Exterior1st').count().rename(columns={'SalePrice':'Count'})

cat_ex1 = cat_ex1.join(count_df).sort_values(by='SalePrice', ascending=False)



cat_ex2 = df[['SalePrice','Exterior2nd']].groupby('Exterior2nd').mean().round(2)

count_df = df[['SalePrice','Exterior2nd']].groupby('Exterior2nd').count().rename(columns={'SalePrice':'Count'})

cat_ex2 = cat_ex2.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,16))

plt.subplot(231)

bbox=[0, 0, 1, 1]

mpl_table = plt.table(cellText = np.round(cat_ex1.values,2), rowLabels = cat_ex1.index, colLabels=cat_ex1.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(232)

plt.xticks(rotation=90)

sns.countplot(df.Exterior1st);

plt.subplot(233)

plt.xticks(rotation=90)

sns.boxplot(x='Exterior1st', y='SalePrice', data=df);



plt.subplot(234)

bbox=[0, 0, 1, 1]

mpl_table = plt.table(cellText = np.round(cat_ex2.values,2), rowLabels = cat_ex2.index, colLabels=cat_ex2.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(235)

plt.xticks(rotation=90)

sns.countplot(df.Exterior2nd);

plt.subplot(236)

plt.xticks(rotation=90)

sns.boxplot(x='Exterior2nd', y='SalePrice', data=df);
cat_mvs = df[['SalePrice','MasVnrType']].groupby('MasVnrType').mean().round(2)

count_df = df[['SalePrice','MasVnrType']].groupby('MasVnrType').count().rename(columns={'SalePrice':'Count'})

cat_mvs = cat_mvs.join(count_df).sort_values(by='SalePrice', ascending=False)



plt.figure(figsize=(22,8))

plt.subplot(131)

bbox=[0.2, 0.2, 0.6, 0.6]

mpl_table = plt.table(cellText = np.round(cat_mvs.values,2), rowLabels = cat_mvs.index, colLabels=cat_mvs.columns,

                     bbox=bbox)

plt.axis('off')

mpl_table.auto_set_font_size(False)

mpl_table.set_fontsize(12)

plt.subplot(132)

plt.xticks(rotation=90)

sns.countplot(df.MasVnrType);

plt.subplot(133)

plt.xticks(rotation=90)

sns.boxplot(x='MasVnrType', y='SalePrice', data=df);
missing_data = df.isnull().sum().sort_values(ascending=False)

missing_data.head(20)
all_data = pd.concat((df.loc[:,'MSSubClass':'SaleCondition'], test_df.loc[:,'MSSubClass':'SaleCondition']))
all_data.loc[all_data.TotalBsmtSF > 0, 'HasBsmt'] = 'Yes'

all_data.loc[all_data.TotalBsmtSF == 0, 'HasBsmt'] = 'No'
all_data.loc[all_data.GarageArea > 0, 'HasGarage'] = 'Yes'

all_data.loc[all_data.GarageArea ==0, 'HasGarage'] = 'No'
all_data.loc[all_data.Fireplaces > 0, 'HasFireplace'] = 'Yes'

all_data.loc[all_data.Fireplaces ==0, 'HasFireplace'] = 'No'
all_data.loc[all_data.PoolArea > 0, 'HasPool'] = 'Yes'

all_data.loc[all_data.PoolArea ==0, 'HasPool'] = 'No'
all_data.loc[all_data['2ndFlrSF'] > 0, 'Has2ndFlr'] = 'Yes'

all_data.loc[all_data['2ndFlrSF'] ==0, 'Has2ndFlr'] = 'No'
all_data['NumOfBaths'] = all_data['BsmtFullBath']+all_data['BsmtHalfBath']+all_data['FullBath']+all_data['HalfBath']
categorical_features = ['MSSubClass', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond','BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',

                       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'NumOfBaths', 'HasBsmt','HasGarage','HasFireplace','HasPool','Has2ndFlr']

for feature in categorical_features:

    all_data[feature] = all_data[feature].astype('category')
df['SalePrice'] = np.log1p(df['SalePrice'])



numeric_feats = all_data.dtypes[(all_data.dtypes == 'int') | (all_data.dtypes=='float') ].index



skewed_feats = all_data[numeric_feats].dropna().skew()

skewed_feats = skewed_feats[skewed_feats>0.75].index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean()) 
X_train = all_data[:df.shape[0]]

X_test = all_data[df.shape[0]:]

y = df.SalePrice
def cv_rmse(model, X=X_train):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))

    return rmse
model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train,y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Лассо выбрал " + str(sum(coef !=0)) + " переменных и удалил " + str(sum(coef ==0)) + " переменных")
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])

plt.figure(figsize=(8,10))

imp_coef.plot(kind='barh');
# xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                     max_depth=3, min_child_weight=0,

#                                     gamma=0, subsample=0.7,

#                                     colsample_bytree=0.7,

#                                     objective='reg:squarederror', nthread=-1,

#                                     scale_pos_weight=1, seed=27,

#                                     reg_alpha=0.00006)
# score = cv_rmse(xgboost)

# print("XGBoost: {:.4f} ({:.4f})".format(score.mean(),score.std()))