# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotnine 

from plotnine import * 

import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train['type'] = 'train'

test['type'] = 'test'

total = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

del total['SalePrice']



agg = pd.DataFrame(total.isnull().sum()).reset_index()

agg.columns = ['column', 'count']

agg['percent'] = 100 * agg['count'] / total.shape[0]

agg = agg[agg['percent'] != 0]
agg = agg.sort_values(by='percent', ascending=False)

(ggplot(data = agg) 

 + geom_bar(aes(x='column', y='percent'), fill = '#49beb7', stat='identity', color='black')

 + scale_x_discrete(limits=agg['column'].values) # sorting columns 

 + theme_light()  

 + labs(title = 'Missing Graph : The percent of columns',

         x = '',

         y = 'Missing Ratio (%)')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
total[(total['GarageCars'].notnull()) & (total['GarageFinish'].isnull())][['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageType', 'GarageCars', 'GarageArea']]['GarageArea'].describe()
agg[agg['column'].isin(['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageType'])]
total[(total['GarageType'].notnull()) & (total['GarageFinish'].isnull())][['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageType', 'GarageCars', 'GarageArea']]
total[(total['Neighborhood'] == 'NAmes') & (total['MSZoning'] == 'RL') & (total['OverallQual'] == 6) & (total['GarageType'] == 'Detchd') & (total['GarageCars'] == 1) & (total['GarageArea'] >= 250) & (total['GarageArea'] <= 450)][['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'GarageYrBlt']]
# 참고로 index는 0부터 시작하기에 index 666는 Id 667을 의미합니다. 

total.loc[2126, 'GarageFinish'] = 'Unf'

total.loc[2126, 'GarageQual'] = 'TA'

total.loc[2126, 'GarageCond'] = 'TA'

total.loc[2126, 'GarageYrBlt'] = total[(total['Neighborhood'] == 'NAmes') & (total['MSZoning'] == 'RL') & (total['OverallQual'] == 6) & (total['GarageType'] == 'Detchd') & (total['GarageCars'] == 1) & (total['GarageArea'] >= 250) & (total['GarageArea'] <= 450)][['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'GarageYrBlt']]['GarageYrBlt'].median()
total.loc[2576, 'GarageType'] = np.NaN
agg = pd.DataFrame(total.isnull().sum()).reset_index()

agg.columns = ['column', 'count']

agg['percent'] = 100 * agg['count'] / total.shape[0]

agg = agg[agg['percent'] != 0]

agg[agg['column'].isin(['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'GarageYrBlt'])]
index = total[total['GarageType'].isnull()].index

for col in ['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'GarageYrBlt']:

    if total[col].dtypes == 'O':

        total.loc[index, col] = 'None'

    else:

        total.loc[index, col] = -1
agg[agg['column'].isin(['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'])]
total[(total['BsmtUnfSF'].notnull()) & (total['BsmtUnfSF'] != 0) & (total['BsmtExposure'].isnull())][['Neighborhood', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtUnfSF']]
for i in [948, 1487, 2348]:

    A = total.loc[i]

    total.loc[i, 'BsmtExposure'] = total[(total['Neighborhood'] == A['Neighborhood']) & (total['MSZoning'] == A['MSZoning']) & (total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtCond'] == 'TA') & (total['BsmtQual'] == 'Gd') & (total['BsmtFinType2'] == 'Unf') & (total['BsmtFinType1'] == 'Unf') & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-125) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+125)]['BsmtExposure'].mode()[0]

        

    # 결측치가 채워진지 확인 

    print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtExposure'])
total[(total['BsmtUnfSF'].notnull()) &(total['BsmtUnfSF'] != 0) & (total['BsmtCond'].isnull()) & (total['BsmtQual'].notnull())][['Neighborhood', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtUnfSF']]
for i in [2185, 2524]:

    A = total.loc[i]

    try:

        total.loc[i, 'BsmtCond'] = total[(total['Neighborhood'] == A['Neighborhood']) & (total['MSZoning'] == A['MSZoning']) & (total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtCond'].mode()[0]

        

        # 결측치가 채워진지 확인 

        print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtCond'])

    except:

        try:

            total.loc[i, 'BsmtCond'] = total[(total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtCond'].mode()[0]

        

            # 결측치가 채워진지 확인 

            print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtCond'])

        except:

            print("No Imputation")
total[(total['BsmtUnfSF'].notnull()) &(total['BsmtUnfSF'] != 0) & (total['BsmtFinType2'].isnull()) & (total['BsmtQual'].notnull())][['Neighborhood', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtUnfSF']]
for i in [332]:

    A = total.loc[i]

    try:

        total.loc[i, 'BsmtFinType2'] = total[(total['Neighborhood'] == A['Neighborhood']) & (total['MSZoning'] == A['MSZoning']) & (total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtFinType2'].mode()[0]

        

        # 결측치가 채워진지 확인 

        print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtFinType2'])

    except:

        try:

            total.loc[i, 'BsmtFinType2'] = total[(total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtFinType2'].mode()[0]

        

            # 결측치가 채워진지 확인 

            print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtFinType2'])

        except:

            print("No Imputation")
total[(total['BsmtUnfSF'].notnull()) &(total['BsmtUnfSF'] != 0) & (total['BsmtQual'].isnull()) & (total['BsmtFinType2'].notnull())][['Neighborhood', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtUnfSF']]
for i in [2217, 2218]:

    A = total.loc[i]

    try:

        total.loc[i, 'BsmtQual'] = total[(total['Neighborhood'] == A['Neighborhood']) & (total['MSZoning'] == A['MSZoning']) & (total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtCond'] == A['BsmtCond']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtQual'].mode()[0]

        

        # 결측치가 채워진지 확인 

        print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtFinType2'])

    except:

        try:

            total.loc[i, 'BsmtQual'] = total[(total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtCond'] == A['BsmtCond']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtQual'].mode()[0]

        

            # 결측치가 채워진지 확인 

            print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtQual'])

        except:

            print("No Imputation")
total[(total['BsmtCond'].isnull()) & (total['BsmtFinType2'].notnull())][['Neighborhood', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'BsmtUnfSF']]
for i in [2040]:

    A = total.loc[i]

    total.loc[i, 'BsmtUnfSF'] = total[(total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1'])]['BsmtUnfSF'].median()

    total.loc[i, 'BsmtCond'] = total[(total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['BsmtExposure'] == A['BsmtExposure']) & (total['BsmtQual'] == A['BsmtQual']) & (total['BsmtFinType2'] == A['BsmtFinType2']) & (total['BsmtFinType1'] == A['BsmtFinType1']) & (total['BsmtUnfSF'] >= A['BsmtUnfSF']-250) & (total['BsmtUnfSF'] <= A['BsmtUnfSF']+250)]['BsmtCond'].mode()[0]

    # 결측치가 채워진지 확인 

    print("Fill {} Missing Values:".format(i), total.loc[i, 'BsmtCond'])
agg = pd.DataFrame(total.isnull().sum()).reset_index()

agg.columns = ['column', 'count']

agg['percent'] = 100 * agg['count'] / total.shape[0]

agg = agg[agg['percent'] != 0]

agg[agg['column'].isin(['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1'])]
index = total[total['BsmtQual'].isnull()].index

for col in ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']:

    if total[col].dtypes == 'O':

        total.loc[index, col] = 'None'

    else:

        total.loc[index, col] = -1
agg[agg['column'].isin(['MasVnrType', 'MasVnrArea'])]
print("2 of Variables Missing :", total[(total['MasVnrType'].isnull()) & (total['MasVnrArea'].isnull())][['MasVnrType', 'MasVnrArea']].shape[0])

print("1 of Variables Missing :", total[(total['MasVnrType'].isnull()) & (total['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']].shape[0] + 

                                total[(total['MasVnrType'].notnull()) & (total['MasVnrArea'].isnull())][['MasVnrType', 'MasVnrArea']].shape[0])
total[total['MiscFeature'].isnull()]['MasVnrType']
index = (total['MasVnrType'].isnull()) & (total['MasVnrArea'].isnull())

total.loc[index, 'MasVnrType'] = 'None'

total.loc[index, 'MasVnrArea'] = 0
total[(total['MasVnrType'].isnull()) & (total['MasVnrArea'].notnull())]
A = total.loc[2610]

total.loc[2610, 'MasVnrType'] = total[(total['Neighborhood'] == A['Neighborhood']) & (total['MSZoning'] == A['MSZoning']) & (total['OverallQual'].isin([A['OverallQual']-1, A['OverallQual'], A['OverallQual']+1])) & (total['Exterior1st'] == A['Exterior1st'])]['MasVnrType'].mode()[0]
agg = pd.DataFrame(total.isnull().sum()).reset_index()

agg.columns = ['column', 'count']

agg['percent'] = 100 * agg['count'] / total.shape[0]

agg = agg[agg['percent'] != 0]

agg[agg['column'].isin(['MasVnrType', 'MasVnrArea'])]
agg[agg['column'].isin(['BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Utilities'])]
total[total['BsmtFullBath'].isnull()][['BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Utilities']]
total[total['Functional'].isnull()][['BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Utilities']]
total[total['BsmtFullBath'].isnull()][['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1','BsmtFullBath','BsmtHalfBath', 'BsmtUnfSF']]
index = total['BsmtFullBath'].isnull()

total.loc[index, 'BsmtFullBath'] = 0

total.loc[index, 'BsmtFullBath'] = 0

total.loc[index, 'BsmtUnfSF'] = 0
index = total[total['Functional'].isnull()].index

for i in index:

    A = total.loc[i]

    agg = total[(total['Neighborhood'] == A['Neighborhood']) & (total['OverallQual'].isin([A['OverallQual']-1,A['OverallQual'], A['OverallQual']+1]))]

    total.loc[i, 'Functional'] = agg['Functional'].mode()[0]

    print("Fill {} Missing Values:".format(i), total.loc[i, 'Functional'])
total.loc[total['Utilities'].isnull()][['Electrical', 'Heating']]
index = total[total['Utilities'].isnull()].index

for i in index:

    A = total.loc[i]

    agg = total[(total['Electrical'] == A['Electrical']) & (total['Heating'] == A['Heating']) & (total['Neighborhood'] == A['Neighborhood']) & (total['OverallQual'].isin([A['OverallQual']-1,A['OverallQual'], A['OverallQual']+1]))]

    try:

        total.loc[i, 'Utilities'] = agg['Utilities'].mode()[0]

        print("Fill {} Missing Values:".format(i), total.loc[i, 'Utilities'])

    except:

        try:

            agg = total[(total['Electrical'] == A['Electrical']) & (total['Heating'] == A['Heating']) & (total['OverallQual'].isin([A['OverallQual']-1,A['OverallQual'], A['OverallQual']+1]))]

            total.loc[i, 'Utilities'] = agg['Utilities'].mode()[0]

            print("Fill {} Missing Values:".format(i), total.loc[i, 'Utilities'])

        except:

            print("No Imputation {}:".format(i))
total[total['GarageArea'].isnull()][['GarageArea', 'GarageCars', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageCond', 'GarageYrBlt']]
total.loc[total['GarageArea'].isnull(), 'GarageArea'] = 0

total.loc[total['GarageCars'].isnull(), 'GarageCars'] = 0
index = total['TotalBsmtSF'].isnull()

total[index][['BsmtQual', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'PoolArea', 'PoolQC', 'MiscFeature', 'Alley', 'Fireplaces', 'FireplaceQu']]
total.loc[index, 'TotalBsmtSF'] = 0

total.loc[index, 'BsmtFinSF1'] = 0

total.loc[index, 'BsmtFinSF2'] = 0

total.loc[index, 'PoolQC'] = 'None'

total.loc[index, 'FireplaceQu'] = 'None'
total['PoolQC'] = total['PoolQC'].fillna('None')

total['MiscFeature'] = total['MiscFeature'].fillna('None')

total['Alley'] = total['Alley'].fillna('None')

total['Fence'] = total['Fence'].fillna('None')
total[(total['FireplaceQu'].isnull()) & (total['Fireplaces'] != 0)]
total['FireplaceQu'] = total['FireplaceQu'].fillna('None')
total[total['BsmtHalfBath'].isnull()][['BsmtQual', 'BsmtFullBath', 'BsmtHalfBath']]
total.loc[total['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0 
for col in ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:

    A = total.loc[total[total[col].isnull()].index[0]]

    B = total[(total['Neighborhood'] == A['Neighborhood']) & (total['OverallQual'].isin([A['OverallQual']-1,A['OverallQual'], A['OverallQual']+1]))][col].mode()[0]

    total.loc[total[total[col].isnull()].index[0], col] = B

    print("Fill {} Missing Values:".format(A.Id - 1), B)
total["LotFrontage"] = total.groupby(["Neighborhood", "OverallQual"])["LotFrontage"].transform(lambda x: x.fillna(x.median()))

total["LotFrontage"] = total.groupby(["Neighborhood"])["LotFrontage"].transform(lambda x: x.fillna(x.median()))
total[total["MSZoning"].isnull()]['Neighborhood']
total[total['MSZoning'].isnull()].index.values
try:

    total["MSZoning"] = total.groupby(["Neighborhood", "OverallQual"])["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))

except:

    try:

        total["MSZoning"] = total.groupby(["Neighborhood"])["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))

    except:

        print("No Imputation {}:".format(total[total['MSZoning'].isnull()].index.values))
agg = pd.DataFrame(total.isnull().sum()).reset_index()

agg.columns = ['column', 'count']

agg['percent'] = 100 * agg['count'] / total.shape[0]

agg = agg[agg['percent'] != 0]

agg
categorical_features = total.select_dtypes(include = ["object"]).columns

for col in categorical_features: 

    total[col], _ = pd.factorize(total[col])
df_train = total[total['type'] == 0].reset_index(drop=True)

df_test = total[total['type'] == 1].reset_index(drop=True)

del df_train['type']

del df_test['type']
target = train['SalePrice']

df_train['SalePrice'] = target
columns = [c for c in df_train.columns.tolist() if c not in ['Id', 'SalePrice']]
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



clf = tree.DecisionTreeRegressor(max_depth = 10)

clf = clf.fit(df_train[columns], target)
import seaborn as sns

# plot the sorted dataframe

importance = pd.DataFrame()

importance['Feature'] = columns 

importance['Importance'] = clf.feature_importances_

importance = importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)

importance = importance[0:10]



(ggplot(data = importance) 

 + geom_bar(aes(x='Feature', y='Importance'), fill = '#49beb7', stat='identity', color='black')

 + scale_x_discrete(limits=importance['Feature'].values) # sorting columns 

 + theme_light()  

 + labs(title = 'Tree Importance Graph : Gini Importance',

         x = '',

         y = 'GINI Importance')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
from sklearn.tree import export_graphviz

clf = tree.DecisionTreeRegressor(max_depth = 3)

clf = clf.fit(df_train[columns], target)

export_graphviz(clf, out_file='tree_limited.dot', 

                feature_names = columns, proportion = False, filled = True)
!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
from IPython.display import Image

Image(filename = 'tree_limited.png')
(ggplot(data = df_train) 

 + geom_point(aes(x='GrLivArea', y='SalePrice'), stat='identity', color='black', size=0.1)

 + geom_smooth(aes(x='GrLivArea', y='SalePrice'), method='lm', color='#49beb7')

 + facet_wrap('OverallQual')

 + theme_light()  

 + labs(title = 'Line Graph of GrLivArea',

         x = 'GrLivArea',

         y = 'SalePrice')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
(ggplot(data = df_train) 

 + geom_point(aes(x='2ndFlrSF', y='SalePrice'), stat='identity', color='black', size=0.1)

 + geom_smooth(aes(x='2ndFlrSF', y='SalePrice'), color='#49beb7')

 + facet_wrap('OverallQual')

 + theme_light()  

 + labs(title = 'Line Graph of 2ndFlrSF',

         x = '2ndFlrSF',

         y = 'SalePrice')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
(ggplot(data = df_train) 

 + geom_point(aes(x='TotalBsmtSF', y='SalePrice'), stat='identity', color='black', size=0.1)

 + geom_smooth(aes(x='TotalBsmtSF', y='SalePrice'), method='lm', color='#49beb7')

 + facet_wrap('OverallQual')

 + theme_light()  

 + labs(title = 'Line Graph of TotalBsmtSF',

         x = 'TotalBsmtSF',

         y = 'SalePrice')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
clf = tree.DecisionTreeRegressor(max_depth = 10)

clf = clf.fit(df_train[columns], target)

oof = clf.predict(df_train[columns])

residual = oof - target
residualDF = pd.DataFrame()

residualDF = pd.concat([residualDF, df_train[columns]], axis=1)

residualDF['residual'] = residual
import scipy as sp



cor_abs = abs(residualDF.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='residual').index # price과 correlation이 높은 column 10개 뽑기(내림차순)



# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(residualDF[cor_cols].values))[0] # 10 x 10



plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
df_train['OverallCond'] = df_train['OverallCond'].astype(str)

(ggplot(data = df_train) 

 + geom_boxplot(aes(x='OverallCond', y='SalePrice'), color='black', fill='#49beb7')

 # + geom_smooth(aes(x='TotalBsmtSF', y='SalePrice'), method='lm', color='#49beb7')

 + facet_wrap('OverallQual')

 + theme_light()  

 + labs(title = 'Boxplot of OverallCond',

         x = 'OverallCond',

         y = 'SalePrice')

 + theme(axis_text_x = element_text(angle=80),

         figure_size=(10,6))

)
import eli5

import shap
# Explain model predictions using shap library:

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(df_test[columns])

shap.summary_plot(shap_values, df_test[columns])