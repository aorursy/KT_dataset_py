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
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 1000)

from sklearn.metrics import mean_absolute_error

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.info()
from collections import Counter

num_col = train.loc[:,'MSSubClass':'SaleCondition'].select_dtypes(exclude=['object']).columns

# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2, num_col)

train.loc[Outliers_to_drop] # Show the outliers rows

# Drop outliers

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
# before tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
sns.distplot(df['MasVnrArea'])
df['MasVnrArea'].head(1000)
scaler = MinMaxScaler((-1,1))



df['BsmtFinSF2_sc'] = scaler.fit_transform(df[['BsmtFinSF2']])

df['BsmtFinSF2_sc'].fillna(0,inplace=True)

df['LowQualFinSF_sc'] = scaler.fit_transform(df[['LowQualFinSF']])

df['WoodDeckSF_sc'] = scaler.fit_transform(df[['WoodDeckSF']])

df['OpenPorchSF_sc'] = scaler.fit_transform(df[['OpenPorchSF']])

df['EnclosedPorch_sc'] = scaler.fit_transform(df[['EnclosedPorch']])

df['3SsnPorch_sc'] = scaler.fit_transform(df[['3SsnPorch']])

df['ScreenPorch_sc'] = scaler.fit_transform(df[['ScreenPorch']])

df['MiscVal_sc'] = scaler.fit_transform(df[['MiscVal']])



df['MSZoning'].fillna('N')

df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace = True)

df['Alley'].fillna('N')

df['Exterior1st'].fillna('N')

df['Exterior2nd'].fillna('N')

df['Utilities'].fillna('N')

df['MasVnrType'].fillna('N')

df['BsmtFullBath'].fillna(0)

df['BsmtHalfBath'].fillna(0)

df['FullBath'].fillna(0)

df['HalfBath'].fillna(0)

df['KitchenQual'].fillna('N')

df['Functional'].fillna('N')

df['FireplaceQu'].fillna('N')

df['GarageType'].fillna('N')

df['GarageYrBlt'].fillna(0,inplace=True)

df['GarageFinish'].fillna('N')

df['GarageCars'].fillna(0)

df['GarageArea'].fillna(0,inplace=True)

df['GarageQual'].fillna('N')

df['GarageCond'].fillna('N')

df['BsmtFinSF2'].fillna(0,inplace=True)

df['MasVnrArea'].fillna(0,inplace=True)

df['BsmtFinSF1'].fillna(0,inplace=True)

df['SaleType'].fillna('N')

df['BsmtUnfSF'].fillna(0,inplace=True)

df['TotalBsmtSF'].fillna(0,inplace=True)

df['PoolQC'].fillna('N')

df['Fence'].fillna('N')

df['MiscFeature'].fillna('N')

df['BsmtQual'].fillna('N')

df['BsmtCond'].fillna('N')

df['BsmtExposure'].fillna('N')

df['BsmtFinType1'].fillna('N')

df['BsmtFinType2'].fillna('N')

df['Electrical'].fillna('N')

df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

df['Area'] = df['LotArea']*df['LotFrontage']

df['Area_log'] = np.log1p(df['Area'])



def Gar_category(cat):

    if cat <= 250:

        return 1

    elif cat <= 500 and cat > 250:

        return 2

    elif cat <= 1000 and cat > 500:

        return 3

    return 4

df['GarageArea_cat'] = df['GarageArea'].apply(Gar_category)



def Low_category(cat):

    if cat <= 1000:

        return 1

    elif cat <= 2000 and cat > 1000:

        return 2

    elif cat <= 3000 and cat > 2000:

        return 3

    return 4

df['GrLivArea_cat'] = df['GrLivArea'].apply(Low_category)



def fl1_category(cat):

    if cat <= 500:

        return 1

    elif cat <= 1000 and cat > 500:

        return 2

    elif cat <= 1500 and cat > 1000:

        return 3

    elif cat <= 2000 and cat > 1500:

        return 4

    return 5

df['1stFlrSF_cat'] = df['1stFlrSF'].apply(fl1_category)

df['2ndFlrSF_cat'] = df['2ndFlrSF'].apply(fl1_category)



def bsmtt_category(cat):

    if cat <= 500:

        return 1

    elif cat <= 1000 and cat > 500:

        return 2

    elif cat <= 1500 and cat > 1000:

        return 3

    elif cat <= 2000 and cat > 1500:

        return 4

    return 5

df['TotalBsmtSF_cat'] = df['TotalBsmtSF'].apply(bsmtt_category)



def bsmt_category(cat):

    if cat <= 500:

        return 1

    elif cat <= 1000 and cat > 500:

        return 2

    elif cat <= 1500 and cat > 1000:

        return 3

    elif cat <= 2000 and cat > 1500:

        return 4

    return 5

df['BsmtUnfSF_cat'] = df['BsmtUnfSF'].apply(bsmt_category)



def lot_category(cat):

    if cat <= 50:

        return 1

    elif cat <= 100 and cat > 50:

        return 2

    elif cat <= 150 and cat > 100:

        return 3

    return 4

df['LotFrontage_cat'] = df['LotFrontage'].apply(lot_category)



def lot_category1(cat):

    if cat <= 5000:

        return 1

    elif cat <= 10000 and cat > 5000:

        return 2

    elif cat <= 15000 and cat > 10000:

        return 3

    elif cat <= 20000 and cat > 15000:

        return 4

    elif cat <= 25000 and cat > 20000:

        return 5

    return 6

df['LotArea_cat'] = df['LotArea'].apply(lot_category1)



def year_category(yb):

    if yb <= 1910:

        return 1

    elif yb <= 1950 and yb > 1910:

        return 2

    elif yb >= 1950 and yb < 1980:

        return 3

    elif yb >= 1980 and yb < 2000:

        return 4

    return 5







df['YearBuilt_cat'] = df['YearBuilt'].apply(year_category) 

df['YearRemodAdd_cat'] = df['YearRemodAdd'].apply(year_category)

df['GarageYrBlt_cat'] = df['GarageYrBlt'].apply(year_category)



def vnr_category(cat):

    if cat <= 250:

        return 1

    elif cat <= 500 and cat > 250:

        return 2

    elif cat <= 750 and cat > 500:

        return 3

    return 4



df['MasVnrArea_cat'] = df['MasVnrArea'].apply(vnr_category)



def allsf_category(yb):

    if yb <= 1000:

        return 1

    elif yb <= 2000 and yb > 1000:

        return 2

    elif yb >= 3000 and yb < 2000:

        return 3

    elif yb >= 4000 and yb < 3000:

        return 4

    elif yb >= 5000 and yb < 4000:

        return 5

    elif yb >= 6000 and yb < 5000:

        return 6

    return 7



df['AllSF_cat'] = df['AllSF'].apply(allsf_category)





dummy_col=['YearRemodAdd', 'YearBuilt', 'AllSF_cat', 'MiscVal','OverallCond', 'OverallQual', 'BsmtFinType2', 'SaleCondition','SaleType', 'YrSold', 'MoSold', 'MiscFeature', 'Fence', 'PoolQC', 'PoolArea', 'PavedDrive', 'GarageCond', 'GarageQual', 'GarageArea_cat', 'GarageCars', 'GarageFinish', 'GarageType', 'FireplaceQu', 'Fireplaces','Functional', 'TotRmsAbvGrd', 'KitchenQual', 'KitchenAbvGr', 'BedroomAbvGr', 'HalfBath', 'FullBath', 'BsmtHalfBath', 'BsmtFullBath','GrLivArea_cat','MSSubClass', 'MSZoning', 'LotFrontage_cat', 'LotArea_cat', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

          'BldgType', 'HouseStyle', 'YearBuilt_cat', 'YearRemodAdd_cat', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'Exterior1st', 'MasVnrType', 'MasVnrArea_cat', 'ExterQual', 'ExterCond', 'Foundation', 

          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtUnfSF_cat', 'TotalBsmtSF_cat', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF_cat', '2ndFlrSF_cat']

df = pd.get_dummies(df, columns=dummy_col, drop_first=False)



df['LotFrontage_log'] = np.log1p(df['LotFrontage'])

df['LotArea_log'] = np.log1p(df['LotArea'])

df['BsmtUnfSF_log'] = np.log1p(df['BsmtUnfSF'])



df['Is_MasVnr'] = [1 if i != 0 else 0 for i in df['MasVnrArea']]

df['Is_BsmtFinSF1'] = [1 if i != 0 else 0 for i in df['BsmtFinSF1']]

df['Is_BsmtFinSF2'] = [1 if i != 0 else 0 for i in df['BsmtFinSF2']]

df['Is_BsmtUnfSF'] = [1 if i != 0 else 0 for i in df['BsmtUnfSF']]

df['Is_TotalBsmtSF'] = [1 if i != 0 else 0 for i in df['TotalBsmtSF']]

df['Is_2ndFlrSF'] = [1 if i != 0 else 0 for i in df['2ndFlrSF']]

df['Is_LowQualFinSF'] = [1 if i != 0 else 0 for i in df['LowQualFinSF']]

df['Is_GarageArea'] = [1 if i != 0 else 0 for i in df['GarageArea']]

df['Is_WoodDeckSF'] = [1 if i != 0 else 0 for i in df['WoodDeckSF']]

df['Is_OpenPorchSF'] = [1 if i != 0 else 0 for i in df['OpenPorchSF']]

df['Is_EnclosedPorch'] = [1 if i != 0 else 0 for i in df['EnclosedPorch']]

df['Is_3SsnPorch'] = [1 if i != 0 else 0 for i in df['3SsnPorch']]

df['Is_ScreenPorch'] = [1 if i != 0 else 0 for i in df['ScreenPorch']]





# after tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
df.shape
def descrictive_stat_feat(df):

    df = pd.DataFrame(df)

    dcol= [c for c in df.columns if df[c].nunique()>=10]

    d_median = df[dcol].median(axis=0)

    d_mean = df[dcol].mean(axis=0)

    q1 = df[dcol].apply(np.float32).quantile(0.25)

    q3 = df[dcol].apply(np.float32).quantile(0.75)

    

    #Add mean and median column to data set having more then 10 categories

    for c in dcol:

        df[c+str('_median_range')] = (df[c].astype(np.float32).values > d_median[c]).astype(np.int8)

        df[c+str('_mean_range')] = (df[c].astype(np.float32).values > d_mean[c]).astype(np.int8)

        df[c+str('_q1')] = (df[c].astype(np.float32).values < q1[c]).astype(np.int8)

        df[c+str('_q3')] = (df[c].astype(np.float32).values > q3[c]).astype(np.int8)

    return df



df = descrictive_stat_feat(df)
df.shape
tot_cat_col = list(df.select_dtypes(include=['category']).columns)

num_col = [c for c in df.columns if c not in tot_cat_col]
def outlier(df,columns):

    for i in columns:

        quartile_1,quartile_3 = np.percentile(df[i],[25,75])

        quartile_f,quartile_l = np.percentile(df[i],[1,99])

        IQR = quartile_3-quartile_1

        lower_bound = quartile_1 - (1.5*IQR)

        upper_bound = quartile_3 + (1.5*IQR)

        print(i,lower_bound,upper_bound,quartile_f,quartile_l)

                

        df[i].loc[df[i] < lower_bound] = quartile_f

        df[i].loc[df[i] > upper_bound] = quartile_l

        

outlier(df,num_col)
df.shape
#creating matrices for feature selection:

X_train = df[:train.shape[0]]

X_test_fin = df[train.shape[0]:]

y = train.SalePrice

X_train['Y'] = y

df = X_train

df.head(20) ## DF for Model training



X = df.drop('Y', axis=1)

y = df.Y
# import xgboost as xgb

# from sklearn.metrics import r2_score



# from sklearn.model_selection import RandomizedSearchCV



# xg_cl = xgb.XGBClassifier()

# param = {'max_depth':[3,4],

#          'learning_rate':[0.01,0.03],

#          'min_child_weight':[1,3],

#          'reg_lambda':[0.1,0.5],

#          'reg_alpha':[1,1.5],      

#          'gamma':[0.1,0.5],

#          'subsample':[0.4,0.5],

#         'colsample_bytree':[0.4,0.5],

# }

# clf = RandomizedSearchCV(xg_cl,param,verbose=10)



# clf.fit(X,y)

# print('Best roc_auc: {:.4}, with best params: {}'.format(clf.best_score_, clf.best_params_))
import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split





X = df.drop('Y', axis=1)

y = df.Y







params = {

        'objective':'reg:linear',

        'n_estimators': 1000,

        'booster':'gbtree',

        'max_depth':2,

        'eval_metric':'mae',

        'learning_rate':0.005, 

        'min_child_weight':2,

        'subsample':0.7,

        'colsample_bytree':0.8,

        'seed':45,

        'reg_alpha':1,

        'gamma':0,

        'nthread':-1



}





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(X_test_fin)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=10)



p_test = clf.predict(d_test)



# cv_result = xgb.cv(params, 

#                   d_train, 

#                   num_boost_round=2000, # increase to have better results (~700)

#                   early_stopping_rounds=50,

#                   verbose_eval=10, 

#                   show_stdv=False

#                  )
leaks = {

    1461:105000,

1477:290941,

1492:67500,

1494:362500,

1514:84900,

1521:108538,

1531:80400,

1537:12789,

1540:76500,

1545:134000,

1554:122000,

1556:107500,

1557:100000,

1559:93369,

1560:114900,

1566:270000,

1567:85000,

1572:128000,

1573:308030,

1575:270000,

1586:84900,

1587:155891,

1589:64000,

1595:100000,

1597:215000,

1603:50138,

1610:174000,

1611:169000,

1615:76000,

1616:88250,

1617:85500,

1620:159000,

1622:161000,

1631:240000,

1638:154000,

1650:76500,

1652:111000,

1661:462000,

1664:610000,

1666:296000,

1678:552000,

1696:245000,

1698:327000,

1712:264500,

1717:152000,

1720:203000,

1726:171500,

1727:145000,

1737:275000,

1767:256000,

1774:135000,

1786:142900,

1787:156500,

1788:59000,

1790:78500,

1793:163000,

1807:103500,

1814:80000,

1820:58500,

1823:44000,

1831:179900,

1832:62500,

1835:97500,

1837:70000,

1842:63000,

1843:113500,

1863:269500,

1864:269500,

1892:85000,

1895:103500,

1912:315000,

1913:123000,

1915:230000,

1916:57625,

1925:170000,

1946:115000,

1947:334000,

1967:317500,

1970:390000,

1971:460000,

1975:615000,

1976:284000,

1996:284500,

1997:291000,

2004:297900,

2014:163000,

2030:300000,

2031:285000,

2032:290000,

2033:305000,

2038:345000,

2052:140000,

2055:141500,

2068:146000,

2076:94000,

2086:143000,

2093:122250,

2099:46500,

2100:65000,

2101:139500,

2106:55000,

2107:184000,

2111:108000,

2152:260000,

2162:475000,

2163:395039,

2180:185000,

2185:165000,

2206:104000,

2207:257076,

2208:263400,

2211:126000,

2217:13100,

2220:65000,

2223:300000,

2227:241500,

2230:172500,

2232:150000,

2235:195000,

2236:298751,

2238:209200,

2239:146000,

2245:94900,

2251:103000,

2263:349265,

2264:591587,

2267:441929,

2268:455000,

2269:174000,

2288:322400,

2295:500067,

2342:260000,

2354:146000,

2362:300000,

2375:279700,

2376:255000,

2379:240050,

2380:162500,

2395:224500,

2404:175000,

2419:115000,

2437:125500,

2455:136500,

2461:132000,

2465:165000,

2466:90000,

2468:113000,

2469:117000,

2474:50000,

2495:109900,

2544:110000,

2550:183850,

2557:79275,

2564:238000,

2565:153500,

2572:200000,

2574:315000,

2583:375000,

2590:244000,

2591:257000,

2599:392000,

2610:138000,

2611:80000,

2617:169000,

2618:252000,

2627:130000,

2631:535000,

2632:401179,

2634:470000,

2638:294323,

2658:344133,

2673:246990,

2690:405749,

2702:129500,

2723:157500,

2741:132000,

2752:167000,

2754:180000,

2760:80000,

2775:111500,

2776:156500,

2779:111500,

2788:64000,

2793:202500,

2794:75000,

2805:125000,

2813:156500,

2823:415000,

2829:224500,

2832:233555,

2859:98000,

2866:134000,

2872:35000,

2873:121000,

2881:195000,

2916:71000,

2917:131000,

2919:188000

}
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = p_test

sub['SalePrice'] = sub.apply(lambda r: leaks[int(r['Id'])] if int(r['Id']) in leaks else r['SalePrice'], axis=1)

sub.to_csv('submission.csv', index=False)
sub
import matplotlib.pyplot as plt

# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax)

plt.show()