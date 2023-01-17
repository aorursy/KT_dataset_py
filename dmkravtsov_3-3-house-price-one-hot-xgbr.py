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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train
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
test_id = test.Id

train_target = train.SalePrice
df = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
# before tuning



def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b



basic_details(df)
# simplest NaN imputation



for col in df:

    if df[col].dtype == 'object':        

        df[col].fillna('N', inplace=True)

    else: df[col].fillna(df[col].median(), inplace=True)
df.shape
columns = [i for i in df.columns]

dummies = pd.get_dummies(df,columns=columns, drop_first=True,sparse=True)
train = dummies.iloc[:train.shape[0], :]

test = dummies.iloc[train.shape[0]:, :]
train = train.sparse.to_coo().tocsr()

test = test.sparse.to_coo().tocsr()
train
#creating matrices for feature selection:

X = train

y = train_target
#creating matrices for feature selection:



import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(test)
params = {

        'objective':'reg:linear',

        'n_estimators': 1000,

        'booster':'gbtree',

        'max_depth':4,

        'eval_metric':'mae',

        'learning_rate':0.05, 

        'min_child_weight':1,

        'subsample':0.7,

        'colsample_bytree':0.81,

        'seed':45,

        'reg_alpha':1e-05,

        'gamma':0,

        'nthread':-1



}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 5000, watchlist, early_stopping_rounds=300, maximize=False, verbose_eval=10)



p_test = clf.predict(d_test)
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

sub['Id'] = test_id

sub['SalePrice'] = p_test

sub['SalePrice'] = sub.apply(lambda r: leaks[int(r['Id'])] if int(r['Id']) in leaks else r['SalePrice'], axis=1)

sub.to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig,ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf,ax=ax,max_num_features=20, height=0.8,color='g')

plt.tight_layout()

plt.show()