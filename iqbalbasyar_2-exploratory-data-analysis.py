import pandas as pd
import numpy as np
print(pd.__version__)
household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv")
household.head()
household.dtypes
household['purchase_time'] = pd.to_datetime(household['purchase_time'])
household.head()
date = pd.Series(['30-01-2020', '31-01-2020', '01-02-2020','02-02-2020'])
date
pd.to_datetime(date)
# Solution 1
pd.to_datetime(date, format="%d-%m-%Y")


# Solution 2
pd.to_datetime(date, dayfirst=True)
## Your code below

## -- Solution code
## Your code below


## -- Solution code
# Reference answer for Knowledge Check
household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv", index_col=1, parse_dates=['purchase_time'])
household.drop(['receipt_id', 'yearmonth', 'sub_category'], axis=1, inplace=True)
household['weekday'] = household['purchase_time'].dt.weekday_name
pd.crosstab(index=household['weekday'], columns='count')
household.dtypes
household['weekday'] = household['weekday'].astype('category', errors='raise')
household.dtypes
## Your code below


## -- Solution code
household.select_dtypes(exclude='object').head()
pd.concat([
    household.select_dtypes(exclude='object'),
    household.select_dtypes(include='object').apply(
        pd.Series.astype, dtype='category'
    )
], axis=1).dtypes
objectcols = household.select_dtypes(include='object')
household[objectcols.columns] = objectcols.apply(lambda x: x.astype('category'))
household.head()
household.dtypes
household = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/household.csv")
household.shape
## Your code below


## -- Solution code
household.sub_category.value_counts(sort=False, ascending=True)
## Your code below


## -- Solution code
pd.crosstab(index=household['sub_category'], columns="count")
pd.crosstab(index=household['sub_category'], columns="count", normalize='columns')
catego = pd.crosstab(index=household['sub_category'], columns="count")
catego / catego.sum()
pd.crosstab(index=household['sub_category'], columns=household['format'])
household.head()
pd.crosstab(index=household['sub_category'], 
            columns=household['format'], 
            margins=True)
## Your code below


## -- Solution code
pd.crosstab(index=household['sub_category'], 
            columns='mean', 
            values=household['unit_price'],
            aggfunc='mean')
pd.crosstab(index=household['sub_category'],
           columns=household['format'],
           values=household['unit_price'],
           aggfunc='median', margins=True)
## Your code below


## -- Solution code
pd.crosstab(index=household['yearmonth'], 
            columns=[household['format'], household['sub_category']], 
            values=household['unit_price'],
            aggfunc='median')
pd.pivot_table(
    data=household,
    index='yearmonth',
    columns=['format','sub_category'],
    values='unit_price',
    aggfunc='median'
)
pd.pivot_table(
    data=household, 
    index='sub_category',
    columns='yearmonth',
    values='quantity'
)
## Your code below


## -- Solution code
import math
x=[i for i in range(32000000, 32000005)]
x.insert(2,32030785)
x
import math
x=[i for i in range(32000000, 32000005)]
x.insert(2,32030785)

household2 = household.head(6).copy()
household2 = household2.reindex(x)
household2 = pd.concat([household2, household.head(14)])
household2.loc[31885876, "weekday"] = math.nan
household2.iloc[2:8,]
household2['weekday'].isna()
household2[household2['weekday'].isna()]
## Your code below


## -- Solution code
household2.isna().sum()
household2.dropna(thresh=6).head()
print(household2.shape)
print(household2.drop_duplicates(keep="first").shape)
## Your code below


## -- Solution code
household3 = household2.copy()
household3.head()
# convert NA categories to 'Missing'
household3[['category', 'format','discount']] = household3[['category', 'format','discount']].fillna('Missing')

# convert NA unit_price to 0
household3.unit_price = household3.unit_price.fillna(0)

# convert NA purchase_time with 'bfill'
household3.purchase_time = household3.fillna(method='bfill')
household3.purchase_time = pd.to_datetime(household3.purchase_time)

# convert NA weekday
household3.weekday = household3.purchase_time.dt.weekday_name

# convert NA quantity with -1
household3.quantity = household3.quantity.replace(np.nan, -1)

household3.head()