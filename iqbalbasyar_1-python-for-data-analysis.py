activity = 'programming'

print(activity)
# activity == Activity
activity == activity
## Your code below:


import pandas as pd
print(pd.__version__)
rice = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/rice.csv", index_col=0)
rice.head()
## Your code below


## -- Solution code
## Your code below


## -- Solution code
print(rice.dtypes)
## Your code below


## -- Solution code
employees = pd.DataFrame({
    'name': ['Anita', 'Brian'],
    'age': [34, 29],
    'joined': [pd.Timestamp('20190410'), pd.Timestamp('20171128')],
    'degree': [True, False],
    'hourlyrate': [35.5, 29],
    'division': ['HR', 'Product']
})
employees.dtypes
employees
## Your code below


## -- Solution code
## Your code below


## -- Solution code
employees.describe()
## Your code below


## -- Solution code
print(rice.shape)
print(rice.size)
## Your code below


## -- Solution code
rice.axes[1]
rice.dtypes
rice['purchase_time'] = rice['purchase_time'].astype('datetime64')
rice[['category', 'sub_category', 'format']] = rice[['category', 'sub_category', 'format']].astype('category')
rice.dtypes
rice.select_dtypes(include='object').head()
rice.select_dtypes(include=['category']).describe()
## Your code below


## -- Solution code
rice.drop(3).head()
rice.drop(['unit_price', 'purchase_time', 'receipt_id'], axis=1).head()
rice[0:4]
rice.iloc[0:5, :]
## Your code below


## -- Solution code
rice = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/rice.csv", index_col=1)
rice = rice.drop('Unnamed: 0', axis=1)
rice.head()
rice.loc[[9643416, 5735850], :]
clients = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/companies.csv", index_col=1)
clients.head()
## Your code below


## -- Solution code
rice = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/rice.csv", index_col=1)
rice = rice.drop('Unnamed: 0', axis=1)
rice[rice.discount != 0].head()
## Your code below


## -- Solution code
rice = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/rice.csv", index_col=1)
rice = rice.drop('Unnamed: 0', axis=1)

rice.head()
rice_july = rice
rice_july['discount'] = 15
rice_july.head()
rice = pd.read_csv("/kaggle/input/algoritma-academy-data-analysis/rice.csv", index_col=1)
rice = rice.drop('Unnamed: 0', axis=1)

rice_july = rice.copy()
rice_july['discount'] = 15
rice_july.head()
rice.head()