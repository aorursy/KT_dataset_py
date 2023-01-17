import numpy as np

import pandas as pd
print('Loading data ...')

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

prop = pd.read_csv('../input/properties_2016.csv')



print('Binding to float32 to save memory')

for c, dtype in zip(prop.columns, prop.dtypes):

	if dtype == np.float64:

		prop[c] = prop[c].astype(np.float32)

        

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

# I didn't clean the data because I am lazy

df_train.fillna(df_train.median(), inplace = True)

  

categorical_columns = [

    'airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid',

    'decktypeid', 'heatingorsystemtypeid', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',

    'propertylandusetypeid', 'regionidcounty', 'storytypeid', 'typeconstructiontypeid'

]

for c in categorical_columns:

    df_train[c] = df_train[c].astype(np.int32)



# One-hot encoding

df_train = pd.get_dummies(data=df_train, columns=categorical_columns)
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)

y_train = df_train['logerror'].values
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)
print(x_train.shape, y_train.shape)
x_train.dtypes