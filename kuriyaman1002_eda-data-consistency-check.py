!ls ../input
input_dir = '../input/'
# workig_dir = '../working/'
# output_dir = '../output/'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
train.shape
train.head()
# Split train data
train_Id = train['Id'] # individual ID
train_idhogar = train['idhogar'] # household ID
train_y = train['Target'] # Target value
train_X = train.drop(['Id', 'Target', 'idhogar'], axis=1) # features
print('We predict these levels')
print(train_y.unique())
print('Num of rows : {}'.format(train.shape[0]))
print('unique ID : {}'.format(train_Id.unique().size))
print('unique idhogar : {}'.format(train_idhogar.unique().size))
train_y_v_c = pd.concat([train_y.value_counts(), (train_y.value_counts() / train_y.shape)], axis=1)
train_y_v_c.columns = ['value_counts', 'frac']

print('Distribution of individual poverty level')
print()
print(train_y_v_c)

print('------')
print()
print('Distribution of household poverty level')
train_y_hh = train['Target'][train['parentesco1']==1]
train_y_hh_c_v = pd.concat([train_y_hh.value_counts(), (train_y_hh.value_counts() / train_y_hh.shape)], axis=1)
train_y_hh_c_v.columns = ['value_counts', 'frac']
print(train_y_hh_c_v)
print('This competition evaluates records with parentesco1==1')
print('**Number of unique idhogar is different from number of parentesco1==1 records**')
print('unique idhogar : {}'.format(train_idhogar.unique().shape[0]))
print('sum of parentesco1 : {}'.format(train.query('parentesco1==1').shape[0]))
print('I assume we have to remove records whose households does not contain head of household in train data')
print('Some households have inconsistent poverty levels in a household.')
train_hh = train[['idhogar', 'Target']].drop_duplicates()
train_hh_c_v = train_hh['idhogar'].value_counts()
print(train_hh_c_v[:5])
print()
print('Number of household with inconsistent Taget values : {}'.format(train_hh_c_v[train_hh_c_v>1].size))
# check the records with idhogar=='5c6f32bbc'
train[['idhogar', 'Id', 'Target']][train['idhogar']=='5c6f32bbc']
print(train_X.dtypes.value_counts())
print()
print('3 columns are "object" type')
train_X.dtypes[train_X.dtypes=='object']
print('dependency unique valuse')
print(train_X['dependency'].unique())
print()
print('edjefa unique valuse')
print(train_X['edjefa'].unique())
print()
print('edjefe unique valuse')
print(train_X['edjefe'].unique())
print('5 colmuns have NaN')
print()
is_null_train_X = train_X.isnull().any()
print('Number of NaN values')
print(train_X.isnull().loc[:,is_null_train_X==True].sum())
print('v2a1 means Monthly rent payment. \n NaN seems to be unknown.')
print('v18q means number of tablets household owns. \n NaN has to be replaced as 0.')
print('rez_esc means Years behind in school. \n I dont know how to replace them...')
train_X.head(10)
train_hh_hgtotal = train[train['parentesco1']==1][['idhogar', 'hogar_total']]
train_hh_hgtotal.index = train_hh_hgtotal['idhogar']
train_hh_hgtotal = train_hh_hgtotal.drop('idhogar', axis=1).sort_index()
train_hh_cnt = train.groupby('idhogar')['idhogar'].count().sort_index()
train_hh_check = pd.concat([train_hh_hgtotal, train_hh_cnt], axis=1)
(train_hh_check['idhogar'] != train_hh_check['hogar_total']).sum()
train.index = train['Id'].values
area_list = ['area1', 'area2']
train_areas = train[area_list]
train_area = train_areas.idxmax(1)
train_area.name = 'area'

region_list = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
train_regions = train[region_list]
train_region = train_regions.idxmax(1)
train_region.name = 'region'

train_geo = pd.concat([train_area, train_region], axis=1)
train_geo.query('area=="area1"').drop_duplicates()
train_geo.query('area=="area2"').drop_duplicates()
instlevel_list = ['instlevel1', 'instlevel2', 'instlevel3',
                  'instlevel4', 'instlevel5', 'instlevel6',
                  'instlevel7', 'instlevel8', 'instlevel9']
train_instlevels = train[instlevel_list]
(train_instlevels.sum(axis=1) != 1).sum()
train_hh = train[train['parentesco1']==1]

tipovivi_list = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
train_tipovivis = train_hh[tipovivi_list]
train_tipovivi = train_tipovivis.idxmax(axis=1)
train_tipovivi.name = 'tipovivi'

train_rent = pd.concat([train_hh['v2a1'], train_tipovivi], axis=1)
train_rent.head()
train_rent[train_rent['v2a1'].isnull()]['tipovivi'].drop_duplicates()
train_rent[train_rent['v2a1'].notnull()]['tipovivi'].drop_duplicates()
train_rent['tipovivi'].value_counts()
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].boxplot(train_rent[train_rent['tipovivi']=='tipovivi2']['v2a1'].values)
ax[0].set_title('tipovivi2: paying in installments')
ax[1].boxplot(train_rent[train_rent['tipovivi']=='tipovivi3']['v2a1'].values)
ax[1].set_title('tipovivi3: rented')
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
ax[0].hist(train_rent[train_rent['tipovivi']=='tipovivi2']['v2a1'].values, bins=25, orientation="horizontal")
ax[0].set_title('tipovivi2: paying in installments')
ax[1].hist(train_rent[train_rent['tipovivi']=='tipovivi3']['v2a1'].values, bins=25, orientation="horizontal")
ax[1].set_title('tipovivi3: rented')
