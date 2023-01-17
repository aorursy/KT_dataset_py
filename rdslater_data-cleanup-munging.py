

import numpy as np

import pandas as pd

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

n=train.shape[0]

test.price_doc=np.nan

ids=test['id']

target=train.price_doc

train=train.append(test)





a=train.sub_area.unique()

for i in a:

    if train[(train['floor'].isnull()) & (train['sub_area']==i)].shape[0]>0:

        train.loc[(train['floor'].isnull()) & (train['sub_area']==i),'floor']=train.loc[(train['floor'].notnull()) & (train['sub_area']==i),'floor'].mode().values[0]

print('floors done',train['product_type'].value_counts())

#fill in nans for max_floor

for i in a:

    if train[(train['max_floor'].isnull()) & (train['sub_area']==i)].shape[0]>0:

        if i=='Poselenie Shhapovskoe':

            train.loc[(train['max_floor'].isnull()) & (train['sub_area']==i),'max_floor']=train.loc[(train['max_floor'].notnull()) & (train['sub_area']==i),'max_floor'].median()

        else:

            train.loc[(train['max_floor'].isnull()) & (train['sub_area']==i),'max_floor']=train.loc[(train['max_floor'].notnull()) & (train['sub_area']==i),'max_floor'].mode().values[0]



#materials



for i in a:

    if train[(train['material'].isnull()) & (train['sub_area']==i)].shape[0]>0:

        train.loc[(train['material'].isnull()) & (train['sub_area']==i),'material']=train.loc[(train['material'].notnull()) & (train['sub_area']==i),'material'].mode().values[0]



#build year

for i in a:

    if train[(train['build_year'].isnull()) & (train['sub_area']==i)].shape[0]>0:

        if i=='Poselenie Voronovskoe':

            train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=2014

        elif i=='Poselenie Shhapovskoe':

            train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=2011

        else:

            train.loc[(train['build_year'].isnull()) & (train['sub_area']==i),'build_year']=train.loc[(train['build_year'].notnull()) & (train['sub_area']==i),'build_year'].mode().values[0]



#state

for i in a:

    if train[(train['state'].isnull()) & (train['sub_area']==i)].shape[0]>0:

        if (i=='Poselenie Klenovskoe' or i=='Poselenie Kievskij'):

            train.loc[(train['state'].isnull()) & (train['sub_area']==i),'state']=2

        else:

            train.loc[(train['state'].isnull()) & (train['sub_area']==i),'state']=train.loc[(train['state'].notnull()) & (train['sub_area']==i),'state'].mode().values[0]



#and now the rail stations

cols=['ID_railroad_station_walk','ID_railroad_station_avto','green_part_2000','metro_km_walk','metro_min_walk']

for j in cols:

    for i in a:

        if train[(train[j].isnull()) & (train['sub_area']==i)].shape[0]>0:

            train.loc[(train[j].isnull()) & (train['sub_area']==i),j]=train.loc[(train[j].notnull()) & (train['sub_area']==i),j].mode().values[0]

cols=['railroad_station_walk_km','railroad_station_walk_min']

for j in cols:

    for i in a:

        if train[(train[j].isnull()) & (train['sub_area']==i)].shape[0]>0:

            train.loc[(train[j].isnull()) & (train['sub_area']==i),j]=train.loc[(train[j].notnull()) & (train['sub_area']==i),j].median()





#doing things a bit differnt for rooms, base it off sq ft.

x=train.loc[(train.full_sq.notnull())&(train.num_room.notnull()),'full_sq']

y=train.loc[(train.full_sq.notnull())&(train.num_room.notnull()),'num_room']

rooms=np.polyfit(x,y,1)

train.loc[train['num_room'].isnull(),'num_room']=train.loc[train['num_room'].isnull(),'full_sq']*rooms[0]+rooms[1]



#kitchen space will be same

x=train.loc[(train.kitch_sq.notnull())&(train.kitch_sq.notnull()),'full_sq']

y=train.loc[(train.kitch_sq.notnull())&(train.kitch_sq.notnull()),'kitch_sq']

kitch=np.polyfit(x,y,1)

train.loc[train['kitch_sq'].isnull(),'kitch_sq']=train.loc[train['kitch_sq'].isnull(),'full_sq']*kitch[0]+kitch[1]



#and fix up the life-sq

x=train.loc[(train.full_sq.notnull())&(train.life_sq.notnull()),'full_sq']

y=train.loc[(train.full_sq.notnull())&(train.life_sq.notnull()),'life_sq']

life=np.polyfit(x,y,1)

train.loc[train['life_sq'].isnull(),'life_sq']=train.loc[train['life_sq'].isnull(),'full_sq']*life[0]+life[1]

#fix some of the off values

train.loc[(train['life_sq']>train['full_sq'])&(train['life_sq']>1000),'life_sq']=train.loc[(train['life_sq']>train['full_sq'])&(train['life_sq']>1000),'life_sq']/1000

train.loc[(train['life_sq']>train['full_sq'])&(train['life_sq']>106),'life_sq']=train.loc[(train['life_sq']>train['full_sq'])&(train['life_sq']>106),'life_sq']/100

print('fits done',train['product_type'].value_counts())



#city /region data

cols=['build_count_1921-1945','build_count_1946-1970','build_count_1971-1995','build_count_after_1995','build_count_before_1920',

'build_count_block','build_count_brick','build_count_foam','build_count_frame','build_count_mix','build_count_monolith',

'build_count_panel','build_count_slag','build_count_wood','cafe_avg_price_1000','cafe_avg_price_1500','cafe_avg_price_2000',

'cafe_avg_price_3000','cafe_avg_price_500','cafe_avg_price_5000','cafe_sum_1000_max_price_avg','cafe_sum_1000_min_price_avg',

'cafe_sum_1500_max_price_avg','cafe_sum_1500_min_price_avg','cafe_sum_2000_max_price_avg','cafe_sum_2000_min_price_avg',

'cafe_sum_3000_max_price_avg','cafe_sum_3000_min_price_avg','cafe_sum_5000_max_price_avg','cafe_sum_5000_min_price_avg',

'cafe_sum_500_max_price_avg','cafe_sum_500_min_price_avg','raion_build_count_with_builddate_info','raion_build_count_with_material_info','prom_part_5000']

for j in cols:

    train.loc[train[j].isnull(),j]=train.loc[train[j].notnull(),j].median()

            

print('city data done',train['product_type'].value_counts())



#hospital and preschoo

i='raion_popul'

j='hospital_beds_raion'

x=train.loc[(train[i].notnull())&(train[j].notnull()),i]

y=train.loc[(train[i].notnull())&(train[j].notnull()),j]

fit=np.polyfit(x,y,1)

train.loc[train[j].isnull(),j]=train.loc[train[j].isnull(),i]*fit[0]+fit[1]



i='children_preschool'

j='preschool_quota'

x=train.loc[(train[i].notnull())&(train[j].notnull()),i]

y=train.loc[(train[i].notnull())&(train[j].notnull()),j]

fit=np.polyfit(x,y,1)

train.loc[train[j].isnull(),j]=train.loc[train[j].isnull(),i]*fit[0]+fit[1]



i='children_school'

j='school_quota'

x=train.loc[(train[i].notnull())&(train[j].notnull()),i]

y=train.loc[(train[i].notnull())&(train[j].notnull()),j]

fit=np.polyfit(x,y,1)

train.loc[train[j].isnull(),j]=train.loc[train[j].isnull(),i]*fit[0]+fit[1]



train.loc[train['product_type'].isnull(),'product_type']='Investment'





binary=[]

for i in train:

    if train[i].dtypes=='object':

        #print(train[i].value_counts())

        if train[i].value_counts().shape[0]==2:

            binary.append(i)

for i in binary:

    train[i]=pd.factorize(train[i])[0]

#change the echology to a 1-4 and ohe for NANs

train.loc[train['ecology']=='no data','ecology_dat']=0

train.loc[train['ecology']!='no data','ecology_dat']=1

train.loc[train['ecology']=='no data','ecology']=2

train.loc[train['ecology']=='poor','ecology']=1

train.loc[train['ecology']=='satisfactory','ecology']=2

train.loc[train['ecology']=='good','ecology']=3

train.loc[train['ecology']=='excellent','ecology']=4

train.ecology=pd.to_numeric(train.ecology)

#sub_area to ohe

train=pd.concat([train,pd.get_dummies(train.sub_area)],axis=1)

train=pd.concat([train,pd.get_dummies(train.material,prefix='material')],axis=1)

train.drop(['sub_area','material'],inplace=True,axis=1)

train.drop(['price_doc','id'],inplace=True,axis=1)



for i in train:

    out=train[i].isnull().sum()

    if out>0:

        print(i)

    

test=train.iloc[n:,]

train=train.iloc[0:n,]

print(test.shape,train.shape,n)