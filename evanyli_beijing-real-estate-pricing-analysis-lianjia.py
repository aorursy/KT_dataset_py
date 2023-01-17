import pandas as pd
lj_data = pd.read_csv('./data/LJdata.csv')
lj_data.columns
lj_data.columns = ['district', 'address', 'title', 'house_type', 'area', 'price', 'floor', 'build_time', 'direction', 'update_time', 'view_num', 'extra_info', 'link']
lj_data.shape
lj_data.info()
lj_data.describe(include='all')

lj_data.head(1)
lj_data.sort_values(by='update_time', ascending=False).head(20)
lj_data.loc[lj_data['update_time']=='2017.07.27',:].shape
lj_data['update_time'].unique()
lj_data.loc[lj_data['update_time']=='2017.07.27',:]
lj_data['view_num'].median()
%matplotlib inline
lj_data['view_num'].value_counts().plot(kind='bar')
lj_data['view_num'].value_counts()
lj_data.head(5)
import numpy as np

def get_house_build_year(x):
    try:
        return int(x[:4])
    except:
        return np.NaN

#lj_data.loc[:,'house_age'] = 2018-lj_data['build_time'].apply(lambda x:x[:4]).astype(int)
lj_data.loc[:,'house_age'] = 2018-lj_data['build_time'].apply(get_house_build_year)
lj_data.head(1)
lj_data.loc[:,'house_area'] = lj_data['area'].apply(lambda x:x[:-2]).astype(float)
lj_data.head()
lj_data.info()
lj_data.nsmallest(columns='house_age', n=20)[['view_num','house_area']].agg('mean')
lj_data['price'].describe()
popular_direction = lj_data.groupby('direction')[['view_num']].agg('sum')
popular_direction.nlargest(columns='view_num', n=1)
house_type_dis = lj_data.groupby('house_type').size()
%matplotlib inline
house_type_dis.plot(kind='pie')
tmp = lj_data.groupby('house_type').agg({'view_num':'sum'})
tmp.reset_index(inplace=True)
tmp[tmp['view_num']==tmp['view_num'].max()]
lj_data.loc[:,'price_per_m2'] = lj_data['price']/lj_data['house_area']
lj_data['price_per_m2'].mean()
lj_data.head()
lj_data[['address','view_num']].groupby('address').sum().nlargest(columns='view_num', n=1)
lj_data['address'].value_counts().head(1)
lj_data.head()
lj_data.loc[:,'center_heating'] = lj_data['extra_info'].apply(lambda x: '集中供暖' in x)
lj_data['center_heating'].value_counts()
lj_data[['center_heating', 'price', 'price_per_m2']].groupby('center_heating').agg('mean')

lj_data[['house_type','house_area']].groupby('house_type').agg(['mean','max','min'])
lj_data.head()
import re
def find_sub_station(x):
    try:
        return re.search(pattern='距离(\d+号线)(.*?站)(\d+?米)', string=x).group(2)
    except:
        return np.NaN
lj_data.loc[:,'sub_station'] = lj_data['extra_info'].apply(find_sub_station)
lj_data.head()
lj_data['sub_station'].value_counts()
def has_sub_station(x):
    return 1 if '距离' in x else 0

lj_data.loc[:,'has_sub_station'] = lj_data['extra_info'].apply(has_sub_station)
lj_data.head()
lj_data[['has_sub_station', 'price', 'price_per_m2']].groupby('has_sub_station').agg('mean')
def get_subway_distance(x):
    try:
        return re.search(pattern='距离(\d+号线)(.*?站)(\d+?)米', string=x).group(3)
    except:
        return np.NaN
lj_data.loc[:,'distance'] = lj_data['extra_info'].apply(get_subway_distance).astype(float)
lj_data.head()
lj_data['distance'].mean()
lj_data.head()
def get_floor(x):
    if '低楼层' in x:
        return '低楼层'
    elif '中楼层' in x:
        return '中楼层'
    else:
        return '高楼层'
    
lj_data.loc[:,'house_floor'] = lj_data['floor'].apply(get_floor)
lj_data['house_floor'].value_counts()

def get_info(x):
    return 1 if '随时看房' in x else 0

lj_data.loc[:,'convenient'] = lj_data['extra_info'].apply(get_info)
lj_data.head()
lj_data['convenient'].value_counts()
def get_elev(x):
    try:
        return int(re.search(pattern='共(\d+)层', string=x).group(1))
    except:
        return np.NaN

lj_data.loc[:,'elev'] = lj_data['floor'].apply(get_elev)
lj_data.head()
lj_data.loc[:,'has_elev'] = lj_data['elev'].apply(lambda x:x>=8)
lj_data['has_elev'].value_counts()
lj_data[['has_elev', 'house_area', 'price', 'price_per_m2', 'direction', 'view_num']].groupby('has_elev').describe(include='all')
# 分割附加信息，怎么合理分列（识别信息类别），然后就可以看覆盖百分比了。
