# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input/japan-trade-statistics"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Loading neccessary files

def load_colls(load_dict, df):

    for item in load_dict:

        # extract the needed arguments

        file_name = load_dict[item][0]

        entry = load_dict[item][1]

        index = load_dict[item][2]

        # now load

        temp_dict = pd.read_csv(file_name, index_col=entry).to_dict()

        temp_dict = temp_dict[index]

        # Temporary shenanigans to fix the entry header

        #if entry == 'hs4%0':

        #   entry = 'hs4'

        # Pass to the dataframe

        df[item] = df[entry].map(temp_dict)

        

    return df



# Map of stuff to load into the dataframe

load_map = {

    'Country_name': ['../input/country_eng.csv', 'Country', 'Country_name'],

    'hs2_name': ['../input/hs2_eng.csv', 'hs2', 'hs2_name'],

    'hs4_name': ['../input/hs4_eng.csv', 'hs4', 'hs4_name'],

    'hs6_name': ['../input/hs6_eng.csv', 'hs6', 'hs6_name'],

    'hs9_name': ['../input/hs9_eng.csv', 'hs9', 'hs9_name']

}



# Trade Period 1988 - 2015

trade_hist = pd.read_csv('../input/year_1988_2015.csv')

trade_hist = load_colls(load_map, trade_hist)

trade_jpbg_hist = trade_hist[trade_hist['Country_name'] == 'Bulgaria']

# Displaying a sample of the dataset

trade_jpbg_hist.head()
# sum all trade by year

trjpbg_gb_year = trade_jpbg_hist.groupby(by=['Year'], as_index=False)['VY'].sum()

plt.bar(trjpbg_gb_year['Year'], trjpbg_gb_year['VY'])

plt.xlabel('Year')

plt.ylabel('VY')

plt.title('Japan-Bulgaria Trade over the years, by year')
print("Minimum trade:" , trjpbg_gb_year['VY'].min())

print("Maximum trade:" ,trjpbg_gb_year['VY'].max())

print("Mean trade:" ,trjpbg_gb_year['VY'].mean())
print ('Minimum Trade Year:' , trjpbg_gb_year[trjpbg_gb_year['VY'] == trjpbg_gb_year['VY'].min()]['Year'].values[0])

print ('Maximum Trade Year:' , trjpbg_gb_year[trjpbg_gb_year['VY'] == trjpbg_gb_year['VY'].max()]['Year'].values[0])
trjpbg_gb_hs4 = trade_jpbg_hist.groupby(by=['hs4_name'], as_index=False)['VY'].sum()

plt.bar(range(1,len(trjpbg_gb_hs4['hs4_name'])+1), sorted(trjpbg_gb_hs4['VY'].values))

plt.xlabel('Good hs4 name')

plt.ylabel('VY')

plt.title('Japan-Bulgaria Trade in Goods')

plt.show()
trjpbg_top_ten_goods = trjpbg_gb_hs4.sort_values('VY', ascending=False).head(10)

trjpbg_top_ten_goods
trjpbg_top_item = trade_jpbg_hist[trade_jpbg_hist['hs4_name'] == trjpbg_top_ten_goods.loc[377].values[0]]

trjpbg_top_item_year = trjpbg_top_item.groupby(by=['Year'], as_index=False)['VY'].sum()

plt.bar(trjpbg_top_item_year['Year'], trjpbg_top_item_year['VY'])

plt.xlabel('Year')

plt.ylabel('VY')

plt.title('JP-BG Trade of item "Motor_cars_and_other_motor_vehicles"')

tr_jb_cpy = trade_jpbg_hist.copy()

years = [1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,

        2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,

        2014, 2015]



tr_jb_cpy['c_dip_d'] = 0



for item in tr_jb_cpy['hs4']:

    selected_indeces = tr_jb_cpy[tr_jb_cpy['hs4'] == item].index

    vy_before = []

    vy_after = []

    tr_jb_cpy_wrk = tr_jb_cpy.loc[selected_indeces, :]

    

    for year in years:

        if year < 2009:

            if year in tr_jb_cpy_wrk['Year'].values:

                year_vals_before = tr_jb_cpy_wrk.loc[tr_jb_cpy_wrk['Year'] == year, 'VY'].values.sum()

                vy_before.append(year_vals_before)

            else:

                vy_before.append(0)

        else:

            if year in tr_jb_cpy_wrk['Year'].values:

                year_vals_after = tr_jb_cpy_wrk.loc[tr_jb_cpy_wrk['Year'] == year, 'VY'].values.sum()

                vy_after.append(year_vals_after)

            else:

                vy_after.append(0)

    

    vy_before_mean = np.mean(vy_before)

    vy_after_mean = np.mean(vy_after)

    

    if vy_before_mean == 0 or vy_after_mean == 0:

        tr_jb_cpy.loc[selected_indeces, 'c_dip_d'] = 0

    else:

        tr_jb_cpy.loc[selected_indeces, 'c_dip_d'] = (vy_before_mean - vy_after_mean) / vy_before_mean * 100

tr_jb_dip_by_item = tr_jb_cpy.groupby(by=['hs4'], as_index=False).mean()
data_dip_trunc = tr_jb_dip_by_item[tr_jb_dip_by_item['c_dip_d']>0].sort_values('c_dip_d', ascending=False)

plt.bar(data_dip_trunc['hs4'].values.astype('str'), data_dip_trunc['c_dip_d'].values)

plt.xlabel('Year')

plt.ylabel('VY')

plt.title('JP-BG Trade of item "Motor_cars_and_other_motor_vehicles"')
tr_jb_dip_by_item.sort_values('c_dip_d', ascending=False).head(10)
trade_jpbg_hist.loc[trade_jpbg_hist['hs4']==5502, 'hs4_name'].values[0]