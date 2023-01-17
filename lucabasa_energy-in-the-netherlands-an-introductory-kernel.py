import numpy as np

import pandas as pd



#visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from os.path import join, isfile

from os import path, scandir, listdir



import gc
def list_all_files(location='kaggle/input/', pattern=None, recursive=True):

    """

    This function returns a list of files at a given location (including subfolders)

    

    - location: path to the directory to be searched

    - pattern: part of the file name to be searched (ex. pattern='.csv' would return all the csv files)

    - recursive: boolean, if True the function calls itself for every subdirectory it finds

    """

    subdirectories= [f.path for f in scandir(location) if f.is_dir()]

    files = [join(location, f) for f in listdir(location) if isfile(join(location, f))]

    if recursive:

        for directory in subdirectories:

            files.extend(list_all_files(directory))

    if pattern:

        files = [f for f in files if pattern in f]

    return files
list_all_files('/kaggle/input/Electricity/', pattern='stedin')
def importer(file_list):

    imported = {}

    for file in file_list:

        yr = file.split('_')[-1].split('.')[0]

        if '0101' in yr:

            yr = yr.replace('0101', '')

        name = file.split('/')[-1].split('_')[0]

        # print(name, yr)

        df = pd.read_csv(file)

        # print(df.shape)

        imported[name + '_' + yr] = df

        del df

    return imported
elec_list = list_all_files('/kaggle/input/Electricity/')

gas_list = list_all_files('/kaggle/input/Gas/')

imp_elec = importer(elec_list)

imp_gas = importer(gas_list)

print('Done!')
def merge_manager(data_dict):

    all_man = pd.DataFrame()

    n_rows = 0

    for key in data_dict.keys():

        df = data_dict[key].copy()

        yr = key.split('_')[1]

        yr = str(int(yr) - 1) # account for the "delayed data issue"

        df = df.rename(columns={'annual_consume' : 'annual_consume_' + yr,

                               'delivery_perc': 'delivery_perc_' + yr,

                               'num_connections': 'num_connections_' + yr,

                               'perc_of_active_connections': 'perc_of_active_connections_' + yr,

                               'annual_consume_lowtarif_perc': 'annual_consume_lowtarif_perc_' + yr,

                               'smartmeter_perc': 'smartmeter_perc_' + yr})

        del df['type_conn_perc']

        del df['type_of_connection']

        del df['net_manager']

        del df['purchase_area']

        n_rows += df.shape[0]

        if len(all_man) == 0:

            all_man = df.copy()

        else:

            del df['street']

            del df['city']

            all_man = pd.merge(all_man, df, on=['zipcode_from', 'zipcode_to'], how='inner') # 'city', 'street',  

        del df

        gc.collect()

    print(f"Total rows before merge: {n_rows}")

    print(f"Total rows after merge: {all_man.shape[0]}")

    return all_man





def merge_yr(data_dict):

    all_yr = pd.DataFrame()

    for manager in ['enexis', 'liander', 'stedin']:

        print(manager)

        tmp = { key: data_dict[key] for key in data_dict.keys() if manager in key}

        all_man = merge_manager(tmp)

        if len(all_yr) == 0:

            all_yr = all_man.copy()

        else:

            all_yr = pd.concat([all_yr, all_man], ignore_index=True, join='inner')

        del all_man

        gc.collect()

        print("_"*40)

    print(f"Final shape: {all_yr.shape}")

    return all_yr
print("Electricity merging...")

elec_full = merge_yr(imp_elec)

print('_'*40)

print('_'*40)

print("Gas merging...")

gas_full = merge_yr(imp_gas)
elec_full.head()
def consume_per_connection(data, consume_list):

    for col in consume_list:

        yr = col.split('_')[-1]

        data['consume_per_conn_'+yr] = data[col] / (data['num_connections_' + yr] * 

                                                   data['perc_of_active_connections_' + yr] / 100)

        data.loc[data['consume_per_conn_' + yr] == np.inf, 'consume_per_conn_' + yr] = 0

    return data
consume = [col for col in elec_full.columns if 'annual_consume_2' in col]

consume.sort()
elec_full = consume_per_connection(elec_full, consume)

elec_full[consume + [col for col in elec_full.columns if 'consume_per_conn_' in col]].describe()
gas_full = consume_per_connection(gas_full, consume)

gas_full[consume + [col for col in gas_full.columns if 'consume_per_conn_' in col]].describe()
fig, ax = plt.subplots(1,2, figsize=(17,8))



for col in consume:

    sns.distplot(elec_full.loc[elec_full[col] < 20000, col], 

                 hist=False, label=col.split('_')[-1], ax=ax[0], axlabel='Annual Consumption')

    sns.distplot(gas_full.loc[gas_full[col] < 6000, col], 

                 hist=False, label=col.split('_')[-1], ax=ax[1], axlabel='Annual Consumption') 



ax[0].set_title('Electricity', fontsize=15)

ax[1].set_title('Gas', fontsize=15)

fig.suptitle('Annual consumption', fontsize=22)

plt.show()
cons_per_conn = [col for col in gas_full.columns if 'consume_per_conn_' in col]

cons_per_conn.sort()
fig, ax = plt.subplots(1,2, figsize=(17,8))



for col in cons_per_conn:

    sns.distplot(elec_full.loc[elec_full[col] < 1000, col], 

                 hist=False, label=col.split('_')[-1], ax=ax[0], axlabel='Annual Consumption')

    sns.distplot(gas_full.loc[gas_full[col] < 400, col], 

                 hist=False, label=col.split('_')[-1], ax=ax[1],  axlabel='Annual Consumption')



ax[0].set_title('Electricity', fontsize=15)

ax[1].set_title('Gas', fontsize=15)

fig.suptitle('Annual consume per connection', fontsize=22)

plt.show()
plt.figure(figsize=(10,10))

sns.pairplot(elec_full[consume].sample(10000), kind="reg")



plt.suptitle('Correlations between electricity consumptions at different years', fontsize=22, y=1.01)

plt.show()
elec_city = elec_full[['city', 'annual_consume_2009']].groupby('city', as_index=False).sum()



for col in consume:

    if col == 'annual_consume_2009':

        continue

    tmp = elec_full[['city', col]].groupby('city', as_index=False).sum()

    elec_city = pd.merge(elec_city, tmp, on='city')



elec_city = elec_city.set_index('city')

elec_city['mean_consume'] = elec_city.mean(axis=1)

elec_city.sample(5)
tmp = elec_city.nlargest(10, 'mean_consume')

del tmp['mean_consume'] # so it doesn't show up in the plot

tmp.columns = tmp.columns.str.replace('annual_consume_', '')

ax = tmp.T.plot(figsize=(10,8))

ax.set_xticklabels(['','2009', '2011','2013','2015', '2017'])

ax.set_ylabel("kWh")

ax.set_xlabel("Year")

ax.set_title("Electricity consumption by year (top 10 cities)", fontsize=18)

del tmp
elec_city = elec_full[['city', 'annual_consume_2009', 'num_connections_2009']].groupby('city', as_index=False).sum()

elec_city['cons_per_con_2009'] = elec_city['annual_consume_2009'] / elec_city['num_connections_2009']

del elec_city['num_connections_2009']

del elec_city['annual_consume_2009']



for col in consume:

    if col == 'annual_consume_2009':

        continue

    yr = col.split('_')[-1]

    tmp = elec_full[['city', col, 'num_connections_'+yr]].groupby('city', as_index=False).sum()

    tmp['cons_per_con_'+yr] = tmp[col] / tmp['num_connections_'+yr]

    del tmp[col]

    del tmp['num_connections_'+yr]

    elec_city = pd.merge(elec_city, tmp, on='city')



elec_city = elec_city.set_index('city')

elec_city['mean_consume'] = elec_city.mean(axis=1)

tmp = elec_city.nlargest(10, 'mean_consume')

del tmp['mean_consume']

tmp.columns = tmp.columns.str.replace('cons_per_con_', '')

ax = tmp.T.plot(figsize=(10,8), title='Electricity consumption per connection by year')

ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax.set_ylabel("kWh")

ax.set_xlabel("Year")

ax.set_title("Electricity consumption per connection by year (top 10 cities)", fontsize=18)

del tmp

plt.show()
gas_city = gas_full[['city', 'annual_consume_2009', 'num_connections_2009']].groupby('city', as_index=False).sum()

gas_city['cons_per_con_2009'] = gas_city['annual_consume_2009'] / gas_city['num_connections_2009']

del gas_city['num_connections_2009']

del gas_city['annual_consume_2009']



for col in consume:

    if col == 'annual_consume_2009':

        continue

    yr = col.split('_')[-1]

    tmp = gas_full[['city', col, 'num_connections_'+yr]].groupby('city', as_index=False).sum()

    tmp['cons_per_con_'+yr] = tmp[col] / tmp['num_connections_'+yr]

    del tmp[col]

    del tmp['num_connections_'+yr]

    gas_city = pd.merge(gas_city, tmp, on='city')



gas_city = gas_city.set_index('city')

gas_city['mean_consume'] = gas_city.mean(axis=1)

tmp = gas_city.nlargest(10, 'mean_consume')

del tmp['mean_consume']

tmp.columns = tmp.columns.str.replace('cons_per_con_', '')

ax = tmp.T.plot(figsize=(10,8), title='Gas consumption per connection by year')

ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax.set_ylabel("m3")

ax.set_xlabel("Year")

ax.set_title("Gas consumption per connection by year (top 10 cities)", fontsize=18)

del tmp
print(elec_city.nlargest(10, 'mean_consume')['mean_consume'])

print('_'*40)

print(gas_city.nlargest(10, 'mean_consume')['mean_consume'])
del elec_full

del gas_full

del elec_city

del gas_city



gc.collect()
def aggr_yr(data, yr):

    # useful features

    data['net_annual_cons_'+yr] = data['annual_consume'] * data['delivery_perc'] / 100

    data['self_production_'+yr] = data['annual_consume'] - data['net_annual_cons_'+yr]

    data['low_tarif_cons_'+yr] = data['annual_consume'] * data['annual_consume_lowtarif_perc'] / 100

    data['active_conn_'+yr] = data['num_connections'] * data['perc_of_active_connections'] / 100

    data['num_smartmeters_'+yr] = data['num_connections'] * data['smartmeter_perc'] / 100

    data = data.rename(columns={'annual_consume': 'annual_consume_'+yr})

    # aggregations

    aggregation = data[['city', 'annual_consume_'+yr, 'net_annual_cons_'+yr,

                        'self_production_'+yr, 'low_tarif_cons_'+yr,

                        'active_conn_'+yr, 'num_smartmeters_'+yr]].groupby('city', as_index=False).sum()

    return aggregation



def aggr_mng(data_dict):

    all_man = pd.DataFrame()

    for key in data_dict.keys():

        df = data_dict[key].copy()

        yr = key.split('_')[-1]

        yr = str(int(yr) - 1) # account for the "delayed data issue"

        if len(all_man) == 0:

            all_man = aggr_yr(df, yr)

        else:

            df = aggr_yr(df,yr)

            all_man = pd.merge(all_man, df, on='city')

        del df

        gc.collect()

    all_man = all_man.set_index('city')

    return all_man



def aggregations(data_dict):

    result = pd.DataFrame()

    for manager in ['enexis', 'liander', 'stedin']:

        print(manager)

        tmp = { key: data_dict[key] for key in data_dict.keys() if manager in key}

        all_man = aggr_mng(tmp)

        if len(result) == 0:

            result = all_man.copy()

        else:

            result = pd.concat([result, all_man], join='inner')

        del all_man

        gc.collect()

        print("_"*40)

    print(f"Final shape: {result.shape}")

    return result
cities_el = aggregations(imp_elec)

cities_el.sample(10)
cities_el.describe()
cities_gas = aggregations(imp_gas)

cities_gas.sample(10)
cities_gas.describe()
consume = [col for col in cities_el.columns if 'annual_consume_' in col]

consume.sort()
fig, ax = plt.subplots(1,2, figsize=(20, 8))



cities_el[consume].sum().plot(title='Total Electricity consumption per year', ax=ax[0])

ax[0].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[0].set_ylabel("kWh")

ax[0].set_xlabel("Year")

cities_gas[consume].sum().plot(title='Total Gas consumption per year', ax=ax[1])

ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[1].set_ylabel("m3")

ax[1].set_xlabel("Year")

plt.show()
tmp = cities_el[consume].copy()

tmp['mean_consume'] = tmp.mean(axis=1)

tmp = tmp.nlargest(10, 'mean_consume')

del tmp['mean_consume']

tmp.columns = tmp.columns.str.replace('annual_consume_', '')



fig, ax = plt.subplots(1,2, figsize=(20, 8))

tmp.T.sum().plot(kind='bar', title='Total Electricity top 10 cities', ax=ax[0])

tmp.T.plot(title='Electricity consumption per year top 10 cities',ax=ax[1])

ax[0].set_ylabel("kWh")

ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[1].set_xlabel("Year")

plt.show()
tmp = cities_gas[consume].copy()

tmp['mean_consume'] = tmp.mean(axis=1)

tmp = tmp.nlargest(10, 'mean_consume')

del tmp['mean_consume']

tmp.columns = tmp.columns.str.replace('annual_consume_', '')



fig, ax = plt.subplots(1,2, figsize=(20, 8))

tmp.T.sum().plot(kind='bar', title='Total Gas top10 cities', ax=ax[0])

tmp.T.plot(title='Gas consumption per year top10 cities',ax=ax[1])

ax[0].set_ylabel("m3")

ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[1].set_xlabel("year")

plt.show()
self_prod = [col for col in cities_el.columns if 'self_production_' in col]

self_prod.sort()
ax = cities_el[self_prod].sum().plot(figsize=(12, 8), fontsize=12)

ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax.set_title('Total Electricity self-produced per year', fontsize=22)

ax.set_ylabel("kWh", fontsize=12)

ax.set_xlabel("year", fontsize=12)

plt.show()
tmp = cities_el[self_prod].copy()

tmp['max_prod'] = tmp.max(axis=1)

tmp = tmp.nlargest(10, 'max_prod')

del tmp['max_prod']

tmp.columns = tmp.columns.str.replace('self_production_', '')



fig, ax = plt.subplots(1,2, figsize=(20, 8))

tmp['2018'].T.plot(kind='bar', title='Electricity self-produced in 2018, top10 cities', ax=ax[0])

tmp.T.plot(title='Electricity self-produced per year top10 cities',ax=ax[1])

ax[0].set_ylabel("kWh")

ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[1].set_xlabel("Year")

plt.show()
smrt = [col for col in cities_el.columns if 'num_smartmeters_' in col]

smrt.sort()
ax = cities_el[smrt].sum().plot(figsize=(12, 8),fontsize=12)

ax.set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax.set_ylabel("Number of smart meters", fontsize=12)

ax.set_xlabel("year", fontsize=12)

ax.set_title('Total smart meters per year', fontsize=22)

plt.show()
tmp = cities_el[smrt].copy()

tmp['max_num'] = tmp.max(axis=1)

tmp = tmp.nlargest(10, 'max_num')

del tmp['max_num']

tmp.columns = tmp.columns.str.replace('num_smartmeters_', '')



fig, ax = plt.subplots(1,2, figsize=(20, 8))

tmp['2018'].T.plot(kind='bar', title='Total number of smart meters in 2018, top10 cities', ax=ax[0])

tmp.T.plot(title='Total number of smart meters per year top10 cities',ax=ax[1])

ax[0].set_ylabel("Number of smart meters")

ax[1].set_xticklabels(['','2009', '2011', '2013','2015','2017', '2019'])

ax[1].set_xlabel("Year")

plt.show()