# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 10000)

pd.set_option('display.max_columns', 1000)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import glob

import os

files = glob.glob('/kaggle/input/us-executive-branch-finances/Pompeo Michael R*.xlsx')

for file in files:

    print(file)

    print(os.path.splitext(file)[0].split('_')[-1])
result = dict()
def get_min_max(ser):

    nan_cond = ser.isnull()

    nans = ser[nan_cond]

    nans['min_val'] = np.nan

    nans['max_val'] = np.nan

    not_nans = ser[~nan_cond]

    cond = not_nans.str.contains('-')

    hyphens = not_nans[cond]

    non_hyphens = not_nans[~cond]

    hdf = hyphens.str.split('-', expand=True)

    for i in range(len(hdf.columns)):

        hdf[i] = hdf[i].str.strip()

        hdf[i] = hdf[i].str.replace('$', '')

        hdf[i] = hdf[i].str.replace(',', '')

        hdf[i] = pd.to_numeric(hdf[i])

    if len(hdf.columns) == 2:

        hdf.columns = ['min_val', 'max_val']

    nhdf = non_hyphens.str.extract(r'None.*\$([\d,]+).*')

    nhdf[0] = nhdf[0].str.strip()

    nhdf[0] = nhdf[0].str.replace('$', '')

    nhdf[0] = nhdf[0].str.replace(',', '')

    nhdf[0] = pd.to_numeric(nhdf[0])

    nhdf.columns = ['max_val']

    nhdf['min_val'] = 0

    return nans[['min_val', 'max_val']].append(hdf[['min_val', 'max_val']].append(nhdf[['min_val', 'max_val']]))
xl = pd.ExcelFile('/kaggle/input/us-executive-branch-finances/Pompeo Michael R_Secretary_Department of State_Annual (2019)_06.14.2019.xlsx')

df = xl.parse('Table 1', header=None)

df
employment_assets = df.loc[10:11][[1, 2, 5, 6, 7]].reset_index(drop=True)

employment_assets.columns = employment_assets.loc[0].values

employment_assets = employment_assets.loc[1:].dropna(axis=0, how='all')

last_part = df.loc[12][0]

employment_assets.loc[1, 'INCOME AMOUNT'] += " " + last_part

#employment_assets



ea_val = get_min_max(employment_assets['VALUE'])

ea = employment_assets.join(ea_val)

ea
spouse_assets = df.loc[17:31].append(df.loc[36:42])[[1, 2, 5, 6, 7]].reset_index(drop=True)

spouse_assets.columns = spouse_assets.loc[0].values

spouse_assets = spouse_assets.loc[1:].dropna(axis=0, how='all')

#last_part = df.loc[38][0]

#spouse_assets.loc[6, 'VALUE'] += " " + last_part

#spouse_assets.loc[12, 'VALUE'] += " " + last_part

#last_part = str(df.loc[36][0])

#spouse_assets.loc[3, 'VALUE'] += " " + last_part

#spouse_assets.loc[4, 'VALUE'] += " " + last_part

#spouse_assets.loc[15, 'VALUE'] += " " + last_part

#spouse_assets



sa_val = get_min_max(spouse_assets['VALUE'])

sa = spouse_assets.join(sa_val)

sa
other_assets = df.loc[45:47][[1, 2, 5, 6, 7]].reset_index(drop=True)

other_assets.columns = other_assets.loc[0].values

other_assets = other_assets.loc[1:].dropna(axis=0, how='all')

#last_part = df.loc[18][0]

#spouse_assets.loc[2, 'VALUE'] += " " + last_part

#spouse_assets.loc[4, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#spouse_assets.loc[15, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#other_assets



oa_val = get_min_max(other_assets['VALUE'])

oa = other_assets.join(oa_val)

oa
assets = pd.concat([ea, sa, oa])

assets
total = assets[['min_val', 'max_val']].sum()

print("Total Asset value :")

asset_value = (total['max_val']-total['min_val'])/2

asset_value
result['6/14/2019'] = asset_value
xl = pd.ExcelFile('/kaggle/input/us-executive-branch-finances/Pompeo Michael R_Director_Central Intelligence Agency_Nominee 278 (12.23.2016)_12.23.2016.xlsx')

xl.sheet_names

df = xl.parse('Table 1', header=None)

df
employment_assets = df.loc[9:11][[1, 2, 4, 5, 6]].reset_index(drop=True)

employment_assets.columns = employment_assets.loc[0].values

employment_assets = employment_assets.loc[1:].dropna(axis=0, how='all')

last_part = df.loc[11][0]

employment_assets.loc[1, 'INCOME AMOUNT'] += " " + last_part

ea_val = get_min_max(employment_assets['VALUE'])

ea = employment_assets.join(ea_val)

ea
spouse_assets = df.loc[15:17].append(df.loc[22:39])[[1, 2, 4, 5, 6]].reset_index(drop=True)

spouse_assets.columns = spouse_assets.loc[0].values

spouse_assets = spouse_assets.loc[1:].dropna(axis=0, how='all')

last_part = df.loc[18][0]

spouse_assets.loc[2, 'VALUE'] += " " + last_part

spouse_assets.loc[4, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

spouse_assets.loc[15, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#spouse_assets

sa_val = get_min_max(spouse_assets['VALUE'])

sa = spouse_assets.join(sa_val)

sa
other_assets = df.loc[41:42].append(df.loc[46:48])[[1, 2, 4, 5, 6]].reset_index(drop=True)

other_assets.columns = other_assets.loc[0].values

other_assets = other_assets.loc[1:].dropna(axis=0, how='all')

#last_part = df.loc[18][0]

#spouse_assets.loc[2, 'VALUE'] += " " + last_part

#spouse_assets.loc[4, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#spouse_assets.loc[15, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#other_assets



oa_val = get_min_max(other_assets['VALUE'])

oa = other_assets.join(oa_val)

oa
assets = pd.concat([ea, sa, oa])

assets
total = assets[['min_val', 'max_val']].sum()

print("Total Asset value :")

asset_value = (total['max_val']-total['min_val'])/2

asset_value
result['12/23/2016'] = asset_value
xl = pd.ExcelFile('/kaggle/input/us-executive-branch-finances/Pompeo Michael R_Secretary_Department Of State_Nominee 278 (03.27.2018)_03.27.2018.xlsx')

df = xl.parse('Table 1', header=None)

df
employment_assets = df.loc[11:13][[1, 2, 5, 6, 7]].reset_index(drop=True)

employment_assets.columns = employment_assets.loc[0].values

employment_assets = employment_assets.loc[1:].dropna(axis=0, how='all')

last_part = df.loc[13][0]

employment_assets.loc[1, 'INCOME AMOUNT'] += " " + last_part

#employment_assets



ea_val = get_min_max(employment_assets['VALUE'])

ea = employment_assets.join(ea_val)

ea
spouse_assets = df.loc[18:35].append(df.loc[43:49])[[1, 2, 4, 5, 6]].reset_index(drop=True)

spouse_assets.columns = spouse_assets.loc[0].values

spouse_assets = spouse_assets.loc[1:].dropna(axis=0, how='all')

last_part = df.loc[38][0]

spouse_assets.loc[6, 'VALUE'] += " " + last_part

spouse_assets.loc[12, 'VALUE'] += " " + last_part

last_part = str(df.loc[36][0])

spouse_assets.loc[3, 'VALUE'] += " " + last_part

spouse_assets.loc[4, 'VALUE'] += " " + last_part

spouse_assets.loc[15, 'VALUE'] += " " + last_part

#spouse_assets



sa_val = get_min_max(spouse_assets['VALUE'])

sa = spouse_assets.join(sa_val)

sa
other_assets = df.loc[52:56][[1, 2, 5, 6, 7]].reset_index(drop=True)

other_assets.columns = other_assets.loc[0].values

other_assets = other_assets.loc[1:].dropna(axis=0, how='all')

#last_part = df.loc[18][0]

#spouse_assets.loc[2, 'VALUE'] += " " + last_part

#spouse_assets.loc[4, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#spouse_assets.loc[15, 'VALUE'] = spouse_assets.loc[3, 'VALUE']

#other_assets



oa_val = get_min_max(other_assets['VALUE'])

oa = other_assets.join(oa_val)

oa
assets = pd.concat([ea, sa, oa])

assets
total = assets[['min_val', 'max_val']].sum()

print("Total Asset value :")

asset_value = (total['max_val']-total['min_val'])/2

asset_value
result['3/27/2018'] = asset_value
result = {'dates': list(result.keys()), 'total_assets': list(result.values())}

result
total_assets = pd.DataFrame(result)

total_assets['dates'] = pd.to_datetime(total_assets['dates'])

total_assets = total_assets.set_index('dates')

total_assets
total_assets.plot()