# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 200)

import matplotlib as plt

import cufflinks as cf

import plotly

import plotly.offline as py

import plotly.graph_objs as go

cf.go_offline() # required to use plotly offline (no account required).

py.init_notebook_mode() # graphs charts inline (IPython).



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





BASE_DIR = "/kaggle/input/personal-finance-of-us-reps/"

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cids = []

for year in range(2012, 2020, 2):

    filename = f"Candidate Ids - {year}.csv" if year == 2018 else f"Candidate IDs - {year}.csv"

    skiprows = 6 if year == 2012 else 13

    cid = pd.read_csv(BASE_DIR + filename, skiprows=skiprows)

    cid = cid.dropna(axis=0, how='all')

    cid = cid.dropna(axis=1, how='all')

    cid['Year'] = year

    cids.append(cid)



cids = pd.concat(cids, axis=0)

cids.head()
assets = pd.read_csv(BASE_DIR + 'PFDasset.csv', parse_dates=['Date'])

assets = assets[assets['Dupe'].str.upper() != 'D']

assets.head()
liability = pd.read_csv(BASE_DIR + 'PFDliability.csv', parse_dates=['LiabilityDate'])

liability = liability[liability['Dupe'].str.upper() != 'D']

liability.head()
assetTypeCodes = pd.read_excel(BASE_DIR + 'CRP_PFDRangeData.xls', sheet_name='AssetTypeCodes')

assetTypeCodes = assetTypeCodes[['AssetTypeCode', 'AssetTypeDescription', 'UseInd']].rename({'UseInd': 'AssetTypeUseIndustry'}, axis=1)

assets['AssetTypeCRP'] = assets['AssetTypeCRP'].str.upper()

assets['AssetTypeCRP'] = assets['AssetTypeCRP'].str.strip()

assetTypeCodes['AssetTypeCode'] = assetTypeCodes['AssetTypeCode'].str.strip()

assets = assets.merge(assetTypeCodes, how='left', left_on='AssetTypeCRP', right_on='AssetTypeCode').drop('AssetTypeCode', axis=1)

assets.head()
assetRanges = pd.read_excel(BASE_DIR + 'CRP_PFDRangeData.xls', sheet_name='RangesAssets')

assetRanges = assetRanges[['Chamber', 'Code', 'MinValue', 'MaxValue']].rename(

    {'MinValue': 'AssetMinValue', 'MaxValue': 'AssetMaxValue'}, axis=1)

assets['Chamber'] = assets['Chamber'].str.upper()

assets['Chamber'] = assets['Chamber'].str.strip()

assets['AssetValue'] = assets['AssetValue'].str.upper()

assets['AssetValue'] = assets['AssetValue'].str.strip()

assetRanges['Chamber'] = assetRanges['Chamber'].str.strip()

assetRanges['Code'] = assetRanges['Code'].str.strip()

assets = assets.merge(assetRanges, how='left', left_on=['Chamber', 'AssetValue'], right_on=['Chamber', 'Code']).drop('Code', axis=1)

assets.head()
liabilityRanges = pd.read_excel(BASE_DIR + 'CRP_PFDRangeData.xls', sheet_name='RangesLiability')

liabilityRanges = liabilityRanges[['Chamber', 'Code', 'MinValue', 'MaxValue']].rename(

     {'MinValue': 'LiabilityMinValue', 'MaxValue': 'LiabilityMaxValue'}, axis=1)

liability['Chamber'] = liability['Chamber'].str.upper()

liability['Chamber'] = liability['Chamber'].str.strip()

liability['LiabilityAmt'] = liability['LiabilityAmt'].str.upper()

liability['LiabilityAmt'] = liability['LiabilityAmt'].str.strip()

liabilityRanges['Chamber'] = liabilityRanges['Chamber'].str.strip()

liabilityRanges['Code'] = liabilityRanges['Code'].str.strip()

liability = liability.merge(liabilityRanges, how='left', left_on=['Chamber', 'LiabilityAmt'],

                            right_on=['Chamber', 'Code']).drop('Code', axis=1)

liability.head()
assetIncomeRanges = pd.read_excel(BASE_DIR + 'CRP_PFDRangeData.xls', sheet_name='Ranges_AssetIncome', skiprows=1)

assetIncomeRanges = assetIncomeRanges[['Chamber', 'Code', 'MinValue', 'MaxValue', 'Display']].rename(

    {'MinValue': 'AssetIncomeMinValue', 'MaxValue': 'AssetIncomeMaxValue', 'Display': 'AssetIncomeDisplay'}, axis=1)

assets['AssetIncomeAmtRange'] = assets['AssetIncomeAmtRange'].str.upper()

assets['AssetIncomeAmtRange'] = assets['AssetIncomeAmtRange'].str.strip()

assetIncomeRanges['Chamber'] = assetIncomeRanges['Chamber'].str.strip()

assetIncomeRanges['Code'] = assetIncomeRanges['Code'].str.strip()

assets = assets.merge(assetIncomeRanges, how='left', left_on=['Chamber', 'AssetIncomeAmtRange'], 

             right_on=['Chamber', 'Code']).drop('Code', axis=1)

assets.head()
industries = pd.read_csv(BASE_DIR + 'CRP Industry Codes.csv', skiprows=5)

industries.head()
assets = assets.merge(industries, how='left', left_on='RealCode', right_on='Catcode').drop('Catcode', axis=1)

assets = assets.merge(industries, how='left', left_on='RealCode2', right_on='Catcode', suffixes=('', '2')).drop('Catcode', axis=1)

assets.head()
liability = liability.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

liability.head()
assets['CalendarYear'] += 2000

liability['CalendarYear'] += 2000

assets.head()
assets['AssetValueDerived'] = np.where(assets['AssetExactValue'].isnull(), (

    assets['AssetMinValue'] + assets['AssetMaxValue'])/2, assets['AssetExactValue'])

assets['AssetIncomeValueDerived'] = np.where(assets['AssetIncomeAmt'].isnull(), (

    assets['AssetIncomeMinValue'] + assets['AssetIncomeMaxValue'])/2, assets['AssetIncomeAmt'])
by_chamber_over_time = assets.groupby(['CalendarYear', 'Chamber']).agg(

    {'AssetValueDerived': 'sum', 'AssetIncomeValueDerived': 'sum'}).reset_index()

for val in [('AssetValueDerived', 'Asset Values'), ('AssetIncomeValueDerived', 'Asset Income Values')]:

    asset_values = by_chamber_over_time[['CalendarYear', 'Chamber', val[0]]].pivot_table(index='CalendarYear', columns='Chamber')

    asset_values = asset_values.rename({'E': 'Executive', 'H': 'House', 'S': 'Senate', 'J': 'Judicial'}, axis=1)

    asset_values.columns = asset_values.columns.droplevel()

    asset_values.iplot(title=f'Total {val[1]} by Chamber')
by_asset_type_over_time = assets.groupby(['CalendarYear', 'AssetTypeDescription']).agg(

    {'AssetValueDerived': 'sum', 'AssetIncomeValueDerived': 'sum'}).reset_index()

for val in [('AssetValueDerived', 'Asset Values'), ('AssetIncomeValueDerived', 'Asset Income Values')]:

    asset_values = by_asset_type_over_time[['CalendarYear', 'AssetTypeDescription', val[0]]].pivot_table(index='CalendarYear', columns='AssetTypeDescription')

    asset_values.columns = asset_values.columns.droplevel()

    asset_values.iplot(title=f'Total {val[1]} by Asset Type')
by_sector_over_time = assets.groupby(['CalendarYear', 'Sector']).agg(

    {'AssetValueDerived': 'sum', 'AssetIncomeValueDerived': 'sum'}).reset_index()

for val in [('AssetValueDerived', 'Asset Values'), ('AssetIncomeValueDerived', 'Asset Income Values')]:

    asset_values = by_sector_over_time[['CalendarYear', 'Sector', val[0]]].pivot_table(index='CalendarYear', columns='Sector')

    asset_values.columns = asset_values.columns.droplevel()

    asset_values.iplot(title=f'Total {val[1]} by Sector')
liability['LiabilityValue'] = (liability['LiabilityMinValue'] + liability['LiabilityMaxValue'])/2

liability.head()
by_chamber_over_time = liability.groupby(['CalendarYear', 'Chamber']).agg({'LiabilityValue': 'sum'}).reset_index()

liability_values = by_chamber_over_time[['CalendarYear', 'Chamber', 'LiabilityValue']].pivot_table(index='CalendarYear', columns='Chamber')

liability_values = liability_values.rename({'E': 'Executive', 'H': 'House', 'S': 'Senate', 'J': 'Judicial'}, axis=1)

liability_values.columns = liability_values.columns.droplevel()

liability_values.iplot(title=f'Total Liability Value by Chamber')
by_sector_over_time = liability.groupby(['CalendarYear', 'Sector']).agg({'LiabilityValue': 'sum'}).reset_index()

liability_values = by_sector_over_time[['CalendarYear', 'Sector', 'LiabilityValue']].pivot_table(index='CalendarYear', columns='Sector')

liability_values.columns = liability_values.columns.droplevel()

liability_values.iplot(title=f'Total Liability Value by Sector')
assets_by_cand = assets.groupby(['CID', 'Chamber', 'CalendarYear']).agg(

    {'AssetValueDerived': 'sum'}).reset_index()

liability_by_cand = liability.groupby(['CID', 'Chamber', 'CalendarYear']).agg(

    {'LiabilityValue': 'sum'}).reset_index()

net_worth = assets_by_cand.merge(liability_by_cand, on=['CID', 'Chamber', 'CalendarYear'])

net_worth['NetWorth'] = net_worth['AssetValueDerived'] - net_worth['LiabilityValue']

net_worth.head()
by_chamber_over_time = net_worth.groupby(['CalendarYear', 'Chamber']).agg({'NetWorth': 'sum'}).reset_index()

net_worth_values = by_chamber_over_time[['CalendarYear', 'Chamber', 'NetWorth']].pivot_table(index='CalendarYear', columns='Chamber')

net_worth_values = net_worth_values.rename({'E': 'Executive', 'H': 'House', 'S': 'Senate', 'J': 'Judicial'}, axis=1)

net_worth_values.columns = net_worth_values.columns.droplevel()

net_worth_values.iplot(title=f'Total Net Worth by Chamber')
net_worth = pd.merge(net_worth, cids, left_on=['CID', 'CalendarYear'], right_on=['CID', 'Year']).drop('Year', axis=1)

net_worth.head()
by_party_over_time = net_worth.groupby(['CalendarYear', 'Party']).agg({'NetWorth': 'sum'}).reset_index()

net_worth_values = by_party_over_time[['CalendarYear', 'Party', 'NetWorth']].pivot_table(index='CalendarYear', columns='Party')

net_worth_values.columns = net_worth_values.columns.droplevel()

net_worth_values.iplot(title=f'Total Net Worth by Party')
net_worth['State'] = net_worth[net_worth['DistIDRunFor'] != 'PRES']['DistIDRunFor'].apply(lambda x: str(x)[:2])

net_worth.head()
net_worth_by_state = net_worth.groupby(['CalendarYear', 'State']).agg({'NetWorth': 'sum', 'CID': 'count'}).reset_index()

net_worth_by_state['NetWorthPerCandidate'] = net_worth_by_state['NetWorth']/net_worth_by_state['CID']

net_worth_by_state.head()
import plotly.express as px



fig = px.choropleth(net_worth_by_state, locations="State", color="NetWorthPerCandidate", hover_name="State", 

                    hover_data=['NetWorthPerCandidate'], animation_frame="CalendarYear", 

                    locationmode='USA-states', scope='usa', title='Net worth per candidate by state over the years')

fig.show()
curr_net_worth = net_worth[net_worth['CalendarYear'] == 2014]

curr_net_worth.nlargest(n=10, columns='NetWorth')
curr_net_worth.nsmallest(n=10, columns='NetWorth')