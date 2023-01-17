# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 100)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

BASE_DIR = "/kaggle/input/personal-finance-of-us-reps/"



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cids = dict()

for year in range(2012, 2020, 2):

    filename = f"Candidate Ids - {year}.csv" if year == 2018 else f"Candidate IDs - {year}.csv"

    skiprows = 6 if year == 2012 else 13

    cid = pd.read_csv(BASE_DIR + filename, skiprows=skiprows, index_col='CID')

    cid = cid.dropna(axis=0, how='all')

    cid = cid.dropna(axis=1, how='all')

    cids[year] = cid



cids[2014]
merged = cids[2012].merge(cids[2014], left_index=True, right_index=True, suffixes=('_2012', '_2014'))

merged[merged['CRPName_2012'] != merged['CRPName_2014']]
merged = cids[2014].merge(cids[2016], left_index=True, right_index=True, suffixes=('_2014', '_2016'))

merged[merged['CRPName_2014'] != merged['CRPName_2016']]
merged = cids[2016].merge(cids[2018], left_index=True, right_index=True, suffixes=('_2016', '_2018'))

merged[merged['CRPName_2016'] != merged['CRPName_2018']]
agreements = pd.read_csv(BASE_DIR + 'PFDagree.csv', parse_dates=['AgreementDate1', 'AgreementDate2'])

agreements = agreements[agreements['Dupe'].str.upper() != 'D']

agreements.head()
assets = pd.read_csv(BASE_DIR + 'PFDasset.csv', parse_dates=['Date'])

assets = assets[assets['Dupe'].str.upper() != 'D']

assets.head()
comp = pd.read_csv(BASE_DIR + 'PFDcomp.csv')

comp = comp[comp['dupe'].str.upper() != 'D']

comp.head()
gifts = pd.read_csv(BASE_DIR + 'PFDgift.csv', parse_dates=['GiftDate'])

gifts = gifts[gifts['Dupe'].str.upper() != 'D']

gifts.head()
honoraria = pd.read_csv(BASE_DIR + 'PFDhonoraria.csv', parse_dates=['HonorariaDate'])

honoraria = honoraria[honoraria['Dupe'].str.upper() != 'D']

honoraria.head()
income = pd.read_csv(BASE_DIR + 'PFDincome.csv')

income = income[income['Dupe'].str.upper() != 'D']

income.head()
liability = pd.read_csv(BASE_DIR + 'PFDliability.csv', parse_dates=['LiabilityDate'])

liability = liability[liability['Dupe'].str.upper() != 'D']

liability.head()
positions = pd.read_csv(BASE_DIR + 'PFDposition.csv', parse_dates=['PositionFromDate', 'PositionToDate'])

positions = positions[positions['Dupe'].str.upper() != 'D']

positions.head()
transactions = pd.read_csv(BASE_DIR + 'PFDtrans.csv', parse_dates=['Asset4Date'])

transactions = transactions[transactions['Dupe'].str.upper() != 'D']

transactions.head()
travel = pd.read_csv(BASE_DIR + 'PFDtravel.csv', parse_dates=['BeginDate', 'EndDate'])

travel = travel[travel['Dupe'].str.upper() != 'D']

travel.head()
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
transactions['Chamber'] = transactions['Chamber'].str.upper()

transactions['Chamber'] = transactions['Chamber'].str.strip()

transactions['Asset4TransAmt'] = transactions['Asset4TransAmt'].str.upper()

transactions['Asset4TransAmt'] = transactions['Asset4TransAmt'].str.strip()

transactions = transactions.merge(assetRanges, how='left', left_on=['Chamber', 'Asset4TransAmt'], right_on=['Chamber', 'Code']).drop('Code', axis=1)

transactions.head()
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
crp_cids = pd.read_excel(BASE_DIR + 'CRP_PFDRangeData.xls', sheet_name='CIDs', skiprows=2, index_col='CID')

crp_cids = crp_cids[['CRPName']].rename({'CRPName': 'CandidateName'}, axis=1)

crp_cids.head()
merged_cids = crp_cids

for year in range(2018, 2010, -2):

    merged_cids = merged_cids.merge(cids[year], left_index=True, right_on='CID', how='outer')

    merged_cids['CandidateName'] = merged_cids['CandidateName'].fillna(merged_cids['CRPName'])

    merged_cids = merged_cids.reset_index()[['CID', 'CandidateName']]

    merged_cids = merged_cids.set_index('CID')

merged_cids.head()
agreements = agreements.merge(merged_cids, left_on='CID', right_index=True, how='left')

agreements.head()
assets = assets.merge(merged_cids, left_on='CID', right_index=True, how='left')

assets.head()
comp = comp.merge(merged_cids, left_on='CID', right_index=True, how='left')

comp.head()
gifts = gifts.merge(merged_cids, left_on='CID', right_index=True, how='left')

gifts.head()
honoraria = honoraria.merge(merged_cids, left_on='CID', right_index=True, how='left')

honoraria.head()
income = income.merge(merged_cids, left_on='CID', right_index=True, how='left')

income.head()
liability = liability.merge(merged_cids, left_on='CID', right_index=True, how='left')

liability.head()
positions = positions.merge(merged_cids, left_on='CID', right_index=True, how='left')

positions.head()
transactions = transactions.merge(merged_cids, left_on='CID', right_index=True, how='left')

transactions.head()
travel = travel.merge(merged_cids, left_on='CID', right_index=True, how='left')

travel.head()
industries = pd.read_csv(BASE_DIR + 'CRP Industry Codes.csv', skiprows=5)

industries.head()
agreements = agreements.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

agreements = agreements.merge(industries, how='left', left_on='Realcode2', right_on='Catcode', suffixes=('', '2')).drop('Catcode', axis=1)

agreements.head()
assets = assets.merge(industries, how='left', left_on='RealCode', right_on='Catcode').drop('Catcode', axis=1)

assets = assets.merge(industries, how='left', left_on='RealCode2', right_on='Catcode', suffixes=('', '2')).drop('Catcode', axis=1)

assets.head()
comp = comp.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

comp.head()
gifts = gifts.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

gifts.head()
honoraria = honoraria.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

honoraria.head()
income = income.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

income.head()
positions = positions.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

positions.head()
transactions = transactions.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

transactions = transactions.merge(industries, how='left', left_on='Realcode2', right_on='Catcode', suffixes=('', '2')).drop('Catcode', axis=1)

transactions.head()
travel = travel.merge(industries, how='left', left_on='Realcode', right_on='Catcode').drop('Catcode', axis=1)

travel.head()