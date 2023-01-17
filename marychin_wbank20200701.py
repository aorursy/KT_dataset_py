import os, glob, pickle

import urllib.request

import zipfile

from collections import Counter

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



if 'kid' in os.getcwd():

    INP = '/home/kid/covid/data'

    WBANK = f'{INP}/wbank'

    OUT = f'{WBANK}/WDIrevision20200701'

elif 'kaggle' in os.getcwd():

    INP = '/kaggle/input/wdi20200701'

    OUT = '/kaggle/working'

    WBANK = f'{OUT}/wbank'

    os.mkdir(WBANK)

else:

    INP = '/home/jupyter/input'

    OUT = '/home/jupyter/output'

    WBANK = f'{INP}/wbank'



# INPUT:

# f'{INP}/WDI_CETS.csv'

# f'{INP}/WDIrevision_additions.csv'

# f'{INP}/WDIrevision_deletions.csv'

# f'{INP}/WDIrevision_codechanges.csv'

# f'{INP}/continent.csv'

# DOWNLOAD:

# f'{WBANK}/API_{code}*.csv'

# 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

# OUTPUT:

# f'{OUT}/wbank01july_meta.csv'

# f'{OUT}/wbank01july_year.csv'

# f'{OUT}/wbank01july_nullRatio.csv'

# f'{OUT}/wbank01july_continentalMedian.csv'

# f'{OUT}/wbank01july_data.csv'

# f'{OUT}/wbank01july_log.txt'
cets = pd.read_csv(f'{INP}/WDI_CETS.csv')

print(len(cets), cets['Series Code'].nunique())

revision = {}

dummy = []

for nam in ['additions', 'deletions', 'codechanges']:

    revision[nam] = pd.read_csv(f'{INP}/WDIrevision_{nam}.csv', delimiter=';', parse_dates=['Date'])
revision['additions'].rename(columns={'Additions': 'add'}, inplace=True)

revision['deletions'].rename(columns={'Deletions': 'drop'}, inplace=True)

revision['codechanges'].rename(columns={'Change in code': 'drop',

                                        'new code': 'add'}, inplace=True)

revised = pd.concat([revision['additions'], revision['deletions'], revision['codechanges']], ignore_index=True)

revised.sort_values(by='Date', inplace=True)

revised
orig_cets = cets.copy()

for row, rowData in revised.iterrows():

    if isinstance(rowData['add'], str):

        cets.loc[len(cets), 'Series Code'] = rowData['add']

    if isinstance(rowData['drop'], str):

        cets = cets.loc[cets['Series Code']!=rowData['drop']]

print(len(cets), cets['Series Code'].nunique())

# CETS contains duplicate series codes. 1273 unique series code altogether.
for code in cets['Series Code'].unique():

    if len(glob.glob(f'{WBANK}/API_{code}*.csv'))==0:

        url = f'http://api.worldbank.org/v2/country/all/indicator/{code}?source=2&downloadformat=csv'

        filename, hdr = urllib.request.urlretrieve(url)

        try:

            with zipfile.ZipFile(filename, 'r') as zf:

                zf.extractall(WBANK)

        except Exception as excep:

            print(code, excep)
def processCountry(df):

    df['Country/Region'].replace({'Taiwan*': 'Taiwan',

                                  'US'     : 'United States'}, inplace=True)

    df.loc[df['Province/State']=='Hong Kong', 'Country/Region'] = 'Hong Kong'

    df = df.loc[(df['Country/Region']!='Diamond Princess') & (df['Country/Region']!='MS Zaandam')]

    df = df[['Country/Region']].drop_duplicates()

    return df



def insertContinent(df):

    continent = pd.read_csv(f'{INP}/continent.csv')

    df = pd.merge(df, continent, on='Country/Region', how='left', validate='many_to_one')    

    print('Countries without <continent>:', df.loc[df['continent'].isnull(), 'Country/Region'].unique(), file=fp)

    return df



def redress(df):

    meta = df[['Indicator Code', 'Indicator Name']].drop_duplicates()

    meta.rename(columns={'Indicator Code': 'Name', 

                         'Indicator Name': 'Variable'}, inplace=True)

    assert len(meta) == 1

    assert 'Country Name' in df.columns and 'Indicator Code' in df.columns and 'Indicator Name' in df.columns

    df.rename(columns={'Country Name': 'Country',

                       'Indicator Code': 'Name', 

                       'Indicator Name': 'Variable'}, inplace=True)

    df.replace({ 'Bahamas, The'                  : 'Bahamas',

                 'Brunei Darussalam'             : 'Brunei',

                 'Myanmar'                       : 'Burma',

                 'Congo, Rep.'                   : 'Congo (Brazzaville)',

                 'Congo, Dem. Rep.'              : 'Congo (Kinshasa)',

                 'Czech Republic'                : 'Czechia',

                 'Egypt, Arab Rep.'              : 'Egypt',

                 'Gambia, The'                   : 'Gambia',

                 'Iran, Islamic Rep.'            : 'Iran',

                 'Korea, Dem. People’s Rep.'     : 'Korea, South',

                 'Kyrgyz Republic'               : 'Kyrgyzstan',

                 'Lao PDR'                       : 'Laos',

                 'Russian Federation'            : 'Russia',

                 'St. Kitts and Nevis'           : 'Saint Kitts and Nevis',

                 'St. Lucia'                     : 'Saint Lucia',

                 'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',

                 'Slovak Republic'               : 'Slovakia',

                 'Syrian Arab Republic'          : 'Syria',

                 'Taiwan, China'                 : 'Taiwan',

                 'Venezuela, RB'                 : 'Venezuela',

                 'Yemen, Rep.'                   : 'Yemen',

                 'Hong Kong SAR, China'          : 'Hong Kong',

                 'Sub-Saharan Africa'           : 'Western Sahara'}, inplace=True)

    df.set_index('Country', inplace=True)

    return df, meta



def insertWBANK(df):

    dfs = {}

    meta, data, record = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for nfile, file in enumerate(glob.glob(f'{WBANK}/API_*.csv')):

        nam = file.split('API_')[1].upper().split('_DS2_EN')[0]

        dff, tismeta = redress(pd.read_csv(file, skiprows=[0, 1, 2, 3]))

        dfs[nam] = dff

        meta = pd.concat([meta, pd.read_csv(file.replace('API', 'Metadata_Indicator_API'), index_col='INDICATOR_CODE')])

        assert tismeta['Name'].unique()[0].upper() == nam

    del meta['Unnamed: 4']

    count = 0

    for nam in dfs.keys():

        dff = dfs[nam]

        flag = False

        for country, countryData in dff.iterrows():

            tmp = ~countryData.isnull()

            year = tmp.index[np.where(tmp)[0].max()]

            if year.isnumeric():

                data.loc[country, countryData['Name']] = countryData[year]

                record.loc[country, countryData['Name']] = year

                flag = True

        if not flag:

            print('nothing from', nam, file=fp)

            meta = meta.loc[meta.index != nam]

        else:

            count += 1

    data.index.name = 'Country'

    record.index.name = 'Country'

    meta.to_csv(f'{OUT}/wbank01july_meta.csv')

    record.to_csv(f'{OUT}/wbank01july_year.csv')

    print('count, data.shape, meta.shape:', count, data.shape, meta.shape)



    df = pd.merge(data, df, how='right', left_on='Country', right_on='Country/Region', validate='one_to_many')

    nullRatio = pd.DataFrame()

    for col in meta.index:

        nullRatio.loc[col, 'INDICATOR_NAME'] = meta.loc[col, 'INDICATOR_NAME']

        nullRatio.loc[col, 'nullRatio'] = np.sum(df[col].isnull())/df.shape[0]

    #   print('{:.2f} {}'.format(np.sum(data[col].isnull())/data.shape[0], meta.loc[col, 'INDICATOR_NAME']))

    nullRatio.sort_values(by='nullRatio', inplace=True, ascending=False)

    nullRatio.index.name = 'INDICATOR_CODE'

    nullRatio.to_csv(f'{OUT}/wbank01july_nullRatio.csv')



    print('*** REMOVING CODES WHERE NULLRATIO EXCEEDS 20%:', df.shape, '-', len(nullRatio.index[nullRatio['nullRatio']>.2]), end=' ')

    df = df[df.columns ^ nullRatio.index[nullRatio['nullRatio']>.2].to_list()]

    print('=', df.shape)



    continentalMedian = df.groupby('continent').median()

    if continentalMedian.isnull().any().any():

        print('data not available for calculating continental median of', continentalMedian.isnull().any().sum(), 'topic codes')

        print('cols with null:', continentalMedian.isnull().sum(axis=1), file=fp)

        continentalMedian.to_csv(f'{OUT}/wbank01july_continentalMedian.csv')



        print('*** REMOVING CODES WHERE CONTINENTAL MEDIAN CANNOT BE CALCULATED:', df.shape, '->')

        df = df[df.columns ^ continentalMedian.columns[continentalMedian.isnull().any()]]

        print(df.shape)

        

    for col in df.columns[df.columns.str.contains(r'\.')]:    

        for continent, continentData in df.groupby('continent'):

            df.loc[continentData.index, col] = df.loc[continentData.index, col].fillna(continentalMedian.loc[continent, col])

        df[col] = df[col].astype('float')

    return df



if 'kid' in os.getcwd():

    df = pd.read_csv(f'{INP}/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', dtype={'Confirmed': 'int'})

else:

    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', dtype={'Confirmed': 'int'})



file = f'{OUT}/wbank01july_log.txt'

if os.path.exists(file):

    os.rename(file, file.replace('log', 'log_last'))

with open(file, 'w') as fp:

    df = (df.pipe(processCountry)

            .pipe(insertContinent)

            .pipe(insertWBANK))

if os.path.exists(file.replace('log', 'log_last')):

    os.system("diff {} {}".format(file, file.replace('log', 'log_last')))

assert df.isnull().any().any()==False



df.set_index('Country/Region', inplace=True)

df.to_csv(f'{OUT}/wbank01july_data.csv')

print(len(df.columns[df.columns.str.contains(r'\.')]))