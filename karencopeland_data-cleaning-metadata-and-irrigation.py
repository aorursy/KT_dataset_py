# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir('/kaggle/input/buildingdatagenomeproject2')

os.listdir()
meta = pd.read_csv('metadata.csv')

print(meta.shape)

meta.head()
meta.info()
test = meta.copy()

test[['eui', 'site_eui', 'source_eui']].head(20)
test['site_eui'] = test['site_eui'].replace('-', np.nan).astype('float64')

test['source_eui'] = test['source_eui'].replace('-', np.nan).astype('float64')

test['eui'] = test['eui'].str.replace(',', '').replace('-', np.nan).astype('float64')

test['source_eui'].unique()
meta[['building_id', 'building_id_kaggle', 'site_id', 'site_id_kaggle']].tail(20)
def binary(df, cols):

    for col in cols:

        df[col] = df[col].replace(np.nan, 0)

        df[col] = df[col].replace("Yes", 1)

    return df
bin_cols = ['electricity', 'hotwater', 'chilledwater', 'water', 'steam', 'irrigation', 'solar', 'gas']

test = binary(meta, bin_cols)
test[bin_cols].nunique()
test['energystarscore'] = meta['energystarscore'].replace('-', np.nan).astype('float64')

test['energystarscore'].unique()
meta['rating'].unique()
meta['leed_level'].unique()
test['date_opened'] = test['date_opened'].astype('datetime64[ns]')
test.dtypes
test.info()
import missingno as msno

msno.matrix(test);
test = test.drop(['date_opened', 'site_eui', 'source_eui'], axis=1)
test['heatingtype'].unique()
heating = pd.get_dummies(test['heatingtype'], drop_first=True, dtype='int64')

heating.head()
heating = heating.rename(columns={'Electric': 'Electric Heating', 

                                  'Electicity': 'Electricity Heating',

                                  'Gas': 'Gas Heating', 

                                  'Oil': 'Oil Heating', 

                                  'Steam': 'Steam Heating'})
heating.head()
primaryspaceusage = test['primaryspaceusage'].unique()

sub_primaryspaceusage = test['sub_primaryspaceusage'].unique()

industries = test['industry'].unique()

subindustries = test['subindustry'].unique()

print(primaryspaceusage)

print(industries)
industries = pd.DataFrame(test['industry'])

industries = industries.rename(columns={'industry': 'usage'})

subindustries = pd.DataFrame(test['subindustry'])

subindustries = subindustries.rename(columns={'subindustry': 'subusage'})

primaryspaceusage = pd.DataFrame(test['primaryspaceusage'])

primaryspaceusage = primaryspaceusage.rename(columns={'primaryspaceusage': 'usage'})

sub_primaryspaceusage = pd.DataFrame(test['sub_primaryspaceusage'])

sub_primaryspaceusage = sub_primaryspaceusage.rename(columns={'sub_primaryspaceusage': 'subusage'})

print(primaryspaceusage.isnull().sum())

print(sub_primaryspaceusage.isnull().sum())

print(industries.isnull().sum())

print(subindustries.isnull().sum())
combine_sub = subindustries.combine_first(sub_primaryspaceusage)

combine_sub.isnull().sum()
combine_sub.head(20)
combine_sub['subusage'].unique()
combine_primary = industries.combine_first(primaryspaceusage)

combine_primary.isnull().sum()
combine_primary.head(20)
(combine_primary['usage']=='Other').sum()
combine_primary['usage'].unique()
test = test.drop(['industry', 'subindustry', 'primaryspaceusage', 'sub_primaryspaceusage', 'heatingtype'], axis=1)

test = pd.concat([test, combine_primary, combine_sub, heating], axis=1)

test.head()
msno.matrix(test)
test = test.drop(['numberoffloors', 'occupants', 'energystarscore'], axis=1)
import math

latlong = test.copy()

for index, row in test.iterrows():

    if not (math.isnan(row['lat'])):

        latlong = latlong.drop(index)

latlong['building_id'].unique()
msno.matrix(test)
test.info()
kaggle = test[test['building_id_kaggle'].notna()]

kaggle = kaggle[kaggle['site_id_kaggle'].notna()]

msno.matrix(kaggle)
no_anonymous = kaggle[kaggle['lat'].notna()]

msno.matrix(no_anonymous)
#save as csv

test.to_csv('/kaggle/working/metadata_cleaned.csv', index=False)

kaggle.to_csv('/kaggle/working/metadata_kaggle_cleaned.csv', index=False)

no_anonymous.to_csv('/kaggle/working/metadata_kaggle_anonymous_cleaned.csv', index=False)
irr = pd.read_csv('irrigation_cleaned.csv')

irr.info()
irr.isnull().sum()
clean = irr.copy()
clean.head(20)
clean['timestamp'] = clean['timestamp'].astype('datetime64[ns]')
msno.matrix(clean)
clean.shape
times = clean['timestamp']

clean['timestamp'].unique()
vals = clean['Panther_lodging_Paulette']

clean['Panther_lodging_Paulette'].unique()
clean.plot.scatter(x='timestamp', y='Panther_lodging_Paulette', figsize=(20,10))
import matplotlib.pyplot as plt

plt.plot(times, vals, '-')

plt.show()
columns = irr.columns.tolist()

zeros = irr.copy()

zeros = zeros.replace(0, np.nan)

drop = [];

ii = 1;

while ii<len(columns):

    if zeros[columns[ii]].isnull().sum() == 17544:

        drop.append(columns[ii])

    ii = ii + 1

    

drop
clean = clean.drop(drop, axis=1)

msno.matrix(clean)
clean = clean.interpolate(method="slinear")

clean.isnull().sum()
msno.matrix(clean)
clean = clean.drop('Panther_lodging_Cora', axis=1)

clean.isnull().sum()
clean_drop = clean.drop(['Panther_lodging_Otis', 'Panther_office_Daina', 'Panther_education_Karri', 'Panther_parking_Lorriane'], axis=1)

msno.matrix(clean_drop)
clean = clean.fillna(method='ffill')

clean.isnull().sum()
msno.matrix(clean)
clean = clean.fillna(method = 'bfill')

clean.isnull().sum()
msno.matrix(clean)
clean.plot.scatter(x='timestamp', y='Panther_parking_Adela', figsize=(20,10))
clean.to_csv('/kaggle/working/interpolated_propogated_irrigation.csv', index=False)

clean.to_csv('/kaggle/working/no_propogation_irrigation.csv', index=False)