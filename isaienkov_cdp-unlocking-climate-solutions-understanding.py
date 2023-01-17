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
check_corporations = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/Full_Corporations_Response_Data_Dictionary.csv')



for index, row in check_corporations.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
check_corporations = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/Full_Corporations_Response_Data_Dictionary copy.csv')

for index, row in check_corporations.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
fccd_2020_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2020_Full_Climate_Change_Dataset.csv')

fccd_2020_df.head()
fccd_2019_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')

fccd_2019_df.head()
fccd_2018_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Climate Change/2018_Full_Climate_Change_Dataset.csv')

fccd_2018_df.head()
fwsd_2020_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2020_Full_Water_Security_Dataset.csv')

fwsd_2020_df.head()
fwsd_2019_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2019_Full_Water_Security_Dataset.csv')

fwsd_2019_df.head()
fwsd_2018_df = pd.read_csv('../input/cdp-unlocking-climate-solutions/Corporations/Corporations Responses/Water Security/2018_Full_Water_Security_Dataset.csv')

fwsd_2018_df.head()
check_corporations = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Responses/Full_Cities_Response_Data_Dictionary.csv')



for index, row in check_corporations.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
cities_2020_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2020_Full_Cities_Dataset.csv")

cities_2020_df.head()
cities_2019_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2019_Full_Cities_Dataset.csv")

cities_2019_df.head()
cities_2018_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Cities/Cities Responses/2018_Full_Cities_Dataset.csv")

cities_2018_df.head()
check = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Water Security/Corporations_Disclosing_to_CDP_Data_Dictionary.csv')



for index, row in check.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
check = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/Corporations_Disclosing_to_CDP_Data_Dictionary.csv')



for index, row in check.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
cdws_2018_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Water Security/2018_Corporates_Disclosing_to_CDP_Water_Security.csv')

cdws_2018_df.head()
cdws_2019_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Water Security/2019_Corporates_Disclosing_to_CDP_Water_Security.csv')

cdws_2019_df.head()
cdws_2020_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Water Security/2020_Corporates_Disclosing_to_CDP_Water_Security.csv')

cdws_2020_df.head()
cdcc_2018_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2018_Corporates_Disclosing_to_CDP_Climate_Change.csv')

cdcc_2018_df.head()
cdcc_2019_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2019_Corporates_Disclosing_to_CDP_Climate_Change.csv')

cdcc_2019_df.head()
cdcc_2020_df = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Corporations/Corporations Disclosing/Climate Change/2020_Corporates_Disclosing_to_CDP_Climate_Change.csv')

cdcc_2020_df.head()
check = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/Cities_Disclosing_to_CDP_Data_Dictionary.csv')



for index, row in check.iterrows():

    print(index, '.', row['field'], ': ', row['description'])
cities_2020_discl = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/2020_Cities_Disclosing_to_CDP.csv')

cities_2020_discl.head()
cities_2019_discl = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/2019_Cities_Disclosing_to_CDP.csv')

cities_2019_discl.head()
cities_2018_discl = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Cities/Cities Disclosing/2018_Cities_Disclosing_to_CDP.csv')

cities_2018_discl.head()
corporations = pd.concat([fccd_2020_df, fccd_2019_df, fccd_2018_df, fwsd_2020_df, fwsd_2019_df, fwsd_2018_df])

corporations
cities = pd.concat([cities_2020_df, cities_2019_df, cities_2018_df])

cities
corp_locations = pd.read_csv('/kaggle/input/cdp-unlocking-climate-solutions/Supplementary Data/Locations of Corporations/NA_HQ_public_data.csv')

corp_locations
svi_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/CDC Social Vulnerability Index 2018/SVI2018_US.csv")

svi_df.head()
us_cities_meta_df = pd.read_csv("../input/cdp-unlocking-climate-solutions/Supplementary Data/Simple Maps US Cities Data/uscities.csv")

us_cities_meta_df