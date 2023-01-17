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
df_sch_water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')
df_sch_water.head()
df_sch_water.shape
df_sch_water.columns
df_sch_water.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(15.7,8.27)})
df_sch_water.groupby('State/UT')['All Schools'].mean()
df_sch_water.groupby('State/UT')['All Schools'].mean().plot(kind='bar')
df_sch_water[df_sch_water['State/UT']=='Arunachal Pradesh']
df_sch_water[df_sch_water['State/UT']=='Assam']
df_sch_water[df_sch_water['State/UT']=='Meghalaya']
df_sch_water[df_sch_water['State/UT']=='Tripura']
df_sch_gt = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
df_sch_gt.shape
df_sch_gt.isnull().sum()
df_sch_gt.head()
df_sch_gt.groupby('State_UT')['All Schools'].mean()
df_sch_gt.groupby('State_UT')['All Schools'].mean().plot(kind='bar')
df_sch_bt = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
df_sch_bt.head()
df_sch_bt.groupby('State_UT')['All Schools'].mean()
df_sch_bt.groupby('State_UT')['All Schools'].mean().plot(kind='bar')
df_sch_bt[df_sch_bt['State_UT']=='Andhra Pradesh']
df_sch_elec = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')
df_sch_elec.head()
df_sch_elec.groupby('State_UT')['All Schools'].mean()
df_sch_elec[df_sch_elec['State_UT']=='Assam']
df_sch_elec.groupby('State_UT')['All Schools'].mean().plot(kind='bar')
df_sch_comp = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')
df_sch_comp.head()
df_sch_comp.groupby('State_UT')['Primary_Only'].mean().plot(kind='bar')
df_sch_comp.groupby('State_UT')['Primary_with_U_Primary_Sec'].mean().plot(kind='bar')
df_sch_comp.groupby('State_UT')['Sec_with_HrSec.'].mean().plot(kind='bar')
df_sch_comp.groupby('State_UT')['All Schools'].mean().plot(kind='bar')
df_sch_er = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')
df_sch_er.head()
df_sch_er.info()
df_sch_er.groupby('State_UT')['Upper_Primary_Total'].mean().plot(kind='barh')
df_sch_dr = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')
df_sch_dr.head()
df_sch_dr.replace('NR',0,inplace=True)
df_sch_dr.replace('Uppe_r_Primary',0,inplace=True)
df_sch_dr['Upper Primary_Boys'].unique()
df_sch_dr['Primary_Boys'] = df_sch_dr['Primary_Boys'].astype(float)

df_sch_dr['Primary_Girls'] = df_sch_dr['Primary_Girls'].astype(float)

df_sch_dr['Primary_Total'] = df_sch_dr['Primary_Total'].astype(float)

df_sch_dr['Upper Primary_Girls'] = df_sch_dr['Upper Primary_Girls'].astype(float)

df_sch_dr['Upper Primary_Boys'] = df_sch_dr['Upper Primary_Boys'].astype(float)

df_sch_dr['Upper Primary_Total'] = df_sch_dr['Upper Primary_Total'].astype(float)

df_sch_dr['Secondary _Boys'] = df_sch_dr['Secondary _Boys'].astype(float)

df_sch_dr['Secondary _Girls'] = df_sch_dr['Secondary _Girls'].astype(float)

df_sch_dr['Secondary _Total'] = df_sch_dr['Secondary _Total'].astype(float)

df_sch_dr['HrSecondary_Boys'] = df_sch_dr['HrSecondary_Boys'].astype(float)

df_sch_dr['HrSecondary_Girls'] = df_sch_dr['HrSecondary_Girls'].astype(float)

df_sch_dr['HrSecondary_Total'] = df_sch_dr['HrSecondary_Total'].astype(float)

df_sch_dr.info()
df_sch_dr.groupby('State_UT')['Primary_Total'].mean().plot(kind='barh')
df_sch_dr.groupby('State_UT')['Upper Primary_Total'].mean().plot(kind='barh')
df_sch_dr.groupby('State_UT')['Secondary _Total'].mean().plot(kind='barh')
df_sch_dr.groupby('State_UT')['HrSecondary_Total'].mean().plot(kind='barh')