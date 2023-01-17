# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r'/kaggle/input/traffic-violations-in-maryland-county/Traffic_Violations.csv')

df.head()
df.columns
sub_df = pd.DataFrame(zip(df['Article'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])

sub_df['alcohol'] = sub_df.alcohol.eq('Yes').mul(1)

sub_df['belts'] = sub_df.belts.eq('Yes').mul(1)

sub_df['fatal'] = sub_df.fatal.eq('Yes').mul(1)

sub_df.set_index('violation').describe()
sub_df1 = pd.DataFrame(zip(df['Violation Type'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['violation','alcohol','belts','fatal'])

sub_df1['alcohol'] = sub_df1.alcohol.eq('Yes').mul(1)

sub_df1['belts'] = sub_df1.belts.eq('Yes').mul(1)

sub_df1['fatal'] = sub_df1.fatal.eq('Yes').mul(1)

table1 = pd.pivot_table(sub_df1, values=['alcohol','belts','fatal'], columns='violation', aggfunc=np.mean)

table1
sub_df1 = pd.DataFrame(zip(df['Date Of Stop'],df['Article'],df['Violation Type'],df['Charge'],df['Alcohol'],df['Belts'],df['Fatal']), columns=['date','article','type','violation','alcohol','belts','fatal'])

sub_df1['alcohol'] = sub_df1.alcohol.eq('Yes').mul(1)

sub_df1['belts'] = sub_df1.belts.eq('Yes').mul(1)

sub_df1['fatal'] = sub_df1.fatal.eq('Yes').mul(1)

sub_df1 = sub_df1[(sub_df1.alcohol > 0) | (sub_df1.belts > 0) | (sub_df1.fatal > 0)]

table1 = pd.pivot_table(sub_df1, values=['alcohol','belts','fatal'], index=['article','type','violation'], aggfunc=np.sum)

table1
table1 = pd.DataFrame(table1, columns=['alcohol','belts','fatal'])

alc_df = table1[table1['alcohol']!=0]

alc_df = alc_df.sort_values('alcohol', ascending=False)

alc_df = pd.DataFrame(alc_df.alcohol,columns=['alcohol'])

alc_df.head(30)
belts_df = table1[table1['belts']!=0]

belts_df = belts_df.sort_values('belts', ascending=False)

belts_df = pd.DataFrame(belts_df.belts,columns=['belts'])

belts_df.head(30)
fatal_df = table1[table1['fatal']!=0]

fatal_df = fatal_df.sort_values('fatal', ascending=False)

fatal_df = pd.DataFrame(fatal_df.fatal,columns=['fatal'])

fatal_df.head(30)