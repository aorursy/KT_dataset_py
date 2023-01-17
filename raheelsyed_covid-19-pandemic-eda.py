# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/ecdc-covid-data/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load ECDC current worldwide COVID-19 statistics (file saved from above link)

df = pd.read_excel('/kaggle/input/geodist/COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx')



# View the first five rows

df.head(5)
country = df['Countries and territories'].unique()



print("Number of countries Impacted: ",len(country))

print("Worldwide Cases Reported (as of March 20, 2020): ", df['Cases'].sum())

print("Worldwide Deaths Reported (as of March 20, 2020): ", df['Deaths'].sum())

print("Fatality Rate: "+"{:.2%}".format((df['Deaths'].sum()/df['Cases'].sum())))
df[df['Countries and territories']=='China']
#df2 = df[['Countries and territories','GeoId','DateRep','Cases']]

df2 = df[['Countries and territories','DateRep','Cases']]

df2.rename(columns={'Countries and territories': 'Country','DateRep': 'DateRp'}, inplace=True)

df2.head()
df2 = df2.drop_duplicates(subset=['Country','DateRp'])

df_pivot = df2.pivot(index='Country',columns='DateRp')

#df_pivot

#df2.pivot_table(df2, index=['Country','GeoId'],columns='DateRp')

df_pivot.columns = df_pivot.columns.to_flat_index()

df_pivot.columns = ['{}'.format(x[1]) for x in df_pivot.columns if x!= 'Country']

#from datetime import datetime

df_pivot.columns = [col.replace('00:00:00', '') for col in df_pivot.columns]

#df_pivot.rename(columns = lambda x: x.strip('00:00:00'))

#df.columns = [col[:-2] for col in df.columns if col[-2:]=='_x' else col]

df_pivot.fillna(0, inplace=True)

df_pivot.head()
geocodes = df[['Countries and territories','GeoId']]

geocodes.drop_duplicates(subset=None, keep="first", inplace=True)

geocodes.head()

merged_left = pd.merge(left=df_pivot, right=geocodes, how='left', left_on='Country', right_on='Countries and territories')

merged_left.head()
merged_left['url'] = 'https://www.countryflags.io/'+ merged_left.GeoId +'/flat/64.png'

merged_left['url']

#merged_left.head()

merged_left.shape

for i in range(0,len(merged_left)):

    for c in range(1,merged_left.shape[1]-3): 

        merged_left.iat[i, c] = merged_left.iat[i, c] + merged_left.iat[i, c-1]

        #print(df_tot.iat[i, c])

        #df_tot.set_value(i,c,val)
merged_left[merged_left['Countries and territories'] == 'China']
merged_left.to_excel('cleandata.xlsx')
import IPython

url = "https://public.flourish.studio/visualisation/1631776/"

iframe = '<iframe src=' + url + ' width=700 height=350></iframe>'

IPython.display.HTML(iframe)