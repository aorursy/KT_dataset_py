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
import pandas as pd

df = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
df.head()
df.pop('Unnamed: 0')
df.head(2)
import matplotlib.pyplot as plt

%matplotlib inline
df.dtypes
df.shape
df.isnull().sum()
df.info()
df.columns

df.loc[df.Deaths == max(df.Deaths )]
df.describe(include=[np.number])
df['Province/State'].unique()
df.Country.unique()
confirmed_con = df.loc[df.Confirmed >=1]

confirmed_con.head()
country_details  = pd.pivot_table(df,index=["Country"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
country_details 
# country_details.sort_values(by='Confirmed',ascending=False).head()
# country_details.sort_values(by='Deaths',ascending=False)
list(zip(country_details.index , country_details.Confirmed))
df.loc[df.Country == 'Australia'].agg('sum')[3:]
df.loc[df.Country == 'Nepal'].agg('sum')[3:]
df['Province/State'].unique()
pd.pivot_table(df,index=["Country","Province/State"] ,aggfunc=np.sum)
ind =[]

for index in range(df.shape[0]):

    

    if df.iloc[index][0]=='0':

        #print('Column Number : ', index)

        ind.append(index)
df.head()
df.loc[ind,'Province/State'] = df.loc[ind].Country
df[df.Country=='Nepal']
pd.pivot_table(df,index=["Country","Province/State"] ,aggfunc=np.sum)
#  country_details  - country
province_state_country = pd.pivot_table(df,index=["Province/State"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
province_state_country[:10]
# province_state_country[:5].plot(kind='pie', subplots=True, figsize=(100, 100))
province_state_country[:10].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12)
province_state_country[1:10].plot(kind='bar' ,figsize=(10, 4), width=2)
country_details[:5]
country_details[0:5].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=2)
country_details[1:6].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=1)
cpd = pd.pivot_table(df,index=["Country","Province/State",'Date last updated'] )
cpd
cpd.loc[('Australia', ), :]
aus = cpd.loc[('Australia', ), :]
aus.loc['Australia'].index
aus.loc['Australia'].sort_index()
aus.loc['Australia'].sort_index().plot.line()
# for china
china = df[df['Country'] == 'Mainland China']

china_d = pd.DataFrame(china.groupby(['Province/State'])['Confirmed','Suspected','Recovered','Deaths'].agg('sum')).reset_index()

china_d.head(35)
import seaborn as sns

china_d.sort_values(by=['Confirmed'], inplace=True,ascending=False)

plt.figure(figsize=(30,15))

plt.title("Patients Confirmed Infected by Corona Virus by States")

sns.barplot(x=china_d['Province/State'],y=china_d['Confirmed'],orient='v')

plt.ylabel("Confirmed Patients")