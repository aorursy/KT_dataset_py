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
##cleaning dataset and preparing data for exploration



filename = '/kaggle/input/starbucks-app-customer-reward-program-data/portfolio.json'

df_portfolio = pd.read_json(filename, lines = True)

#print (df_portfolio.head())



filename = '/kaggle/input/starbucks-app-customer-reward-program-data/profile.json'

df_profile =  pd.read_json(filename, lines = True)

#print (df_profile.head())



filename = '/kaggle/input/starbucks-app-customer-reward-program-data/transcript.json'

df_transcript = pd.read_json(filename, lines = True)

#df_transcript.head()



##dividing the value field in separate columns with offer id and amount

df_transcript['record'] = df_transcript.value.apply(lambda x: list(x.keys())[0])

df_transcript['id'] = df_transcript.value.apply(lambda x: list(x.values())[0])

df_transcript.drop(['value'], axis=1, inplace=True)

df_transcript.head()



##filtering out offer received from this dataframe 

df_offer_received = (df_transcript[df_transcript.event.isin(['offer received'])])

print("number of customers who received offers : " ,len(df_offer_received))

#df_offer_received.groupby('record_value').count().head(5)

##df_offer_received.head(10)



#merge this dataframe with df_portfolio to get the offer name

merged_data = df_offer_received.merge(df_portfolio).rename(columns = {'id':'offer_id'})

merged_data.groupby('offer_id').head(1)

#now getting demographical analysis, how many male/female customer received these offers

# this will require merging transcript and profile dataframe , after renaming the person and id fields resp.



#renaming person to customer id

df_offer_recd = df_offer_received.rename(columns = {'person' : 'customer_id'})

#print (df_offer_recd.head())



#renaming profile id to customer_id

df_profile = df_profile.rename(columns = {'id' : 'customer_id'})

#print (df_profile.head())



#merge both df to get the male/female count

merged_data_demo = df_offer_recd.merge(df_profile,on = 'customer_id')

#print (merged_data_demo.head())



##printing how many were females among offer received

import matplotlib.pyplot as plt

plot_demo = merged_data_demo.groupby('gender',as_index = False).count()

plot_demo

plot_demo.plot(x ='gender', y='customer_id', kind = 'bar')

plt.show()