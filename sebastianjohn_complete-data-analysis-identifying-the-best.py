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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/facebook-ad-campaign/data.csv')
df.shape
df.info()
print(list(df.columns))
df.head()
df.isnull().sum()
df= df.dropna()
df.shape
df = df.drop(['reporting_start','reporting_end','fb_campaign_id'], axis=1)
df.head()
df.gender.value_counts()
df.age.value_counts()
df.interest1.value_counts()
df.interest2.value_counts()
df.interest3.value_counts()
sns.countplot(df.age)
sns.countplot(df.gender)
sns.distplot(df.clicks)
sns.distplot(df.spent)
sns.countplot(df.approved_conversion)
# cost analysis

print('Campaign wise clicks')

print((df.groupby(['campaign_id'])).clicks.sum())

print('-------------------------')



print('Campaign wise amount spent')

print((df.groupby(['campaign_id'])).spent.sum())

print('--------------------------')





print('Campaign wise total conversions')

print((df.groupby(['campaign_id'])).total_conversion.sum())

print('---------------------------')



print('Campaign wise ad count')

print((df.groupby(['campaign_id'])).ad_id.count())

print('===========================')
campaign_1178_clicks = 9577

campaign_1178_cost = 16577.159998

campaign_1178_conv = 1050

campaign_1178_adcount = 243

campaign_1178_cpc = (campaign_1178_cost/campaign_1178_clicks)

campaign_1178_cpco = (campaign_1178_cost/campaign_1178_conv)

campaign_1178_cpad = (campaign_1178_cost/campaign_1178_adcount)



print('The cost per click of campaign_1178 is '+ str(campaign_1178_cpc))

print('The cost per conversion of campaign_1178 is '+ str(campaign_1178_cpco))

print('The cost per ad in campaign_1178 is '+ str(campaign_1178_cpad))

print('---------------------------------------------------------------')





campaign_936_clicks = 1984

campaign_936_cost = 2893.369999

campaign_936_conv = 537

campaign_936_adcount = 464

campaign_936_cpc = (campaign_936_cost/campaign_936_clicks)

campaign_936_cpco = (campaign_936_cost/campaign_936_conv)

campaign_936_cpad = (campaign_936_cost/campaign_936_adcount)



print('The cost per click of campaign_936 is '+ str(campaign_936_cpc))

print('The cost per conversion of campaign_936 is '+ str(campaign_936_cpco))

print('The cost per ad in campaign_936 is '+ str(campaign_936_cpad))

print('---------------------------------------------------------------')



campaign_916_clicks = 113

campaign_916_cost = 149.710001

campaign_916_conv = 58

campaign_916_adcount = 54

campaign_916_cpc = (campaign_916_cost/campaign_916_clicks)

campaign_916_cpco = (campaign_916_cost/campaign_916_conv)

campaign_916_cpad = (campaign_916_cost/campaign_916_adcount)



print('The cost per click of campaign_916 is '+ str(campaign_916_cpc))

print('The cost per conversion of campaign_916 is '+ str(campaign_916_cpco))

print('The cost per ad in campaign_916 is '+ str(campaign_916_cpad))

print('---------------------------------------------------------------')
dfn = df.query('campaign_id =="916"')

dfn.head()
dfm = df.query('campaign_id =="1178"')

dfm.head()
# gender analysis
print('Gender based analysis')

print((df.groupby(['gender'])).total_conversion.sum())

print((df.groupby(['gender'])).ad_id.count())

print((dfn.groupby(['gender'])).total_conversion.sum())

print((dfn.groupby(['gender'])).ad_id.count())

print((dfm.groupby(['gender'])).total_conversion.sum())

print((dfm.groupby(['gender'])).ad_id.count())
#age analysis
print((df.groupby(['age'])).total_conversion.sum())

print((df.groupby(['age'])).ad_id.count())

print((dfn.groupby(['age'])).total_conversion.sum())

print((dfn.groupby(['age'])).ad_id.count())

print((dfm.groupby(['age'])).total_conversion.sum())

print((dfm.groupby(['age'])).ad_id.count())
# Interests analysis
(dfn.groupby(['interest1'])).total_conversion.sum()
(dfn.groupby(['interest2'])).total_conversion.sum()
(dfn.groupby(['interest3'])).total_conversion.sum()
(dfm.groupby(['interest1'])).total_conversion.sum()
(dfm.groupby(['interest2'])).total_conversion.sum()
(dfm.groupby(['interest3'])).total_conversion.sum()