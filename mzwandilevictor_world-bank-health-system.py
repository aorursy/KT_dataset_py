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
import matplotlib as mlt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

%matplotlib inline
df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")
df.head()
## Data cleaning and manipulation
df.info()
df.describe()
df.shape
#The data have 210 rows and 14 columns
df.isna().sum()
## The data have missing values but except the world bank name column
df.tail()
df2 = df.drop(["Province_State"], axis = 1)
df2

# I have droped the Province_State column to have an accurate analysis has the column had 196 missing data out of 210
# check for duplications in the data 
df2.duplicated()
## The data has no duplication in it
df2.shape
# After i dropped the column the data shows that the data have 210 rows and 13 columns
df2.isnull().any()
df2.median()
# will be replacing every null value with the meadian value expect for Country_Region  column
df2["Country_Region"].fillna('no country_region', inplace = True)
df2
# For country_relion i have replaced all the null with no counry religion
df3 =df2.fillna(df2.median())
df3.isna().sum()
#They is no missing values in the data 
for column in df4:
    plt.figure()
    df4.boxplot([column])
## we can see that they is outliers in the data
# df4 is the dataset with numeric values
df4.head()

# will use heatmap

correlations = df3.corr()
sns.heatmap(data=correlations, square=True, cmap="bwr")

plt.yticks(rotation=0)
plt.xticks(rotation =90)
## we can see that they is more behavior on health_exp_per_capita_usd 2016 and per_capit_exp_ppp_2016, The red colour shows maximum correlation and 
## the blue shows the minimum correlation
### Gonna focus on the health_exp_per_capita_usd 2016 and per_capit_exp_ppp_2016 to further understand there relationship

df3[['Health_exp_per_capita_USD_2016', 'per_capita_exp_PPP_2016']].groupby(['per_capita_exp_PPP_2016']).describe().unstack()
## we can see the count and max of both columns grouped by per capita exp ppp 2016
df3[['Health_exp_per_capita_USD_2016', 'per_capita_exp_PPP_2016']].corr()
# looking for relationship against per capita xep ppp 2016

fig,axs = plt.subplots(1,5, sharey = True)
df3.plot(kind = "scatter", x = "Health_exp_per_capita_USD_2016", y = "per_capita_exp_PPP_2016", ax = axs[0],figsize=(22,10))
df3.plot(kind = "scatter", x = "External_health_exp_pct_2016", y = "per_capita_exp_PPP_2016", ax = axs[1])
df3.plot(kind = "scatter", x = "Health_exp_public_pct_2016", y="per_capita_exp_PPP_2016", ax=axs[2])
df3.plot(kind="scatter", x ="Health_exp_out_of_pocket_pct_2016", y = "per_capita_exp_PPP_2016", ax=axs[3])
df3.plot(kind = 'scatter', x = 'Health_exp_per_capita_USD_2016', y = "per_capita_exp_PPP_2016", ax = axs[4])
# the relationship against Specialist_surgical_per_1000_2008

fig,axs = plt.subplots(1,2, sharey = True)
df3.plot(kind = "scatter", x = "Physicians_per_1000_2009-18", y = "Specialist_surgical_per_1000_2008-18", ax = axs[0],figsize=(22,10), color = 'g')
df3.plot(kind = "scatter", x = "Nurse_midwife_per_1000_2009-18", y = "Specialist_surgical_per_1000_2008-18", ax = axs[1],color = 'g')
# we can see the relationships of each column against the one column
