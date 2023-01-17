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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



pd.set_option('display.max_rows', None)
df=pd.read_csv("../input/singapore/singapore-residents.csv")

df.head()
#checking for missing value 

len(df[df['value']=='na'])
df_total=df[(df['value']!='na') & (~df['level_2'].isin(['65 Years & Over','70 - 74 Years','75 - 79 Years','75 Years & Over','80 - 84 Years','80 Years & Over','85 Years & Over']))]
df_total['value']=df_total['value'].astype(int)
df_total.reset_index(drop=True,inplace=True)
df_ethnic=df_total[df_total['level_1'].isin(['Total Indians','Total Chinese','Total Malays','Other Ethnic Groups (Total)'])]
df_1=df_ethnic[df_ethnic['year']==2018]

df_1.groupby(['level_1']).sum().plot(kind='pie',y='value',figsize=(10,7), autopct='%1.1f%%')

plt.title("Proportion of ethnicity ")
df_chinese=df_ethnic[df_ethnic['level_1']=='Total Chinese']

df_chinese.groupby('year')['value'].mean().plot(kind='line',figsize=(10,7))

plt.ylabel('avg.population')

plt.title('Population trend 1957-2018')
df_1.groupby(['level_2']).sum().plot(kind='pie',y='value',figsize=(11,9), autopct='%1.1f%%')

plt.title('Proportion of population by Age Groups')
df_age=df_ethnic[df_ethnic['level_2'].isin(['45 - 49 Years', '50 - 54 Years'])]

plt.figure(figsize=(10,7))

sns.lineplot(df_age['year'],df_age['value'],hue=df_age['level_2'])

plt.title('Population Growth of high density Age groups')
def growth_rate1(past_year,present_year,diff_years):

   # new list for growth rates

    

    growth_rate = []

   # for population in list

    for pop in range(0, len(past_year)):

        gnumbers = round(((present_year[pop] - past_year[pop]) * 100.0 / past_year[pop]) / diff_years,3)

        growth_rate.append(gnumbers)



    return growth_rate    



x=df_ethnic[df_ethnic['year'].isin([1957,2018])]

y_1957=x[x['year']==1957]

y_2018=x[x['year']==2018]
yr_1957=list(y_1957.groupby('level_2')['value'].sum())

yr_2018=list(y_2018.groupby('level_2')['value'].sum())

growth_rate=growth_rate1(yr_1957,yr_2018,2018-1957)
groups=['0  -  4 Years', '10 - 14 Years', '15 - 19 Years',

       '20 - 24 Years', '25 - 29 Years', '30 - 34 Years', '35 - 39 Years',

       '40 - 44 Years', '45 - 49 Years', '5  -  9 Years', '50 - 54 Years', '55 - 59 Years',

       '60 - 64 Years', '65 - 69 Years', '70 Years & Over']


r1 =[0.1,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  #--x

bars1 = growth_rate #--y

plt.figure(figsize=(18,9))

sns.barplot(groups,growth_rate,ci=None,palette='CMRmap_r')

for i in range(len(groups)):

    plt.text(x = r1[i]-0.3 , y = bars1[i]+0.2, s = growth_rate[i], size = 11)
et_1957=list(y_1957.groupby('level_1')['value'].sum())

et_2018=list(y_2018.groupby('level_1')['value'].sum())

growth_rate_et=growth_rate1(et_1957,et_2018,2018-1957)
labels=['Other Ethnic Groups (Total)','Total Chinese','Total Indians','Total Malays']
r1 =[0.1,1,2,3]  #--x

bars1 = growth_rate_et #--y

plt.figure(figsize=(9,7))

sns.barplot(labels,growth_rate_et,ci=None,palette='RdGy_r')

for i in range(len(labels)):

    plt.text(x = r1[i]-0.2 , y = bars1[i]+0.05, s = growth_rate_et[i], size = 11)
df_gender=df_total[df_total['level_1'].isin(['Total Male Residents','Total Female Residents'])]

y=df_gender[df_gender['year'].isin([1957,2018])]

g_1957=y[y['year']==1957]

g_2018=y[y['year']==2018]
gd_1957=list(g_1957.groupby('level_1')['value'].sum())

gd_2018=list(g_2018.groupby('level_1')['value'].sum())

growth_rate_gd=growth_rate1(gd_1957,gd_2018,2018-1957)
cat=['Female Residents','Male Residents']
r1 =[0.1,1]  #--x

bars1 = growth_rate_gd #--y

plt.figure(figsize=(8,6))

sns.barplot(cat,growth_rate_gd,ci=None)

for i in range(len(cat)):

    plt.text(x = r1[i]-0.2 , y = bars1[i]+0.05, s = growth_rate_gd[i], size = 11)
plt.figure(figsize=(10,8))

sns.lineplot(df_ethnic['year'],df_ethnic['value'],hue=df_ethnic['level_1'])
df_male=df_total[df_total['level_1'].isin(['Total Male Malays','Total Male Chinese','Total Male Indians','Other Ethnic Groups (Males)'])]

df_female=df_total[df_total['level_1'].isin(['Total Female Malays','Total Female Chinese','Total Female Indians','Other Ethnic Groups (Females)'])]
fig, axs = plt.subplots(1,2,figsize=(14,9))

plt.figure(figsize=(10,8))



sns.barplot(x=df_male['value'],y=df_male['level_2'],hue=df_male['level_1'],ax=axs[0])



sns.barplot(x=df_female['value'],y=df_female['level_2'],hue=df_female['level_1'],ax=axs[1])