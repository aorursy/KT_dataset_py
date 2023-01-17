import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import seaborn as sns

plt.style.use('fivethirtyeight')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/marathon_results_2016.csv')
def time_to_min(string):

    if string is not '-':

        time_segments = string.split(':')

        hours = int(time_segments[0])

        mins = int(time_segments[1])

        sec = int(time_segments[2])

        time = hours*60 + mins + np.true_divide(sec,60)

        return time

    else:

        return -1



print(time_to_min(df.loc[1,'Half']))
df['Half_min'] = df.Half.apply(lambda x: time_to_min(x))

df['Full_min'] = df['Official Time'].apply(lambda x: time_to_min(x))

df['split_ratio'] = (df['Full_min'] - df['Half_min'])/(df['Half_min'])

df_split = df[df.Half_min > 0]

sns.kdeplot(df_split.split_ratio)

plt.xlim([0.7,1.7])

plt.xlabel('Split Ratio')

plt.title('Split Distribution (Negative split when < 1)')
plt.plot(df_split.Overall,df_split.split_ratio,'o', alpha = 0.2)

plt.ylim([0.5,3])

plt.xlabel('Overall Rank')

plt.ylabel('Split')

plt.title('Split and performance')
sns.distplot(df_split.split_ratio[df_split['M/F'] == 'M'], np.arange(0.6,3,0.01))

sns.distplot(df_split.split_ratio[df_split['M/F'] == 'F'],np.arange(0.6,3,0.01))

plt.legend(['Males','Females'])

plt.xlim([0.5,2])
median = np.median(df_split.Age)

sns.distplot(df_split.split_ratio[df_split.Age > median],np.arange(0.6,3,0.01))

sns.distplot(df_split.split_ratio[df_split.Age < median],np.arange(0.6,3,0.01))

plt.xlim([0.5,2])

plt.legend(['Age >' + str(median), 'Age <' + str(median)])
sns.distplot(df_split.split_ratio[df_split.Age >50],np.arange(0.6,3,0.01))

sns.distplot(df_split.split_ratio[df_split.Age <30],np.arange(0.6,3,0.01))

plt.xlim([0.5,2])

plt.legend(['Age > 50', 'Age < 30'])
sns.distplot(df_split.split_ratio[df_split.Overall < 1000],np.arange(0.6,3,0.01))

sns.distplot(df_split.split_ratio[df_split.Overall > 10000],np.arange(0.6,3,0.01))

plt.legend(['Overall <1000','Overall>10000'])

plt.xlim([0.5,2])
plt.plot(df_split.Overall[(df_split.Country is not 'ETH') & (df_split.Country is not 'KEN') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country is not 'ETH') & (df_split.Country is not 'KEN') & (df_split.Overall<100)],'o')

plt.plot(df_split.Overall[(df_split.Country == 'ETH') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country == 'ETH')  & (df_split.Overall<100)],'o', color = 'r')

plt.plot(df_split.Overall[(df_split.Country == 'KEN') & (df_split.Overall<100)], df_split.split_ratio[(df_split.Country == 'KEN')  & (df_split.Overall<100)],'o', color = 'r')

plt.xlabel('Overall Rank')

plt.ylabel('Split Ratio')

plt.legend(['Others','Kenya and Ethiopia'])
df['5K_mins'] = df['5K'].apply(lambda x: time_to_min(x))

df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))

df['10K_mins'] = df['10K_mins'] - df['5K_mins'] 

df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))

df['15K_mins'] = df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))

df['20K_mins'] = df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))

df['25K_mins'] = df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))

df['30K_mins'] = df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))

df['35K_mins'] = df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))

df['40K_mins'] = df['40K_mins'] -  df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']
columns = ['40K_mins','35K_mins','30K_mins','25K_mins','20K_mins','15K_mins','10K_mins','5K_mins']

df['avg'] = df[columns].mean(axis = 1)

df['stdev'] = df[columns].std(axis = 1)

df_split = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))]

plt.plot(df_split.avg,df_split.stdev,'o', alpha = 0.1)

plt.ylim([0,30])

plt.xlim(15,45)

plt.xlabel('Average Time for 5K along the race')

plt.ylabel('Standard Deviation of segments pace')

plt.title('Faster runners are also more stable')
df_split['10_dif'] = df_split['10K_mins'] - df_split['5K_mins'] 

df_split['15_dif'] = df_split['15K_mins'] - df_split['10K_mins']

df_split['20_dif'] = df_split['20K_mins'] - df_split['15K_mins']

df_split['25_dif'] = df_split['25K_mins'] - df_split['20K_mins']

df_split['30_dif'] = df_split['30K_mins'] - df_split['25K_mins']

df_split['35_dif'] = df_split['35K_mins'] - df_split['30K_mins']

df_split['40_dif'] = df_split['40K_mins'] - df_split['25K_mins']
sns.distplot(df_split['10_dif'],np.arange(-5,10,0.08), kde = False)

sns.distplot(df_split['20_dif'],np.arange(-5,10,0.08), kde = False)

sns.distplot(df_split['30_dif'],np.arange(-5,10,0.08), kde = False)

sns.distplot(df_split['40_dif'],np.arange(-5,10,0.08), kde = False)

plt.legend(['10K','20K','30K','40K'])

plt.xlabel('Slowing Down Compared to previous segment')
sns.distplot(df_split['25_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['30_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['35_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['40_dif'],np.arange(-5,10,0.05), kde = False)

plt.xlim([-5,10])

plt.legend(['25K','30K','35K','40K'])

plt.xlabel('Slowing Down Compared to previous segment')
sns.distplot(df_split['10_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['15_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['20_dif'],np.arange(-5,10,0.05), kde = False)

sns.distplot(df_split['25_dif'],np.arange(-5,10,0.05), kde = False)

plt.xlim([-5,10])

plt.legend(['10K','15K','20K','25K'])

plt.xlabel('Slowing Down Compared to previous segment')