import pandas as pd # data processing

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 1_2')



df1 = df_raw.copy()[3:23]

df1.columns=['Country','Males 2017', 'Females 2017', 'Total 2017','Males 2018', 'Females 2018', 'Total 2018']

df1 = df1.set_index('Country')



df1[['Total 2017','Total 2018']][:10].plot(kind='bar',figsize=(15,10),rot=0)

plt.xlabel('Country')

plt.ylabel('Number of foreign students')

plt.title('Number of foreign students by country in 2017-2018')

plt.show()
fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(15,10))

df1.loc['Italy'][['Females 2017', 'Females 2018']].plot(rot=0, kind='bar', x='LABEL', ax=ax, title=('Female students from Italy'))

df1.loc['Italy'][['Males 2017', 'Males 2018']].plot(rot=0,kind='bar', x='LABEL',  legend=False, ax=ax2, title='Male students from Italy', color='grey')

plt.show()
df2 = df_raw.copy()[32:]

df2.columns = df2.iloc[0]

df2.rename(columns={ df2.columns[0]: "Age" }, inplace=True)

df2 = df2[1:7].set_index('Age')

df2.columns.name='Country'

df2.plot(kind='bar', figsize=(15,10),rot=0)



plt.xlabel('Age group')

plt.ylabel('Number of foreign students')

plt.title('Number of foreign students by country and age in 2018')

plt.show()

df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 5')



## data preparation



df3 = df_raw.copy()[1:]

df3.columns = df3.iloc[0]

df3 = df3.set_index('Citizenship')

df3.columns.name='Type of Course'

df3=df3.iloc[1:21,[1,2,3]]

df3.rename(columns={'English specific purposes2':'English specific purposes','Other3':'Other'}, index={'Other countries4':'Other countries'},inplace=True)



## plotting



df3[:10].plot(kind='bar',figsize=(15,10),rot=0)

plt.xlabel('Country')

plt.ylabel('Number of foreign students')

plt.title('Foreign students by type of course followed and citizenship')

plt.show()
df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 6')



## data preparation



df4 = df_raw.copy()

df4.columns = df4.iloc[2]

df4.rename(columns={ df4.columns[0]: "Month" }, inplace=True)

df4 = df4.set_index('Month')

df4.columns.name='Gender'

df4=df4.iloc[3:15,[0,1]]



df4.plot.bar(figsize=(15,10),stacked=True,rot=0)

plt.xlabel('Month')

plt.ylabel('Number of foreign students')

plt.title('Number of English students per month and gender')

plt.show()



df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 9')



## data preparation



df5 = df_raw.copy()

df5.columns = df5.iloc[1]

df5 = df5.set_index('Citizenship')

df5=df5.iloc[2:22,[0,1,2,3,4,5]]

df5.columns.name='Age'

df5.rename(index={'Other countries2':'Other countries'},inplace=True)



## plotting



df5[:5].plot(kind='bar',figsize=(15,10),rot=0)

plt.xlabel('Country')

plt.ylabel('Total number of foreign student weeks')

plt.title('Total number of foreign student weeks by country and age group')

plt.show()
df_raw = pd.read_excel('/kaggle/input/News2019_042.xlsx','Table 12')



## data preparation



df6 = df_raw.copy()

df6.columns = df6.iloc[2]

df6.rename(columns={ df6.columns[0]: "Type of employment" }, inplace=True)

df6 = df6.set_index('Type of employment')

df6.columns.name='Gender'

df6=df6.iloc[3:6,[6,7]]







df6.plot(kind='bar',rot=0,figsize=(15,8))

plt.xlabel('Type of employment')

plt.ylabel('Number of staff members')

plt.title('Number of staff members by type of employment and gender')

plt.show()