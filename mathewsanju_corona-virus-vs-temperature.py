import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind



sns.set(style="whitegrid", color_codes=True)

sns.set_palette("Set1")

%matplotlib inline
df = pd.read_excel('/kaggle/input/corona-data/Corona Data.xlsx')

df.head()
df_Cases = pd.DataFrame()

Tot_Cases = (100/df['Total Cases'])

df_Cases['Country'] = df['Country']

df_Cases['Recovered'] = Tot_Cases * df['Total Recovered']

df_Cases['Dead'] = Tot_Cases * df['Total Deaths']

df_Cases = df_Cases.fillna(0)

df_Cases['Active Cases'] = (Tot_Cases * (df['Total Cases'])) - (df_Cases['Recovered'] + df_Cases['Dead'])

df_Cases['Temp'] = df['Avg. Temp Feb 2020']

df_Cases = df_Cases.sort_values(by = ['Temp'], ascending = True)



plt.figure(figsize=(15,5))

con = np.arange(len(df_Cases['Country']))

barWidth = 1

p1 = plt.bar(con, df_Cases['Dead'], bottom=[i+j for i,j in zip(df_Cases['Recovered'], df_Cases['Active Cases'])], color='r', edgecolor='white', width=barWidth)

p2 = plt.bar(con, df_Cases['Recovered'], bottom=df_Cases['Active Cases'], color='g', edgecolor='white', width=barWidth)

p3 = plt.bar(con, df_Cases['Active Cases'], color='b', edgecolor='white', width=barWidth)



plt.ylabel('Distribution Percentage')

plt.title('Percentage of Deaths, Recovered and Active Cases sorted by average temperature of country', size = 15)

plt.xticks(con, df_Cases['Country'])

plt.xticks(rotation=90)

plt.legend((p1[0], p2[0], p3[0]), ('Deaths', 'Recovered', 'Active Cases'), loc='lower left')

print('Sorted by average temperature of country')
plt.figure(figsize=(15,5))

sns.barplot(df[df['Recovery Rate'] > 0]['Country'], df['Recovery Rate'], color='g')

plt.xticks(rotation=60)

print('Recovery Rate Vs Country')
plt.figure(figsize=(15,5))

sns.barplot(df[df['Death Rate'] > 0]['Country'], df['Death Rate'], color='r')

plt.xticks(rotation=60)

print('Death Rate Vs Country')
f, ax = plt.subplots(1, 3, sharey = True, figsize = (15,6))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)

ax[0].set_ylim(-10, 110)

ax[0].set_xlim(-15, 30)

ax[1].set_xlim(-15, 30)

ax[2].set_xlim(-15, 30)

ax[0].set_title("Deaths upon Total Cases", size = 15, weight = "bold")

ax[1].set_title("Recovered upon Total Cases", size = 15, weight = "bold")

ax[2].set_title("Recovered upon (Recovered + Deaths)", size = 15, weight = "bold")

sns.scatterplot(x=df['Avg. Temp Feb 2020'], y=df['Death Rate Tot'], color = 'r', ax = ax[0])

sns.scatterplot(x=df['Avg. Temp Feb 2020'], y=df['Recovery Rate Tot'], color = 'g', ax = ax[1])

sns.scatterplot(x=df['Avg. Temp Feb 2020'], y=df['Recovery Rate'], color = 'b', ax = ax[2])

print('Recovery Rate & Death Rate Vs Avg. Temperature')
corr = df[['Avg. Temp Feb 2020', 'Death Rate Tot', 'Recovery Rate Tot', 'Death Rate', 'Recovery Rate']].corr()

plt.figure(figsize=(12, 6))

plt.title('Pearson Correlation of attributes', y=1.05, size=19)

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

sns.heatmap(corr, mask=mask, cmap='YlGnBu', annot=True, linewidths=.5, fmt= '.2f', center = 1)
def box(Temp):

    global df_Temp

    df_Temp = pd.DataFrame()

    df_Temp = df_Temp.append(pd.DataFrame({'Status': 'Above', 'Recovery Rate':df[df['Avg. Temp Feb 2020'] > Temp]['Recovery Rate']}))

    df_Temp = df_Temp.append(pd.DataFrame({'Status': 'Below', 'Recovery Rate':df[df['Avg. Temp Feb 2020'] <= Temp]['Recovery Rate']}))
Threshold = [(0,0),(1,5),(2,10),(3,15),(4,20),(5,25)]



print('Recovery Rate Distribution')

f, ax = plt.subplots(1, len(Threshold), sharey=True, figsize = (15,6))

for i,j in Threshold:

    box(j)

    ax[i].set_title("Threshold %d°C"%(j), size = 15)

    sns.boxplot(y = df_Temp['Recovery Rate'], x = df_Temp.Status, ax = ax[i])
df_Recovery_Temp = pd.DataFrame()

for i in range(-10,28):

    box(i)

    Below = round(df_Temp[df_Temp['Status'] == 'Below']['Recovery Rate'].mean(),1)

    Above = round(df_Temp[df_Temp['Status'] == 'Above']['Recovery Rate'].mean(),1)

    df_Recovery_Temp = df_Recovery_Temp.append({'Temp':'%d°C'%(i), 'Below Mean':Below, 'Above Mean':Above}, ignore_index=True)

df_Recovery_Temp

plt.figure(figsize=(15, 6))

plt.title('Mean Recovery Rate based on Temperature', size = 15)

plt.xlim([0, 37])

plt.ylim([80, 105])

plt.xticks(rotation=45)

p1 = plt.plot(df_Recovery_Temp['Temp'], df_Recovery_Temp['Below Mean'], color = 'b')

p2 = plt.plot(df_Recovery_Temp['Temp'], df_Recovery_Temp['Above Mean'], color = 'r')

plt.legend((p1[0], p2[0]), ('Mean Recovery Rate below threshold temperature', 'Mean Recovery Rate above threshold temperature'), loc='lower left')
# Calculate p value

group1 = df[df['Avg. Temp Feb 2020'] <= 11]['Recovery Rate']

group2 = df[df['Avg. Temp Feb 2020'] > 11]['Recovery Rate']

print(' Mean recovery rate for patients from countries below or equal to 11°C:', round(group1.mean(),2),"\n",'Mean recovery rate for patients from countries above 11°C:', round(group2.mean(),2))



t_statistic, p_value = ttest_ind(group1.dropna(), group2.dropna())

print(' t-statistic is %1.6f' %(t_statistic), "\n", 'p-value is %1.6f' %(p_value))
# Calculate p value

group1 = df[df['Avg. Temp Feb 2020'] <= 20]['Recovery Rate']

group2 = df[df['Avg. Temp Feb 2020'] > 20]['Recovery Rate']

print(' Mean recovery rate for patients from countries below or equal to 20°C:', round(group1.mean(),2),"\n",'Mean recovery rate for patients from countries above 20°C:', round(group2.mean(),2))



t_statistic, p_value = ttest_ind(group1.dropna(), group2.dropna())

print(' t-statistic is %1.6f' %(t_statistic), "\n", 'p-value is %1.6f' %(p_value))