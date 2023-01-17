# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
population = pd.read_csv("../input/for-watercup-analysis/maharastra population districtwise - Sheet1.csv")

Winners = pd.read_csv("/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv")

print("Population dataset")

print("--------------------")

print(population.head())

print()

print('Winners dataset')

print('----------------')

print(Winners.head())
districts = Winners['District'].unique()

print("Districts which have got top 3 from 2016 to 2019")

print("-----------------------------------------------")

print(districts)
winning = population[population['District'].isin(districts)]

winning
notwinning = pd.concat([population,winning]).drop_duplicates(keep=False)

notwinning
plt.figure(figsize=(16,16))

plt.subplot(231)

y=[winning['population_2001'].mean(),notwinning['population_2001'].mean()]

x=['Districts which Won','District which not Won']

plot=sb.barplot(x,y);

sb.despine(left=True,bottom=True);

v=0

for i in (y):

    plt.text(v-0.2,i+10000, str(round(i,2)));

    v=v+1

plot.set_yticks([]);

plot.set_title('average population in 2001');



plt.subplot(232)

y=[winning['population_2011'].mean(),notwinning['population_2011'].mean()]

x=['Districts which Won','District which not Won']

plot=sb.barplot(x,y);

sb.despine(left=True,bottom=True);

v=0

for i in (y):

    plt.text(v-0.2,i+10000, str(round(i,2)));

    v=v+1

plot.set_yticks([]);

plot.set_title('average population in 2011');



plt.subplot(233)

y=[winning['Area(sq km)'].mean(),notwinning['Area(sq km)'].mean()]

x=['Districts which Won','District which not Won']

plot=sb.barplot(x,y);

sb.despine(left=True,bottom=True);

v=0

for i in (y):

    plt.text(v-0.2,i+100, str(round(i,2)));

    v=v+1

plot.set_yticks([]);

plot.set_title('average area in Sq km');



plt.subplot(234)

y=[winning['literacy'].mean(),notwinning['literacy'].mean()]

x=['Districts which Won','District which not Won']

plot=sb.barplot(x,y);

sb.despine(left=True,bottom=True);

v=0

for i in (y):

    plt.text(v-0.2,i+1, str(round(i,2)));

    v=v+1

plot.set_yticks([]);

plot.set_title('average literacy rates');



plt.subplot(235)

y=[winning['Sex ratio(per 1000 boys)'].mean(),notwinning['Sex ratio(per 1000 boys)'].mean()]

x=['Districts which Won','District which not Won']

plot=sb.barplot(x,y);

sb.despine(left=True,bottom=True);

v=0

for i in (y):

    plt.text(v-0.2,i+10, str(round(i,2)));

    v=v+1

plot.set_yticks([]);

plot.set_title('Sex ratio(per 1000 boys)');
plt.figure(figsize=(18,18))

plt.subplot(221)

avgar = winning['Area(sq km)'].mean()

clrs = ['grey' if (x < avgar) else 'coral' for x in winning['Area(sq km)']]

plot=sb.barplot(winning['District'],winning['Area(sq km)'],palette=clrs);

i=0;

for y in (winning['Area(sq km)']):

    plt.text(i-0.3,y+100, str(round(y,1)));

    i=i+1; 

sb.despine(left=True,bottom=True);

plt.text(7,15000,str('Average='+str(avgar)));

plot.set_yticks([]);

plt.title('Area for every District in winning dataset');



plt.subplot(222)

avgar = winning['literacy'].mean()

clrs = ['grey' if (x < avgar) else 'coral' for x in winning['literacy']]

plot=sb.barplot(winning['District'],winning['literacy'],palette=clrs);

i=0;

for y in (winning['literacy']):

    plt.text(i-0.3,y+1, str(round(y,2)));

    i=i+1; 

sb.despine(left=True,bottom=True);

plt.text(1,85,str('Average='+str(round(avgar,2))));

plot.set_yticks([]);

plt.title('literacy for every District in winning dataset');





plt.subplot(212)

winning['increase'] = ((winning['population_2011']-winning['population_2001'])/winning['population_2001'])*100;

avg = winning['increase'].mean()

winning = winning.sort_values('District')

ax=sb.lineplot(winning['District'],winning['increase'], marker='o');

sb.despine(left=True,bottom=True);

i=0;

for y in (winning['increase']):

    if(y>avg):

        plt.text(i-0.1,y+0.2, str(round(y,2)),fontsize=12,color='coral');

    else:

        plt.text(i-0.1,y+0.2, str(round(y,2)),fontsize=12,color='grey');

    i=i+1; 

ax.set_yticks([]);

plt.text(6,15,'Average='+str(round(avg,2)),fontsize=12);

plt.title('Increase in population rate from 2001 to 2011');