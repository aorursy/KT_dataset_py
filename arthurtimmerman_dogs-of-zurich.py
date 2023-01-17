import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from datetime import date

from operator import itemgetter

import matplotlib.ticker as ticker



print("Setup complete")
dog17_filepath = "../input/dogs-of-zurich/20170308hundehalter.csv"

dog17data = pd.read_csv(dog17_filepath,index_col='HALTER_ID')

dog17data['ALTER'].dropna()

df = pd.read_csv(dog17_filepath,index_col='HALTER_ID')

df.dropna(subset = ["ALTER"], inplace=True)

print('nans dropped and ready to go')

df.head()
df.describe(include = 'all')
df.dtypes
# making a new df so as not to mess with the primary one

# goal = get rid of the one row with nan value for ALTER, although it does have dog geburtsjahr.

# the above problem caused graph to not work because length of the y and x axis did not match, so had to get rid of whole row.

df = pd.read_csv(dog17_filepath,index_col='HALTER_ID')

df.dropna(subset = ["ALTER"], inplace=True)

# then I made a dog age variable, to make it easier to show age.

df['dog_age'] = 2017 - df['GEBURTSJAHR_HUND']

# further made b and a variable to get columns needed for graph.

b = df['dog_age'][(df['dog_age'] >= 0) & (df['dog_age'] <= 25)]

# 'a' variable was changed so that x values for ALTER are sorted from youngest to oldest, to be able to later make regression line.



f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)},figsize=(15,10))



mean=b.mean()

median=b.median()

# mode=b.mode().get_values()[0]



sns.set_style("darkgrid")



sns.boxplot(b, ax=ax_box)

ax_box.axvline(mean, color='r', linestyle='--')

ax_box.axvline(median, color='g', linestyle='-')

# ax_box.axvline(mode, color='b', linestyle='-')



sns.distplot(a=b,hist=True, bins=np.arange(min(b)-0.5, max(b)+1, 1), ax = ax_hist)

ax_hist.axvline(mean, color='r', linestyle='--')

ax_hist.axvline(median, color='g', linestyle='-')

# ax_hist.axvline(mode, color='b', linestyle='-')



plt.xticks(np.arange(min(b), max(b)+1, 1))

plt.xlabel('Age Dog', fontsize=12)

plt.title("Distribution of dog age", fontsize=20)

plt.xlim(min(b), max(b) + 1)



plt.legend({'Mean':mean,'Median':median})

ax_box.set(xlabel='')

plt.show()
df['dog_age'] = 2017 - df['GEBURTSJAHR_HUND']

owner = df['ALTER']

s_owner = sorted(owner)

age = df['dog_age'][(df['dog_age'] >= 0) & (df['dog_age'] <= 25)]

sex = df['GESCHLECHT_HUND']



plt.figure(figsize=(15,5))

ax = sns.barplot(x=s_owner, y=age,hue=sex)



l = plt.legend()

l.get_texts()[0].set_text('Female dog')

l.get_texts()[1].set_text('Male dog')



plt.xlabel('Owner age', fontsize=12)

plt.ylabel('Dog age', fontsize=12)

plt.title('Average dog age per dog sex and owner age group', fontsize=20)



plt.show()
df['dog_age'] = 2017 - df['GEBURTSJAHR_HUND']

owner = df['ALTER']

owner_sex = df['GESCHLECHT']

s_owner = sorted(owner)

age = df['dog_age'][(df['dog_age'] >= 0) & (df['dog_age'] <= 25)]

sex = df['GESCHLECHT_HUND']



plt.figure(figsize=(15,5))

ax = sns.barplot(x=s_owner, y=age,hue=owner_sex)



l = plt.legend()

l.get_texts()[0].set_text('Female owner')

l.get_texts()[1].set_text('Male owner')



plt.xlabel('Owner age', fontsize=12)

plt.ylabel('Dog age', fontsize=12)

plt.title('Average dog age per owner sex and owner age group', fontsize=20)



plt.show()
# first, convert float column of stadtkreis to integers.

df['STADTKREIS'] = df['STADTKREIS'].fillna(0.0).astype(int)



# second, make separate dfs for all city districts.

district_1 = df[df.STADTKREIS == 1]

district_2 = df[df.STADTKREIS == 2]

district_3 = df[df.STADTKREIS == 3]

district_4 = df[df.STADTKREIS == 4]

district_5 = df[df.STADTKREIS == 5]

district_6 = df[df.STADTKREIS == 6]

district_7 = df[df.STADTKREIS == 7]

district_8 = df[df.STADTKREIS == 8]

district_9 = df[df.STADTKREIS == 9]

district_10 = df[df.STADTKREIS == 10]

district_11 = df[df.STADTKREIS == 11]

district_12 = df[df.STADTKREIS == 12]
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(nrows=4, ncols=3, figsize=(30,20))

plt.subplots_adjust(hspace=0.5)



# district 1 plot

d1_races = district_1['RASSE1'].value_counts()

d1_races = d1_races/sum(d1_races)

d1_races_percent = (d1_races*100).head(5)

ax1.bar(d1_races_percent.index , d1_races_percent)

ax1.set_title('District 1')

ax1.set_ylabel('% race in district')

ax1.set_xticklabels(d1_races_percent.index, rotation=30)



# district 2 plot

d2_races = district_2['RASSE1'].value_counts()

d2_races = d2_races/sum(d2_races)

d2_races_percent = (d2_races*100).head(5)

ax2.bar(d2_races_percent.index , d2_races_percent)

ax2.set_title('District 2')

ax2.set_ylabel('% race in district')

ax2.set_xticklabels(d2_races_percent.index, rotation=30)



# district 3 plot

d3_races = district_3['RASSE1'].value_counts()

d3_races = d3_races/sum(d3_races)

d3_races_percent = (d3_races*100).head(5)

ax3.bar(d3_races_percent.index , d3_races_percent)

ax3.set_title('District 3')

ax3.set_ylabel('% race in district')

ax3.set_xticklabels(d3_races_percent.index, rotation=30)



# district 4 plot

d4_races = district_4['RASSE1'].value_counts()

d4_races = d4_races/sum(d4_races)

d4_races_percent = (d4_races*100).head(5)

ax4.bar(d4_races_percent.index , d4_races_percent)

ax4.set_title('District 4')

ax4.set_ylabel('% race in district')

ax4.set_xticklabels(d4_races_percent.index, rotation=30)



# district 5 plot

d5_races = district_5['RASSE1'].value_counts()

d5_races = d5_races/sum(d5_races)

d5_races_percent = (d5_races*100).head(5)

ax5.bar(d5_races_percent.index , d5_races_percent)

ax5.set_title('District 5')

ax5.set_ylabel('% race in district')

ax5.set_xticklabels(d5_races_percent.index, rotation=30)



# district 6 plot

d6_races = district_6['RASSE1'].value_counts()

d6_races = d6_races/sum(d6_races)

d6_races_percent = (d6_races*100).head(5)

ax6.bar(d6_races_percent.index , d6_races_percent)

ax6.set_title('District 6')

ax6.set_ylabel('% race in district')

ax6.set_xticklabels(d6_races_percent.index, rotation=30)



# district 7 plot

d7_races = district_7['RASSE1'].value_counts()

d7_races = d7_races/sum(d7_races)

d7_races_percent = (d7_races*100).head(5)

ax7.bar(d7_races_percent.index , d7_races_percent)

ax7.set_title('District 7')

ax7.set_ylabel('% race in district')

ax7.set_xticklabels(d7_races_percent.index, rotation=30)



# district 8 plot

d8_races = district_8['RASSE1'].value_counts()

d8_races = d8_races/sum(d8_races)

d8_races_percent = (d8_races*100).head(5)

ax8.bar(d8_races_percent.index , d8_races_percent)

ax8.set_title('District 8')

ax8.set_ylabel('% race in district')

ax8.set_xticklabels(d8_races_percent.index, rotation=30)



# district 9 plot

d9_races = district_9['RASSE1'].value_counts()

d9_races = d9_races/sum(d9_races)

d9_races_percent = (d9_races*100).head(5)

ax9.bar(d9_races_percent.index , d9_races_percent)

ax9.set_title('District 9')

ax9.set_ylabel('% race in district')

ax9.set_xticklabels(d9_races_percent.index, rotation=30)



# district 10 plot

d10_races = district_10['RASSE1'].value_counts()

d10_races = d10_races/sum(d10_races)

d10_races_percent = (d10_races*100).head(5)

ax10.bar(d10_races_percent.index , d10_races_percent)

ax10.set_title('District 10')

ax10.set_ylabel('% race in district')

ax10.set_xticklabels(d10_races_percent.index, rotation=30)



# district 11 plot

d11_races = district_11['RASSE1'].value_counts()

d11_races = d11_races/sum(d11_races)

d11_races_percent = (d11_races*100).head(5)

ax11.bar(d11_races_percent.index , d11_races_percent)

ax11.set_title('District 11')

ax11.set_ylabel('% race in district')

ax11.set_xticklabels(d11_races_percent.index, rotation=30)



# district 12 plot

d12_races = district_12['RASSE1'].value_counts()

d12_races = d12_races/sum(d12_races)

d12_races_percent = (d12_races*100).head(5)

ax12.bar(d12_races_percent.index , d12_races_percent)

ax12.set_title('District 12')

ax12.set_ylabel('% race in district')

ax12.set_xticklabels(d12_races_percent.index, rotation=30)



fig.suptitle('Most popular dog races per district', fontsize=25)



plt.show()
df2 = df[['RASSE1','GEBURTSJAHR_HUND']].set_index('GEBURTSJAHR_HUND')



line_data = df2.groupby(['GEBURTSJAHR_HUND','RASSE1']).size().unstack()

line_data[np.isnan(line_data)] = 0



five = line_data[['Chihuahua', 'Mischling klein', 'Labrador Retriever', 'Jack Russel Terrier','Französische Bulldogge']]

plt.figure(figsize=(20,7))

sns.lineplot(data=five)

plt.xlabel('Year',fontsize=12)

plt.ylabel('Number of newly registered dogs', fontsize=12)

plt.title('Popularity of selected dog races over time - 1', fontsize=20)

plt.show()
df2 = df[['RASSE1','GEBURTSJAHR_HUND']].set_index('GEBURTSJAHR_HUND')



line_data = df2.groupby(['GEBURTSJAHR_HUND','RASSE1']).size().unstack()

line_data[np.isnan(line_data)] = 0



five = line_data[['Jack Russel Terrier', 'Mischling gross', 'Yorkshire Terrier', 'Border Collie','Malteser','Mops']]

plt.figure(figsize=(20,7))

sns.lineplot(data=five)

plt.xlabel('Year',fontsize=12)

plt.ylabel('Number of newly registered dogs', fontsize=12)

plt.title('Popularity of selected dog races over time - 2', fontsize=20)

plt.show()