# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

forever=pd.read_csv('../input/the-demographic-rforeveralone-dataset/foreveralone.csv')

gune=pd.read_csv('../input/gun-deaths-in-the-us/guns.csv')





# I will use 3 different data for practicing.
gun=gune[gune.intent=='Suicide'] # we are looking for only suicides
gun.head()
data.rename(columns={'suicides/100k pop':'suicid_ratio',' gdp_for_year ($)':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capita','HDI for year':'HDI_for_year'},inplace=True)
data.info()
country_list=list(data['country'].unique()) # for each country

con_suicide_ratio = []



for i in country_list:

    x = data[data['country']==i]

    z = sum(x.suicid_ratio)/len(x) # we summit all the suicide ration and divide it countries numbers For finding mean.

    con_suicide_ratio.append(z)   # we can use also mean function here



con_percapita=[]

for i in country_list:           # with the same way I want to find each's GDP per capita

    a= data[data['country']==i]

    b=sum(a.gdp_per_capita)/len(a)

    con_percapita.append(b)



# I need a another DataFrame so I create it.

data2 = pd.DataFrame({'country_list': country_list,'con_suicide_ratio':con_suicide_ratio,'Gdp_per_capita':con_percapita})

new_index = (data2['con_suicide_ratio'].sort_values(ascending=False)).index.values

sorted_data = data2.reindex(new_index)
data2.head()
plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['country_list'], y=sorted_data['con_suicide_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('Countries')

plt.ylabel('Suicide Rate')

plt.title('Suicide Rate of Countries')

data.info()
country_list=list(data['country'].unique())

con_sui_75_over_ratio = []



for i in country_list:

    over = data[(data['country']== i)&(data.age == '75+ years')]  # over 75 years old people who suicide

    senieur=sum(over.suicid_ratio)/len(over)

    con_sui_75_over_ratio.append(senieur)



data2 = pd.DataFrame({'country_list': country_list,'con_suicide_ratio':con_suicide_ratio,'Gdp_per_capita':con_percapita,'over_75_suicide':con_sui_75_over_ratio})    

data2.head()
country_list=list(data['country'].unique())

femalelist=[]

malelist=[]

for i in country_list:

    femel=data[(data['country']== i)&(data.sex=='female')]  # sex= female

    mal=data[(data['country']== i)&(data.sex=='male')]      #sex=male

    women=femel.suicid_ratio.mean()

    femalelist.append(women)

    men=mal.suicid_ratio.mean()

    malelist.append(men)

    

    

data2['male_ratio']=malelist

data2['female_ratio']=femalelist
forever.head()
forever['bodyweight'].unique()

forever.loc[forever['bodyweight'] == 'Normal weight'] = 'Normal'

forever['bodyweight'].unique()
weig_count= Counter(forever.bodyweight)   # now I will look the other data.

mweig_count=weig_count.most_common(4)     #we count the bodyweight item so we will make a grafic for visuliasition.

x,y = zip(*mweig_count)

x,y = list(x),list(y)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Bodyweights type')

plt.ylabel('Frequency')

plt.title('Bodyweights&suicide attempts')
forever.head()
emp_count= Counter(forever.employment)

most_comun_emp= emp_count.most_common(5)

most_comun_emp

x,y = zip(*most_comun_emp)

x,y = list(x),list(y)

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Employes')

plt.ylabel('Frequency')

plt.title('Most common 5 employes who suicide')
data3=data2.head(5)
plt.figure(figsize=(15,10))

sns.lmplot(x="con_suicide_ratio", y="Gdp_per_capita", col="male_ratio", hue="female_ratio", data=data3,

           col_wrap=2, ci=None, palette="muted", height=4,

           scatter_kws={"s": 50, "alpha": 1})
data3
# Set up the matplotlib figure

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)



# Generate some sequential data



sns.barplot(x="country_list", y="con_suicide_ratio", palette="rocket",data=data3, ax=ax1)

ax1.axhline(0, color="k", clip_on=False)

ax1.set_ylabel("con_suicide_ratio")



# Center the data to make it diverging



sns.barplot(x="country_list", y="over_75_suicide", palette="vlag",data=data3, ax=ax2)

ax2.axhline(0, color="k", clip_on=False)

ax2.set_ylabel("over_75_suicide")



# Randomly reorder the data to make it qualitative



sns.barplot(x="country_list", y="female_ratio", palette="deep",data=data3, ax=ax3)

ax3.axhline(0, color="k", clip_on=False)

ax3.set_ylabel("female_ratio")



# Finalize the plot

sns.despine(bottom=True)

plt.setp(f.axes, yticks=[])

plt.tight_layout(h_pad=2)
data2.head()
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(data2.male_ratio, data2.female_ratio,

                hue="Gdp_per_capita", size="con_suicide_ratio",

                palette="ch:r=-.2,d=.3_r",

                sizes=(1, 8), linewidth=0,

                data=data2, ax=ax)
data2.head()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

sns.despine(left=True)

sns.distplot(data2.con_suicide_ratio, kde=False, color="b", ax=axes[0, 0])

sns.distplot(data2.over_75_suicide, hist=False, rug=True, color="r", ax=axes[0, 1])

sns.distplot(data2.male_ratio, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

sns.distplot(data2.female_ratio, color="m", ax=axes[1, 1])



plt.setp(axes, yticks=[])

plt.tight_layout()

data3


g = sns.catplot(x="country_list", y="con_suicide_ratio", data=data2,

                height=15, kind="bar", palette="muted")

g.despine(left=False)

g.set_ylabels("suicide ratio")

g.set_xticklabels(rotation=90)
data3


# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(15, 9))

sns.heatmap(data2.corr(), annot=True, linewidths=.5, ax=ax)
sns.jointplot(data2.male_ratio, data2.female_ratio,data= data2, kind="hex", color="#4CB391")
data2.head()
g = sns.lmplot(x="con_suicide_ratio", y="over_75_suicide", hue="country_list",

               truncate=True, height=10, data=data2)



# Use more informative axis labels than are provided by default

g.set_axis_labels("con_suicide_ratio", "over_75_suicide")
sns.set(style="ticks")



sns.pairplot(data2, hue="country_list")
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

ax = sns.scatterplot(x="Gdp_per_capita", y="con_suicide_ratio",

                     hue="male_ratio", 

                     palette=cmap, sizes=(10, 200),

                     data=data2)
data2.head()



sns.set(style="darkgrid")



g = sns.jointplot("male_ratio", "female_ratio", data=data2, kind="reg",

                   color="m", height=7)
gun.head()
f, ax = plt.subplots()

sns.despine(bottom=True, left=True)



# Show each observation with a scatterplot

sns.stripplot(x="month", y="place", hue="sex",

              data=gun, dodge=True, jitter=True,

              alpha=.25, zorder=1)



# Show the conditional means

sns.pointplot(x="month", y="place", hue="sex",

              data=gun, dodge=.532, join=False, palette="dark",

              markers="d", scale=.75, ci=None)



# Improve the legend 

handles, labels = ax.get_legend_handles_labels()

ax.legend( title="place",

          handletextpad=0, columnspacing=1,

          loc="lower right", ncol=3, frameon=True)