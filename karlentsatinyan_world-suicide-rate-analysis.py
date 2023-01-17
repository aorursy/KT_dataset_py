import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



data = pd.read_csv("../input/master.csv")

file = data.copy()

file.drop('country-year', axis = 1, inplace = True)

file.head()
file.describe(include = "all")
file.info()
file[" gdp_for_year ($) "] = file[" gdp_for_year ($) "].str.replace(',' , '')

file['GDP for year'] = file[' gdp_for_year ($) '].astype("int64")

file['GDP per capita'] = file['gdp_per_capita ($)'].astype("int64")



file.drop([' gdp_for_year ($) ', 'gdp_per_capita ($)'], axis = 1, inplace = True)

file[["country", "sex", "generation", "age"]] = file[["country", "sex", "generation", "age"]].apply(lambda x: x.astype('category'))

file.info()
file.head()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



grouped_gen1=file.groupby("generation")['suicides_no'].sum()

x1=grouped_gen1.index.tolist()

y1=grouped_gen1.values.tolist()

ax1.pie(y1,labels=x1, autopct='%1.1f%%', shadow=True, pctdistance=0.7, textprops={'fontsize': 14})

ax1.set_title("SUICIDES", fontsize=18, color='k')

grouped_gen2=file.groupby("generation")['suicides/100k pop'].sum()

x2=grouped_gen2.index.tolist()

y2=grouped_gen2.values.tolist()

ax2.pie(y2,labels=x2, autopct='%1.1f%%', shadow=True, pctdistance=0.7, textprops={'fontsize': 14})

ax2.set_title("SUICIDES/100k pop", fontsize=18, color='k')

plt.tight_layout()

plt.draw()
file.hist(bins=10, color="cadetblue")

plt.tight_layout(rect=(0, 0, 3, 3)) 
ax1=plt.subplot(1,2,1)

file['population'].plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax1.set_xlabel("population")



ax2=plt.subplot(1,2,2)

np.log(file["population"]).plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax2.set_xlabel("logged population")

plt.suptitle("Population Distribution")

plt.show()
ax1=plt.subplot(1,2,1)

file['GDP for year'].plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax1.set_xlabel("GDP for year")



ax2=plt.subplot(1,2,2)

np.log(file["GDP for year"]).plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax2.set_xlabel("logged DGP for year")

plt.suptitle("Distribution for yearly GDP")

plt.show()
ax1=plt.subplot(1,2,1)

file['GDP per capita'].plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax1.set_xlabel("GDP per capita")



ax2=plt.subplot(1,2,2)

np.log(file["GDP per capita"]).plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax2.set_xlabel("logged GDP per capita")

plt.suptitle("Distribution for GDP per capita")

plt.show()
ax1=plt.subplot(1,2,1)

file['suicides/100k pop'].plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax1.set_xlabel("suicides/100k pop")



ax2=plt.subplot(1,2,2)

np.log(file["suicides/100k pop"]).plot.hist(bins=50, edgecolor="white", figsize=(12,6), range=[-5, 6])

ax2.set_xlabel("logged suicides/100k pop")

plt.suptitle("Distribution for suicides/100k pop")

plt.show()

ax1=plt.subplot(1,2,1)

file['suicides_no'].plot.hist(bins=50, edgecolor="white", figsize=(12,6))

ax1.set_xlabel("suicides_no")



ax2=plt.subplot(1,2,2)

np.log(file["suicides_no"]).plot.hist(bins=25, edgecolor="white", figsize=(12,6), range=[-0.5, 10])

ax2.set_xlabel("logged suicides_no")

plt.suptitle("Distribution for suicides_no")

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))

grouped_sex = file.groupby("sex")['suicides_no'].sum()

x = grouped_sex.index.tolist()

y = grouped_sex.values.tolist()

ax1.pie(y,labels=x, autopct='%1.1f%%', shadow=True, startangle=90, pctdistance=0.5, textprops={'fontsize': 14})

grouped_age = file.groupby("age")['suicides_no'].sum()

x = grouped_age.index.tolist()

y = grouped_age.values.tolist()

ax2.pie(y,labels=x, autopct='%1.1f%%', shadow=True, startangle=140, pctdistance=0.7, textprops={'fontsize': 14})

f.suptitle("SUICIDE DISTRIBUTION in GENDER and AGE GROUPS", fontsize=18, color = 'k')

f.subplots_adjust(top=0.92)

plt.tight_layout()

plt.show()
sns.set_style("whitegrid")

g=sns.factorplot("year", data=file, aspect=3, kind='count', color="steelblue")

g.set_xticklabels(rotation=60)  

plt.draw()
fig=plt.figure(figsize=(12,6))

file_no_zero=file.copy()

file_no_zero["year_no_zero"]=file_no_zero.loc[(file_no_zero["suicides_no"]!=0), "year"]

plt.hist(file_no_zero["year"], color='blue', bins=32)

plt.hist(file_no_zero["year_no_zero"], color='green', bins=32)

plt.legend(labels=["with zeros","without zeros"], loc="upper left")

plt.draw()
fig, ax = plt.subplots(figsize=(10, 6))

corr = file.corr()

hm = sns.heatmap(round(corr,2), annot = True, ax = ax, cmap = "coolwarm",fmt = '.2f',

                 linewidths = .05)

fig.subplots_adjust(top = 0.92)

fig.suptitle('Attributes Correlation Heatmap', fontsize = 20)

plt.show()
total_suicides = file.groupby("country")["suicides_no"].sum()

total_suicides_df = pd.DataFrame(total_suicides).sort_values(by = "suicides_no", ascending = False)

ind_list = list(total_suicides_df.index)

plt.figure(figsize=(10,40))

plt.subplots_adjust(left=-7, bottom=0.05, right=-3.8, top=1.05, wspace=-0.5, hspace=0.05)

ax=sns.barplot(y=ind_list, x=total_suicides_df.iloc[:,0].values, data=total_suicides_df, ci=None)

plt.xticks(fontsize=22)

plt.yticks(fontsize=25)



for p in ax.patches:

    width = p.get_width()

    plt.text(p.get_width(), p.get_y()+.6*p.get_height(),'{:.0f}'.format(width), ha='left', va='center', fontsize=25)
plt.figure(figsize=(20,6))

for c in ("Russian Federation", "United States", "Japan"):

    file_c=file[file['country']==c]

    file_c_grouped=file_c.groupby("year")[["suicides_no"]].sum()

    x=file_c_grouped.index.tolist()

    y=file_c_grouped["suicides_no"]

    plt.plot(x,y, linewidth=5, label=c)

    plt.yticks(np.arange(0,65000, step=10000), fontsize=12)

    plt.xticks(np.arange(1985,2016, step=1), fontsize=12)

    plt.xlabel("Years",fontsize=14)

    plt.ylabel("Number of suicides per 100k pop", fontsize=14)

plt.legend()

plt.show()
total_suicides_pop=file.groupby("country")["suicides/100k pop"].sum()

total_suicides_pop_df=pd.DataFrame(total_suicides_pop).sort_values(by="suicides/100k pop", ascending=False)

ind_list_pop = list(total_suicides_pop_df.index)

plt.figure(figsize=(10,40))

plt.subplots_adjust(left=-1.05, bottom=0.05, right=1.05, top=1.05, wspace=0.05, hspace=0.05)

ax=sns.barplot(y=ind_list_pop, x=total_suicides_pop_df.iloc[:,0].values, data=total_suicides_pop_df, ci=None)

plt.xticks(fontsize=22)

plt.yticks(fontsize=22)



for p in ax.patches:

    width = p.get_width()

    plt.text(p.get_width(), p.get_y()+.6*p.get_height(),'{:.0f}'.format(width), ha='left', va='center', fontsize=22)
plt.figure(figsize=(20,6))

for c in ["Russian Federation", "Lithuania", "Hungary", "United States"]:

    file_c=file[file['country']==c]

    file_c_grouped=file_c.groupby("year")[['suicides/100k pop']].sum()

    x=file_c_grouped.index.tolist()

    y=file_c_grouped['suicides/100k pop']

    plt.plot(x,y, linewidth=5, label=c)

    plt.yticks(np.arange(0,700, step=100), fontsize=12)

    plt.xticks(np.arange(1985,2017, step=1), fontsize=12)

    plt.xlabel("Years",fontsize=14)

    plt.ylabel("Number of suicides per 100k pop", fontsize=14)

plt.legend()

plt.show()