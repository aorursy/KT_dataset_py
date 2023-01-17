#Load the Librarys

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))

Kick2018= pd.read_csv('../input/ks-projects-201801.csv')

print(Kick2018.shape)

print(Kick2018.info())

Kick2018.head(n=10)


sns.set(style="whitegrid")

state_count = Kick2018.state.value_counts()

source_data = pd.DataFrame(state_count).reset_index()

print(source_data)

fig=sns.barplot(x="index", y="state" , data=source_data, palette="GnBu_d")

fig.set(xlabel='State', ylabel='Campaigns')

fig.grid(False)
categ=Kick2018.main_category.value_counts()

cdata = pd.DataFrame(categ).reset_index()

print(cdata)

figs=sns.barplot(x="index", y="main_category" , data=cdata, palette="GnBu_d")

figs.grid(False)

figs.set(xlabel='', ylabel='Campaigns')

figs.set_xticklabels(figs.get_xticklabels(),rotation=90)
failed = Kick2018.loc[Kick2018.state=='failed']

successful = Kick2018.loc[Kick2018.state=='successful']

canceled = Kick2018.loc[Kick2018.state=='canceled']

suc_categ=successful.main_category.value_counts()

suc_data = pd.DataFrame(suc_categ).reset_index()

print(suc_data)

figss=sns.barplot(x="index", y="main_category" , data=suc_data, palette="GnBu_d")

figss.grid(False)

figss.set(xlabel='', ylabel='Campaigns')

figss.set_title('Number of Successful Campaigns in The Main Categories')

figss.set_xticklabels(figss.get_xticklabels(),rotation=90)
suc_top=successful.category.value_counts()[:20]

suct_data = pd.DataFrame(suc_top).reset_index()

print(suct_data)

figss=sns.barplot(x="index", y="category" , data=suct_data, palette="GnBu_d")

figss.grid(False)

figss.set(xlabel='', ylabel='Campaigns')

figss.set_title('Top 20 successful categories')

figss.set_xticklabels(figss.get_xticklabels(),rotation=90)
successful = Kick2018.loc[(Kick2018.state == 'successful') & (Kick2018.main_category == 'Technology')]

succ_data = pd.DataFrame(successful.usd_pledged_real).reset_index()

succ_data1 = pd.DataFrame(successful.category).reset_index()

x=succ_data.usd_pledged_real

y=succ_data1.category





f, axes= plt.subplots(1,2,figsize=(26, 4))



successful2 = Kick2018.loc[(Kick2018.state == 'successful') & (Kick2018.category == 'Video Games')]

succ_data2 = pd.DataFrame(successful2.usd_pledged_real).reset_index()

succ_data3 = pd.DataFrame(successful2.category).reset_index()

x2=succ_data2.usd_pledged_real

y2=succ_data3.category



successful = Kick2018.loc[(Kick2018.state == 'successful')]

groupby_main_category = successful.groupby(['main_category']).mean()

mean= pd.DataFrame(groupby_main_category.usd_goal_real).reset_index()

mean_ord=mean.sort_values(by="usd_goal_real",ascending=False)

print(mean_ord)

x=mean.main_category

y=mean.usd_goal_real

figz=sns.barplot(x="main_category", y="usd_goal_real" , data=mean_ord, palette="GnBu_d")

figz.grid(False)

figz.set(xlabel='', ylabel='Average Pledged (USD)')

figz.set_xticklabels(figz.get_xticklabels(),rotation=90)

figz.set_title('Average USD Pledged Among Successful Campaigns')





means= pd.DataFrame(groupby_main_category.usd_goal_real).reset_index()

mean_ord=means.sort_values(by="usd_goal_real",ascending=False)

print(mean_ord)

x=mean.main_category

y=mean.usd_goal_real

figz=sns.barplot(x="main_category", y="usd_goal_real" , data=mean_ord, palette="GnBu_d")

figz.grid(False)

figz.set(xlabel='', ylabel='Average Pledged (USD)')

figz.set_xticklabels(figz.get_xticklabels(),rotation=90)

figz.set_title('Average USD Pledged Among Successful Campaigns')
sum_b= pd.DataFrame(groupby_main_category.backers).reset_index()



print(sum_b)

sum_back=sum_b.sort_values(by="backers",ascending=False)

print(sum_back)

figzs=sns.barplot(x="main_category", y="backers" , data=sum_back, palette="GnBu_d")

figzs.grid(False)

figzs.set_xticklabels(figzs.get_xticklabels(),rotation=90)

figzs.set_title('Average Number of Backers Among Successful Campaigns')

groupby_main_category = successful.groupby(['main_category']).mean()

groupby_main_category.backers