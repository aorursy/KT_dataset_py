import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import numpy as np

import seaborn as sns

from matplotlib.patches import Polygon

from matplotlib import colors as colorm

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

import datetime

from dateutil.relativedelta import relativedelta

import re

import itertools

from datetime import timedelta



plt.style.use('seaborn')



calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

calendar["date"] = pd.to_datetime(calendar["date"])
def hist_percentage_zero_values(sales_train_validation):



    sales_train_id_stacked = sales_train_validation.set_index(['id']).drop(['item_id','store_id','dept_id', 'cat_id', 'state_id'],axis=1).T

    sales_train_id_stacked = sales_train_id_stacked.merge(calendar[["date","year","d"]],right_on="d",left_index=True).set_index(["date","year"]).drop("d",axis=1)

    percent_of_zeros = np.abs(sales_train_id_stacked.clip(0,1).groupby(level="year").mean() -1)

    

    names = [2011,2012,2013,2014,2015,2016]

    fig, ax = plt.subplots(6, 2,sharex="col",gridspec_kw={'width_ratios': [2, 0.5]},figsize=(15, 10),sharey='col')

    for i,t in enumerate(names):

        zero_tmp = percent_of_zeros.loc[t,:]

        ax_dist = ax[i,0]

        ax[i, 0].set_yscale('log')

        ax[i, 0].hist(zero_tmp,bins=60,color="lightsteelblue")

        ax[i, 0].set_ylabel(str(t))

        vals = ax[i, 0].get_xticks()

        ax[i, 0].set_xticklabels(['{:,.0%}'.format(x) for x in vals])

        



        zero_share = pd.Series(index=["Inactive products","Active products"])

        zero_share["Inactive products"] = zero_tmp.loc[zero_tmp==1].count()/zero_tmp.count()

        zero_share["Active products"] = 1 - zero_share["Inactive products"]

        ax[i, 1].pie(zero_share, autopct='%1.0f%%', pctdistance=1.5,textprops={'fontsize': 10},colors=["lightsteelblue","cornflowerblue"])

#"peru","brown"



    plt.title("Histogram of percentage of zero values",y=7.3,x=-4)

    plt.legend(labels=zero_share.index,bbox_to_anchor=(0.9,0.04), loc="lower right",

                    bbox_transform=fig.transFigure, ncol=1, fontsize=11)

    

    plt.tight_layout()

    return plt.show()

hist_percentage_zero_values(sales_train_validation)
# drop all zero values before the first value >=1

sales_train_id_stacked = sales_train_validation.set_index(['id']).drop(['item_id','store_id','dept_id', 'cat_id', 'state_id'],axis=1).T.astype("float")



pre_start_period = ((sales_train_id_stacked>0).cumsum()==0)



sales_train_id_stacked.iloc[pre_start_period] = np.nan



# drop also all values from 25th december (all shops are closed)

christmas_days = calendar.query('(date.dt.month==12)&(date.dt.day==25)')["d"]

sales_train_id_stacked.drop(christmas_days,inplace=True)

#sales_train_id_stacked.loc[sales_train_id_stacked.index.isin(christmas_days)] = np.nan



sales_train_id_stacked = sales_train_id_stacked.T



sales_train_validation_active = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].merge(sales_train_id_stacked,left_on="id",right_index=True)





# Histogram of percentage of zero values only of active products

hist_percentage_zero_values(sales_train_validation_active)



group_names_prod = sales_train_validation["cat_id"].unique()

group_names_prod.sort()

group_size_prod = sales_train_validation["cat_id"].value_counts().sort_index().reset_index(drop=True)

group_names_prod = [(group_names_prod[i] + " n= "+ str(group_size_prod[i]//10)) for i in range(0,3)]



subgroup_names_prod = [1,2,3,1,2,1,2]

subgroup_size_prod = sales_train_validation["dept_id"].value_counts().sort_index().reset_index(drop=True)

# number of sales per category

sales_train_id_stacked = sales_train_validation_active.set_index(['item_id','store_id','dept_id', 'cat_id']).drop(['id', 'state_id'],axis=1).T

# sales_train_categories = sales_train_id_stacked.mean().unstack(level=[2,3]).sum().sort_values().reset_index()

group_names_sales = sales_train_id_stacked.columns.get_level_values(3).unique().values

group_names_sales.sort()

group_size_sales = sales_train_id_stacked.mean().unstack(level=3).sum().sort_index().reset_index(drop=True)

group_names_sales = [(group_names_sales[i] + " n= "+ str(int(group_size_sales[i]//10))) for i in range(0,3)]



subgroup_names_sales = [1,2,3,1,2,1,2]

subgroup_size_sales = sales_train_id_stacked.mean().unstack(level=2).sum().sort_index().reset_index(drop=True)//10





# Create colors

a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]



# First Ring (outside)

fig, ax = plt.subplots(ncols=2,figsize=(15, 9),gridspec_kw={'width_ratios': [2, 2]})

ax[0].axis('equal')

mypie = list([0,1])

mypie[0], _ = ax[0].pie(group_size_prod, radius=1.3, labels=group_names_prod, colors=[a(0.6), b(0.8), c(0.6)])

ax[1].axis('equal')

mypie[1], _ = ax[1].pie(group_size_sales, radius=1.3, labels=group_names_sales, colors=[a(0.6), b(0.8), c(0.6)])

plt.setp( mypie[0], width=0.3, edgecolor='white')

plt.setp( mypie[1], width=0.3, edgecolor='white')





# Second Ring (Inside)

# fig, ax = plt.subplots(ncols=2)

# ax[0].axis('equal')

mypie2 = list([0,1])

mypie2[0], _ = ax[0].pie(subgroup_size_prod, radius=1.3-0.3,

labels=subgroup_names_prod, labeldistance=0.7, colors=[a(0.5), a(0.4),

a(0.3), b(0.5), b(0.4), c(0.6), c(0.5)])



mypie2[1], _ = ax[1].pie(subgroup_size_sales, radius=1.3-0.3,

labels=subgroup_names_sales, labeldistance=0.7, colors=[a(0.5), a(0.4),

a(0.3), b(0.5), b(0.4), c(0.6), c(0.5)])

# ax[0].title.set_text("Number of offered items in one store by category and department",pad=1.08)

# ax[1].title.set_text("Average number of daily sales in one store by category and department",pad=1.08)

ax[0].set_title("Number of offered items in one \n store by category and department\n(n= {})".format(group_size_prod.sum()//10))

ax[1].set_title("Average number of daily sales in one \n store by category and department\n(n= {})".format(int(group_size_sales.sum()//10)))



plt.setp( mypie2[0], width=0.4, edgecolor='white')

plt.setp( mypie2[1], width=0.4, edgecolor='white')

plt.margins(0,0)

plt.show()
sales_train_id_stacked = sales_train_validation_active.set_index(['item_id','store_id','dept_id', 'cat_id']).drop(['id', 'state_id'],axis=1).T

sales_train_id_stacked = sales_train_id_stacked.merge(calendar[["date","year","month","d"]],right_on="d",left_index=True).set_index(['date',"year","month"]).drop("d",axis=1)

sales_train_id_stacked.columns = pd.MultiIndex.from_tuples(sales_train_id_stacked.columns,names=['item_id','store_id', 'dept_id','cat_id'])

sales_train_id_stacked = sales_train_id_stacked.groupby(level=["year","month"],axis=0).mean().groupby(level=["store_id","dept_id"],axis=1).sum()

sales_train_id_stacked.index = ['-'.join(map(str,i)) for i in sales_train_id_stacked.index.tolist()]

store_id = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2','WI_3']

a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

colors = [a(0.7), a(0.5), a(0.9),b(0.9), b(0.6), c(0.9), c(0.6)]

#colors=[a(0.5), a(0.4),a(0.3), b(0.5), b(0.4), c(0.6), c(0.5)]

fig, axes = plt.subplots(10, figsize=(15,40),sharey=True)

for s, ax in zip(store_id,axes):

    store_sales = sales_train_id_stacked.loc[:,s]

    total_sales = store_sales.sum(axis=1)

    prop_sales = store_sales.div(total_sales,axis=0)

    # second axis

    ax2 = ax.twinx()

    #ax.bar(x=prop_sales.index, height=prop_sales["FOODS_1"])

    ax2 = prop_sales.plot(kind="bar",stacked=True,color = colors,ax=ax2,legend=False,alpha=0.7)

    ax.plot(total_sales)

    ax.set_xticklabels([t if (i%12==0) else "" for i,t in enumerate(prop_sales.index)]

,rotation=70)

    ax.set_ylabel("Average daily sales")

    ax.set_title(s)

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40),

          ncol=3, fancybox=True, shadow=True)

#plt.legend(ax2, fontsize=12, ncol=4, framealpha=0, fancybox=True)

#fig.legend(labels=prop_sales.columns, loc="lower center", ncol=4)

#plt.xticks(prop_sales.index,rotation=90)



plt.show()
sales_train_id_stacked = sales_train_validation_active.set_index(['cat_id','id']).drop(['item_id','store_id','dept_id', 'state_id'],axis=1).T

sales_train_id_stacked = sales_train_id_stacked.merge(calendar[["date","d"]],right_on="d",left_index=True).set_index(["date"]).drop("d",axis=1)

sales_train_id_stacked.columns = pd.MultiIndex.from_tuples(sales_train_id_stacked.columns,names=['cat_id','id'])

#drop products which are active in the first 60 days of start

sales_train_id_stacked = sales_train_id_stacked.loc[:,(slice(None),(sales_train_id_stacked.droplevel(level=0,axis=1).iloc[:60].sum()==0))]

base_colors = [plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

years = [2011,2012,2013]

fig, ax = plt.subplots(3,figsize=(18, 20))

for i1,c in enumerate(["FOODS","HOBBIES","HOUSEHOLD"]):

    color_cat = base_colors[i1]

    color_cat = [color_cat(0.5), color_cat(0.7), color_cat(0.9)]

    cat_sales = sales_train_id_stacked.loc[:,c]

    active = list()

    for i2,t in enumerate(years):

        color = color_cat[i2]

        item_names = cat_sales.loc[(cat_sales.index.year==t),(~cat_sales.columns.isin(active))].isna()

        item_names = item_names.loc[:,item_names.sum()<len(item_names)].columns

        mean_sales = cat_sales.loc[(cat_sales.index.year==(t+1)),item_names].mean()+1

        item_sales = ((cat_sales.loc[:,item_names]+1).div(mean_sales)-1).rolling(360).mean().median(axis=1)

        item_sales = item_sales.iloc[(334+364*(i2+1)):]

        item_sales.plot.line(ax=ax[i1],color=color)

        active.extend(item_names)

    ax[i1].set_xlabel("")

    ax[i1].legend(years)

    ax[i1].set_title(c)

fig.suptitle("Relative change in sales by year of introduction",size=20,y=0.91)

plt.show()
sales_train_id_stacked = sales_train_validation_active.set_index(['id','cat_id']).drop(['item_id', 'state_id','store_id','dept_id', ],axis=1).T

last_time_sold =(sales_train_id_stacked.loc[::-1,:]>0).cumsum()[::-1]

last_time_sold = last_time_sold==0

last_time_sold = last_time_sold.sum().reset_index()

last_time_sold = pd.pivot_table(last_time_sold,values=0,aggfunc='mean',columns="cat_id",index="id")

last_time_sold.reset_index(drop=True,inplace=True)

last_time_sold +=1



q75 = last_time_sold.sum(axis=1).quantile(0.75)

q90 = last_time_sold.sum(axis=1).quantile(0.90)

q99 = last_time_sold.sum(axis=1).quantile(0.99)

q999 = last_time_sold.sum(axis=1).quantile(0.999)



a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

plt.figure(figsize=(18, 10))

plt.hist([last_time_sold["FOODS"].dropna(),last_time_sold["HOBBIES"].dropna(),last_time_sold["HOUSEHOLD"].dropna()] ,bins = 10 ** np.linspace(0, np.log10(2000), 50),

         color= [a(0.6), b(0.8), c(0.6)],label=("FOODS","HOBBIES","HOUSEHOLD"))

plt.axvline(q75, color='k', linestyle='dashed', linewidth=1)

plt.text(q75*1.1, 2000, '75% quantile',rotation=90)

plt.axvline(q90, color='k', linestyle='dashed', linewidth=1)

plt.text(q90*1.1, 2000, '90% quantile',rotation=90)

plt.axvline(q99, color='k', linestyle='dashed', linewidth=1)

plt.text(q99*1.1, 2000, '99% quantile',rotation=90)

plt.axvline(q999, color='k', linestyle='dashed', linewidth=1)

plt.text(q999*1.1, 2000, '99.9% quantile',rotation=90)

plt.legend(prop={'size': 11})

# plt.plot([q75, q75], [0, 1500], 'k-', lw=2,)

# plt.legend([last_time_sold["FOODS"].dropna(),last_time_sold["HOBBIES"].dropna(),last_time_sold["HOUSEHOLD"].dropna()],["FOODS","HOBBIES","HOUSEHOLD"],loc='upper left')

plt.gca().set_xscale("log")

plt.yscale('log')

plt.xlabel("days")

plt.ylabel("number of item ID's")

plt.title('Histogram of days a item was last time sold')

plt.show()



sales_train_id_stacked = sales_train_validation_active.set_index(['item_id', 'cat_id']).drop(['id','dept_id', 'store_id', 'state_id'],axis=1).T



average_product_sale = sales_train_id_stacked.mean().reset_index()

average_product_sale = pd.pivot_table(average_product_sale,values=0,index="item_id",columns="cat_id")

average_product_sale.reset_index(drop=True,inplace=True)



a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

q50 = average_product_sale.sum(axis=1).quantile(0.5)

q75 = average_product_sale.sum(axis=1).quantile(0.75)

q90 = average_product_sale.sum(axis=1).quantile(0.90)

q99 = average_product_sale.sum(axis=1).quantile(0.99)

q999 = average_product_sale.sum(axis=1).quantile(0.999)

plt.figure(figsize=(18, 10))

plt.hist([average_product_sale["FOODS"].dropna(),average_product_sale["HOBBIES"].dropna(),average_product_sale["HOUSEHOLD"].dropna()]

         ,bins = 70, range=(0,6),

         color= [a(0.6), b(0.8), c(0.6)],label=("FOODS","HOBBIES","HOUSEHOLD"))

# = 10 ** np.linspace(0, np.log10(100), 50),

plt.axvline(q50, color='k', linestyle='dashed', linewidth=1)

plt.text(q50+0.1, 80, '50% quantile',rotation=90)

plt.axvline(q75, color='k', linestyle='dashed', linewidth=1)

plt.text(q75+0.1, 80, '75% quantile',rotation=90)

plt.axvline(q90, color='k', linestyle='dashed', linewidth=1)

plt.text(q90+0.1, 80, '90% quantile',rotation=90)

# plt.axvline(q99, color='k', linestyle='dashed', linewidth=1)

# plt.text(q99*1.1, 70, '99% quantile',rotation=90)

# plt.axvline(q999, color='k', linestyle='dashed', linewidth=1)

# plt.text(q999*1.1, 70, '99.9% quantile',rotation=90)



# plt.gca().set_xscale("log")

# plt.yscale('log')



plt.legend(prop={'size': 11})

plt.xlabel("average number of daily sales")

plt.ylabel("number of item ID's")

plt.title('Histogram of average number of daily sales')

plt.show()
sales_train_id_stacked = sales_train_validation_active.set_index(['item_id','store_id', 'cat_id']).drop(['id','dept_id', 'state_id'],axis=1).T

mean_sales = sales_train_id_stacked.groupby(level=[0,2],axis=1).mean().mean()

fig, ax = plt.subplots(nrows= 3,figsize=(18, 20))

for i,c in enumerate(["FOODS","HOBBIES","HOUSEHOLD"]):

    q = mean_sales.loc[mean_sales.index.get_level_values("cat_id")==c].quantile(0.50)

    top_prod = mean_sales.loc[(mean_sales.index.get_level_values("cat_id") == c) & (mean_sales >= q)].index.get_level_values("item_id")

    shop_corr = sales_train_id_stacked.loc[:,top_prod].groupby(level=["item_id"],axis=1).apply(lambda x:x.corr().droplevel("item_id"))

    shop_corr_mean = shop_corr.groupby(level="store_id", axis=1).mean().droplevel(1)

    mask = np.zeros_like(shop_corr_mean)

    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(shop_corr_mean, mask=mask,annot=False,vmin=.04,vmax=.23, square=True,ax=ax[i])

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

    ax[i].set_title(c)
mean_sales_hobbies = mean_sales.loc[mean_sales.index.get_level_values(1)=="HOBBIES"]

mean_sales_hobbies = mean_sales_hobbies.loc[mean_sales_hobbies>mean_sales_hobbies.quantile(0.50)].sort_values()



mean_sales_foods = mean_sales.loc[mean_sales.index.get_level_values(1)=="FOODS"].sort_values()

mean_sales_household = mean_sales.loc[mean_sales.index.get_level_values(1)=="HOUSEHOLD"].sort_values()





# replace duplicated index values with the next possible previous value

def replace_duplicate(index_with_dup):

    index = []

    for i in index_with_dup:

        if i not in index:

            index.append(i)

        else:

            while i in index:

                i-=1

            index.append(i)

    return index



index_foods = np.searchsorted(mean_sales_foods,mean_sales_hobbies,side="left")

index_foods = replace_duplicate(index_foods)

index_household = np.searchsorted(mean_sales_household,mean_sales_hobbies,side="left")

index_household = replace_duplicate(index_household)



mean_sales_foods = mean_sales_foods.iloc[index_foods]

mean_sales_household = mean_sales_household.iloc[index_household]



a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

plt.hist([mean_sales_foods,mean_sales_hobbies,mean_sales_household],color= [a(0.6), b(0.8), c(0.6)],bins=20)

plt.legend(["FOODS","HOBBIES","HOUSEHOLD"])

plt.xlabel("average number of daily sales")

plt.title("Histogram of average number of daily sales (sanity check)")

plt.show()
item_id = pd.DataFrame(columns=["FOODS","HOBBIES","HOUSEHOLD"])

item_id["FOODS"] = mean_sales_foods.index.get_level_values("item_id")

item_id["HOUSEHOLD"] = mean_sales_household.index.get_level_values("item_id")

item_id["HOBBIES"] = mean_sales_hobbies.index.get_level_values("item_id")





fig, ax = plt.subplots(nrows= 3,figsize=(18, 20))

for i,c in enumerate(["FOODS","HOBBIES","HOUSEHOLD"]):

    prod = item_id[c]

    shop_corr = sales_train_id_stacked.loc[:,prod].groupby(level=["item_id"],axis=1).apply(lambda x:x.corr().droplevel("item_id"))

    shop_corr_mean = shop_corr.groupby(level="store_id", axis=1).mean().droplevel(1)

    mask = np.zeros_like(shop_corr_mean)

    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(shop_corr_mean, mask=mask,annot=False,vmin=.04,vmax=.23, square=True,ax=ax[i])

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

    ax[i].set_title(c)
sales_train_id_stacked = sales_train_validation_active.set_index(['item_id','state_id','store_id', 'dept_id']).drop(['id','cat_id'],axis=1).T

mean_sales = sales_train_id_stacked.groupby(level=[0,1,2,3],axis=1).mean().mean()

# rolling mean corr

dept_id = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1',

       'FOODS_2', 'FOODS_3']

for i,c in enumerate(dept_id):

    q = mean_sales.loc[mean_sales.index.get_level_values(3)==c].quantile(0.50)

    top_prod = mean_sales.loc[(mean_sales.index.get_level_values(3) == c) & (mean_sales >= q)].index.get_level_values(0)

    shop_corr = sales_train_id_stacked.loc[:,top_prod].groupby(level=["item_id"],axis=1).apply(lambda x:x.corr().droplevel("item_id"))

    shop_corr_mean = shop_corr.groupby(level=["state_id","store_id"], axis=1).mean().droplevel(2)

    mask = np.zeros_like(shop_corr_mean)

    mask[np.triu_indices_from(mask)] = True

    mask = mask==1

    shop_corr_mean.iloc[mask] = np.nan

    shop_corr_mean = shop_corr_mean.stack().stack()

    shop_corr_mean.index = shop_corr_mean.index.set_names(['state_id1','store_id1','store_id2','state_id2'])

    if i==0:

        shop_corr_mean_subcat = pd.DataFrame(columns=dept_id,index=shop_corr_mean.index)

    shop_corr_mean_subcat.loc[shop_corr_mean.index,c] = shop_corr_mean

    
#shop_corr_mean_subcat.mean(axis=1).sort_values()

for i,s in enumerate(["CA","TX","WI"]):

    if i==0:

        state_group_corr = shop_corr_mean_subcat.loc[(s,slice(None),slice(None),s)].sort_values("FOODS_3",ascending=False)

    else:

        state_group_corr = state_group_corr.append(shop_corr_mean_subcat.loc[(s,slice(None),slice(None),s)].sort_values("FOODS_3",ascending=False))



tmp = state_group_corr.droplevel([0,3])

tmp.index = ['-'.join(map(str,i)) for i in tmp.index.tolist()]

tmp = tmp.stack()

# Create colors

a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

colors = [b(0.9), b(0.6), c(0.9), c(0.6),a(0.7), a(0.5), a(0.9)]

markers = ("o", "o","^", "^","s","s","s")

cat_groups = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1',

       'FOODS_2', 'FOODS_3']

fig, ax = plt.subplots(figsize=(18, 10))

for i,g in enumerate(cat_groups):

    tmp_group = tmp.loc[(slice(None),g)]

    markers_group = markers[i]

    color_group = colors[i]

    ax.scatter(x=tmp_group.index.get_level_values(0),y=tmp_group,c=color_group,marker=markers_group,

                  label=g)



ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          ncol=4, fancybox=True, shadow=True)

ax.grid(True)

plt.title("Correlations of department sales within state")

plt.show()
sales_train_id_stacked = sales_train_validation_active.set_index(['item_id','store_id','cat_id']).drop(['id','dept_id', 'state_id'],axis=1).T

# active in the first 60 days

item_names_drop = sales_train_id_stacked.loc[:,(sales_train_id_stacked.iloc[:60].sum()>0)].columns.get_level_values(0).unique()

sales_train_id_stacked = sales_train_id_stacked.loc[:,(~sales_train_id_stacked.columns.get_level_values(0).isin(item_names_drop))]

prod_active_time = sales_train_id_stacked.cumsum().clip(0,1).sum()

prod_active_diff_time = prod_active_time -prod_active_time.groupby(level=["item_id"]).transform("max")



# prod_active_diff_time = prod_active_diff_time.unstack(level=[1,2])

cat = prod_active_diff_time.index.levels[2]

fig, ax = plt.subplots(3, 1,figsize=(15, 10))

for i, t in enumerate(cat):

    tmp_prod_active_diff_time = prod_active_diff_time.loc[(slice(None), slice(None), t)]

    tmp_prod_active_diff_time = tmp_prod_active_diff_time.droplevel(0).reset_index()



    sns.boxplot(x="store_id", y=0, data=tmp_prod_active_diff_time,ax=ax[i])

    ax[i].set_yscale('symlog')

    ax[i].set_ylim(top=0)

    ax[i].set_title(t)

    ax[i].label_outer()



fig.tight_layout()

plt.show()

events_pre_pos = pd.DataFrame([['Christmas',3,5],

                              ['Easter',3,3],

                              ['Halloween',2,2],

                              ['IndependenceDay',2,1],

                              ['LaborDay',3,1],

                              ['MartinLutherKingDay',3,1],

                              ['MemorialDay',2,1],

                              ["Mother's day",1,1],

                              ['NewYear',2,1],

                              ['StPatricksDay',1,1],

                              ['SuperBowl',2,1],

                              ['Thanksgiving',2,5],

                              ['ValentinesDay',3,3]]

                              ,columns=["event_name","prelude","postlude"])



events_df = calendar.loc[~calendar["event_name_1"].isna(),["date","event_name_1","event_type_1"]]

events_df.columns = ["date","event_name","event_type"]   

events_tmp_2 = calendar.loc[~calendar["event_name_2"].isna(),["date","event_name_2","event_type_2"]]

events_tmp_2.columns = ["date","event_name","event_type"]   

events_df = events_df.append(events_tmp_2)

events_df = events_df.sort_values("date")



eff_ev_days = events_df.merge(events_pre_pos,how="left",right_on="event_name",left_on="event_name")

eff_ev_days["prelude"].fillna(0,inplace=True)

eff_ev_days["postlude"].fillna(0,inplace=True)



eff_ev_days_l = list()

for i,r in eff_ev_days.iterrows():

    eff_ev_days_l.append(pd.date_range(end=r["date"],periods=r["prelude"]+1))

    eff_ev_days_l.append(pd.date_range(start=r["date"],periods=r["postlude"]+1))

eff_ev_days_l = list(set(np.concatenate(eff_ev_days_l).ravel()))

# compare snap and non-snap sales

calendar_span = calendar[["date",'d']].copy()

last_d = sales_train_validation_active.columns[-1]

last_d = int(re.findall("\d+",last_d)[0])

calendar_span = calendar_span.loc[:(last_d-1),:]



calendar_span.loc[(calendar_span["date"].dt.day ==1)&(calendar_span["date"].dt.month%4==2),"span_1"] = 2

calendar_span["span_1"] = calendar_span["span_1"].fillna(0).cumsum()

calendar_span.loc[(calendar_span["date"].dt.month%2==1),"span_1"] -= 1

calendar_span.loc[(calendar_span["span_1"]<1)|(calendar_span["span_1"]>30),"span_1"] = np.nan



calendar_span.loc[(calendar_span["date"].dt.day ==1)&(calendar_span["date"].dt.month%4==0),"span_2"] = 2

calendar_span["span_2"] = calendar_span["span_2"].fillna(0).cumsum()

calendar_span.loc[(calendar_span["date"].dt.month%2==1),"span_2"] -= 1

calendar_span.loc[(calendar_span["span_2"]<1)|(calendar_span["span_2"]>30),"span_2"] = np.nan







sales_train_id_stacked = sales_train_validation_active.set_index(['cat_id','dept_id','state_id','store_id','item_id']).drop(['id'],axis=1).T

sales_train_id_stacked = sales_train_id_stacked.merge(calendar_span,right_on="d",left_index=True).set_index(['date','span_1', 'span_2']).drop("d",axis=1)

sales_train_id_stacked.columns = pd.MultiIndex.from_tuples(sales_train_id_stacked.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])



# set nan sales on event days and for some events the day around

sales_train_id_stacked.loc[sales_train_id_stacked.index.get_level_values(0).isin(eff_ev_days_l)] = np.nan





# test only ids of each department in the upper 80% avg sales quantile

# and are active for longer than 2 years

mean_tmp = sales_train_id_stacked.loc[:,sales_train_id_stacked.clip(1,1).sum()>720].mean()

cat_store_quantiles = mean_tmp.groupby(level=["cat_id","store_id"]).quantile(0.80)

#cat_store_quantiles = mean_tmp.groupby(level=["store_id"]).quantile(0.80)



item_ids = list()

for i, r in cat_store_quantiles.iteritems():

    item_ids.append([mean_tmp.loc[mean_tmp>=r].loc[(i[0],slice(None),slice(None),i[1])].droplevel([0,1]).index,i[1]])



# 5096 total number of items

# 2368 items in FOODS

        

tmp_top_sales = pd.DataFrame(index=sales_train_id_stacked.index)

for cat_store_item in item_ids:

    tmp_top_sales = tmp_top_sales.merge(sales_train_id_stacked.loc[:,(slice(None),slice(None),slice(None),cat_store_item[1],cat_store_item[0])],right_index=True,left_index=True)

sales_train_id_stacked = tmp_top_sales.copy()

del tmp_top_sales



#     

def avg_sales(period_sales):

    date_target = period_sales.index.get_level_values("date")[1] +pd.to_timedelta(1,unit="M")

    month_target = date_target.month

    year_target = date_target.year



    mean_sales = period_sales.mean().to_frame()

    mean_sales = mean_sales.T.set_index([pd.Index([month_target]),pd.Index([year_target])])

    

    mean_sales.loc[:,period_sales.clip(1,1).sum()<35] = np.nan

    return mean_sales



relative_diff_sales = pd.DataFrame()

for i in [1,2]:

    relative_diff_sales = relative_diff_sales.append(sales_train_id_stacked.groupby(level=i,axis=0).apply(lambda x:avg_sales(x)))

    

relative_diff_sales = relative_diff_sales.droplevel(0)

relative_diff_sales.index.set_names(["month","year"],inplace=True)



sales_train_id_stacked = sales_train_id_stacked.droplevel([1,2])

sales_train_id_stacked.set_index([sales_train_id_stacked.index.month,sales_train_id_stacked.index.year],append=True,inplace=True)

sales_train_id_stacked.index.set_names(["date","month","year"],inplace=True)

sales_train_id_stacked = sales_train_id_stacked.reorder_levels(["month","year","date"])



relative_diff_sales = relative_diff_sales.reindex(sales_train_id_stacked.index)



relative_diff_sales = relative_diff_sales.loc[relative_diff_sales.isna().sum(axis=1)<4580]

relative_diff_sales.columns = pd.MultiIndex.from_tuples(relative_diff_sales.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])



sales_train_id_stacked[relative_diff_sales==0] += 0.1

relative_diff_sales[relative_diff_sales==0] = 0.1



relative_diff_sales = sales_train_id_stacked.loc[relative_diff_sales.index].div(relative_diff_sales)

relative_diff_sales = relative_diff_sales.droplevel([0,1])







relative_diff_sales[relative_diff_sales>3] = 3

relative_diff_sales.columns = pd.MultiIndex.from_tuples(relative_diff_sales.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])

relative_diff_sales.index = pd.to_datetime(relative_diff_sales.index)





diff_sales_snap = relative_diff_sales.merge(calendar[["date","snap_CA","snap_TX","snap_WI"]],left_index=True,right_on="date").set_index(["date","snap_CA","snap_TX","snap_WI"])

diff_sales_snap.columns = pd.MultiIndex.from_tuples(diff_sales_snap.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])
def set_day_groups(x):

    if x < 8:

        return "DG: 1"

    if x < 16:

        return "DG: 2"

    if x < 23:

        return "DG: 3"

    else:

        return "DG: 4"



day_groups_index = diff_sales_snap.index.get_level_values(0).day.map(lambda x: set_day_groups(x))

diff_sales_snap.set_index(day_groups_index,append=True,inplace=True)

weekend_index = diff_sales_snap.index.get_level_values(0).weekday.map(lambda x:"WE: F" if x<5 else "WE: True")

diff_sales_snap.set_index(weekend_index,append=True,inplace=True)

weekday_index = diff_sales_snap.index.get_level_values(0).weekday

diff_sales_snap.set_index(weekday_index,append=True,inplace=True)





diff_sales_snap.index.set_names(['date', 'snap_CA', 'snap_TX', 'snap_WI', 'day_group', 'weekend','weekday'],inplace=True)



grouped_weekday_state_sales = diff_sales_snap.groupby(level="weekday").mean()

grouped_weekday_state_sales = grouped_weekday_state_sales.droplevel("state_id",axis=1)





cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

a, b, c=[plt.cm.Oranges, plt.cm.Blues, plt.cm.Greys]

colors=[a(0.6), b(0.8), c(0.6)]

fig, ax = plt.subplots(10, 3,figsize=(15, 15),sharey=True)

for col, c in enumerate(cat):

    color = colors[col]

    tmp_grouped_weekday_state_sales = grouped_weekday_state_sales.loc[:,c]

    # level: use dept_id instead of store_id

    tmp_grouped_weekday_state_sales = tmp_grouped_weekday_state_sales.groupby(level=["store_id"],axis=1).mean()

    tmp_grouped_weekday_state_sales = tmp_grouped_weekday_state_sales.div(tmp_grouped_weekday_state_sales.min())

    tmp_grouped_weekday_state_sales -=1

    for row, s in enumerate(tmp_grouped_weekday_state_sales.columns):

        ax[row,col].bar(tmp_grouped_weekday_state_sales.index,tmp_grouped_weekday_state_sales[s],color=color)

        ax[row,1].set_title(s, fontsize=10)

fig.legend(labels=['FOODS', 'HOBBIES', 'HOUSEHOLD'],loc='upper center', bbox_to_anchor=(0.5, -0.01),

                    bbox_transform=fig.transFigure, ncol=3, fontsize=11)

#plt.subplots_adjust(hspace = 0.5,wspace = 0.03)

plt.tight_layout()



# I eddited it for easier use with date-time index



# ----------------------------------------------------------------------------

# Author:  Nicolas P. Rougier

# License: BSD

# ----------------------------------------------------------------------------



def calmap(ax,year, data, cmap):

    ax.tick_params('x', length=0, labelsize="medium", which='major')

    ax.tick_params('y', length=0, labelsize="x-small", which='major')



    ax.pcolormesh(data, edgecolors='grey', linewidth=0.2,cmap=cmap,vmin=-1, vmax=1)

    # Month borders

    xticks, labels = [], []

    start = datetime.datetime(year, 1, 1).weekday()

    for month in range(1, 13):

        first = datetime.datetime(year, month, 1)

        last = first + relativedelta(months=1, days=-1)



        y0 = first.weekday()

        y1 = last.weekday()

        x0 = (int(first.strftime("%j")) + start - 1) // 7

        x1 = (int(last.strftime("%j")) + start - 1) // 7



        P = [(x0, y0), (x0, 7), (x1, 7),

             (x1, y1 + 1), (x1 + 1, y1 + 1), (x1 + 1, 0),

             (x0 + 1, 0), (x0 + 1, y0)]

        xticks.append(x0 + (x1 - x0 + 1) / 2)

        labels.append(first.strftime("%b"))

        poly = Polygon(P, edgecolor="k", facecolor="None",

                       linewidth=1.5, zorder=20, clip_on=False)

        ax.add_artist(poly)



    ax.set_xticks(xticks)

    ax.set_xticklabels(labels)

    ax.set_yticks(0.5 + np.arange(7))

    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])



def calprep(data):

    year = data.first_valid_index().year

    # Add days to overlapping weeks at the begin and end of year

    first_week = datetime.datetime(year, 1, 1) - pd.to_timedelta(datetime.datetime(year, 1, 1).weekday(),unit="d")

    first_nan_period = pd.date_range(first_week, data.first_valid_index(),closed="left")

    data = data.append(pd.DataFrame(index=first_nan_period))

    last_week = datetime.datetime(year, 12, 31) + pd.to_timedelta(6-datetime.datetime(year, 12, 31).weekday(),unit="d")

    last_nan_period = pd.date_range(data.last_valid_index(), last_week,closed="right")

    data = data.append(pd.DataFrame(index=last_nan_period)).sort_index()

    num_weeks = len(data)//7

    data = data.values.reshape(num_weeks,7).T

    return data

cal_tmp = calendar.loc[calendar["year"]==2015,["snap_CA","date"]].set_index("date")

array_cal = calprep(cal_tmp)

fig = plt.figure(figsize=(15, 8))

ax = plt.subplot(311, xlim=[0,array_cal.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap(ax, 2015, array_cal,"RdBu")

ax.set_title("{} Snap dates CA".format(2015), weight="semibold")



cal_tmp = calendar.loc[calendar["year"]==2015,["snap_TX","date"]].set_index("date")

array_cal = calprep(cal_tmp)

ax = plt.subplot(312, xlim=[0,array_cal.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap(ax, 2015, array_cal,"RdBu")

ax.set_title("{} Snap dates TX".format(2015), weight="semibold")



cal_tmp = calendar.loc[calendar["year"]==2015,["snap_WI","date"]].set_index("date")

array_cal = calprep(cal_tmp)

ax = plt.subplot(313, xlim=[0,array_cal.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap(ax, 2015, array_cal,"RdBu")

ax.set_title("{} Snap dates WI".format(2015), weight="semibold")

plt.show()
grouped_state_sales = pd.DataFrame()

for state in ['CA','TX','WI']:

    state_sales_tmp = diff_sales_snap.loc[:,(slice(None),slice(None),state)]

    state_sales_tmp = state_sales_tmp.groupby(level=["day_group", "weekend", ("snap_" + state)]).mean()

    snap_index = state_sales_tmp.index.get_level_values(2).map(lambda x: "SN: T" if x == 1 else "SN: F")

    state_sales_tmp.set_index(snap_index, append=True, inplace=True)

    state_sales_tmp = state_sales_tmp.droplevel(2)

    state_sales_tmp.index.set_names(['day_group', 'weekend','snap'],inplace=True)

    grouped_state_sales = pd.concat([state_sales_tmp,grouped_state_sales],axis=1)

    

grouped_state_sales = grouped_state_sales.droplevel("state_id",axis=1)



cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

fig, ax = plt.subplots(3, 1,figsize=(15, 12))

for i, c in enumerate(cat):

    tmp_grouped_state_sales = grouped_state_sales.loc[:,c]

    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=["dept_id"],axis=1).mean()

    tmp_grouped_state_sales = tmp_grouped_state_sales.div(tmp_grouped_state_sales.min())

    tmp_grouped_state_sales -=1

    

    sns.heatmap(tmp_grouped_state_sales,ax=ax[i], vmin=0,vmax=.55,annot=True,fmt=".1%")

    ax[i].hlines([4, 8, 10], *ax[i].get_xlim(),color='w',linestyle='--',linewidth=2.5)

    ax[i].hlines([2, 6, 9, 11], *ax[i].get_xlim(),color='w',linestyle=':',linewidth=1)

    ax[i].set_title(c)

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

plt.show()
diff_daily = relative_diff_sales.copy()

diff_daily.set_index([diff_daily.index.day],inplace=True)

diff_daily.index.set_names(["day"],inplace=True)

diff_daily = diff_daily.groupby(level=[0]).mean()

# tmp.groupby(level=[0,1],axis=1).mean()

diff_daily = diff_daily.groupby(level=[0,2,3],axis=1).mean().droplevel("state_id",axis=1)



diff_daily = diff_daily.div(diff_daily.min())

diff_daily -=1



plt.figure(figsize=(15, 10))

sns.heatmap(diff_daily)

plt.title("Relative difference in sales in a average month")

plt.show()


cat = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

fig, ax = plt.subplots(3, 1,figsize=(15, 12))

for i, c in enumerate(cat):

    tmp_grouped_state_sales = grouped_state_sales.loc[:,c]

    # level: use dept_id instead of store_id

    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=["store_id"],axis=1).median()

    tmp_grouped_state_sales = tmp_grouped_state_sales.div(tmp_grouped_state_sales.min())

    tmp_grouped_state_sales -=1



    sns.heatmap(tmp_grouped_state_sales,ax=ax[i], vmin=-.01,vmax=.55,annot=True,fmt=".1%")

    ax[i].hlines([4, 8, 10], *ax[i].get_xlim(),color='w',linestyle='--',linewidth=2.5)

    ax[i].hlines([2, 6, 9, 11], *ax[i].get_xlim(),color='w',linestyle=':',linewidth=1)

    ax[i].set_title(c)

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

plt.show()
# dept_group grouped_state_sales.columns.get_level_values(1).unique()

dept_group = ['FOODS_1', 'FOODS_2', 'FOODS_3']

fig, ax = plt.subplots(3, 1,figsize=(15, 12))

for i, c in enumerate(dept_group):

    tmp_grouped_state_sales = grouped_state_sales.droplevel(0,axis=1)

    tmp_grouped_state_sales = tmp_grouped_state_sales.loc[:,c]

    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=[0],axis=1).mean()

    tmp_grouped_state_sales = tmp_grouped_state_sales.div(tmp_grouped_state_sales.min())

    tmp_grouped_state_sales -=1

    

    sns.heatmap(tmp_grouped_state_sales,ax=ax[i], vmin=0,vmax=.55,annot=True,fmt=".1%")

    ax[i].hlines([4, 8, 10], *ax[i].get_xlim(),color='w',linestyle='--',linewidth=2.5)

    ax[i].hlines([2, 6, 9, 11], *ax[i].get_xlim(),color='w',linestyle=':',linewidth=1)

    ax[i].set_title(c)

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

plt.show()



#sns.heatmap(tmp_grouped_state_sales.groupby(level=[0,1]).diff(),annot=True,fmt=".1%")



#tmp_grouped_state_sales.groupby(level=[0,1]).diff(
# dept_group grouped_state_sales.columns.get_level_values(1).unique()

dept_group = ['FOODS_1', 'FOODS_2', 'FOODS_3']

fig, ax = plt.subplots(3, 1,figsize=(15, 6))

for i, c in enumerate(dept_group):

    tmp_grouped_state_sales = grouped_state_sales.droplevel(0,axis=1)

    tmp_grouped_state_sales = tmp_grouped_state_sales.loc[:,c]

    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=[0],axis=1).mean()

#    tmp_grouped_state_sales = tmp_grouped_state_sales.div(tmp_grouped_state_sales.min())

#    tmp_grouped_state_sales -=1

    tmp_grouped_state_sales = tmp_grouped_state_sales.iloc[0:8]

    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=[0,1]).apply(lambda x: x.iloc[1].div(x.iloc[0]))-1



#    tmp_grouped_state_sales = tmp_grouped_state_sales.groupby(level=[0,1]).diff(axis=0)

#    tmp_grouped_state_sales = tmp_grouped_state_sales.droplevel(2)

#    tmp_grouped_state_sales = tmp_grouped_state_sales.iloc[[1,3,5,7]]



   # tmp_grouped_state_sales.index.set_levels(["","diff SNAP/non-SNAP"],level=2,inplace=True)

    

    sns.heatmap(tmp_grouped_state_sales,ax=ax[i], vmin=0,vmax=.40,annot=True,fmt=".1%")

    ax[i].hlines([2], *ax[i].get_xlim(),color='w',linestyle='--',linewidth=2.5)

    ax[i].hlines([1,3], *ax[i].get_xlim(),color='w',linestyle=':',linewidth=1)

    ax[i].set_title(c)

    ax[i].set_ylabel("")

    ax[i].set_xlabel("")

fig.suptitle("Difference between SNAP and non-SNAP groups",y=1.02,size=18)

plt.tight_layout()

plt.show()







# I eddited it for easier use with date-time index and changed it for visualisation of categories



# ----------------------------------------------------------------------------

# Author:  Nicolas P. Rougier

# License: BSD

# ---------

def calmap_cat(ax,year, data, cmap,bounds):

    ax.tick_params('x', length=0, labelsize="medium", which='major')

    ax.tick_params('y', length=0, labelsize="x-small", which='major')

    norm = colorm.BoundaryNorm(bounds, cmap.N)



    ax.pcolormesh(data, edgecolors='grey', linewidth=0.2,cmap=cmap, norm=norm,shading='flat')



    # Month borders

    xticks, labels = [], []

    start = datetime.datetime(year, 1, 1).weekday()

    for month in range(1, 13):

        first = datetime.datetime(year, month, 1)

        last = first + relativedelta(months=1, days=-1)



        y0 = first.weekday()

        y1 = last.weekday()

        x0 = (int(first.strftime("%j")) + start - 1) // 7

        x1 = (int(last.strftime("%j")) + start - 1) // 7



        P = [(x0, y0), (x0, 7), (x1, 7),

             (x1, y1 + 1), (x1 + 1, y1 + 1), (x1 + 1, 0),

             (x0 + 1, 0), (x0 + 1, y0)]

        xticks.append(x0 + (x1 - x0 + 1) / 2)

        labels.append(first.strftime("%b"))

        poly = Polygon(P, edgecolor="k", facecolor="None",

                       linewidth=1.5, zorder=20, clip_on=False)

        ax.add_artist(poly)



    ax.set_xticks(xticks)

    ax.set_xticklabels(labels)

    ax.set_yticks(0.5 + np.arange(7))

    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])





def calprep(data):

    year = data.first_valid_index().year

    # Add days to overlapping weeks at the begin and end of year

    first_week = datetime.datetime(year, 1, 1) - pd.to_timedelta(datetime.datetime(year, 1, 1).weekday(),unit="d")

    first_nan_period = pd.date_range(first_week, data.first_valid_index(),closed="left")

    data = data.append(pd.DataFrame(index=first_nan_period))

    last_week = datetime.datetime(year, 12, 31) + pd.to_timedelta(6-datetime.datetime(year, 12, 31).weekday(),unit="d")

    last_nan_period = pd.date_range(data.last_valid_index(), last_week,closed="right")

    data = data.append(pd.DataFrame(index=last_nan_period)).sort_index()

    num_weeks = len(data)//7

    data = data.values.reshape(num_weeks,7).T

    return data

cal_cat = calendar.loc[:,["event_type_1","date"]].set_index("date")

cal_cat = cal_cat["event_type_1"].astype("category")

cal_cat = cal_cat.cat.codes

cal_14 = cal_cat.loc[cal_cat.index.year==2014]

cal_15 = cal_cat.loc[cal_cat.index.year==2015]

cal_16 = cal_cat.loc[cal_cat.index.year==2016]







array_14 = calprep(cal_14)

array_15 = calprep(cal_15)

array_16 = calprep(cal_16)



#cat_names = ["", 'National', 'Religious', 'Cultural', 'Sporting']

#cmap = colors.ListedColormap(['gainsboro', 'red','blue','yellow','green'])

cmap = colorm.ListedColormap(['gainsboro', 'indianred','royalblue','yellow','forestgreen'])



bounds=[-1,0,1,2,3,4]

legend_elements = [Patch(facecolor='indianred',label='Cultural'),

                  Patch(facecolor='royalblue',label='National'),

                  Patch(facecolor='yellow',label='Religious'),

                  Patch(facecolor='forestgreen',label='Sporting')]



fig = plt.figure(figsize=(15, 8))

ax = plt.subplot(311, xlim=[0,array_14.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap_cat(ax, 2014, array_14,cmap,bounds)

ax.set_title("{} Events ".format(2014), weight="semibold")





fig = plt.figure(figsize=(15, 8))

ax = plt.subplot(311, xlim=[0,array_15.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap_cat(ax, 2015, array_15,cmap,bounds)

ax.set_title("{} Events ".format(2015), weight="semibold")



fig = plt.figure(figsize=(15, 8))

ax = plt.subplot(311, xlim=[0,array_16.shape[1]], ylim=[0,7], frameon=False, aspect=1)

calmap_cat(ax, 2016, array_16,cmap,bounds)

ax.set_title("{} Events ".format(2016), weight="semibold")



ax.legend(handles=legend_elements,bbox_to_anchor=(0.7,-.4), loc="lower right",ncol=4)

plt.show()



events_pre_pos

grouped_state_sales = pd.DataFrame()

for state in ['CA','TX','WI']:

    state_sales_tmp = diff_sales_snap.loc[:,(slice(None),slice(None),state)]

    state_sales_tmp = state_sales_tmp.groupby(level=["day_group", "weekend", ("snap_" + state)]).mean()

    snap_index = state_sales_tmp.index.get_level_values(2).map(lambda x: "SN: T" if x == 1 else "SN: F")

    state_sales_tmp.set_index(snap_index, append=True, inplace=True)

    state_sales_tmp = state_sales_tmp.droplevel(2)

    state_sales_tmp.index.set_names(['day_group', 'weekend','snap'],inplace=True)

    grouped_state_sales = pd.concat([state_sales_tmp,grouped_state_sales],axis=1)

    

# drop all zero values before the first value >=1

sales_train_id_stacked = sales_train_validation.set_index(['id']).drop(['item_id','store_id','dept_id', 'cat_id', 'state_id'],axis=1).T.astype("float")



pre_start_period = ((sales_train_id_stacked>0).cumsum()==0)



sales_train_id_stacked.iloc[pre_start_period] = np.nan



sales_train_id_stacked = sales_train_id_stacked.T



sales_train_validation_active = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].merge(sales_train_id_stacked,left_on="id",right_index=True)

sales_train_id_stacked = sales_train_validation_active.set_index(['cat_id','dept_id','state_id','store_id','item_id']).drop(['id'],axis=1).T

sales_train_id_stacked = sales_train_id_stacked.merge(calendar[["date","snap_CA","snap_TX","snap_WI","d"]],right_on="d",left_index=True).set_index(["date","snap_CA","snap_TX","snap_WI"]).drop("d",axis=1)

#sales_train_id_stacked = sales_train_id_stacked.merge(calendar[["date","d"]],right_on="d",left_index=True).set_index(['date']).drop("d",axis=1)

sales_train_id_stacked.columns = pd.MultiIndex.from_tuples(sales_train_id_stacked.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])



day_groups_index = sales_train_id_stacked.index.get_level_values(0).day.map(lambda x: set_day_groups(x))

sales_train_id_stacked.set_index(day_groups_index,append=True,inplace=True)

weekend_index = sales_train_id_stacked.index.get_level_values(0).weekday.map(lambda x:"WE: F" if x<5 else "WE: True")

sales_train_id_stacked.set_index(weekend_index,append=True,inplace=True)

weekday_index = sales_train_id_stacked.index.get_level_values(0).weekday

sales_train_id_stacked.set_index(weekday_index,append=True,inplace=True)





# test top 30% products

tmp_top_sales = pd.DataFrame(index=sales_train_id_stacked.index)

for cat_store_item in item_ids:

    tmp_top_sales = tmp_top_sales.merge(sales_train_id_stacked.loc[:,(slice(None),slice(None),slice(None),cat_store_item[1],cat_store_item[0])],right_index=True,left_index=True)

sales_train_id_stacked = tmp_top_sales.copy()

del tmp_top_sales

sales_train_id_stacked.columns = pd.MultiIndex.from_tuples(sales_train_id_stacked.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])





sales_train_id_stacked.index.set_names(['date',"snap_CA","snap_TX","snap_WI",'day_group', 'weekend',"weekday"],inplace=True)



adapted_sales = pd.DataFrame(index=sales_train_id_stacked.index.get_level_values(0))



for state in ['CA','TX','WI']:

    state_sales_tmp = sales_train_id_stacked.loc[:,(slice(None),slice(None),state)]

    i=0

    grouped_sales = state_sales_tmp.groupby(level=["day_group", "weekend", ("snap_" + state)])

    adapted_sales_tmp = pd.DataFrame()

    for name,group in grouped_sales:

        i+=1

       # print(i)

        name = list(name)

        name[2] = "SN: T" if name[2] == 1 else "SN: F"

        tmp = group/(grouped_state_sales.loc[tuple(name),(slice(None),slice(None),state)])

        tmp = tmp.droplevel(["snap_CA","snap_TX","snap_WI",'day_group','weekend','weekday'])

        adapted_sales_tmp = adapted_sales_tmp.append(tmp)

    adapted_sales = adapted_sales.merge(adapted_sales_tmp,how='left',left_index=True,right_index=True)

adapted_sales.columns = pd.MultiIndex.from_tuples(adapted_sales.columns,names=['cat_id','dept_id','state_id','store_id','item_id'])



adapted_sales = adapted_sales.loc[:,"FOODS"]

events_df = calendar.loc[~calendar["event_name_1"].isna(),["date","event_name_1"]]

events_df.columns = ["date","event_name"]   

events_tmp_2 = calendar.loc[~calendar["event_name_2"].isna(),["date","event_name_2"]]

events_tmp_2.columns = ["date","event_name"]   

events_df = events_df.append(events_tmp_2)

events_df = events_df.sort_values("date")

events_df = events_df.loc[events_df["date"]<pd.datetime(2016,4,25)]

events_df = events_df.loc[events_df["date"]>pd.datetime(2011,2,15)]





for i in range(-5,6):

    events_df = events_df.groupby(["date","event_name"]).apply(lambda gr: gr.append(gr.tail(1).assign(ev_shift=i))).droplevel([0,1])

events_df.dropna(inplace=True)

events_df.loc[:,"ev_shift_date"] = events_df["date"] + pd.to_timedelta(events_df["ev_shift"],unit="day")

events_df.loc[:,"ev_shift_weekday"] = events_df["ev_shift_date"].dt.weekday 

events_df.set_index(["event_name","date","ev_shift","ev_shift_date","ev_shift_weekday"],inplace=True)







events_exp = events_df.copy()

wd_mean_list = list()

wd_std_list = list()

for name,group in events_exp.groupby(level=["date","event_name"]):

    # 8 weeks -56 days

    prev_period = pd.date_range(start=name[0],periods=90)

    future_period = pd.date_range(end=name[0],periods=90)

    period = prev_period.append(future_period)

    # only dates which are not in examination period

    period = [x for x in period if x not in group.index.get_level_values(3)]

    period =  list(set(period)-set(eff_ev_days_l))

    tmp_sales = adapted_sales.loc[period]

    tmp_sales.index = tmp_sales.index.weekday

    tmp_sales.loc[:,tmp_sales.clip(1,1).sum()<105] = np.nan

    wd_mean = tmp_sales.groupby(level=0).mean()

    

    wd_mean_tmp = wd_mean.copy()

    tmp_sales[wd_mean==0] += 0.1

    wd_mean_tmp[wd_mean_tmp==0] = 0.1

    wd_std = np.std(tmp_sales/wd_mean_tmp-1)

    wd_std = wd_std.to_frame()

    wd_std = wd_std.T.set_index([pd.Index([name[1]]),pd.Index([name[0]])])

#    wd_std = wd_std/(tmp_sales.mean().clip(0,20))

    wd_std = wd_std/(np.log1p(tmp_sales.mean()))



    wd_mean = group.merge(wd_mean,how="left",right_index=True,left_on=group.index.get_level_values("ev_shift_weekday"))

    wd_mean.columns = pd.MultiIndex.from_tuples(adapted_sales.columns,names=['dept_id','state_id','store_id','item_id'])





    wd_mean_list.append(wd_mean)

    wd_std_list.append(wd_std)



events_exp = pd.concat(wd_mean_list)

std_exp = pd.concat(wd_std_list)



std_exp[std_exp==0] = 4

std_exp = 1/std_exp

std_exp.index.set_names(['date',"event_name"],inplace=True)

std_exp = std_exp.reindex(events_exp.index)



events_df = events_df.merge(adapted_sales,how="left",left_on=events_df.index.get_level_values("ev_shift_date"),right_index=True)

events_df.columns = pd.MultiIndex.from_tuples((events_df.columns),names=['dept_id','state_id','store_id','item_id'])



events_df[events_exp==0] += 0.1

events_exp[events_exp==0] = 0.1



events_res = events_df/(events_exp) -1

events_res[events_res>2] = 2



cat_res = list()

for name, group in events_res.groupby(level="store_id",axis=1):

    weight_cat = std_exp.loc[:,(slice(None),slice(None),name)]

    weight_cat = weight_cat.div(weight_cat.sum(axis=1),axis=0)

    group = group * weight_cat

    group = group.sum(axis=1).groupby(["event_name","date","ev_shift"]).mean()

    group = group.rename(name).to_frame()

    cat_res.append(group.T)

events_res2 = pd.concat(cat_res).T.mean(axis=1)#.loc["Mother's day"]

# test without special weighting

# events_res2 = events_res.groupby(level="store_id",axis=1).mean().mean(axis=1).droplevel([3,4])

# test end

events_res2 = events_res2.to_frame()

events_res2.set_index(events_res2.index.get_level_values(1).year,append=True,inplace=True)

events_res2.index = events_res2.index.set_names(["event_name","date","ev_shift","year"])

events_res2 = events_res2.droplevel(["date"])



events_res2 = events_res2.unstack(level=2)

events_res2 = events_res2.droplevel(0,axis=1)

events_res2["mean"] = events_res2.mean(axis=1)



events_cat = events_res2.index.get_level_values(0).unique()



a, b=[plt.cm.Greys,plt.cm.Oranges]

colors = [a(0.4), a(0.6), a(0.8), a(0.4), a(0.6), a(0.8), b(0.9)]



markers = ("o", "o","o","^","^","^","s")

cat_groups = [2011,2012,2013,2014,2015,2016,"mean"]



fig, ax = plt.subplots(10, 3,figsize=(15, 30))#,sharey=True)

for col, e in enumerate(events_cat):

    #color = colors[col]

    plot_index = ((col)//3,(col)%3)

    event = events_res2.loc[e].stack()

    ax[plot_index].set_title(e)

    ax[plot_index].axhline(0, color='w', linewidth=4,zorder=-1)



    for i,g in enumerate(cat_groups):

        if g not in event.index.get_level_values(1):

            continue

        tmp_group = event.loc[(slice(None),g)]

        markers_group = markers[i]

        color_group = colors[i]

      

        ax[plot_index].scatter(x=tmp_group.index,y=tmp_group,marker=markers_group,c=color_group)



        

legend_elements = [Line2D([0],[0],color='w',markerfacecolor=colors[i],marker=markers[i],label=cat_groups[i]) for i in range(0,7)]



fig.legend(handles=legend_elements, labels=cat_groups,loc='upper center', bbox_to_anchor=(0.5, -0.01),

                    bbox_transform=fig.transFigure, ncol=7, fontsize=11)



fig.suptitle("Relative effect of event days on sales",size=20,y=1.02)

plt.tight_layout()

plt.show() 
overlaping_events = events_exp.reset_index(level=["event_name","ev_shift_date"])[["event_name","ev_shift_date"]].droplevel([1,2,3],axis=1)

overlaping_events_matrix = overlaping_events.loc[overlaping_events.duplicated("ev_shift_date",keep=False)].reset_index(drop=True)

#overlaping_events_matrix["event_name_ov"] = overlaping_events_matrix["event_name"]

#overlaping_events_matrix = overlaping_events_matrix.append([overlaping_events_matrix.assign(event_name_ov = overlaping_events_matrix.loc[overlaping_events_matrix["ev_shift_date"]==date,"event_name"]) for i,date in enumerate(overlaping_events_matrix["ev_shift_date"])])



overlaping_events_matrix = overlaping_events_matrix.append([overlaping_events_matrix.loc[overlaping_events_matrix["ev_shift_date"]==date].assign(event_name_ov = overlaping_events_matrix.loc[i,"event_name"]) for i,date in enumerate(overlaping_events_matrix["ev_shift_date"])])





overlaping_events_matrix = overlaping_events_matrix.dropna()

overlaping_events_matrix = overlaping_events_matrix.loc[overlaping_events_matrix["event_name"]!=overlaping_events_matrix["event_name_ov"]]

overlaping_events_matrix = pd.crosstab(overlaping_events_matrix.event_name, overlaping_events_matrix.event_name_ov)



plt.figure(figsize=(18, 10))

sns.heatmap(overlaping_events_matrix,linewidths=.5,cmap='Reds')

plt.xlabel("")

plt.ylabel("")

plt.title("Count of overlapping event window days")

plt.show()



events_effective_pre_pos = pd.DataFrame([['Christmas',2,4],

                              ['Easter',2,4],

                              ['Halloween',1,0],

                              ['IndependenceDay',3,0],

                              ['LaborDay',1,0],

                              ['MemorialDay',2,0],

                              ["Mother's day",0,0],

                              ['NewYear',1,0],

                              ['SuperBowl',1,0],

                              ['Thanksgiving',2,4],

                              ['ValentinesDay',2,0]]

                              ,columns=["event_name","prelude","postlude"])

events_effective_pre_pos