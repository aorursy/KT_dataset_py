!pip install missingno
import numpy as np
import pandas as pd
import math
import missingno as msno
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set2')

import os
# df_kiva_loans = pd.read_csv("../kiva/kiva_loans.csv")
# df_mpi = pd.read_csv("../kiva/kiva_mpi_region_locations.csv")
df_kiva_loans = pd.read_csv("/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_mpi = pd.read_csv("/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
df_kiva_loans.head(5)
msno.bar(df_kiva_loans)
msno.matrix(df_kiva_loans)
df_kiva_loans.dtypes
df_kiva_loans.describe(include=[np.number])
df_kiva_loans.describe(include=[np.object])
countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts(normalize=True)> 0.005]
list_countries = list(countries.index)
countries
plt.figure(figsize=(20,10))
plt.title("Количество займов в разрезе стран", fontsize=16)
plt.tick_params(labelsize=14)
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.show()
df_mpi.head()
df_mpi_grouped = df_mpi\
    .groupby(['ISO', 'country', 'world_region'])['MPI']\
    .mean()\
    .fillna(0)\
    .reset_index()
df_kiva_loans = df_kiva_loans.merge(df_mpi_grouped, how='left', on='country')
regions = df_kiva_loans['world_region'].value_counts()
regions_list = regions.index.to_list()
plt.figure(figsize=(20,10))
sns.barplot(y=regions.index, x=regions.values, alpha=0.6)
plt.title("Количество займов в разрезе макрорегионов", fontsize=18)
plt.tick_params(labelsize=16)
plt.show();
df_kiva_loans['borrower_genders'].value_counts()
df_kiva_loans['borrower_genders'] = [elem if elem in ['female','male'] else 'group' for elem in df_kiva_loans['borrower_genders'] ]

borrowers = df_kiva_loans['borrower_genders'].value_counts()

plot = borrowers.plot.pie(fontsize=16, autopct='%1.0f%%', labeldistance=1.2, radius=2)
df_gender_by_country = \
    df_kiva_loans[df_kiva_loans['country'].isin(countries.index.values)]\
    .groupby(['country', 'borrower_genders'])['borrower_genders']\
    .count()\
    .groupby(level=0).apply(lambda x: 100 * x / x.sum())\
    .unstack('borrower_genders')\
    .fillna(0)\
    .sort_values(by=['female', 'male'])[-20:]

df_gender_by_country = df_gender_by_country[['female', 'male', 'group']]
plot = df_gender_by_country.plot.barh(
        figsize=(20,10)
        , fontsize=16
        , stacked=True
        , title='Гендерная структура заемщиков')
plot.title.set_size(18)
plot.legend(loc=1, bbox_to_anchor=(1.2, 1), fontsize=16)
plot.set(ylabel=None, xlabel=None)
plt.show()
normalized_country_counts = df_kiva_loans['country'].value_counts(normalize=True)> 0.005

df_male_loans = df_kiva_loans[df_kiva_loans['borrower_genders'] == "male"]
df_male_by_country = df_male_loans["country"].value_counts()[normalized_country_counts]
plt.figure(figsize=(20,10))
plt.title("Количество мужчин заемщиков в разрезе стран", fontsize=18)
plt.xlabel("Количество мужчин", fontsize=14)
plt.tick_params(labelsize=14)
sns.barplot(y=df_male_by_country.index, x=df_male_by_country.values, alpha=0.6)
plt.show()
df_group_loans = df_kiva_loans[df_kiva_loans['borrower_genders'] == "group"]
df_group_by_country = df_group_loans["country"].value_counts()[normalized_country_counts]
plt.figure(figsize=(20,10))
plt.title("Количество групп заемщиков в разрезе стран", fontsize=18)
plt.xlabel("Количество групп", fontsize=14)
plt.tick_params(labelsize=14)
sns.barplot(y=df_group_by_country.index, x=df_group_by_country.values, alpha=0.6)
plt.show()
sectors = df_kiva_loans['sector'].value_counts()

plt.figure(figsize=(20,10))
plt.title("Количество займов в разрезе секторов", fontsize=16)
# plt.xlabel('Number of loans', fontsize=16)
# plt.ylabel("Sectors", fontsize=16)
plt.tick_params(labelsize=15)

sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.show()
activities = df_kiva_loans['activity'].value_counts().head(30)

plt.figure(figsize=(20,10))
plt.title("Количество займов в разрезе видов деятельности", fontsize=16)
plt.tick_params(labelsize=14)

sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.show();
activities = df_kiva_loans['use'].value_counts().head(25)
activities.head(25)
loans_by_activity_sector = \
    df_kiva_loans[df_kiva_loans['sector'].isin(sectors[:9].index.values)]\
    .groupby(['sector', 'activity'])['loan_amount']\
    .count()\
    .reset_index()
fig,axes = plt.subplots(3,3, sharex=False, squeeze=False, figsize=(30,15))

for ax,q in zip(axes.ravel(), loans_by_activity_sector.sector.unique()):
    tmp_df = loans_by_activity_sector[loans_by_activity_sector.sector.eq(q)]\
                .dropna()\
                .sort_values(by='loan_amount')[-10:]
    ax.set_title(q, fontsize=20)  
    ax.yaxis.label.set_visible(False)
    plt.tight_layout()
    tmp_df.plot.barh(x='activity', ax=ax, legend=None, fontsize=18)
plt.figure(figsize=(20,10))
plt.title("Распределение суммы займа", fontsize=16)
plt.tick_params(labelsize=14)

sns.distplot(df_kiva_loans['loan_amount'], axlabel=False)
plt.show()
# Наблюдаемое отклонение от среднего
dev = (df_kiva_loans['loan_amount']-df_kiva_loans['loan_amount'].mean()).abs()
# Стандартное отклонение
std = df_kiva_loans['loan_amount'].std()
# Фильтруем исходный набор данных
df_kiva_loans_trimmed = df_kiva_loans[~(dev>3*std)]

plt.figure(figsize=(20,10))
plt.title("Распределение суммы займа (без выбросов)", fontsize=16)
plt.tick_params(labelsize=14)
plt.xticks(np.arange(0, max(df_kiva_loans_trimmed['loan_amount']+1), 200))

sns.distplot(df_kiva_loans_trimmed['loan_amount'], axlabel=False)
plt.show()
plt.figure(figsize=(20,10))
plt.title("Суммы займов по секторам", fontsize=18)
plt.xticks(np.arange(0, max(df_kiva_loans['loan_amount']+1), 10000))
plt.tick_params(labelsize=14)

sns.boxplot(x='loan_amount', y="sector", data=df_kiva_loans).set(ylabel=None, xlabel=None)
plt.show()
big_loans = df_kiva_loans[df_kiva_loans["loan_amount"] > 40000]
use_big_loans = big_loans.groupby("use")["loan_amount"].max()
use_big_loans_sort = use_big_loans.sort_values(ascending=False)[:10]

plt.figure(figsize=(15, 5))
plt.title("Назначение займов на сумму более 40к", fontsize=16)
plt.xlabel("Сумма займов, долл. США", fontsize=14)
plt.tick_params(labelsize=14)
sns.barplot(y=use_big_loans_sort.index, x=use_big_loans_sort.values, alpha=0.6)
plt.show();
country_big_loans = big_loans["country"].value_counts()
plt.figure(figsize=(20, 8))
plt.title("Количество займов на сумму более 40к в разрезе стран", fontsize=18)
plt.xlabel("Количество займов", fontsize=14)
plt.tick_params(labelsize=14)
sns.barplot(y=country_big_loans.index, x=country_big_loans.values, alpha=0.6)
plt.show();
genders_big_loans = big_loans["borrower_genders"].value_counts()

plt.figure(figsize=(10, 3))
plt.title("Количество займов на сумму более 40к в разрезе гендерного состава", fontsize=16)
plt.xlabel("Количество займов", fontsize=14)
plt.tick_params(labelsize=14)
sns.barplot(y=genders_big_loans.index, x=genders_big_loans.values, alpha=0.6)
plt.show();
sector_mean_median = df_kiva_loans\
    .groupby(['sector'])['loan_amount']\
    .agg(median='median', mean='mean')\
    .sort_values(by='median', ascending=False)

sort_order = sector_mean_median.index.to_list()

sector_mean_median
plt.figure(figsize=(20,10))
plt.title("Суммы займов по секторам", fontsize=18)
plt.xticks(np.arange(0, max(df_kiva_loans['loan_amount']+1), 500))
plt.tick_params(labelsize=14)

sns.boxplot(x='loan_amount', y="sector", order=sort_order, data=df_kiva_loans_trimmed).set(ylabel=None, xlabel=None)
plt.show()
plt.figure(figsize=(16,10))
plt.title("Суммы займов по макрорегионам", fontsize=18)
plt.xticks(np.arange(0, max(df_kiva_loans['loan_amount']+1), 500))
plt.tick_params(labelsize=14)

sns.boxplot(x='loan_amount', y="world_region", order=regions_list, data=df_kiva_loans_trimmed)\
    .set(ylabel=None, xlabel=None)
# plt.legend(loc=1, bbox_to_anchor=(1.15, 1), fontsize=16)
plt.show()
macro = df_kiva_loans.groupby(["world_region", "borrower_genders"])["loan_amount"].median().reset_index()
fig, axes = plt.subplots(3, 2, sharex=False, squeeze=False, figsize=(15, 10))

for ax, region in zip(axes.ravel(), macro.world_region.unique()):
    ax.set_title(region, fontsize=16)
    ax.yaxis.label.set_visible(False)
    ax.set_xlabel("Количество заемщиков", fontsize=14)
    df_reg = macro[macro["world_region"] == region]
    df_reg.plot.barh(x="borrower_genders", ax=ax, legend=None, fontsize=14)
plt.tight_layout(5)
# drop na (missing) values
df_loans_dates = df_kiva_loans_trimmed.dropna(subset=['disbursed_time', 'funded_time'], how='any', inplace=False)

# dates columns:
dates = ['posted_time','disbursed_time']

# format dates:
df_loans_dates[dates] = df_loans_dates[dates].applymap(lambda x : x.split('+')[0])
df_loans_dates[dates] = df_loans_dates[dates].apply(pd.to_datetime)

# calculate time interval
df_loans_dates.loc[:, 'time_funding'] = df_loans_dates['disbursed_time']-df_loans_dates['posted_time']
df_loans_dates.loc[:, 'time_funding'] = df_loans_dates['time_funding'] / timedelta(days=1) 

# remove outliers +-3 sigma
dev = (df_loans_dates['time_funding']-df_loans_dates['time_funding'].mean()).abs()
std = df_loans_dates['time_funding'].std()
# Keep rows where time_funding interval > 0 only
df_loans_dates_trimmed = df_loans_dates[~(dev>3*std) & (df_loans_dates.loc[:, 'time_funding']>0)]
plt.figure(figsize=(20,10))
plt.title("Количество дней до полного финансирования заявки", fontsize=16)
plt.xticks(np.arange(0, max(df_loans_dates_trimmed['time_funding']+1), 5))
plt.tick_params(labelsize=14)

sns.distplot(df_loans_dates_trimmed['time_funding']).set(ylabel=None, xlabel=None)
plt.show()
p = sns.jointplot(x="time_funding", y="loan_amount", data=df_loans_dates_trimmed, kind='kde', height=10, ratio=7, xlim=[0,40] , ylim=[0,2200])\
        .set_axis_labels("Количество дней", "Сумма займа", size=18)

# p.ax_joint.set_xticks(np.arange(0, max(df_loans_dates_trimmed['time_funding']+1), 5))
# p.ax_joint.set_yticks(np.arange(0, max(df_loans_dates_trimmed['loan_amount']+1), 250))
p.ax_joint.tick_params(labelsize=14)

plt.show()
df_country_median = df_loans_dates_trimmed.groupby(['world_region', 'country'])\
    .agg({'loan_amount' : 'median', 'time_funding' : 'median', 'term_in_months' : 'median'})\
    .reset_index()

df_country_median = df_country_median[df_country_median.country.isin(list_countries)].sort_values(by='time_funding')
f,ax=plt.subplots(1, 2, sharey=True, figsize=(25,10))

sns.barplot(y='country', x='time_funding', data=df_country_median, alpha=0.6, ax=ax[0])
ax[0].set_title("Медиана количества дней до полного финансирования заявки", fontsize=20)
ax[0].set_xlabel('Количество дней', fontsize=18)
ax[0].set_ylabel(None)
ax[0].tick_params(labelsize=16)

sns.barplot(y='country', x='loan_amount', data=df_country_median, alpha=0.6, ax=ax[1])
ax[1].set_title("Медиана суммы займа", fontsize=20)
ax[1].set_xlabel('Сумма в долл. США', fontsize=18)
ax[1].set_ylabel(None)
ax[1].tick_params(labelsize=16)

plt.tight_layout()
plt.show()
df_country_median = df_country_median.sort_values(by='term_in_months')
df_country_median['monthly_repayment'] = df_country_median['loan_amount'] / df_country_median['term_in_months']
f,ax=plt.subplots(1, 3, sharey=True, figsize=(25,10))

sns.barplot(y='country', x='term_in_months', data=df_country_median, alpha=0.6, ax=ax[0])
ax[0].set_title("Медиана срока выплаты займа", fontsize=20)
ax[0].set_xlabel('Количество месяцев', fontsize=18)
ax[0].set_ylabel(None)
ax[0].tick_params(labelsize=16)

sns.barplot(y='country', x='loan_amount', data=df_country_median, alpha=0.6, ax=ax[1])
ax[1].set_title("Медиана суммы займа", fontsize=20)
ax[1].set_xlabel('Сумма в долл. США', fontsize=18)
ax[1].set_ylabel(None)
ax[1].tick_params(labelsize=16)

sns.barplot(y='country', x='monthly_repayment', data=df_country_median, alpha=0.6, ax=ax[2])
ax[2].set_title("Медиана ежемесячного платежа", fontsize=20)
ax[2].set_xlabel('Сумма в долл. США', fontsize=18)
ax[2].set_ylabel(None)
ax[2].tick_params(labelsize=16)

plt.tight_layout()
plt.show()
genders_by_region = df_kiva_loans.groupby(
    ["borrower_genders", "world_region"], as_index=False)["loan_amount"].count()
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 5))

for ax, gender in zip(axes.ravel(), genders_by_region["borrower_genders"].unique()):
    
    df_reg = genders_by_region[genders_by_region["borrower_genders"] == gender]
    
    sns.barplot(y='world_region', x='loan_amount', data=df_reg, alpha=0.6, ax=ax)

    ax.set_title(gender, fontsize=20)
    ax.yaxis.label.set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel('Количество заемщиков', fontsize=18)
    plt.tight_layout()
genders_by_sector = df_kiva_loans.groupby(
    ["borrower_genders", "sector"], as_index=False)["loan_amount"].count()
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(20, 6))

for ax, gender in zip(axes.ravel(), genders_by_sector["borrower_genders"].unique()):
    
    df_sec = genders_by_sector[genders_by_sector["borrower_genders"] == gender]
    
    sns.barplot(y='sector', x='loan_amount', data=df_sec, alpha=0.6, ax=ax)

    ax.set_title(gender, fontsize=20)
    ax.yaxis.label.set_visible(False)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Количество заемщиков', fontsize=16)
    plt.tight_layout()
loans_not_null_lender = df_kiva_loans[df_kiva_loans.lender_count > 0]
loans_by_lender_count = loans_not_null_lender.groupby("lender_count")["loan_amount"].median()
loans_by_lender_count = loans_by_lender_count[loans_by_lender_count < 100000]

plt.figure(figsize=(20, 8))
plt.title("Влияние количества кредиторов на суммы займов", fontsize=18)
plt.xlabel("Количество кредиторов", fontsize=14)
plt.ylabel("Сумма займа, долл. США", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.scatter(loans_by_lender_count.index, loans_by_lender_count.values, c="b");
term_by_lender_count = loans_not_null_lender.groupby("lender_count")["term_in_months"].median()
term_by_lender_count = term_by_lender_count[term_by_lender_count.index < 2000]

plt.figure(figsize=(20, 8))
plt.title("Влияние количества кредиторов на сроки займов", fontsize=18)
plt.xlabel("Количество кредиторов", fontsize=14)
plt.ylabel("Срок займа, месяцев", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.scatter(term_by_lender_count.index, term_by_lender_count.values, c="b"); 
plt.figure(figsize=(20, 8))
plt.title("Влияние показателя MPI на суммы займов", fontsize=18)
plt.xlabel("Показатель MPI", fontsize=14)
plt.ylabel("Сумма займа, долл. США", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.scatter(df_kiva_loans.MPI, df_kiva_loans.loan_amount, c="b");
plt.figure(figsize=(20, 8))
plt.title("Влияние показателя MPI на сроки займов", fontsize=18)
plt.xlabel("Показатель MPI", fontsize=14)
plt.ylabel("Срок займа, месяцев", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.scatter(df_kiva_loans.MPI, df_kiva_loans.term_in_months, c="b");
loan_amount_by_world_region = df_kiva_loans.groupby("world_region")["loan_amount"].median()

term_in_months_by_world_region = df_kiva_loans.groupby("world_region")["term_in_months"].median()

ft_tmp = df_kiva_loans[["world_region", "posted_time", "funded_time"]].dropna()
ft_tmp["delta_time"] = pd.to_datetime(ft_tmp.funded_time) - pd.to_datetime(ft_tmp.posted_time)
ft_tmp.delta_time = ft_tmp.delta_time.round("D").dt.days.astype("int")
funding_time_by_world_region = ft_tmp.groupby("world_region")["delta_time"].median()

mr_temp = df_kiva_loans[["world_region", "loan_amount", "term_in_months"]].dropna()
mr_temp["monthly_repayment"] = mr_temp.loan_amount / mr_temp.term_in_months
monthly_repayment_by_world_region = mr_temp.groupby("world_region")["monthly_repayment"].median()

f, ax = plt.subplots(1, 4, sharey=True, figsize=(25, 5))


sns.barplot(
    y=loan_amount_by_world_region.index, 
    x=loan_amount_by_world_region.values, 
    ax=ax[0]
)
ax[0].set_title("Медиана суммы займа", fontsize=16)
ax[0].set_xlabel('Сумма в долл. США', fontsize=14)
ax[0].set_ylabel(None)
ax[0].tick_params(labelsize=12)

sns.barplot(
    y=term_in_months_by_world_region.index, 
    x=term_in_months_by_world_region.values, 
    ax=ax[1]
)
ax[1].set_title("Медиана срока выплаты займа", fontsize=16)
ax[1].set_xlabel('Количество месяцев', fontsize=14)
ax[1].set_ylabel(None)
ax[1].tick_params(labelsize=12)

sns.barplot(
    y=funding_time_by_world_region.index, 
    x=funding_time_by_world_region.values, 
    ax=ax[2]
)
ax[2].set_title("Медиана времеми финансирования заявки", fontsize=16)
ax[2].set_xlabel('Количество дней', fontsize=14)
ax[2].set_ylabel(None)
ax[2].tick_params(labelsize=12)

sns.barplot(
    y=monthly_repayment_by_world_region.index, 
    x=monthly_repayment_by_world_region.values, 
    ax=ax[3]
)
ax[3].set_title("Медиана ежемесячного платежа", fontsize=16)
ax[3].set_xlabel('Сумма в долл. США', fontsize=14)
ax[3].set_ylabel(None)
ax[3].tick_params(labelsize=12)

f.suptitle("В разрезе макрорегионов", fontsize=22)
plt.tight_layout(5)
plt.show()