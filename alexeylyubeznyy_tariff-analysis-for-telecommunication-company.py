import pandas as pd

import seaborn as sns

sns.set(color_codes=True)



import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as st

from functools import reduce

idx = pd.IndexSlice

from scipy import stats as st



calls = pd.read_csv('../input/calls.csv', index_col=0)

internet = pd.read_csv('../input/internet.csv', index_col=0)

messages = pd.read_csv('../input/messages.csv', index_col=0)

tariffs = pd.read_csv('../input/tariffs.csv', index_col=0)

users = pd.read_csv('../input/users.csv', index_col=0)



for data in [calls, internet, messages, users, tariffs]: print(data.info(), end='\n\n')
calls['call_date'] = pd.to_datetime(calls['call_date'], format='%Y-%m-%d')

internet['session_date'] = pd.to_datetime(internet['session_date'], format='%Y-%m-%d')

messages['message_date'] = pd.to_datetime(messages['message_date'], format='%Y-%m-%d')

users['churn_date'] = pd.to_datetime(users['churn_date'], format='%Y-%m-%d')

users['reg_date'] = pd.to_datetime(users['reg_date'], format='%Y-%m-%d')
print('Количество звонков с длительностью, равной нулю:', calls.loc[calls['duration'] == 0, 'duration'].value_counts()[0])
# добавим столбец со значением месяца

calls['month'] = calls['call_date'].dt.month



# построим график зависимости количества нулевых значений от общего числа звонков за каждый месяц

calls.loc[calls['duration'] == 0, 'nulls_count'] = 1

null_calls = calls.pivot_table(index='month', values=['id', 'nulls_count'], aggfunc='count')

null_calls['null_ratio'] = round(null_calls['nulls_count'] / null_calls['id'], 3)



sns.relplot(x='id', y='nulls_count', data=null_calls)

plt.xlabel('Общее число звонков по месяцам')

plt.ylabel('Число звонков с нулевой продолжительностью')

_ = plt.title('Зависимость количества нулевых значений от общего числа звонков за каждый месяц')
calls['duration'] = calls['duration'].apply(np.ceil)

calls.loc[calls['duration'] == 0, 'duration'] = 1
# добавим столбец со значением месяца

internet['month'] = internet['session_date'].dt.month



# построим график зависимости количества нулевых значений от общего числа Интернет-сессий за каждый месяц

internet.loc[internet['mb_used'] == 0, 'nulls_count'] = 1

null_mb = internet.pivot_table(index='month', values=['id', 'nulls_count'], aggfunc='count')

null_mb['null_ratio'] = round(null_mb['nulls_count'] / null_mb['id'], 3)



sns.relplot(x='id', y='nulls_count', data=null_mb, color='tab:red')

plt.xlabel('Общее число интернет-сессий по месяцам')

plt.ylabel('Число интернет-сессий с нулевым трафиком')

_ = plt.title('Зависимость количества нулевых значений от общего числа интернет-сессий за каждый месяц')
internet['mb_used'] = internet['mb_used'].apply(np.ceil)

internet.loc[internet['mb_used'] == 0, 'mb_used'] = 1
# для подсчета при создании сводной таблицы сгенерируем уникальный код из имени и населенного пункта

users['unique_name'] = users['city'] + users['first_name'] + users['last_name']



# сводная таблица по звонкам

calls_temp = calls.merge(users, on='user_id', how='inner')

calls_grouped = calls_temp.pivot_table(values=['unique_name', 'duration'], 

                       index=['user_id', 'month'], 

                       aggfunc={'unique_name': 'count', 'duration': 'sum'})

calls_grouped.columns = ['duration', 'calls']



# сводная таблица по сообщениям

messages['month'] = messages['message_date'].dt.month

messages_temp = messages.merge(users, on='user_id', how='inner')

messages_grouped = messages_temp.pivot_table(values='unique_name', 

                       index=['user_id', 'month'], 

                       aggfunc={'unique_name': 'count'})

messages_grouped.columns = ['messages']



# сводная таблица по интернет-трафику

internet['month'] = internet['session_date'].dt.month

internet_temp = internet.merge(users, on='user_id', how='inner')

internet_grouped = internet_temp.pivot_table(values='mb_used', 

                       index=['user_id', 'month'], 

                       aggfunc={'mb_used': 'sum'})

internet_grouped.columns = ['mb_used']



# объединяем все три сводные таблицы в одну

dfs = [calls_grouped, messages_grouped, internet_grouped]

grouped_data = reduce(lambda left,right: pd.merge(left,right,on=['user_id', 'month'], how='outer'), dfs)

grouped_data.head(10)
# пропущенные значения заменяем на нули

grouped_data.loc[grouped_data['duration'].isna(), 'duration'] = 0

grouped_data.loc[grouped_data['messages'].isna(), 'messages'] = 0

grouped_data.loc[grouped_data['mb_used'].isna(), 'mb_used'] = 0



# добавим наименование используемого тарифа для каждого пользователя

for user in grouped_data.index:

    grouped_data.loc[user, 'tariff'] = users.loc[user[0]-1000, 'tariff']



# функция подсчета выручки с каждого пользователя в месяц

def det_revenue(row):

    messages = row['messages']

    mb_used = row['mb_used']

    tariff = row['tariff']

    duration = row['duration']

    calls = row['calls']

    

    if tariff == 'smart':

        extra_duration = duration - tariffs.loc[0, 'minutes_included']

        extra_mb = mb_used - tariffs.loc[0, 'mg_per_month_included']

        extra_messages = messages - tariffs.loc[0, 'messages_included']

        

        if extra_duration < 0: extra_duration = 0

        if extra_mb < 0: extra_mb = 0

        if extra_messages < 0: extra_messages = 0

        

        return (tariffs.loc[0, 'rub_per_message'] * extra_messages + 

                   (tariffs.loc[0, 'rub_per_gb'] / 1024) * extra_mb + 

                   tariffs.loc[0, 'rub_per_minute'] * extra_duration + 

                   tariffs.loc[0, 'rub_monthly_fee']

                  ) 

    else:

        extra_duration = duration - tariffs.loc[1, 'minutes_included']

        extra_mb = mb_used - tariffs.loc[1, 'mg_per_month_included']

        extra_messages = messages - tariffs.loc[1, 'messages_included']

        

        if extra_duration < 0: extra_duration = 0

        if extra_mb < 0: extra_mb = 0

        if extra_messages < 0: extra_messages = 0

        

        return (tariffs.loc[1, 'rub_per_message'] * extra_messages + 

                   (tariffs.loc[1, 'rub_per_gb'] / 1024) * extra_mb + 

                   tariffs.loc[1, 'rub_per_minute'] * extra_duration + 

                   tariffs.loc[1, 'rub_monthly_fee']

                  )



grouped_data['revenue'] = grouped_data.apply(det_revenue, axis=1)

grouped_data.head(10)
medians = grouped_data.pivot_table(index='user_id', values=['duration', 'messages', 'mb_used', 'revenue'], aggfunc='median')

for user_id in users['user_id']:

    medians.loc[user_id, 'tariff'] = users.loc[user_id-1000, 'tariff']

medians.head()
# описательная статистика:

medians.dropna(subset=['duration'], inplace=True)

desc_stat = medians.pivot_table(index=['tariff'], values=['duration', 'mb_used', 'messages'], 

                    aggfunc={'duration': [np.median, np.var, np.std], 

                             'mb_used': [np.median, np.var, np.std], 

                             'messages': [np.median, np.var, np.std]})

desc_stat
for column in ['duration', 'mb_used', 'messages']:    

    sns.catplot(x="tariff", y=column, kind="box", data=medians, orient='v')
for tariff in ['ultra', 'smart']:

    sns.distplot(medians.query('tariff == @tariff')['duration'], kde=False, label=tariff)

_ = plt.legend(['ultra', 'smart'])
for tariff in ['ultra', 'smart']:

    sns.distplot(medians.query('tariff == @tariff')['mb_used'], kde=False)

_ = plt.legend(['ultra', 'smart'])
for tariff in ['ultra', 'smart']:

    sns.distplot(medians.query('tariff == @tariff')['messages'], kde=False, label=tariff)

_ = plt.legend(['ultra', 'smart'])
ultra = grouped_data.query('tariff == "ultra"')['revenue']

smart = grouped_data.query('tariff == "smart"')['revenue']



alpha = .01



results = st.ttest_ind(

    ultra, 

    smart, 

    equal_var=False)



print('p-значение:', results.pvalue)



if (results.pvalue < alpha):

    print("Отвергаем нулевую гипотезу")

else:

    print("Не получилось отвергнуть нулевую гипотезу")
grouped_data.pivot_table(index='tariff', values='revenue', aggfunc='median')
_ = sns.catplot(x="tariff", y='revenue', kind="box", data=grouped_data, orient='v')
# добавим наименование населенного пункта для каждого пользователя

for user in grouped_data.index:

    grouped_data.loc[user, 'city'] = users.loc[user[0]-1000, 'city']



moscow = grouped_data.query('city == "Москва"')['revenue']

regions = grouped_data.query('city != "Москва"')['revenue']



alpha = .01



results = st.ttest_ind(

    moscow, 

    regions, 

    equal_var=False)



print('p-значение:', results.pvalue)



if (results.pvalue < alpha):

    print("Отвергаем нулевую гипотезу")

else:

    print("Не получилось отвергнуть нулевую гипотезу")
def det_region(city):

    if city == 'Москва': return 'Москва'

    else: return 'Другой регион'

    

grouped_data['region'] = grouped_data['city'].apply(det_region)

grouped_data.pivot_table(index='region', values='revenue', aggfunc='median')
_ = sns.catplot(x="region", y='revenue', kind="box", data=grouped_data, orient='v')