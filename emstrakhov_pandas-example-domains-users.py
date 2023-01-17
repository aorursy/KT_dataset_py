# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
users = pd.read_excel('/kaggle/input/domains-and-users/Test.xlsx')

users.head()
users.shape
countries = pd.read_excel('/kaggle/input/domains-and-users/Additional Information.xlsx')

countries.head()
# 1. Возможные стоимости

values = np.arange(5.5, 13.6, 0.5)

# 2. Количество уникальных доменов

num_of_domains = users['Media'].nunique()

# 3. Случайная выборка из values нужного размера (по числу уникальных доменов)

sample_values = np.random.choice(values, num_of_domains)

# 4. Множество уникальных доменов

set_of_domains = users['Media'].unique()

# 5. Составляем словарь из пар "домен":"стоимость"

domain_dict = dict(zip(set_of_domains, sample_values))

# 6. Добавляем колонку "стоимость" к таблице users

users['value'] = users['Media'].map(domain_dict)

users.head()
# 1. "Отрезаем" от адреса часть после последней точки

# 2. Переводим в верхний регистр

# 3. Даём хостингам .com, .net, .tv, .org имя Worldwide

def get_host(address):

    host_tail = address.split('.')[-1].upper() # 1 и 2

    if host_tail in ('COM', 'NET', 'TV', 'ORG'): # 3

        host_tail = 'Worldwide'

    return host_tail
# Создаём новую колонку в таблице users по построенному правилу

users['host'] = users['Media'].apply(get_host)

users.head()
users.tail()
# Определяем страну путём слияния (merge) двух таблиц

# Аналоги операции --- в Spreadsheets/Excel? --- в SQL?

users = users.merge(countries, how='left', 

                    left_on='host', right_on='ISO code')

users.head()
users.tail()
# Заполняем пропуски

users['country'].fillna('Worldwide', inplace=True)

users.head()
# Удаляем лишнюю колонку ISO code

users.drop('ISO code', axis=1, inplace=True)

users.head()
# Фильтр по условию

ua = users[ users['country']=='Ukraine' ]

ua.head()
# Сортировка и топ-5

ua_sorted = ua.sort_values('avails', ascending=False)

ua_sorted.head(5)
users.groupby('Media')['avails'].sum()
users_sorted = users.sort_values(['avails', 'country', 'Media'], 

                                 ascending=[False, True, True])

users_sorted.head(10)
users.groupby('country')['avails'].sum()
users.groupby(['country', 'Media'])['avails'].sum()
top5_by_country = users.sort_values(['country', 'avails', 'Media'], 

                                    ascending=[True, False, True]).groupby('country').head(5)
top5_by_country.head(20)
# Для инфографики возьмём Украину, Польшу и Германию

ua_de_pl = top5_by_country[ top5_by_country['country'].isin(['Ukraine', 'Poland', 'Germany']) ]

ua_de_pl
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 15))

for i, cntry in zip(range(3), ua_de_pl['country'].unique()):

    df = ua_de_pl[ ua_de_pl['country']==cntry ]

    sns.barplot(y='Media', x='avails', data=df, palette="Blues_d", ax=ax[i])

    ax[i].set_title(cntry)

plt.show()