# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# строка 71083 -- убрать лишний разделитель



reestr = pd.read_csv('../input/alfa sql challenge/pdv_actual_28-05-2019.csv', 

                     sep=';', quotechar='"')

reestr.head()
reestr.info()
# строка 876133 -- пофиксить



borg = pd.read_csv('../input/alfa sql challenge/borg_01052019.csv', 

                   sep=';')

borg.head()
borg.info()
anul = pd.read_csv('../input/alfa sql challenge/pdv_anul_28-05-2019.csv', 

                   sep=';', quotechar='"')

anul.head()
anul.info()
vpp = pd.read_csv('../input/alfa sql challenge/dodatok_do_nakazu_reiestr_2019_zmini.csv', 

                  sep=';', index_col=0, dtype='object')

vpp.head()
vpp.info()
reestr_ms = pd.read_csv('../input/alfa sql challenge/reiestr_-mc_22-05-19.csv', 

                        sep=';')

reestr_ms.head()
reestr_ms.info()
import xml.etree.ElementTree as etree



def parse_XML(xml_file, attributes): 

    """Parse the input XML file and store the result in a pandas DataFrame 

    with the given columns."""

    

    df = []

    for event, elem in etree.iterparse(xml_file, events=('start', 'end')):

        if event == 'start' and elem.tag == 'RECORD':

            row = {f : None for f in attributes}

        elif event == 'end' and elem.tag == 'RECORD':

            df.append(row)

        elif event == 'start' and elem.tag in attributes:

            row[elem.tag] = elem.text



    out_df = pd.DataFrame(df, columns=attributes) 

    

    return out_df
# Работает примерно 10 минут



uop_attributes = ["NAME", "SHORT_NAME", "EDRPOU", "ADDRESS", "BOSS", "KVED", "STAN"]

fop_attributes = ["FIO", "ADDRESS", "KVED", "STAN"]

bankrupt_attributes = ["RN_NUM", "REG_DATE", "REG_NUM", "DATE_START", "NAME",

                       "STAN_PROV", "PROC_BORG_NAME", "INIC_SPR"]



# Юридические лица

uop_df = parse_XML('../input/alfa sql challenge/15.1-EX_XML_EDR_UO.xml', 

                   uop_attributes)



# ФОПы

fop_df = parse_XML('../input/alfa sql challenge/15.2-EX_XML_EDR_FOP.xml', 

                   fop_attributes)



# Банкроты

bankrupt_df = parse_XML('../input/alfa sql challenge/25-EX_XML_RBA.xml', 

                        bankrupt_attributes)

fop_df.info()
uop_df.info()
bankrupt_df.info()
# Соедним реестр юридических лиц и "больших" налогоплательщиков по коду ЕДРПОУ



# Переименуем колонки в vpp

vpp.columns = ['edrpou', 'name']

vpp.head()
# Соединяем



df_1 = pd.merge(vpp, uop_df, left_on='edrpou', right_on='EDRPOU')

df_1.drop('edrpou', axis=1, inplace=True)

df_1.head()
df_1['KVED'].value_counts().sort_values(ascending=False).head(20)
df_1['KVED'].value_counts().sort_values(ascending=False).head(20).plot(kind='barh');
# Соединим таблицу должников с df_1 по ЕДРПОУ



# Есть проблема с пропусками в TIN_S

borg['TIN_S'].fillna(0, inplace=True)
# Переведём TIN_S сначала в целое число, потом в строку

borg['TIN_S'] = borg['TIN_S'].astype(int).astype(str)
# Проверим

borg['TIN_S']
# Соединяем



df_2 = pd.merge(borg, uop_df, left_on='TIN_S', right_on='EDRPOU')

df_2.head()
# Какая сфера встречается чаще среди должников



df_2['KVED'].value_counts().sort_values(ascending=False).head(10)
fop_df['FIO'].head(5)
pib = set(borg['PIB'])

fio = set(fop_df['FIO'])

len(pib & fio)
fop_borg = fop_df[fop_df['FIO'].isin(borg['PIB'])]

fop_borg.shape[0]
# Смотрим должников среди ФОП



df_3 = pd.merge(borg, fop_borg, left_on='PIB', right_on='FIO')

df_3.head()
# Какая сфера встречается чаще среди должников ФОП



df_3['KVED'].value_counts().sort_values(ascending=False).head(10)
df_3['FIO'].value_counts().sort_values(ascending=False).head(20)
# А теперь по сумме долга



# Для этого нужно перевести числа с запятой в числа с точкой

comma_dot = lambda s: s.replace(',', '.')

df_2['SUM_D'] = df_2['SUM_D'].apply(comma_dot).astype(float)

df_2['SUM_M'] = df_2['SUM_M'].apply(comma_dot).astype(float)
# Проверим

df_2['SUM_D'].dtype, df_2['SUM_M'].dtype
# Группируем по КВЕД и вычисляем сумму долгов по предприятиям

df_2.groupby('KVED')['SUM_D'].sum().sort_values(ascending=False).head(10)
# Группируем по КВЕД и вычисляем сумму долгов по предприятиям

df_2.groupby('KVED')['SUM_M'].sum().sort_values(ascending=False).head(10)
# А теперь по сумме долга



# Для этого нужно перевести числа с запятой в числа с точкой

comma_dot = lambda s: s.replace(',', '.')

df_3['SUM_D'] = df_3['SUM_D'].apply(comma_dot).astype(float)

df_3['SUM_M'] = df_3['SUM_M'].apply(comma_dot).astype(float)
# Группируем по КВЕД и вычисляем сумму долгов по предприятиям

df_3.groupby('KVED')['SUM_D'].sum().sort_values(ascending=False).head(10)
# Группируем по КВЕД и вычисляем сумму долгов по предприятиям

df_3.groupby('KVED')['SUM_M'].sum().sort_values(ascending=False).head(10)
# Посчитаем количество больших налогоплательщиков среди должников 



# Для этого нужно соединить таблицы vpp и borg по ЕДРПОУ и найти кол-во строк

df_3 = pd.merge(vpp, borg, left_on='edrpou', right_on='TIN_S')

df_3.shape[0]
bankrupt_df.head()
bankrupt_df['RN_NUM'].describe()
bankrupt_df['RN_NUM'].value_counts()