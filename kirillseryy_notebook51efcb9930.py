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
#импортируем необходимые библиотеки
import bq_helper
from bq_helper import BigQueryHelper
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import graph_objects as go
#изменим опции отображения таблиц
pd.set_option('max_colwidth', 1000)
pd.set_option('max_columns', 50)
#создадим объект BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
#изучим данные
display(bq_assistant.list_tables())
display(bq_assistant.table_schema("crime"))
display(bq_assistant.head("crime", num_rows=3))
#напишем запрос на получение столбцов с битами и датами
QUERY = """
        SELECT beat, date
        FROM `bigquery-public-data.chicago_crime.crime`
        """
display(bq_assistant.estimate_query_size(QUERY))
df = bq_assistant.query_to_pandas_safe(QUERY)
#проверка
df.head()
#извлечем часы из каждой даты
df['hour'] = pd.DatetimeIndex(df['date']).hour
#вычислим, в какие часы опаснее всего появляться на улицах Чикаго
dangerous_hours = df.groupby('hour',as_index=False).agg({'date':'count'}).sort_values(by='date', ascending=False)
#проверка
dangerous_hours
#напишем функцию для построения столбчатого графика
def barplot_constructor(data, x, y, title, x_ax, y_ax):
    fig = px.bar(data, x=x, y=y, title=title, labels={x:x_ax,y:y_ax})
    fig.show()
#визуализируем построенную таблицу
barplot_constructor(dangerous_hours,'hour','date','Распределение преступлений по часам','Час','Количество совершаемых преступлений')
#определим наиболее востребованный временной интервал для патрулирования
table = pd.DataFrame(columns=('interval','total_crimes'))
total_crimes = 0
for interval in range(0,12):
    hour1 = interval
    hour2 = interval+1
    hour3 = interval+2
    hour4 = interval+12
    hour5 = interval+13
    hour6 = interval+14
    if interval == 10:
        hour6 = interval-10
    if interval == 11:
        hour5 = interval-11
        hour6 = interval-10
    for hour in (hour1,hour2,hour3,hour4,hour5,hour6):
        total_crimes += dangerous_hours.loc[hour]['date']
    table.loc[interval,'interval'] = str(hour1)+':00-'+str(hour3)+':59',str(hour4)+':00-'+str(hour6)+':59'
    table.loc[interval,'total_crimes'] = total_crimes
display(table.sort_values(by='total_crimes',ascending=False))
#сформируем датафрейм из данных, соответствующих часам патрулирования, определенным в предыдущем разделе
patrol_hours = df.query('hour in [11,12,13,23,0,1]')
#проверка
patrol_hours.groupby('hour',as_index=False).agg({'beat':'count'})
#построим сводную таблицу числа совершаемых преступлений в каждом бите в каждый час патрулирования
beats_pivot = patrol_hours.pivot_table(index='beat',columns='hour',values='date',aggfunc='count')
#проверка
beats_pivot
#визуализируем построенную сводную таблицу в виде тепловой карты
fig_dims = (25, 30)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(data=beats_pivot, ax=ax)
#создадим список 50 самых опасных районов в часы патрулирования
top50_beats = patrol_hours.groupby('beat').agg({'date':'count'}).sort_values(by='date',ascending=False).head(50)
#визуализируем построенную таблицу
plt.subplots(figsize=(15,10))
top50_beats['date'].plot(kind='barh',grid=True,ax=plt.subplot(111))
plt.show()
#создадим список из названий 50 самых опасный районов
top50_beats = top50_beats.reset_index()
top50_beats_list = top50_beats['beat'].to_numpy()
#вывод
display(top50_beats_list)
#напишем запрос на получение количества преступлений по типам
QUERY = """
        SELECT primary_type, COUNT(unique_key)
        FROM `bigquery-public-data.chicago_crime.crime`
        GROUP BY primary_type
        """
display(bq_assistant.estimate_query_size(QUERY))
crimes_by_types = bq_assistant.query_to_pandas_safe(QUERY)
crimes_by_types = crimes_by_types.rename(columns={'f0_':'crimes_number'})
#проверка
crimes_by_types
#создадим список всех преступлений
crimes_list = crimes_by_types['primary_type'].to_numpy()
#вывод
display(crimes_list)
#вручную рассортируем список преступлений на те, в которых потерпевшим наносится физический вред, и те, где этого нет
crimes_with_violence = ('CRIM SEXUAL ASSAULT','BATTERY','KIDNAPPING','OFFENSE INVOLVING CHILDREN','ASSAULT','WEAPONS VIOLATION','HOMICIDE',
                           'SEX OFFENSE','CRIMINAL SEXUAL ASSAULT','HUMAN TRAFFICKING','PUBLIC INDECENCY','DOMESTIC VIOLENCE')
crimes_without_violence = ('THEFT', 'DECEPTIVE PRACTICE', 'CRIMINAL TRESPASS', 'NARCOTICS', 'OTHER OFFENSE', 'ARSON', 'BURGLARY', 'ROBBERY', 'MOTOR VEHICLE THEFT', 
                          'INTERFERENCE WITH PUBLIC OFFICER','STALKING', 'PUBLIC PEACE VIOLATION', 'CRIMINAL DAMAGE','OBSCENITY', 'GAMBLING', 'LIQUOR LAW VIOLATION',
                           'NON-CRIMINAL (SUBJECT SPECIFIED)','INTIMIDATION', 'OTHER NARCOTIC VIOLATION','PROSTITUTION', 'CONCEALED CARRY LICENSE VIOLATION',
                           'NON-CRIMINAL','RITUALISM','NON - CRIMINAL')
#добавим в таблицу отметки о том, нужно ли доставать оружие при конкретном типе преступлнения
for i in range(len(crimes_by_types)):
    if crimes_by_types.loc[i,'primary_type'] in crimes_with_violence:
        crimes_by_types.loc[i,'violent'] = True
    else:
        crimes_by_types.loc[i,'violent'] = False
#проверка
crimes_by_types
#определим общее количество и долю жестоких преступлений
violent_crimes = crimes_by_types.groupby('violent',as_index=False).agg({'crimes_number':'sum'})
#вывод
display(violent_crimes)
print("Доля преступлений с причинением физического насилия жертве: {:.2%}"\
                                                              .format((violent_crimes.query('violent==True')['crimes_number']/sum(violent_crimes['crimes_number']))[1]))
#визуализируем построенную таблицу
fig = go.Figure(data=[go.Pie(labels=violent_crimes['violent'], values=violent_crimes['crimes_number'], title='Распределение преступлений по причастности к жестоким')])
fig.show()
