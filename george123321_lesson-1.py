import pandas as pd # для работы с таблицами

import numpy as np # для работы со списками



import datetime # для работы с датами

from dateutil.relativedelta import relativedelta # для работы с датами
path='../input/fin111/table.xlsx' # путь к фаилу

shn='Лист1' # название листа



d=pd.read_excel(path,sheet_name=shn)

print(d)
# оставить все кроме первой строчки

d=d.iloc[1:,:]

print(d)



d['Дата операции']=pd.to_datetime(d['Дата операции'],format='%d.%m.%Y %H:%M') 

# поменять формат столбца на формат даты       





d['Дата выдачи']=pd.to_datetime(d['Дата выдачи'],format='%d.%m.%Y') 

# поменять формат столбца на формат даты       







d['Дата операции']=pd.to_datetime(d['Дата операции'].map(lambda x: x.date()  ))

# оставить только день-мес-год















d['delta']=d['Дата операции']-d['Дата выдачи']

d['delta']=d['delta'].apply(lambda l: l.days)

# найти разницу в днях





d['paymentDate']=d['Дата выдачи'].map(lambda x: datetime.datetime.now()- datetime.timedelta(days=7))

# вычесть из текущей даты 7 дней и получившуюся дату записать в результат





d1=d.loc[(d['Дата операции']>d['Дата выдачи']+ datetime.timedelta(days=30))|(d['Дата операции']<d['Дата выдачи']+datetime.timedelta(days=1)),:]

# выбрать все строчки в которых:

# дата операции более date +7 дней

# или

# дата операции менее date +1 дней



print(d)
d=d.sort_values(['Дата операции','Ident'])

# сортировка значений по колонкам: дата операции и Договор - по обоим по убыванию



d.columns=['Договор', 'Дата операции', 'Итого сумма', 'Дата выдачи', 'Дата расчёта',

       'Мес выдачи', 'delta', 'paymentDate']

# присловить всем плонкам новые названия



d=d[['Договор','Дата операции', 'Итого сумма' ]]

# выбрать только некоторые колонки



print(d)
d.to_excel('res.xlsx',index=False)

# записать таблицу в фаил





writer = pd.ExcelWriter('res2.xlsx', engine='xlsxwriter')

d[['Договор','Дата операции']].to_excel(writer, sheet_name='лист1',index=False)

d[['Договор','Итого сумма' ]].to_excel(writer, sheet_name='лист2',index=False)

writer.save()

# записать две таблицы в один фаил



print(d)