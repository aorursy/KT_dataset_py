



import pandas as pd # для работы с таблицами

import numpy as np # для работы со списками





path='../input/table.xlsx' # путь к фаилу

shn='Лист1' # название листа



df=pd.read_excel(path, sheet_name=shn)

print(df)



# работа со строчками



# убрать дубликаты (нет пропусков)

df=df.drop_duplicates()





# убрать те строчки где нет значений

df=df.dropna()

                 
    

# разность колонок

df['ост сумма']=df[['Сумма выдачи','Итого сумма']].diff(axis=1).iloc[:,1]



# деление колонок

df['доходность']=df['Итого сумма']/df['Сумма выдачи']





# найти все строчки 

df=df.loc[df['ID клиента'].isin([124577,124578])]





# взять все договора, очистив

ig=df['Ident'].dropna().tolist()

ig=[float(''.join([ii for ii in i if ii.isdigit()]) ) if type(i) is str else i for i in ig]

print(ig)
# Добавить строчку

new=pd.DataFrame({'ID клиента': 124891,

 'Ident': 765666,

 'Дата выдачи': '13.09.2019',

 'Дата операции': '09.09.2019 13:20',

 'Дата расчёта': '01.10.2019',

 'Итого сумма': 2300,

 'Мес выдачи': '01.05.2019'

 },index=[df.shape[0]])





df=df.append(new,'sort=True')




#Деление перменных на категории



# найти все строчки где Итого сумма более нуля

df=df.loc[df['Сумма выдачи']>30000]



# поделить на 5 равных 

df["bins"] = pd.cut(df['Сумма выдачи'], bins = 5).apply(lambda x: x.mid)





df=df.drop('Сумма выдачи',axis=1)

        


#колонки в несколько уровней

va='займ '

s=df.columns.tolist()

tuples = list(zip(*[[va]*len(s)]+[s]))

multi_index= pd.MultiIndex.from_tuples(tuples, names=['тип','колонка'])

df.columns=multi_index

df.to_excel('res.xlsx')


