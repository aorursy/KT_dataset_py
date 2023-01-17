



import pandas as pd

import numpy as np



from dateutil.relativedelta import relativedelta

    

import datetime





df=pd.read_excel('../input/table.xlsx')    





df['Дата выдачи']=pd.to_datetime(df['Дата выдачи'],format='%d.%m.%Y')



## нумерование займов клиентов

d0=df[['ID клиента','Дата выдачи']].sort_values(['ID клиента','Дата выдачи']).drop_duplicates()

d0['count']=d0.groupby('ID клиента')['Дата выдачи'].transform(lambda x: np.arange(x.shape[0] )+1 ) 

df=df.merge(d0,on=['ID клиента','Дата выдачи'])



print(df)



# займ клиента с самым длинным названием

d1=df.groupby(['ID клиента'])['Ident'].apply(lambda x : 

    

    x.iloc[np.argmax(x.map(lambda y: len(str(y)) ).tolist())] 

    

    ).reset_index()



# x.map(lambda y: len(str(y)) - длинна Ид для каждой строчки

# np.argmax(x.map(lambda y: len(str(y)) - номер строки с самой большой длинной      

# x.iloc[np.argmax(x.map(lambda y: len(str(y)) ).tolist())] - значение Ид по номеру строки

print(d1)

    
# статистика по займам

    

df['Дата операции']=pd.to_datetime(df['Дата операции'],format='%d.%m.%Y %H:%M')





d2=df.groupby('Ident').apply(lambda x: pd.Series({

          

    'средний платеж': x['Итого сумма'].mean(),    

    

    'станд откл платежей':  x['Итого сумма'].std(),

    

    'кол-во платежных дней': x['Дата операции'].nunique(),

    

    'сумма платежей за первый месяц': x.loc[x['Дата операции']<x['Дата выдачи']+ datetime.timedelta(days=30),'Итого сумма'].sum(),

    

    'сумма первого платежа': x.loc[x['Дата операции']==min(x['Дата операции']),'Итого сумма'].iloc[0]

    

 }) )

    

print(d2)

    



    
# кол-во мес между первым и вторым займами клиента





df1=df.groupby('ID клиента')['count'].max()

ids=df1[df1==2].index.tolist()



df1=df.loc[df['ID клиента'].isin(ids)]





def delta(x):

    

    a=x.loc[x['count']==1,'Дата выдачи'].iloc[0]

    

    b=x.loc[x['count']==2,'Дата выдачи'].iloc[0]

    

    y=[b,a]

    y=relativedelta(y[0],y[1]).months

    

    return y





d3=df1.groupby('ID клиента').apply(lambda x: pd.Series({

                        'delta months': delta(x)

                        }))

    

print(d3)