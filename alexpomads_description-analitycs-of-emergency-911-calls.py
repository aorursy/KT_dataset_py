import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd 

from datetime import datetime
df_call = pd.read_csv('../input/911.csv')

df_call.head()
df_call['twp'].value_counts()

name_pol = [x for x in list(set(df_call['twp'])) if str(x) != 'nan']

x_twp=np.arange(len(df_call['twp'].value_counts()))

pol_stat_ind=pd.DataFrame.transpose(pd.DataFrame(name_pol,x_twp))

pol_stat_ind.index=['Township']

#Index of Police Station for the Bar Plot

pol_stat_ind
plt.bar(x_twp,df_call['twp'].value_counts(),width=0.5)

plt.xlabel("Township")

plt.ylabel("Number of Calls")
time_call=df_call['timeStamp']

aux=[]

for i in time_call:

    aux.append(i[0:10])

time_call=[[x,aux.count(x)]for x in set(aux)]

time_call=sorted(time_call)



aux=pd.DataFrame(time_call)

tab_date=pd.DataFrame.transpose(pd.DataFrame(aux))

tab_date.index=['Date','Call Numbers']

tab_date
date_str=list(aux[0])

count=list(aux[1])

date=[]

for i in date_str:

    date.append(datetime.strptime(i,'%Y-%m-%d').date())

fig, ax = plt.subplots()

ax.plot_date(date, count, '-')

fig.autofmt_xdate()

plt.xlabel('Date')

plt.ylabel('Number of Calls')

plt.grid(True)

plt.show()