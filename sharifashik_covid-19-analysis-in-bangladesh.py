import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/COVID-19-Bangladesh.csv')
data.sample(10)
data.info()
x=data['new_confirmed'].max()
print('Highest affected a single day = ',x)
data[(data['new_confirmed']==x)]['date']

print(data['daily_collected_sam'].sum())

#maximum sam collection in a day
data[(data['daily_collected_sam']==data['daily_collected_sam'].max())]['date']
data.plot(x='date',y='total_confirmed',figsize=(30,15),kind='bar',grid='bold',fontsize=15,title='Sequence of affected')
plt.xlabel('Date',fontsize=30)
plt.ylabel('Total Affected',fontsize=30)
#sns.lineplot(x="total_recovered", y="total_deaths",hue='date' ,data= data, palette = 'BuGn')
# sns.set_style(style='whitegrid')
# sns.countplot(x='total_recovered',hue='date',data=data,palette='rainbow')
plt.figure(figsize=(20,5))
plt.plot('date','total_recovered',data=data,c='red')
plt.plot('date','total_deaths',data=data,c='green')
plt.tick_params(axis='x',labelrotation=90.0,labelsize=12.0)
plt.tick_params(axis='y', labelsize = 12.0,pad=20.0)
plt.legend(('total_recovered','total_deaths'))
plt.show()
plt.figure(figsize=(20,5))
plt.scatter('date','new_deaths',data=data,c='red')
plt.tick_params(axis='x',labelrotation=90.0)

plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
#ax = sns.countplot(x=data['daily_collected_sam'], data=data)
sns.barplot(x=data['daily_collected_sam'],y=data['new_confirmed'], data=data)
total_quarantine=data['total_quarantine'].sum()
now_in_quarantine=data['now_in_quarantine'].sum()
released_from_quarantine=data['released_from_quarantine'].sum()
quaraintine=now_in_quarantine,released_from_quarantine
colors = ["green", "red"]
label='Present quarantine','Relesed quarantine'
plt.pie(quaraintine,labels=label,colors=colors,shadow=True,autopct='%1.1f%%',startangle=140)
print('Total quarantine =',total_quarantine)
#data['total_quarantine'].plot(x=data['date'],kind='bar',figsize=(30,20),fontsize='12',color='red')
data.plot(x='date',y='total_quarantine',figsize=(30,15),kind='bar',grid='bold',fontsize=20,color='red')
plt.title('Total Quarantine',fontdict={'fontsize':20})



