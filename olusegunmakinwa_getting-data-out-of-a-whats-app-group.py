import numpy as np ##for linear algebra
import pandas as pd #data cleaning and wrangling 
import re ##to extract strings using a parttern 
import emoji #to acess emoji
import matplotlib.pyplot as plt #visualisation
import seaborn as sns #visualisation
d=open(r"C:\Users\User\Downloads\WhatsApp Chat with Discussing Football",mode='r', encoding = 'utf-8')
data=d.read()
data
datep=re.compile('\d+/\d+/\d+')
date=re.findall(datep, data)
date
timep=re.compile('\d+:\d+')
time=re.findall(timep, data)
time
senderp=re.compile('-.*?:')
sender1=re.findall(senderp, data)
sender1
messdata=re.sub('\d+/\d+/\d+,\s\d+:\d+\s-','',data)
messdata
messagep=re.compile('\d+/\d+/\d+,\s\d+:\d+\s-.*:.*')
dtu=re.findall(messagep, data)
dtu
date=list(map(lambda t : t.split("-")[0],dtu))
#split the lines into 2 using the '-' sign
len(date)
date
time=list(map(lambda x:x.split(',')[1].strip(' '),date))
time
#Seperate the time with the comma and put it in a list 
len(time)
date=list(map(lambda x:x.split(',')[0].strip(' '),date))
date
sender_message=list(map(lambda t : t.split("-")[1],dtu))
sender_message
#getting the names and the messengers 
sender=list(map(lambda sm: sm.split(':')[0].lstrip(' '), sender_message))
sender
message=list(map(lambda sm: sm.split(':')[1].lstrip(' '), sender_message))
message
#getting the messages 
df=pd.DataFrame({'Date':date, 'Time':time, 'Sender':sender, 'Message':message})
#creating a dataframe from what was collected 
df.index+=1
df
df["Letter_Count"]=df['Message'].apply(lambda w: len(w.replace(' ','')))
df["Word_Count"]=df['Message'].apply(lambda w: len(w.split(' ')))
df['Avg_Word_length']=df['Letter_Count']//df["Word_Count"]
df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf-8-sig')
df[df['Word_Count']==df['Word_Count'].max()]
longest_sentences=df.sort_values(by=['Word_Count'],ascending=False).head(5)
df.sort_values(by=['Letter_Count'],ascending=False).head(5)
df.sort_values(by=['Avg_Word_length'],ascending=False).head(5)
number_of_messages=df.Sender.value_counts()
Active_members=number_of_messages.head(5)
least_activeMembers=number_of_messages.tail(5)
df.query('Message=="You deleted this message"')
df=df.drop([5078,19199,19200], axis=0)
todrop=df.query('Message=="This message was deleted"')
df['Message'].value_counts()["This message was deleted"]
todrop["Sender"].value_counts().head(5)
df['Hour']=df['Time'].apply(lambda t:t.split(':')[0]).astype('int64')
df["Hour"]
df['Hour'].value_counts().head(10)
df['Hour'].value_counts().tail(5)
df['Hour'].dtypes
nocturnals=df[df['Hour']<6]
nocturn=nocturnals['Sender'].value_counts()
df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf-8-sig',index=False)

plt.axes([1,1,1,0.98])
plt.grid(True)
nocturn.plot.bar()
plt.xlabel('Guys That Message at Night')
plt.ylabel('No. of Messages')
plt.xticks(rotation=90)
plt.show()
%matplotlib inline
rush_hours=df['Hour'].value_counts().head(10)
rush_hours.sort_index().plot.bar()
plt.xlabel('hours')
plt.ylabel('No. of Messages')
plt.xticks(rotation=0)
plt.show()
%matplotlib inline
rush_hours.sort_index()

df['Hour'].value_counts().tail(10).sort_index().plot.bar()
plt.xlabel('hours')
plt.ylabel('No. of Messages')
plt.xticks(rotation=0)
plt.show()
%matplotlib inline
df.groupby(['Date']).mean().sort_values(by='Date')
no_msg_pday=df.groupby(['Date']).count()
no_msg_pday['Message'].mean()
df['Message'].count()
df.to_csv(r"C:\Users\User\Downloads\Discussing Football.csv", encoding='utf8-sig')
