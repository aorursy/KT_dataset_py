import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pandas as pd

df=pd.read_excel("../input/portuguesebankmarketing/bank-full.xlsx")
df
df.isnull().sum()
df.info()
df['y'].value_counts()
df_update=df.copy()
df_update
a1=df_update[(df_update['age']>=18) & (df_update['age']<=28)]
a2=df_update[(df_update['age']>=29) & (df_update['age']<=39)]
a3=df_update[(df_update['age']>=40) & (df_update['age']<=50)]
a4=df_update[(df_update['age']>=51) & (df_update['age']<=60)]
a5=df_update[(df_update['age']>=61)]
total=[]
list1=[a1,a2,a3,a4,a5]
for i in list1:
  total.append(i.shape[0])
yes_count=[]
for i in list1:
  yes_count.append(i[i['y']=='yes'].count()['y'])
age_final=pd.DataFrame()
age_final['Age Groups']=['18-28','29-39','40-50','51-60','61+']
age_final['Total']=total
age_final['Yes Count']=yes_count
age_final['Percetage']=(age_final['Yes Count']/age_final['Total'])*100
age_final.sort_values(by='Percetage',ascending=False)
plt.figure(figsize=(20,20))
plt.subplot(3,2,1)
sns.swarmplot(a1['age'],a1['campaign'],hue=a1['y'])
plt.subplot(3,2,2)
sns.swarmplot(a2['age'],a2['campaign'],hue=a2['y'])
plt.subplot(3,2,3)
sns.swarmplot(a3['age'],a3['campaign'],hue=a3['y'])
plt.subplot(3,2,4)
sns.swarmplot(a4['age'],a4['campaign'],hue=a4['y'])
plt.subplot(3,2,5)
sns.swarmplot(a5['age'],a5['campaign'],hue=a5['y'])


df_update['job'].value_counts().index
total_count_in_each_group=[]
yes_count=[]
no_count=[]
title=[]
for i in df_update['job'].value_counts().index:
  df_job=pd.DataFrame()
  df_job=df_update[df_update['job']==i]
  title.append(i)
  total_count_in_each_group.append(df_job.shape[0])
  yes_count.append(df_job[df_job['y']=='yes'].count()['y'])
  no_count.append(df_job[df_job['y']=='no'].count()['y'])
df_jobs=pd.DataFrame()
df_jobs['Job Title']=title
df_jobs['Total']=total_count_in_each_group
df_jobs['Yes']=yes_count
df_jobs['No']=no_count
df_jobs
labels=df_jobs['Job Title']
positions = np.arange(0,12,1)
df_jobs.plot(kind='bar',figsize=(20,5))
plt.xticks(positions, labels)
df_jobs_final=pd.DataFrame()
df_jobs_final['Job Title']=title
df_jobs_final['Percentage_yes']=(df_jobs['Yes']/df_jobs['Total'])*100
df_jobs_final['Percentage_no']=(df_jobs['No']/df_jobs['Total'])*100
df_jobs_final=df_jobs_final.sort_values('Percentage_yes',ascending=False)
df_jobs_final
labels=df_jobs_final['Job Title'].tolist()
position=df_jobs_final['Job Title'].index
df_jobs_final.plot(kind='bar',figsize=(20,5))
plt.xticks(positions, labels)
df_married=df_update[df_update['marital']=='married']
df_single=df_update[df_update['marital']=='single']
df_divorced=df_update[df_update['marital']=='divorced']
married=df_update['marital'].value_counts().to_frame()
married
yes_count=[]
yes_count.append(df_married[df_married['y']=='yes'].count()[1])
yes_count.append(df_single[df_single['y']=='yes'].count()[1])
yes_count.append(df_divorced[df_divorced['y']=='yes'].count()[1])
married['Subscription count']=yes_count
married.plot(kind='bar')
married['Percentage']=(married['Subscription count']/married['marital'])*100
married.sort_values(by='Percentage',ascending=False)
education=df_update['education'].value_counts().to_frame()
education
df_secondary=df_update[df_update['education']=='secondary']
df_tertiary=df_update[df_update['education']=='tertiary']
df_primary=df_update[df_update['education']=='primary']
df_unknown=df_update[df_update['education']=='unknown']
yes_count=[]
yes_count.append(df_secondary[df_secondary['y']=='yes'].count()[1])
yes_count.append(df_tertiary[df_tertiary['y']=='yes'].count()[1])
yes_count.append(df_primary[df_primary['y']=='yes'].count()[1])
yes_count.append(df_unknown[df_unknown['y']=='yes'].count()[1])
education['Subscription count']=yes_count
education.plot(kind='bar')
education['Percentage']=(education['Subscription count']/education['education'])*100
education.sort_values(by='Percentage',ascending=False)
success=df_update[(df_update['poutcome']=='success')].count()['poutcome']
failure=df_update[(df_update['poutcome']=='failure')].count()['poutcome']
unknown=df_update[(df_update['poutcome']=='unknown')].count()['poutcome']
other=df_update[(df_update['poutcome']=='other')].count()['poutcome']
previous_campaign=pd.DataFrame({'Status':['Success','Failure','Unknown','Others'],"Values":[success,failure,unknown,other]})
previous_campaign['Percentage']=(previous_campaign['Values']/45211)*100
previous_campaign
df_update[df_update['pdays']==-1]
df_update[(df_update['pdays']!=-1) & (df_update['poutcome']=='unknown')]
sns.barplot(df_update['y'],df_update['balance'])
yes=df_update[df_update['y']=='yes']
no=df_update[df_update['y']=='no']
print(yes['balance'].mean())
print(no['balance'].mean())

sns.scatterplot(df_update['balance'],df_update['age'],hue=df_update['y'])
sns.boxplot(df['balance'],df['y'])
default_yes=df_update[df_update['default']=='yes']
default_no=df_update[df_update['default']=='no']
total=df_update.shape[0]
yes=default_yes.count()['y']
no=default_no.count()['y']
sns.countplot(df_update['default'])
print("Total Count",total)
print((yes/total)*100)
print((no/total*100))
yes=default_yes[default_yes['y']=='yes'].count()['y']
no=default_yes[default_yes['y']=='no'].count()['y']
total=default_yes.count()['y']
print("Default")
print(total)
print((yes/total)*100)
print((no/total)*100)
yes=default_no[default_no['y']=='yes'].count()['y']
no=default_no[default_no['y']=='no'].count()['y']
total=default_no.count()['y']
print("Non default")
print(total)
print((yes/total)*100)
print((no/total)*100)
plt.figure(figsize=(20,5))
sns.countplot(df_update['default'],hue=df_update['y'])
default_yes=df_update[df_update['housing']=='yes']
default_no=df_update[df_update['housing']=='no']

total=df_update.shape[0]
yes=default_yes.count()['y']
no=default_no.count()['y']
sns.countplot(df_update['housing'])

print("Total counts")
print("We have ",(yes/total)*100," % who have house loan and count is ", yes)
print("We have ",(no/total)*100," % who have dont house loan and count is ",no)
yes=default_yes[default_yes['y']=='yes'].count()['y']
no=default_yes[default_yes['y']=='no'].count()['y']
total=default_yes.count()['y']
print("Total that have housing loan are",total)
print("Out of the total",(yes/total)*100," % have subscribed to Term Deposit Plan")
print("Out of the total",(no/total)*100," % have not subscribed to Term Deposit Plan")
yes=default_no[default_no['y']=='yes'].count()['y']
no=default_no[default_no['y']=='no'].count()['y']
total=default_no.count()['y']
print("Total number of people that don't housing loan are",total)
print("Out of the total",(yes/total)*100," % have subscribed to Term Deposit Plan")
print("Out of the total",(no/total)*100," % have not subscribed to Term Deposit Plan")
plt.figure(figsize=(20,5))
sns.countplot(df_update['housing'],hue=df_update['y'])
default_yes=df_update[df_update['loan']=='yes']
default_no=df_update[df_update['loan']=='no']

total=df_update.shape[0]
yes=default_yes.count()['y']
no=default_no.count()['y']
sns.countplot(df_update['loan'])

print("Total counts")
print("We have ",(yes/total)*100," % who have personal loan and count is ", yes)
print("We have ",(no/total)*100," % who have dont personal loan and count is ",no)
yes=default_yes[default_yes['y']=='yes'].count()['y']
no=default_yes[default_yes['y']=='no'].count()['y']
total=default_yes.count()['y']
print("Total that have Personal loan are",total)
print("Out of the total",(yes/total)*100," % have subscribed to Term Deposit Plan")
print("Out of the total",(no/total)*100," % have not subscribed to Term Deposit Plan")
yes=default_no[default_no['y']=='yes'].count()['y']
no=default_no[default_no['y']=='no'].count()['y']
total=default_no.count()['y']
print("Total number of people that don't have personal loan are",total)
print("Out of the total",(yes/total)*100," % have subscribed to Term Deposit Plan")
print("Out of the total",(no/total)*100," % have not subscribed to Term Deposit Plan")
plt.figure(figsize=(20,5))
sns.countplot(df_update['housing'],hue=df_update['y'])
df_update['contact'].value_counts()
sns.countplot(df['contact'],hue=df['y'])
df_update['month'].value_counts()
df_update.loc[df_update['month']=='jan','month']=1
df_update.loc[df_update['month']=='feb','month']=2
df_update.loc[df_update['month']=='mar','month']=3
df_update.loc[df_update['month']=='apr','month']=4
df_update.loc[df_update['month']=='may','month']=5
df_update.loc[df_update['month']=='jun','month']=6
df_update.loc[df_update['month']=='jul','month']=7
df_update.loc[df_update['month']=='aug','month']=8
df_update.loc[df_update['month']=='sep','month']=9
df_update.loc[df_update['month']=='oct','month']=10
df_update.loc[df_update['month']=='nov','month']=11
df_update.loc[df_update['month']=='dec','month']=12
dict1=dict(list(df_update.groupby(['month','y'])))
list1=[1,2,3,4,5,6,7,8,9,10,11,12]
no=[]
yes=[]
months=[]
for i in list1:
  months.append(i)
  for j in ['no','yes']:
    if(j=='no'):
      no.append(dict1[i,j].count()['y'])
    else:
      yes.append(dict1[i,j].count()['y'])

total_count_per_month=[]
dict2=dict(list(df_update.groupby(['month'])))
for i in list1:
  total_count_per_month.append(dict2[i].count()['y'])

      
print(months)
print(total_count_per_month)
print(yes)
print(no)
month_wise=pd.DataFrame()
month_wise['Months']=months
month_wise['Total ENteries per month']=total_count_per_month
month_wise['Count of Subscribed']=yes
month_wise['Count of Not Sub']=no
month_wise['Subscription Rate']=(month_wise['Count of Subscribed']/month_wise['Total ENteries per month'])*100
month_wise['Not Sub Rate']=(month_wise['Count of Not Sub']/month_wise['Total ENteries per month'])*100
month_wise=month_wise.sort_values("Subscription Rate",ascending=False)
month_wise
plt.figure(figsize=(20,5))
labels=['Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec']
position=np.arange(0,13,1)
sns.pointplot(month_wise['Months'],month_wise['Subscription Rate'],color='g')
plt.xticks(position,labels)
plt.legend(['Subscription %'])
plt.figure(figsize=(20,5))
labels=['Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec']
position=np.arange(0,13,1)
sns.pointplot(month_wise['Months'],month_wise['Not Sub Rate'],color='r')
plt.xticks(position,labels)
plt.legend(['Not subscribed %'])
plt.figure(figsize=(20,5))
labels=['Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec']
position=np.arange(0,13,1)
sns.pointplot(month_wise['Months'],month_wise['Total ENteries per month'],color='r')
plt.xticks(position,labels)
plt.legend(['Total count- contacted per month'])
month_final=month_wise[['Months','Subscription Rate','Not Sub Rate']]
month_final=month_final.set_index('Months')
month_final.plot(kind='bar',figsize=(15,5))

sns.boxplot(df_update['duration'],df_update['y'])
total=[]
no=[]
yes=[]

total.append(df_update[(df_update['duration']<=500)].count()['duration'])
no.append(df_update[(df_update['duration']<=500) & (df_update['y']=='no')].count()['y'])
yes.append(df_update[(df_update['duration']<=500) & (df_update['y']=='yes')].count()['y'])


total.append(df_update[(df_update['duration']<=1000) & (df_update['duration']>500)].count()['duration'])
no.append(df_update[(df_update['duration']<=1000) & (df_update['duration']>500) & (df_update['y']=='no')].count()['y'])
yes.append(df_update[(df_update['duration']<=1000) & (df_update['duration']>500) & (df_update['y']=='yes')].count()['y'])

total.append(df_update[(df_update['duration']<=1500) & (df_update['duration']>1000)].count()['duration'])
no.append(df_update[(df_update['duration']<=1500) & (df_update['duration']>1000) & (df_update['y']=='no')].count()['y'])
yes.append(df_update[(df_update['duration']<=1500) & (df_update['duration']>1000) & (df_update['y']=='yes')].count()['y'])

total.append(df_update[(df_update['duration']<=2000) & (df_update['duration']>1500)].count()['duration'])
no.append(df_update[(df_update['duration']<=2000) & (df_update['duration']>1500) & (df_update['y']=='no')].count()['y'])
yes.append(df_update[(df_update['duration']<=2000) & (df_update['duration']>1500) & (df_update['y']=='yes')].count()['y'])

total.append(df_update[(df_update['duration']>2000)].count()['duration'])
no.append(df_update[(df_update['duration']>2000) & (df_update['y']=='no')].count()['y'])
yes.append(df_update[(df_update['duration']>2000) & (df_update['y']=='yes')].count()['y'])


duration=pd.DataFrame()
duration['Duration']=['Till 500','501-1000','1001-1500','1501-2000','2001+']
duration['total']=total
duration['Yes']=yes
duration['No']=no
duration['Yes percent']=(duration['Yes']/duration['total'])*100
duration.sort_values('Yes percent',ascending=False)
plt.figure(figsize=(20,5))
plt.xticks(np.arange(0,5000,150))
sns.scatterplot(df_update['duration'],df['campaign'],hue=df_update['y'])
plt.figure(figsize=(25,10))
sns.countplot(df['campaign'],hue=df['y'])
df_15=df_update[(df_update['campaign']<=15) ]
df_30=df_update[(df_update['campaign']>=16) & (df_update['campaign']<=30) ]
df_31=df_update[(df_update['campaign']>=31) ]
total_calls=[]
yes=[]
no=[]
for i in[df_15,df_30,df_31]:
  total_calls.append(i.count()['campaign'])
  yes.append(i[i['y']=='yes'].count()['y'])
  no.append(i[i['y']=='no'].count()['y'])

print(total_calls)
print(yes)
print(no)
campaigns=pd.DataFrame()
campaigns['No.of call']=['Till 15','16- 30','30+ more']
campaigns['Total no of Calls']=total_calls
campaigns['Took Subscription']=yes
campaigns['No Subscription']=no
campaigns['Rate of Sub']=(campaigns['Took Subscription']/campaigns['Total no of Calls'])*100
campaigns
sns.barplot(campaigns['No.of call'],campaigns['Rate of Sub'])
df_update['day']
plt.figure(figsize=(20,5))
sns.set(style='darkgrid')
sns.countplot(df_update['day'],hue=df_update['y'])

dict1=dict(list(df_update.groupby(['day','y'])))
list1=np.arange(1,32,1)
no=[]
yes=[]
days=[]
for i in list1:
  days.append(i)
  for j in ['no','yes']:
    if(j=='no'):
      no.append(dict1[i,j].count()['y'])
    else:
      yes.append(dict1[i,j].count()['y'])

total_count_per_day=[]
dict2=dict(list(df_update.groupby(['day'])))
for i in list1:
  total_count_per_day.append(dict2[i].count()['y'])

day_wise=pd.DataFrame()
day_wise['Day']=days
day_wise['Total ENteries per day']=total_count_per_day
day_wise['Count of Subscribed']=yes
day_wise['Count of Not Sub']=no
day_wise['Subscription Rate']=(day_wise['Count of Subscribed']/day_wise['Total ENteries per day'])*100
day_wise['Not Sub Rate']=(day_wise['Count of Not Sub']/day_wise['Total ENteries per day'])*100
day_wise=day_wise.sort_values('Subscription Rate',ascending=False)
plt.figure(figsize=(20,5))
sns.pointplot(day_wise['Day'],day_wise['Subscription Rate'])
plt.figure(figsize=(20,5))
sns.set(style='darkgrid')
sns.pointplot(day_wise['Day'],day_wise['Total ENteries per day'])
plt.figure(figsize=(20,5))
sns.set(style='darkgrid')
sns.pointplot(day_wise['Day'],day_wise['Not Sub Rate'])
plt.ylabel('Not subscribed percentage')
