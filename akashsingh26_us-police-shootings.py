#importing the essentials

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/us-police-shootings/shootings.csv')
data.head()
#feature engineering (Adding col Age Group)
bins = [0,18,45,60,100]
group_names = ['Teenager','Adult','Old','Very Old']
data['Age Group'] = pd.cut(data['age'], bins, labels= group_names)

#converting date column from str to date
data['date']=pd.to_datetime(data['date'])
data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month
data['month_year']= pd.to_datetime(data['date']).dt.to_period('M')

count_year= data.groupby(['year']).agg('count')['id'].to_frame(name='count').reset_index()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,7))
sns.barplot(x=count_year['year'], y=count_year['count'], data=count_year)
plt.xlabel("")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Killings by Year')
plt.ylabel("")
plt.show()
plt.style.use('bmh')
data['month_year']= data.month_year.astype(str)
line_chart = data.groupby(['month_year']).agg('count')['id'].to_frame(name='count').reset_index()
plt.figure(figsize=(20,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
plt.plot(line_chart['month_year'],line_chart['count'])
plt.title('Killings by Month')
plt.xticks(ticks = line_chart['month_year'],rotation=90)
plt.show()
avg_shot_per_day = (count_year['count'][0:5].sum())/(365*5)
print('Avg death count per day',avg_shot_per_day)
avg_per_month = (count_year['count'][0:5].sum())/(12*5)
print('Avg death count per month',avg_per_month)
line_chart.columns = ['Month_Year', 'Victim_Count']
max_1 = pd.DataFrame(line_chart[line_chart.Victim_Count == line_chart.Victim_Count.max()].reset_index(drop=True))
min_1 = pd.DataFrame(line_chart[line_chart.Victim_Count == line_chart.Victim_Count.min()].reset_index(drop=True))
print('Max amount of Death is in Month of\n',max_1)
print('**********************************************')
print('Min amount of Death is in Month of\n',min_1)
shot_or_taser = data.groupby(['Age Group','manner_of_death']).agg('count')['id'].to_frame(name='count').reset_index()
shot_or_taser = shot_or_taser.rename(columns = {'manner_of_death':'Manner of Death', 0:'Count'})
shot_or_taser = shot_or_taser.sort_values(by=['count'],ascending=False)
plt.style.use('seaborn-pastel')
plt.figure(figsize=(20,10))
sns.barplot(x="Age Group", y="count",hue="Manner of Death", data=shot_or_taser)
plt.xlabel("")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Killings by Age Category')
plt.ylabel("")
plt.show()
plt.style.use('seaborn-pastel')
plt.figure(figsize=(15,7))
sns.barplot(x="Age Group", y="count",hue="Manner of Death", data=shot_or_taser,log=True)
plt.xlabel("")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Killings by Age Group(On Log Axis)')
plt.ylabel("")
plt.show()
a=shot_or_taser.groupby(['Age Group']).sum().reset_index()

list_percent = []
total = a['count']
temp = 0
temp_2 = 0
for i in range(4):
    for j in range(2):
        per=(shot_or_taser['count'][temp])/total[temp_2]
        list_percent.append(per)
        temp=temp+1
    temp_2=temp_2+1

list_1 = ['Shot','Shot and Tasered']
teenager = list_percent[0:2]
adult = list_percent[2:4]
old = list_percent[4:6]
very_old = list_percent[6:8]
#For Teenager and Adult
plt.style.use('seaborn-pastel')
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (15,15))
plt.rcParams.update({'font.size': 18})
ax1.pie(teenager, labels=list_1, shadow=True, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})

ax2.pie(adult, labels=list_1, shadow=True, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})

ax1.set_title("Teenager")
ax2.set_title("Adult")
fig.tight_layout()
plt.show()
plt.style.use('seaborn-pastel')
fig, (ax3,ax4) = plt.subplots(1,2, figsize = (15,15))
plt.rcParams.update({'font.size': 18})
ax3.pie(old, labels=list_1, shadow=True, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
ax4.pie(very_old, labels=list_1, shadow=True, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})

ax3.set_title("Old")
ax4.set_title("Very Old")
plt.tight_layout()
plt.show()
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
sns.swarmplot(data=data, x="year", y="age",hue="manner_of_death")
plt.xlabel("")
plt.ylabel("")
plt.title('Manner of Death by Year and Age')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
data.groupby(['year','manner_of_death']).count()['id'].reset_index()
#finding top 5(what ammunitions used)

armed_or_not=data.groupby(['armed']).size()
armed_or_not=armed_or_not.sort_values()
armed_or_not = armed_or_not.to_frame().reset_index()
armed_or_not = armed_or_not.rename(columns={'armed':'Armed', 0:'Count'})
armed_or_not = armed_or_not.sort_values(by = ['Count'],ascending=False)
top_5= armed_or_not.head(5)
plt.figure(figsize=(20,10))
plt.style.use('fivethirtyeight')
plt.bar(top_5.Armed,top_5.Count)
sns.barplot(x='Armed', y='Count', data=top_5)
plt.title('Armed or Not')
plt.ylabel('Number of Victims')
plt.xlabel('')
plt.show()
records = data.shape[0]
armed_or_not_pie = armed_or_not.head(4)
list_p = []
for i in (range(len(armed_or_not_pie))):
    temp=(armed_or_not_pie['Count'].values[i]/records)*100
    list_p.append(temp)

Others_p= 100-sum(list_p)
percentages=pd.Series(list_p)
armed_or_not_pie['percent'] = percentages.values
armed_or_not_pie.drop('Count',axis=1,inplace=True)
Others_df = ['Others',Others_p]
other_series = pd. Series(Others_df, index = armed_or_not_pie.columns)
armed_or_not_pie = armed_or_not_pie. append(other_series, ignore_index=True)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 18})
plt.pie(armed_or_not_pie['percent'], labels=armed_or_not_pie['Armed'], shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
plt.title("Armed or Not")
plt.tight_layout()
plt.show()
#full df
gender_shoot = data.groupby(['year','gender']).agg('count')['id'].to_frame(name='count').reset_index()

#for Male
male = gender_shoot.loc[gender_shoot['gender']=='M']

#for Female
female = gender_shoot.loc[gender_shoot['gender']=='F']
#plotting the data
plt.figure(figsize=(20,10))
plt.style.use('fivethirtyeight')
x_indexes = np.arange(len(male['year']))
width = 0.40

plt.bar( x_indexes,male['count'],width = width,label = 'Male')
plt.bar( x_indexes+width,female['count'],width = width, label = 'Female')
plt.title('Gender Wise Killings per Year')
plt.xticks(ticks = x_indexes, labels = female['year'])
plt.tight_layout()
plt.legend()
plt.show()
df =data.groupby(['Age Group','gender']).agg('count')['id'].to_frame(name='count').reset_index()
df=df.sort_values(by=['count'],ascending=False)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,8))
sns.barplot(x='Age Group', y='count', data=df)
plt.title('Killings by Age Group')
plt.ylabel('Number of Victims')
plt.show()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
sns.barplot(x='Age Group', y='count', hue='gender', data=df,log=True)
plt.title('Killings by Age Group(on log)')
plt.ylabel('Number of Victims')
plt.show()
race_df = data.groupby(['year','race']).agg('count')['id'].to_frame(name='count').reset_index()
race_df = race_df.sort_values(by='count',ascending=False)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
sns.barplot(x="year", y="count",hue="race", data=race_df)
plt.title('Killings by Race by Year')
plt.ylabel('Number of Victims')
plt.xlabel('')
plt.show()
plt.style.use('fivethirtyeight')
fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.stripplot(
    data=data, x="age", y="race",s=20,
    alpha=0.20,jitter=0.30
)
plt.title('Killings by Race by Age')
plt.ylabel('')
plt.xlabel('Age')
plt.show()
killings_by_race =data.groupby(['year','month_year','race']).count()['id'].reset_index()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
sns.boxplot(x=killings_by_race['race'],y=killings_by_race['id'],dodge=False)
plt.title('Killings by Race(By Month Average)')
plt.ylabel('Number of Victims')
plt.xlabel('')
plt.show()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
sns.boxplot(x=data['race'],y=data['age'],dodge=False)
plt.title('Killings by Race by Age')
plt.ylabel('Age')
plt.xlabel('')
plt.show()
#Comparing White V/S Black V/S Hispanic
white_black_hispanic = data[data['race']. isin (['Black','White','Hispanic'])]
white_black_hispanic=white_black_hispanic.groupby(['year','race']).agg('count')['id'].to_frame(name='count').reset_index()
white_black_hispanic['year']= white_black_hispanic['year'].astype(str)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))

colors = ["#2874A6", "#B9770E","#922B21"]
sns.set_palette(sns.color_palette(colors))

sns.lineplot( data=white_black_hispanic,x="year", y="count", hue="race")
plt.title('Killings in Top 3 Races by Year')
plt.ylabel('Number of Victims')
plt.xlabel('')
plt.show()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
plt.bar(data['state'].value_counts().index,data['state'].value_counts().values)
plt.title('Killings by different States')
plt.ylabel('Number of Victims')
plt.xticks(fontsize=11)
plt.show()
top_5_states = data[data['state'] . isin(['CA','TX','FL','AZ','CO'])]
top_5_states = top_5_states.groupby(['state','flee']).agg('count')['id'].to_frame(name='count').reset_index().sort_values('count',ascending=False)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
sns.barplot(x='state', y='count',hue='flee', data=top_5_states)
plt.ylabel('Number of Victims')
plt.title('Killings by States with different flee Status')
plt.xlabel("")
plt.show()
plt.style.use('fivethirtyeight')

fig_dims = (20, 20)
fig, ax = plt.subplots(figsize=fig_dims)
sns.stripplot(
    data=data, x="age", y="race",hue="signs_of_mental_illness",s=25,
    alpha=0.5,jitter=0.40
)
plt.title('Killings by Race on basis of Mental Status')
plt.axis('tight')
plt.xlabel('Age')
plt.show()
#Let's check by pie chart
pie_chart =data.groupby(['race','signs_of_mental_illness']).count()['id'].reset_index()
black_pie_chart = pie_chart[pie_chart['race']=='Black']
white_pie_chart = pie_chart[pie_chart['race']=='White']
hispanic_pie_chart = pie_chart[pie_chart['race']=='Hispanic']
plt.style.use('seaborn-pastel')
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (15,15))
plt.rcParams.update({'font.size': 18})
ax1.pie(black_pie_chart['id'], labels=black_pie_chart['signs_of_mental_illness'], shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
ax1.set_title("Race: Black")

ax2.pie(white_pie_chart['id'], labels=white_pie_chart['signs_of_mental_illness'], shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
ax2.set_title("Race - White")

ax3.pie(hispanic_pie_chart['id'], labels=hispanic_pie_chart['signs_of_mental_illness'], shadow=True,
        startangle=90, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'})
ax3.set_title("Race - Hispanic")

fig.tight_layout()

plt.show()