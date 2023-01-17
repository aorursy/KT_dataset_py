#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

from IPython.display import Markdown as md
# Import and read files

# First import India dataset
agedetails = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
individualdetails = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
covid_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
population = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")

# Then import Italy dataset
province = pd.read_csv("../input/covid19-in-italy/covid19_italy_province.csv")
region = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
individualdetails
# Getting a brife about the files
individualdetails.info()
individualdetails.drop(columns=['government_id','age','gender','detected_city','nationality'],inplace = True)
a = individualdetails.groupby('detected_state',as_index=False).id.count()
df = pd.DataFrame(a)
df
plt.figure(figsize=(20,8))
chart = sns.barplot(x = 'detected_state',y = 'id' , data = a)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
for index, row in a.iterrows():
    chart.text(row.name,row.id, round(row.id,0), ha="center")
plt.xlabel('States')
plt.ylabel('Count')
plt.title('State vs. Count')
plt.show()
md("***`%s`*** Has the most no.of Cases with `%i` and ***`%s`*** has the Least no.of Cases with `%i`"%([i for i in a[a.id == max(a.id)].detected_state][0],[i for i in a[a.id == max(a.id)].id][0],[i for i in a[a.id == min(a.id)].detected_state][0],[i for i in a[a.id == min(a.id)].id][0]))
a = individualdetails.groupby('current_status',as_index=False).id.count()
df = pd.DataFrame(a)
df
a = individualdetails.groupby(['detected_state','current_status']).id.count()
df = pd.DataFrame(a)
df.head(60)
df.reset_index(inplace=True)
plt.figure(figsize=(15,25))
chart = sns.barplot(x = 'detected_state',y = 'id' , hue = 'current_status' , data = df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.xlabel('States')
plt.ylabel('Count')
plt.title('Statewise Count vs. Status')
plt.show()
individualdetails['Month'] = 0
individualdetails['Date'] = 0
for i in range(len(individualdetails.diagnosed_date)):
    individualdetails['Month'][i] = int(individualdetails.diagnosed_date[i].split('/')[1])
    individualdetails['Date'][i] = int(individualdetails.diagnosed_date[i].split('/')[0])
a = pd.DataFrame(individualdetails.groupby(['Month','Date'],as_index=False).id.count())
a['DDate'] = 0
for i in range(len(a.Date)):
    a['DDate'][i] = str(a['Date'][i]) + '/' + str(a['Month'][i])  
plt.figure(figsize=(30,15))
chart = sns.lineplot(x = 'DDate' ,y = 'id' ,sort = False ,data = a)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Day vs. Count')
plt.show()
a = individualdetails.groupby(['Month','current_status'],as_index=False).id.count()
plt.figure(figsize=(20,5))
sns.barplot(x='Month',y='id',hue='current_status',data=a)
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Monthwise Status vs. Count')
plt.show()
a = pd.DataFrame(individualdetails.groupby(['detected_state','Month','Date'],as_index=False).id.count())
a['DDate'] = 0
for i in range(len(a.Date)):
    a['DDate'][i] = str(a['Date'][i]) + '/' + str(a['Month'][i])  
a.sort_values(['Month','Date'] , inplace = True)
a.reset_index()
plt.figure(figsize=(25,10))
chart = sns.lineplot(x = 'DDate' ,y = 'id' ,hue = 'detected_state' ,sort = False ,data = a)
plt.show()
a = individualdetails.groupby(['Month','detected_state','current_status'],as_index=False).id.count()
for i in a.Month.unique():
    plt.figure(figsize=(15,4))
    chart = sns.barplot(x='detected_state',y='id',hue='current_status',data=a[a.Month == i])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title('Status vs. Count for {}.Month'.format(i))
    plt.show()