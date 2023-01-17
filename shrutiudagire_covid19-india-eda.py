import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
df
df['Name of State / UT'].value_counts()
df.loc[(df['Name of State / UT']=='Union Territory of Jammu and Kashmir'),'Name of State / UT']='Jammu and Kashmir'
df.loc[(df['Name of State / UT']=='Union Territory of Ladakh'),'Name of State / UT']='Ladakh'
df.loc[(df['Name of State / UT']=='Union Territory of Chandigarh'),'Name of State / UT']='Chandigarh'
df['Name of State / UT'].value_counts()
# Calculating datewise count of confirmed cases
date=dict(list(df.groupby('Date')))
dates_df=pd.DataFrame()
count=[]
dates=[]
for i in date:
    dates.append(i)
    count.append(date[i]['Total Confirmed cases'].sum())
dates_df['Date']=dates
dates_df['Count']=count 
dates_df=dates_df.sort_values(by='Count',ascending=True)
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)
sns.barplot(dates_df['Date'],dates_df['Count'])
plt.figure(figsize=(25,10))
plt.xticks(rotation=90)
sns.pointplot(dates_df['Date'],dates_df['Count'])
months=[]
for i in df['Date']:
    months.append(i.split("-")[1])
df['Months']=months
print(df['Name of State / UT'].value_counts().count())
print(df['Name of State / UT'].value_counts().index)

dict1=dict(list(df.groupby('Months')))
total_count_per_state=[]
state_name=[]
death=[]
recovered=[]
dict_names=dict(list(dict1['01'].groupby('Name of State / UT')))
for i in dict_names.keys():
    total_count_per_state.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Total Confirmed cases'])
    death.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Death'])
    recovered.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Cured/Discharged/Migrated'])
    state_name.append(i)
    
jan=pd.DataFrame()
jan['State']=state_name
jan['Cases per state']=total_count_per_state
jan['Death']=death
jan['Cured/Discharged/Migrated']=recovered


not_in_jan_state=[]
no_case=[]
for i in df['Name of State / UT'].value_counts().index:
    if i not in jan['State'].tolist():
        not_in_jan_state.append(i)
        no_case.append(0)
jan_missing_state=pd.DataFrame()
jan_missing_state['State']=not_in_jan_state
jan_missing_state['Cases per state']=no_case
jan_missing_state['Death']=no_case
jan_missing_state['Cured/Discharged/Migrated']=no_case

jan=pd.concat([jan,jan_missing_state])
jan=jan.sort_values('State')
jan=jan.reset_index()
jan.drop('index',axis=1,inplace=True)
jan
# Feb
total_count_per_state=[]
state_name=[]
death=[]
recovered=[]
dict_names=dict(list(dict1['02'].groupby('Name of State / UT')))
for i in dict_names.keys():
    total_count_per_state.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Total Confirmed cases'])
    death.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Death'])
    recovered.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Cured/Discharged/Migrated'])
    state_name.append(i)
    
feb=pd.DataFrame()
feb['State']=state_name
feb['Cases per state']=total_count_per_state
feb['Death']=death
feb['Cured/Discharged/Migrated']=recovered

not_in_feb_state=[]
no_case=[]
for i in df['Name of State / UT'].value_counts().index:
    if i not in feb['State'].tolist():
        not_in_feb_state.append(i)
        no_case.append(0)
feb_missing_state=pd.DataFrame()
feb_missing_state['State']=not_in_feb_state
feb_missing_state['Cases per state']=no_case
feb_missing_state['Death']=no_case
feb_missing_state['Cured/Discharged/Migrated']=no_case

feb=pd.concat([feb,feb_missing_state])
feb=feb.sort_values('State')
feb=feb.reset_index()
feb.drop('index',axis=1,inplace=True)

feb
# MArch
total_count_per_state=[]
state_name=[]
death=[]
recovered=[]
dict_names=dict(list(dict1['03'].groupby('Name of State / UT')))
for i in dict_names.keys():
    total_count_per_state.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Total Confirmed cases'])
    state_name.append(i)
    death.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Death'])
    recovered.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Cured/Discharged/Migrated'])
    
march=pd.DataFrame()
march['State']=state_name
march['Cases per state']=total_count_per_state
march['Death']=death
march['Cured/Discharged/Migrated']=recovered

not_in_march_state=[]
no_case=[]
for i in df['Name of State / UT'].value_counts().index:
    if i not in march['State'].tolist():
        not_in_march_state.append(i)
        no_case.append(0)
march_missing_state=pd.DataFrame()
march_missing_state['State']=not_in_march_state
march_missing_state['Cases per state']=no_case
march_missing_state['Death']=no_case
march_missing_state['Cured/Discharged/Migrated']=no_case

march=pd.concat([march,march_missing_state])
march=march.sort_values('State')
march=march.reset_index()
march.drop('index',axis=1,inplace=True)
march
total_count_per_state=[]
state_name=[]
death=[]
recovered=[]
dict_names=dict(list(dict1['04'].groupby('Name of State / UT')))
for i in dict_names.keys():
    total_count_per_state.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Total Confirmed cases'])
    state_name.append(i)
    death.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Death'])
    recovered.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Cured/Discharged/Migrated'])
    
april=pd.DataFrame()
april['State']=state_name
april['Cases per state']=total_count_per_state
april['Death']=death
april['Cured/Discharged/Migrated']=recovered

not_in_april_state=[]
no_case=[]
for i in df['Name of State / UT'].value_counts().index:
    if i not in april['State'].tolist():
        not_in_april_state.append(i)
        no_case.append(0)
april_missing_state=pd.DataFrame()
april_missing_state['State']=not_in_april_state
april_missing_state['Cases per state']=no_case
april_missing_state['Death']=no_case
april_missing_state['Cured/Discharged/Migrated']=no_case


april=pd.concat([april,april_missing_state])
april=april.sort_values('State')
april=april.reset_index()
april.drop('index',axis=1,inplace=True)
april
total_count_per_state=[]
state_name=[]
death=[]
recovered=[]
dict_names=dict(list(dict1['05'].groupby('Name of State / UT')))
for i in dict_names.keys():
    total_count_per_state.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Total Confirmed cases'])
    state_name.append(i)
    death.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Death'])
    recovered.append(dict_names[i].sort_values('Date',ascending=True).iloc[-1]['Cured/Discharged/Migrated'])
    
may=pd.DataFrame()
may['State']=state_name
may['Cases per state']=total_count_per_state
may['Death']=death
may['Cured/Discharged/Migrated']=recovered

not_in_may_state=[]
no_case=[]
for i in df['Name of State / UT'].value_counts().index:
    if i not in may['State'].tolist():
        not_in_may_state.append(i)
        no_case.append(0)
may_missing_state=pd.DataFrame()
may_missing_state['State']=not_in_may_state
may_missing_state['Cases per state']=no_case
may_missing_state['Death']=no_case
may_missing_state['Cured/Discharged/Migrated']=no_case

may=pd.concat([may,may_missing_state])
may=may.sort_values('State')
may=may.reset_index()
may.drop('index',axis=1,inplace=True)
Final=pd.DataFrame()
Final=may.copy()
for i in ['Cases per state','Death','Cured/Discharged/Migrated']:
  feb[i]=feb[i]-jan[i]
  march[i]=march[i]-feb[i]
  april[i]=april[i]-march[i]
  may[i]=may[i]-april[i]
Final_january=jan
Final_February=feb
Final_March=march
Final_April=april
Final_May=may
labels=Final_january['State'].values
position=Final_january['State'].index
Final_january.plot(kind='bar',figsize=(20,5))
plt.xticks(position, labels)
plt.title("New Cases for January month")
labels=Final_February['State'].values
position=Final_February['State'].index
Final_February.plot(kind='bar',figsize=(20,5))
plt.xticks(position, labels)
plt.yticks(np.arange(0,3,1))
plt.title("New Cases for February month")
labels=Final_March['State'].values
position=Final_March['State'].index
Final_March.plot(kind='bar',figsize=(20,5))
plt.xticks(position, labels)
plt.title("New Cases for March month")
Final_April.plot(kind='bar',figsize=(20,5))
labels=Final_April['State'].values
position=Final_April['State'].index
plt.xticks(position, labels)
plt.title("New Cases for April month")
labels=Final_May['State'].values
position=Final_May['State'].index
Final_May.plot(kind='bar',figsize=(20,5))
plt.xticks(position, labels)
plt.title("New Cases for May month")
labels=Final_May['State'].values
position=Final_May['State'].index
Final.plot(kind='bar',figsize=(20,10))
plt.xticks(position,labels)
plt.title("All Cases State wise at the end of the May Month-Total Confirmed Cases,Deaths and Cured")