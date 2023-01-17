#Lets import required libraries and methods
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mlt
import matplotlib.pyplot as plt
import matplotlib.style as style 
import matplotlib.cm as cm #colormap

from datetime import datetime, timedelta, date
from dateutil import relativedelta

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
##Code to create the assembly seats visualization - 5 concentric eclipses that accomodate 250 points
x=[]
y=[]
#Generate polar coordinates to plot the circles
radius=[10,12,14,16,18]
theta=np.linspace(0,360,50)
#Generate polar coordinates
for i in radius:
    x.append(i*np.cos(theta))
    y.append(i*np.sin(theta))
#Transpose array
x_t=np.transpose(x)
y_t=np.transpose(y)
#reshape to 1-D array to plot
x=np.reshape(x_t,len(x)*len(x[1]))
y=np.reshape(y_t,len(y)*len(y[1]))
#reindexing helps us to plot similar points together
reindex=np.reshape([[np.linspace(5*i+1,5*i+5,5)] for i in np.append(np.reshape(np.transpose(np.reshape(range(48),(8,6))),48),[48,49])],250).astype(int)
#generate (X,Y) from reindexed array
x=x[reindex-1]
y=y[reindex-1]
#use dark background for visualization
style.use('dark_background')
#load the 2018 member details file into Pandas Dataframe
df_members=pd.read_excel('../input/rajyasabha_member_details_2018.xlsx')
##view data
df_members.head(5)
#Plot the total seats in Rajya Sabha first
fig, axes = plt.subplots(figsize=(10,7.5))
plt.plot(x, y, 'o', color='w',alpha=0.65)
plt.plot(x[245:250], y[245:250], 'o', color='black')
plt.axis('off')
plt.title('Total seats in Rajya Sabha = 245',fontsize=20)
#PLot the number of vacant seats
fig, axes = plt.subplots(figsize=(10,7.5))
counter=0
idx=0
count_by_party=df_members.groupby('Party').size()
count_by_party.sort_values(ascending=True,inplace=True)
vacant=pd.Series([count_by_party['No Party']],index=(['No Party']))

vacant['Filled Seats']=sum(count_by_party[key] for key in count_by_party.keys() if key not in ['No Party'])
vacant.sort_values(ascending=True,inplace=True)
key_vacant=vacant.keys()
c=['r','w']
symbol=['X','o']
alpha=[1,0.65]
labels=['Vacant Seats','Filled Seats']

for i in vacant.values+1:
    #print(i,counter,counter+i-1)
    plt.plot(x[counter:counter+i-1], y[counter:counter+i-1], symbol[idx], color=c[idx],label=labels[idx],alpha=alpha[idx])
    counter=counter+i
    idx=idx+1

#As 250 points are plotted, hide 5 points
plt.plot(x[245:250], y[245:250], 'o', color='black') 
plt.legend(loc='best')
plt.axis('off')
plt.title('Currently there are '+ str(vacant[0]) +' vacant seats in Rajya Sabha.',fontsize=20)
#Plot the count of ministers
fig, axes = plt.subplots(figsize=(10,7.5))
counter=0
idx=0
count_of_ministers=df_members.groupby('MinisterY_N').size()
count_of_ministers.sort_values(ascending=True,inplace=True)
key_by_ministers=count_of_ministers.keys()
c=['r','w']
labels=['Cabinet Minister','Member of Parliment']
symbol=['^','o']
alpha=[1.0,0.65]

for i in count_of_ministers.values+1:
    #print(i,counter,counter+i-1)
    plt.plot(x[counter:counter+i-1], y[counter:counter+i-1], symbol[idx], color=c[idx],label=labels[idx],alpha=alpha[idx])
    counter=counter+i
    idx=idx+1
#As 250 points are plotted, hide 5 points
plt.plot(x[245:250], y[245:250], 'o', color='black') 

plt.legend(loc='best')
plt.axis('off')
plt.title('Currently '+ str(count_of_ministers[0]) +' members in Rajya Sabha serve as Cabinet Ministers',fontsize=20)
#Plot count of women members
fig, axes = plt.subplots(figsize=(10,7.5))
counter=0
idx=0
count_of_women=df_members.groupby('FemaleY_N').size()
count_of_women.sort_values(ascending=True,inplace=True)
key_by_women=count_of_women.keys()
labels=['Female Member','Male Member']
c=['r','w']
symbol=['D','o']
alpha=[1.0,0.65]

for i in count_of_women.values+1:
    #print(i,x[counter:counter+i-1], y[counter:counter+i-1])
    plt.plot(x[counter:counter+i-1], y[counter:counter+i-1], symbol[idx], color=c[idx],label=labels[idx],alpha=alpha[idx])
    counter=counter+i
    idx=idx+1
#As 250 points are plotted, hide 5 points
plt.plot(x[245:250], y[245:250], 'o', color='black') 

plt.legend(loc='best')
plt.axis('off')
plt.title('Currently there are '+ str(count_of_women[0]) +' women members in Rajya Sabha',fontsize=20)
#Plot count by party
fig, axes = plt.subplots(figsize=(10,7.5))
counter=0
idx=0
count_by_party=df_members.groupby('Party').size()
count_by_party.sort_values(ascending=True,inplace=True)
party_reduced=pd.Series([count_by_party['BJP'],count_by_party['INC'],count_by_party['AIADMK'],count_by_party['AITC'],
                         count_by_party['SP'],count_by_party['NOM.']],index=('BJP','INC','AIADMK','AITC','SP','NOM.'))

party_reduced['Others']=sum(count_by_party[key] for key in count_by_party.keys() if key not in ['BJP','INC','AIDMK','AITC','SP','NOM.'])
key_by_party=party_reduced.keys()
c=['orange','b','g','r','purple','y','white']

for i in party_reduced.values+1:
    #print(i,counter,counter+i-1)
    plt.plot(x[counter:counter+i-1], y[counter:counter+i-1], 'o', color=c[idx],label=key_by_party[idx])
    counter=counter+i
    idx=idx+1
#As 250 points are plotted, hide 5 points
plt.plot(x[245:250], y[245:250], 'o', color='black') 

plt.legend(loc='best')
plt.axis('off')
plt.title('Member distribution according to party affiliations',fontsize=20)
#Import cartopy to plot maps
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#Load the 2011 Census file into a dataframe
StateCensus=pd.read_csv('../input/India_2011census_state.csv')
#Plot the distribution of representatives by state
style.use('seaborn-white')
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([62.4, 97.96,4.9, 37.8], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
#ax.add_feature(cfeature.STATES, linestyle=':')

for i in range(StateCensus.shape[0]):
    lat,lon,density,rep,state = StateCensus[['Latitude','Longitude','Density','Rajyasabha representation','State']].iloc[i]
    
    if rep>0:
        ax.plot(lon, lat, marker='o', color='blue', markersize=rep*2.5, alpha=0.5)
    else:
        ax.plot(lon, lat, marker='X', color='red',markersize=15)
    ax.text(lon, lat, state,verticalalignment='top', horizontalalignment='center',fontsize=7,
            bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))
plt.title('Representation in Rajya Sabha by state',fontsize=20)    
plt.show()
#Load the member details for all years
members_allyears=pd.read_csv('../input/rajyasabha_member_details_allyears.csv',engine='python')
#States on timescale
state_timescale=members_allyears[['STATE','TERM_From','TERM_To']]
state_timescale['TERM_From']=pd.to_datetime(state_timescale['TERM_From'], format='%d/%m/%Y')
state_timescale['TERM_To']=pd.to_datetime(state_timescale['TERM_To'], format='%d/%m/%Y')
#Find the minimum and maximum years of representation per state
state_gantt_df=state_timescale.groupby('STATE').min()['TERM_From'].reset_index()
state_gantt_df['TERM_To']=state_timescale.groupby('STATE').max()['TERM_To'].reset_index()['TERM_To']
#Color code the gantt chart bars
diff=[]
color_encode=[]
for i in range(state_gantt_df.shape[0]):
    start_date,end_date = state_gantt_df[['TERM_From','TERM_To']].iloc[i]
    diff.append((end_date - start_date).days)
    if start_date.year==1952 and end_date.year>=2018:
        color_encode.append('orange')
    elif start_date.year==1952 and end_date.year<2018:
        color_encode.append('lightgrey')
    else: color_encode.append('green')
    
state_gantt_df['TERM_Diff']=diff
state_gantt_df['Color_encode']=color_encode
state_gantt_df.sort_values(by=['Color_encode','STATE'],ascending=True,inplace=True)
#import mpatches to plot legend
import matplotlib.patches as mpatches
#Plot Gantt chart for duration of representation of states in Rajya Sabha
#Reference - http://www.clowersresearch.com/main/gantt-charts-in-matplotlib/
fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111)
ylabels=state_gantt_df.STATE
ilen=len(ylabels)
pos = np.arange(0.5,ilen*0.5+0.5,0.5)
for i in range(len(ylabels)):
    start_date,end_date = state_gantt_df[['TERM_From','TERM_To']].iloc[i]
    ax.barh((i*0.5)+0.5, state_gantt_df.TERM_Diff.iloc[i], left=start_date, height=0.4, align='center', edgecolor=state_gantt_df.Color_encode.iloc[i], 
            color=state_gantt_df.Color_encode.iloc[i], alpha = 0.75)
locsy, labelsy = plt.yticks(pos,ylabels)
plt.setp(labelsy, fontsize = 16)
ax.axis('tight')
ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
ax.grid(axis='x',color = 'black', linestyle = ':')
plt.xticks(fontsize=16)
ax.xaxis_date()
or_patch = mpatches.Patch(color='orange',alpha = 0.75, label='States from 1952-2018')
grey_patch = mpatches.Patch(color='lightgrey',alpha = 0.75, label='States no longer present')
gr_patch = mpatches.Patch(color='green',alpha = 0.75, label='New States formed after 1952')
plt.legend(handles=[or_patch,grey_patch,gr_patch],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=16)
plt.title('Duration for representation of Indian States in Rajya Sabha(1952-2018)',fontsize=25)
#Aggregate state counts in 3 year timeslots
gantt_year=np.linspace(1953,2019,24).astype(int)
gantt_angle=np.linspace(0,360,24).astype(int)
gantt_state_cnt=[]
for i in range(len(gantt_year)-1):
    start_date=date(gantt_year[i], 1, 1)
    end_date=date(gantt_year[i+1]-1, 12, 31)
    query_str=str('TERM_From<=\''+str(start_date)+'\' & TERM_To>=\''+str(end_date)+'\'')
    gantt_state_cnt.append(state_gantt_df.query(query_str).shape[0])
#Plot circular bar graph
#Reference: https://stackoverflow.com/questions/46874689/getting-labels-on-top-of-bar-in-polar-radial-bar-chart-in-matplotlib-python3
iN = len(gantt_state_cnt)
theta=np.arange(0,2*np.pi,2*np.pi/iN)
#width = (2*np.pi)/iN *0.9

fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.5, 0.5], polar=True)
bars = ax.bar(theta, gantt_state_cnt, width=0.1, bottom=2,color='navy',alpha=0.7)
ax.set_xticks(theta)
plt.axis('off')
bottom = 8
rotations = np.rad2deg(theta)
for x, bar, rotation, label, state_count  in zip(theta, bars, rotations, gantt_year,gantt_state_cnt):
    lab = ax.text(x,bottom+4+bar.get_height() , label , ha='center', va='bottom',fontsize=15 )  
    lab = ax.text(x,bottom-3+bar.get_height() , state_count , ha='center', va='bottom',color='red',fontsize=15 ) 
#Political parties on timescale
PP_timescale=members_allyears[['PARTY','TERM_From','TERM_To']]
PP_timescale['TERM_From']=pd.to_datetime(PP_timescale['TERM_From'], format='%d/%m/%Y')
PP_timescale['TERM_To']=pd.to_datetime(PP_timescale['TERM_To'], format='%d/%m/%Y')
PP_gantt_df=PP_timescale.groupby('PARTY').min()['TERM_From'].reset_index()
PP_gantt_df['TERM_To']=PP_timescale.groupby('PARTY').max()['TERM_To'].reset_index()['TERM_To']
#Create Gantt chart
diff=[]
color_encode=[]
for i in range(PP_gantt_df.shape[0]):
    start_date,end_date = PP_gantt_df[['TERM_From','TERM_To']].iloc[i]
    diff.append((end_date - start_date).days)
    if end_date.year>=2018:
        color_encode.append('green')
    else: color_encode.append('orange')
PP_gantt_df['TERM_Diff']=diff
PP_gantt_df['Color_encode']=color_encode
PP_gantt_df.sort_values(by=['TERM_From','Color_encode'],ascending=False,inplace=True)
#plot Gantt chart
fig = plt.figure(figsize=(30,20))
ax = fig.add_subplot(111)
ylabels=PP_gantt_df.PARTY
ilen=len(ylabels)
pos = np.arange(0.5,ilen*0.5+0.5,0.5)
for i in range(len(ylabels)):
    start_date,end_date = PP_gantt_df[['TERM_From','TERM_To']].iloc[i]
    ax.barh((i*0.5)+0.5, PP_gantt_df.TERM_Diff.iloc[i], left=start_date, height=0.4, align='center', edgecolor='lightgreen', 
            color=PP_gantt_df.Color_encode.iloc[i], alpha = 0.8)
locsy, labelsy = plt.yticks(pos,ylabels)
plt.setp(labelsy, fontsize = 18)
plt.xticks(fontsize=18)
ax.axis('tight')
ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
ax.grid(color = 'g', linestyle = ':')
ax.xaxis_date()
or_patch = mpatches.Patch(color='orange',alpha = 0.8, label='Political Parties no longer active')
gr_patch = mpatches.Patch(color='green',alpha = 0.8, label='Political Parties existing in 2018')
plt.legend(handles=[or_patch,gr_patch],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=18)
plt.title('Duration for representation of Political Parties in Rajya Sabha(1952 - 2018)',fontsize=24)
#plot donut chart showing proportion of categories
plt.pie(PP_gantt_df.groupby('Color_encode').size(),colors=list(PP_gantt_df.groupby('Color_encode').size().index),
        counterclock=True)
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
#histogram of member terms
from datetime import datetime, timedelta
from dateutil import relativedelta
term_len=[]
for i in range(members_allyears.shape[0]):
    term_from = datetime.strptime(members_allyears.TERM_From.iloc[i], '%d/%m/%Y')
    term_to = datetime.strptime(members_allyears.TERM_To.iloc[i], '%d/%m/%Y') + timedelta(days=1)
    r = relativedelta.relativedelta(term_to, term_from)
    term_len.append(np.round((r.years*365+r.months*30+r.days)/365,1))
members_allyears['Term_length_mo']=term_len
#add up term lengths for each member
term_lens=members_allyears.groupby('NAME OF MEMBER').sum()
#plot histogram for term lengths
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(1, 1, 1)
plt.hist(x=term_lens.Term_length_mo,bins=max(term_lens.Term_length_mo.astype('int')),bottom=10,
         histtype='stepfilled',orientation='horizontal',color='blue')
ax1.set_ylabel('Number of years served as Member of Parliament (MP)',fontsize=18)
ax1.set_xlabel('Number of members with this tenure',fontsize=18)
ax1.text(1200,7.5, 'Single term (6 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.text(400,13.5, 'Two terms (12 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.text(250,19.5, 'Three terms (18 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.text(200,24, 'Four terms (24 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.text(200,30, 'Five terms (30 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.text(200,36, 'Six terms (36 years)',verticalalignment='top', horizontalalignment='center',fontsize=18,
            bbox=dict(facecolor='grey', alpha=0.5, boxstyle='round'))
ax1.grid(axis='x')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Distribution of tenure for members of Rajya Sabha(1952-2018)',fontsize=20)
#plot Barchart showing number of representatives per state
rep=StateCensus[StateCensus['Rajyasabha representation']>0][['State','Rajyasabha representation']]
rep.columns=['State','rep']
#update representation for Andhra Pradesh
rep=rep.set_value(col='rep',value=11,index=rep[rep.State=='Andhra Pradesh'].index)
#update Delhi NCR -> National Capital Territory of Delhi
rep=rep.set_value(col='State',value='National Capital Territory of Delhi',index=rep[rep.State=='Delhi NCR'].index)
#add states values - NA, Nominated, Telengana
rep=rep.append({'State':'NA','rep':0},ignore_index=True)
rep=rep.append({'State':'Nominated','rep':12},ignore_index=True)
rep=rep.append({'State':'Telangana','rep':7},ignore_index=True)
rep.sort_values(by=['State'],ascending=False,inplace=True)

plt.barh(y=range(rep.shape[0]), width=rep.rep, align='center',height=0.9,color='green',alpha=0.4)
plt.yticks(range(rep.shape[0]), list(rep.State),fontsize=30)
#plt.gcf().set_size_inches(70,70)
plt.axis('off')
plt.show()
#create a copy of member name df to use as member lookup
lkp_member_name=members_allyears.copy()
#Standardize the name format in member lookup to match with the name format in question and answers.
name_formatted=[]
for i in range(lkp_member_name.shape[0]):
    name,title=lkp_member_name[['NAME OF MEMBER','Title_derivedcol']].iloc[i]
    name_split=name.replace(title,str("," + title)).split(',')
    #print(name_split,title)
    name_formatted.append(str.strip(name_split[1] + ' ' + name_split[0]))

#Add formatted name as a column to the dataframe
lkp_member_name['Name_Formatted']=name_formatted
lkp_member_name['Name_idx']=[str.lower(name) for name in name_formatted]
#Set formated name as the dataframe index for easy access to member attributes
lkp_member_name.set_index('Name_idx',inplace=True)
#Load session details
session_df=pd.read_csv('../input/rajyasabha_session.csv',engine='python')
session_df['start_date_formatted']=pd.to_datetime(session_df.session_start_date,dayfirst=False)
session_df['end_date_formatted']=pd.to_datetime(session_df.session_end_date,dayfirst=False)
session_df.pop('session_start_date')
session_df.pop('session_end_date')
session_df.set_index('session_no',inplace=True)
#function to lookup session number
def get_session(df_year,df_session):
    
    
    min_ans_date=np.min(df_year.answer_date_formatted)
    max_ans_date=np.max(df_year.answer_date_formatted)
    min_session_num=df_session.query(str('start_date_formatted<=\'' + str(min_ans_date) + '\'')).index.max()
    max_session_num=df_session.query(str('end_date_formatted>=\'' + str(max_ans_date) + '\'')).index.min()
    #initialize session_num as an array of min session nums of length df_year - because year 2014 has answer dates outside of the session dates range
    session_num=[min_session_num]*df_year.shape[0]
    print(min_ans_date,max_ans_date,min_session_num,max_session_num)
    for i in range(df_year.shape[0]):
        for j in range(min_session_num,max_session_num+1):
            if session_df.start_date_formatted.loc[j]<=df_year.answer_date_formatted.iloc[i]<=session_df.end_date_formatted.loc[j]:
                session_num[i]=j
                break
    return session_num
%%time
#Load all yearwise question and answer files
start_year=2009
years=range(start_year+1,2018)
df_complete=pd.read_csv(str('../input/rajyasabha_questions_and_answers_' + str(start_year) + '.csv'))
df_complete.columns=['question', 'answer_date', 'ministry', 'question_type', 'question_no',
       'question_by', 'question_title', 'question_description', 'answer']
df_complete['answer_date_formatted']=pd.to_datetime(df_complete.answer_date,dayfirst=True)
df_complete.drop(columns=['question','question_type','question_no','question_description','answer','answer_date'],inplace=True)
df_complete['session_num']=get_session(df_complete,session_df)
df_complete['Year']=[start_year]*df_complete.shape[0]
for year in years:
    
    filename=str('../input/rajyasabha_questions_and_answers_' + str(year) + '.csv')
    df_year=pd.read_csv(filename)
    #print(df_year.head(1))
    #standardizing column names as files for some years have have different column names
    df_year.columns=['question', 'answer_date', 'ministry', 'question_type', 'question_no',
       'question_by', 'question_title', 'question_description', 'answer']
    df_year['answer_date_formatted']=pd.to_datetime(df_year.answer_date,dayfirst=True)
    df_year.drop(columns=['question','question_type','question_no','question_description','answer','answer_date'],inplace=True)
    df_year['session_num']=get_session(df_year,session_df)
    df_year['Year']=[year]*df_year.shape[0]
    df_complete=pd.concat([df_complete,df_year],ignore_index=True)
    #print(year,df_year.shape[0],df_complete.shape[0])
#Consolidate all member information in df_member_profile
df_member_profile=pd.DataFrame(df_complete.question_by.unique(),columns=['member_name'])

#remove Names = NaN
NaN_idx=df_member_profile[df_member_profile.member_name.isnull()].index
df_member_profile.drop(NaN_idx,axis=0,inplace=True)

#count of questions per session
countby_session=df_complete.groupby(['question_by','session_num']).size()
countby_session_df=countby_session.reset_index().set_index('question_by')

member_state=[]
member_party=[]
member_term_start=[]
member_term_end=[]
member_session=[]
member_qcount=[]
lkp_idx=lkp_member_name.index
for i in range(df_member_profile.shape[0]):
    name=df_member_profile.member_name.iloc[i]
    if str.lower(name) in lkp_idx:
        member_state.append(np.unique(lkp_member_name.STATE.at[str.lower(name)]))
        member_party.append(np.unique(lkp_member_name.PARTY.at[str.lower(name)]))
        member_term_start.append(lkp_member_name.TERM_From.at[str.lower(name)]) 
        member_term_end.append(lkp_member_name.TERM_To.at[str.lower(name)]) 
        member_session.append(countby_session_df.session_num.at[name])
        member_qcount.append(countby_session_df[0].at[name])

    else:
        member_state.append('NA')
        member_party.append('NA')
        member_term_start.append('NA')
        member_term_end.append('NA')
        member_session.append('NA')
        member_qcount.append('NA')
        
df_member_profile['State']=member_state
df_member_profile['Party']=member_party
df_member_profile['Term_start']=member_term_start
df_member_profile['Term_end']=member_term_end
df_member_profile['Session']=member_session
df_member_profile['Qcounts']=member_qcount

df_member_profile.set_index('member_name',inplace=True)
#Add the member name column back into data frame for further use
df_member_profile['member_name']=list(df_member_profile.index)
#Lookup and add member state
member_state=[]
lkp_idx=df_member_profile.index
for i in range(df_complete.shape[0]):
    name=df_complete.question_by.iloc[i]
    #member_names=df_member_profile['Name_Formatted'].unique()
    #print(name)
    #Some member names are not present in the lkp
    if name in lkp_idx:
        #members can be elected from multiple states during different terms so lookup can either return a single state or 
        #an array of states for members with multiple terms.
        #Taking only one state value for sankey chart if multiple found
        state_name=df_member_profile.State.at[name]
        
        if isinstance(state_name,np.ndarray):
            member_state.append(state_name[0])
        else:
            member_state.append(state_name)
    else:
        member_state.append('NA')
df_complete['Member_State']=member_state
#pip install pySankey
#https://github.com/anazalea/pySankey
from pySankey import sankey
#plot sankey chart for all questions
sankey_all=df_complete[['Member_State','ministry','Year']]
#drop NaN records as Sankey cannot handle Nulls
NaN_idx=sankey_all[sankey_all.Member_State.isnull()].index
sankey_all.drop(NaN_idx,axis=0,inplace=True)
NaN_idx=sankey_all[sankey_all.ministry.isnull()].index
sankey_all.drop(NaN_idx,axis=0,inplace=True)
#Remove dups
ministry_dups=['AGRICULTURE','CONSUMER AFFAIRS AND PUBLIC DISTRIBUTION','ENVIRONMENT AND FORESTS','SKILL DEVELOPMENT AND ENTREPRENEURSHIP','WATER RESOURCES']
ministry_replace=['AGRICULTURE AND FARMERS WELFARE','CONSUMER AFFAIRS, FOOD AND PUBLIC DISTRIBUTION','ENVIRONMENT, FOREST AND CLIMATE CHANGE','SKILL DEVELOPMENT ENTERPRENEURSHIP, YOUTH AFFAIRS','WATER RESOURCES, RIVER DEVELOPMENT AND GANGA REJUVENATION']

ministry_fix=sankey_all.ministry.apply(lambda x:str.strip(x))
ministry_fix=ministry_fix.apply(lambda x:' '.join(x.split('  ')))
sankey_all.ministry=ministry_fix

for i in range(len(ministry_dups)):
    sankey_all.loc[sankey_all.ministry==ministry_dups[i],'ministry']=ministry_replace[i]
    
sankey_all.sort_values(by=['Member_State','ministry'],ascending=False,inplace=True)
sankey.sankey(sankey_all['Member_State'], sankey_all['ministry'], fontsize=40)
plt.gcf().set_size_inches(70,70)
plt.title('Flow of questions from State to Ministries',fontsize=70)
# import libraries for interactive plots
#https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html#interactive
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
#Interactive sankey plot - user can select state or ministry to zoom in details
#plt.gcf().set_size_inches(35,35)
states_sk=list(np.sort(sankey_all.Member_State.unique()))
ministry_sk=list(np.sort(sankey_all.ministry.unique()))
x_widget = widgets.Dropdown(options=['All','By State','By Ministry'], value='All', description='Select Key:', disabled=False,)
y_widget = widgets.Dropdown(options=['NA'], value='NA', description='Key Values:', disabled=False,)

play = widgets.Play(
#     interval=10,
    value=2009,
    min=2009,
    max=2017,
    step=1,
    description="Press play",
    disabled=False
)

slider = widgets.IntSlider(value=2009,min=2009,max=2017,step=1,description='Year:')


def update_keys(*args):
    if x_widget.value=='All':
        y_widget.options = 'NA'        
    elif x_widget.value=='By Ministry':
        y_widget.options = ministry_sk
    else:
        y_widget.options = states_sk
x_widget.observe(update_keys, 'value')

def plot_sankey(*args):
    q_year=slider.value
    print(slider.value)
#query_str='Member_State==\'Maharashtra\''
    if x_widget.value=='All':
        query_str=str('Year==' + str(slider.value))
        plt_title=str('Volume of questions from all states to all Ministries for Year ' + str(slider.value))
    elif x_widget.value=='By Ministry':
        query_str=str('ministry==\''+y_widget.value+'\' & Year==' + str(slider.value))
        plt_title=str('Volume of questions from all states to MINISTRY OF '+ y_widget.value + ' for Year ' + str(slider.value))
    else:
        query_str=str('Member_State==\''+y_widget.value+'\' & Year==' + str(slider.value))
        plt_title=str('Volume of questions from '+ y_widget.value + ' to all Ministries for Year ' + str(slider.value))
    select_state=sankey_all.query(query_str)[['Member_State','ministry']]
    select_state.sort_values(by=['Member_State','ministry'],ascending=False,inplace=True)
    #state_ordered=pd.Series(np.sort(select_state['Member_State'].unique()))
    #ministry_ordered=pd.Series(np.sort(select_state['ministry'].unique()))
    sankey.sankey(left=select_state['Member_State'], right=select_state['ministry'], fontsize=25)
                  
    plt.gcf().set_size_inches(35,35)
    plt.title(str(plt_title),fontsize=50)
    
slider.observe(plot_sankey,'value')

def plot_fig(x,y,z):
    plot_sankey()
#widgets.jslink((play, 'value'), (slider, 'value'))
#widgets.HBox([x_widget,y_widget,play, slider])
interact(plot_fig,x=x_widget, y=y_widget,z=slider);
#display(widgets.HBox([x_widget,y_widget,play, slider]))

#Import NLP libraries from NLTK
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn 
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
#define stop words
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
#Define Regex pattern to identify topics
pattern_topical_nouns = r"""
  NP: {<NN.*|VB.*|CC|JJ.*|RP>*<IN|TO>}   # chunk determiner/possessive, adjectives and noun
      {<NN.*|VB.*|CC|JJ.*|RP>*<IN|TO>*<NN.*|VB.*|CC|JJ.*|RP>*}
"""
cp = nltk.RegexpParser(pattern_topical_nouns)
#define function to preprocess data
def nlp_preprocess(df):
    q_token_arr=[]
    q_token_POS=[]
    q_token_NP=[]
    for q in df.question_title:
        q_token= nltk.tokenize.word_tokenize(q,preserve_line=False)
        #remove stop words
        #q_tokenized=[str(word) for word in q_token if str(word).lower() not in stop]
        q_token=[lemmatizer.lemmatize(q_token[i].lower()) for i in range(len(q_token))]
        q_pos=nltk.pos_tag(q_token) 
        q_token_arr.append(q_token)
        q_token_POS.append(q_pos)
        q_token_NP.append(cp.parse(q_pos))
    df['q_token']=q_token_arr
    df['q_pos']=q_token_POS
    df['q_pos_NP']=q_token_NP
    return df
#define function to extract topics
def extract_topics(df):
    w=[]
    tag_dict={}
    for i in range(df.shape[0]):
        tags=[]
        t = list(df.q_pos_NP[i].subtrees(filter=lambda x: x.label()=='NP'))
            #print(t[len(t)-2],t[len(t)-1])
        
        if len(t)==0:
            tags.append('NA')
        elif len(t)>1:
            for j in [2,1]:
                tags.append([word for word,pos in t[len(t)-j] if pos in['NNP','NNS','NN']])
        else:
            #print(t)
            tags.append([word for word,pos in t[len(t)-1] if pos in['NNP','NNS','NN']])
        tag_dict[df['index'][i]]=tags
        w.append(tags)
    flattened_list = [z for x in w for y in x for z in y]
    return flattened_list,tag_dict
#define function to generate word frequencies
def dict_bag_of_words(flattened_list):
    dict_bag_of_words={}
    for idx in flattened_list:
        if idx in dict_bag_of_words.keys():
            dict_bag_of_words[idx]+=1 
        else:
            dict_bag_of_words[idx]=1
    return dict_bag_of_words
#define functions to plot word frequencies as bar chart
def plot_word_freq(D,top_count,title_word):
    if top_count==None:
        top_count=25
        
    df=pd.DataFrame.from_dict(D,orient='index',columns=['Counts'])
    df.sort_values(by=['Counts'],ascending=False,inplace=True)
    df_top=df.head(top_count)
    df_top.sort_index(ascending=False,inplace=True)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    my_range=range(1,len(df_top.index)+1)
    plt.barh(my_range, width=df_top.Counts, height=0.4, color='blue',alpha=0.3)
    plt.plot(df_top.Counts, my_range, "o",color='blue')
    plt.yticks(my_range, df_top.index)
    plt.grid(color='black',axis='x',linestyle = ':')
    plt.title('Word Frequencies for top ' + str(top_count) + ' words for '+ str(title_word))
    plt.show()
#Create dataframe to consolidate word frequencies by ministry
ministry=df_complete.ministry.unique()
years=list(df_complete.Year.unique())
level_ministry=np.repeat(ministry,len(years),axis=0)
level_year=np.array(years*len(ministry))
df_ministry_wc=pd.DataFrame(level_ministry,columns=['Ministry'])
df_ministry_wc['Year']=level_year
NaN_idx=df_ministry_wc[df_ministry_wc.Ministry.isnull()].index
df_ministry_wc.drop(NaN_idx,axis=0,inplace=True)
%%time
#Compute word frequencies by ministry

#df_temp.set_index('question_by',inplace=True)

tags_wc=[]

for i in range(df_ministry_wc.shape[0]):
    tags_dict={}
    ministry=df_ministry_wc.Ministry.iloc[i]
    year=str(df_ministry_wc.Year.iloc[i])
    query=str('ministry==\''+ ministry +'\' & Year==' + year)
    q_df=df_complete.query(query).question_title.reset_index()
    q_df_processed=nlp_preprocess(q_df)
    flattened_list,tags_dict_inc=extract_topics(q_df_processed)
    
    if len(flattened_list)>0:
        tags_dict.update(tags_dict_inc)
        dict_word_cnt=dict_bag_of_words(flattened_list)
        #wc = WordCloud().generate_from_frequencies(dict_word_cnt)
        tags_wc.append(dict_word_cnt)
    else :
        #wc=WordCloud().generate_from_frequencies({'No Data':100})
        tags_wc.append({'No Data':0})
df_ministry_wc['word_cloud']=tags_wc
#plot word chart for top 10 question topics per ministry
style.use('dark_background')
fig = plt.figure(figsize=(50,10))
ax = fig.add_subplot(1, 1, 1)
plt.scatter([0,3000],[0,3000],alpha=0.1)
h_offset=0
j=0
cmap = plt.cm.get_cmap("tab20", 11)
for year in range(2010,2018):
    v_offset=100    
    query=str('Ministry==\''+ str('FINANCE') +'\' & Year==' + str(year))
    idx=df_ministry_wc.query(query).index[0]
    #print(year)
    #d=df_ministry_wc.word_cloud.at[idx]
    df=pd.DataFrame.from_dict(df_ministry_wc.word_cloud.at[idx],orient='index',columns=['Counts'])
    df.sort_values(by=['Counts'],ascending=False,inplace=True)
    df_top=df.head(10)
    df_top.sort_values(by=['Counts'],ascending=False,inplace=True)
    df_top.reset_index(inplace=True)
    df_top.columns=['Word','Counts']
    j+=1
    for i in range(df_top.shape[0]):
        t=np.random.randint(1)
        count=df_top.Counts.at[i]
        if count>70:
            count=70
        elif count<15:
            count=15
        ax.text(h_offset+count/2,v_offset+count, df_top.Word.at[i],verticalalignment='top', horizontalalignment='center',
                fontsize=count,color=cmap(j))
        v_offset+=300
    h_offset+=400
plt.axis('off')
#change the plotting style
style.use('seaborn-white')
%%time
#df_temp.set_index('question_by',inplace=True)
tags_dict={}
tags_wc=[]
for i in range(df_member_profile.shape[0]):
    name=df_member_profile.member_name.iloc[i]
    query=str('question_by==\''+ name +'\'')
    q_df=df_complete.query(query).question_title.reset_index()
    q_df_processed=nlp_preprocess(q_df)
    flattened_list,tags_dict_inc=extract_topics(q_df_processed)
    tags_dict.update(tags_dict_inc)
    dict_word_cnt=dict_bag_of_words(flattened_list)
    #wc = WordCloud().generate_from_frequencies(dict_word_cnt)
    tags_wc.append(dict_word_cnt)
df_member_profile['word_cloud']=tags_wc
#Interactive member profile
states=np.unique(lkp_member_name.STATE)
states[-1]='All States'
members=np.unique(lkp_member_name.Name_Formatted)
print('Narrow down the search result for member names by selecting the state first')
x_widget = widgets.Dropdown(options=states, value='All States', description='State:', disabled=False,)
y_widget = widgets.Dropdown(options=members, value=members[0], description='Member:', disabled=False,)

def update_names(*args):
    if x_widget.value!='All States':
        query_str=str('STATE==\''+x_widget.value+'\'')
        y_widget.options = np.unique(lkp_member_name.query(query_str).Name_Formatted)
    else:
        y_widget.options=members
        
x_widget.observe(update_names, 'value')

def printer(x, y):
    if y in df_member_profile.index:
        print('Member Profile')
        print('Name: ',y)
        print('State Represented: ',df_member_profile.State.at[y])
        print('Political Party: ',df_member_profile.Party.at[y])
        
        if isinstance(df_member_profile.Term_start.at[y],np.ndarray):
            for i in range(len(df_member_profile.Term_start.at[y])):
                print('Term ',i+1,': ',df_member_profile.Term_start.at[y][i],'-',df_member_profile.Term_end.at[y][i])
        else:
            print('Term 1: ',df_member_profile.Term_start.at[y],'-',df_member_profile.Term_end.at[y])
        #print('Trend of questions asked per session:')
        plt.bar(df_member_profile.Session.at[y], height=df_member_profile.Qcounts.at[y], width=0.3, align='center', edgecolor='lightgreen', 
            color='blue', alpha = 0.8)
        plt.xticks(df_member_profile.Session.at[y])
        plt.title('Trend of questions asked per session')
        #plt.yticks(np.arange(0, 350, 50))
        plt.show()
        #print('Word frequency of most common topics raised by this member: ')
        plot_word_freq(df_member_profile.word_cloud.at[y],25,y)
        
    else:
        print('Profile not found.')
        
interact_manual(printer,x=x_widget, y=y_widget);