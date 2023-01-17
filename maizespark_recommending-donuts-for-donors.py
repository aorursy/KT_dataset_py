# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Visualization Libraries
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn

resources=pd.read_csv('../input/Resources.csv',index_col=False)
print(resources.info())
resources.head()
#from wordcloud import WordCloud, STOPWORDS
#wordcloud = WordCloud( max_font_size=50, 
#                       stopwords=STOPWORDS,
#                       background_color='white',
#                     ).generate(str(resources['Resource Item Name'].tolist()))
#import matplotlib
#import matplotlib.pyplot as plt # for plotting
#plt.figure(figsize=(14,7))
#plt.title("Wordcloud for Top Keywords in Project Title", fontsize=35)
#plt.imshow(wordcloud)
#plt.axis('off')
#plt.show()
resources['Resource Total']=resources['Resource Quantity']*resources['Resource Unit Price']
print(resources[['Resource Item Name','Resource Total']].groupby(['Resource Item Name'])['Resource Total'] \
                                              .sum() \
                                              .reset_index(name='sum') \
                                              .sort_values(['sum'], ascending=False))
resources=resources.fillna(-1)
#resources['Resource Item Name']=pd.Categorical(resources['Resource Item Name'])
#resources['Resource Item Name']=resources['Resource Item Name'].cat.codes
#resources['Resource Vendor Name']=pd.Categorical(resources['Resource Vendor Name'])
#resources['Resource Vendor Name']=resources['Resource Vendor Name'].cat.codes
resources['Resource Quantity']=resources['Resource Quantity'].astype(int)

schools=pd.read_csv('../input/Schools.csv',index_col=False)
print(schools.info())
schools.head()
schools=schools.fillna(-1)
schools['School Percentage Free Lunch']=schools['School Percentage Free Lunch'].astype(int)
donors=pd.read_csv('../input/Donors.csv',index_col=False)
print(donors.info())
donors.head()
donors['Donor Zip']=donors['Donor Zip'].fillna('-1')
donors['Donor Zip'][donors['Donor Zip'].apply(np.isreal)]=donors['Donor Zip'][donors['Donor Zip'].apply(np.isreal)].apply(str).str.replace('.0','')
donors['Donor Zip'][~donors['Donor Zip'].str.isdigit()]='-1'
donors['Donor Zip']=pd.to_numeric(donors['Donor Zip'])
donations=pd.read_csv('../input/Donations.csv',index_col=False)
print(donations.info())
donations.head()
teachers=pd.read_csv('../input/Teachers.csv',index_col=False)
print(teachers.info())
teachers.head()
teachers['Teacher First Projected Posted Year']=pd.to_numeric(teachers['Teacher First Project Posted Date'].str[:4])
teachers['Teacher First Projected Posted Month']=pd.to_numeric(teachers['Teacher First Project Posted Date'].str[4:6])
del teachers['Teacher First Project Posted Date']
projects=pd.read_csv('../input/Projects.csv',index_col=False)
print(projects.info())
projects.head()
#projects['Project Cost']=projects['Project Cost'].str.replace('$','').str.replace(',','')
#projects['Project Cost']=pd.to_numeric(projects['Project Cost'])
projects['Project Posted Year']=pd.to_numeric(projects['Project Posted Date'].str[:4])
projects['Project Posted Month']=pd.to_numeric(projects['Project Posted Date'].str[4:6])
del projects['Project Posted Date']
donations=donations.merge(projects,how='left',on='Project ID')
del projects
donations=donations.merge(teachers,how='left',on='Teacher ID')
del teachers
donations=donations.merge(donors,how='left',on='Donor ID')
del donors
donations=donations.merge(schools,how='left',on='School ID')
del schools
donations['Teacher Project Posted Sequence']=donations['Teacher Project Posted Sequence'].fillna(-1).astype(int)
donations['Project Posted Year']=donations['Project Posted Year'].fillna(-1).astype(int)
donations['Project Posted Month']=donations['Project Posted Month'].fillna(-1).astype(int)
donations['Teacher First Projected Posted Year']=donations['Teacher First Projected Posted Year'].fillna(-1).astype(int)
donations['Teacher First Projected Posted Month']=donations['Teacher First Projected Posted Month'].fillna(-1).astype(int)
donations['School Percentage Free Lunch']=donations['School Percentage Free Lunch'].fillna(-1).astype(int)
donations['Donor Zip']=donations['Donor Zip'].fillna(-1).astype(int)
donations['School Zip']=donations['School Zip'].fillna(-1).astype(int)
donations.info()
temp = donations[['Donation Amount','Donor ID']].groupby(['Donor ID'])['Donation Amount'] \
                                                    .agg(['sum']) \
                                                    .sort_values('sum', ascending=False)
temp['Donor ID']=temp.index
temp['label']=pd.cut(temp['sum'],100000)
temp1=temp[['label','Donor ID']].groupby(['label'])['Donor ID'].agg(['count'])
temp1.head(20).plot(kind='bar')
temp2=temp[['label','sum']].groupby(['label'])['sum'].agg(['sum'])
temp2.head(100).plot(kind='bar')
del temp1
del temp2
temp = donations[['Donation Amount','Donor Cart Sequence']].groupby(['Donor Cart Sequence'])['Donation Amount'] \
                                                    .agg(['sum']) \
                                                    .sort_values('sum', ascending=False)
temp.head(10).plot(kind='pie',subplots=True,title='Donor Cart Sequence', autopct='%.1f%%',legend=False)

#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()
print(temp.head(5).sum())
print(temp.tail(len(temp)-5).sum())
temp1=pd.concat([temp.head(5)['sum'],temp.tail(len(temp)-5).sum().to_frame()])
temp1.plot(kind='pie',subplots=True,title='Donor Cart Sequence', autopct='%.1f%%',legend=False)
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()
del temp
del temp1
temp = donations[['Teacher ID','Donation Amount']].groupby(['Teacher ID'])['Donation Amount'].agg(['sum']).sort_values('sum', ascending=False)
temp.head(10).plot(kind='pie',title='Top Teachers',subplots=True,sharex=False,sharey=False,legend=False,autopct='%.1f%%')
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()
print(temp.head(50).sum())
print(temp.tail(len(temp)-50).sum())
temp1=pd.concat([temp.head(50)['sum'],temp.tail(len(temp)-50).sum().to_frame()])
temp1.plot(kind='pie',subplots=True,title='Donor Cart Sequence', autopct='%.1f%%',legend=False)
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show() 
teacher_dict=donations.groupby(['Teacher ID'])['Teacher First Projected Posted Year'].agg(['mean']).sort_values('mean').to_dict()['mean']
temp=donations.groupby(['Teacher ID'])['Donation Amount'].agg(['sum']).sort_values('sum', ascending=False)
temp['Teacher ID']=temp.index
temp['Teacher Year']=temp['Teacher ID'].map(teacher_dict)
temp['Teacher Year']=temp['Teacher Year'].fillna(-1).astype(int)
teacher_yr_count=temp.groupby(['Teacher Year'])['Teacher ID'].agg('count')
donations_by_teacher_exp = donations.groupby(['Teacher First Projected Posted Year'])['Donation Amount'].sum().sort_values()
donations_by_teacher_exp.plot(kind='pie',title='Donation by Teacher Experience',subplots=True,sharex=False,sharey=False,legend=False,autopct='%.1f%%')
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()
teacher_dict=donations.groupby(['Teacher ID'])['Teacher First Projected Posted Year'].agg(['mean']).sort_values('mean').to_dict()['mean']
temp1=donations.groupby(['Teacher ID'])['Donation Amount'].sum().sort_values()
temp1=pd.DataFrame(temp1)
temp1['Teacher ID']=temp1.index
temp1['Teacher Year']=temp1['Teacher ID'].map(teacher_dict)
teacher_yr_count=temp1.groupby(['Teacher Year'])['Teacher ID'].agg('count')
don_to_teacher_by_exp = np.divide(donations_by_teacher_exp.sort_index(),teacher_yr_count)
don_to_teacher_by_exp=don_to_teacher_by_exp.drop(-1)
don_to_teacher_by_exp.plot(kind='bar',title='Donation to Teacher by Experience',subplots=True,sharex=False,sharey=False,legend=False)
del temp1
donations['Fully Funded']=(donations['Project Current Status']=='Fully Funded')
donations['Fully Funded'][donations['Project Current Status']=='Live']=-1
proj_fully_funded=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Fully Funded'].mean()
don_per_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Donation Amount'].sum()
cost_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Project Cost'].mean()
proj_fund=pd.concat([don_per_proj,cost_by_proj,proj_fully_funded],axis=1)
proj_fund['Full Coverage']=proj_fund['Donation Amount']>=proj_fund['Project Cost']
proj_fund[['Fully Funded','Full Coverage']].mean().plot(kind='bar')
proj_fully_funded=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Fully Funded','Teacher First Projected Posted Year'].mean()
fully_funded_ratio_by_teacher_exp=proj_fully_funded.groupby(['Teacher First Projected Posted Year'])['Fully Funded'].mean()
fully_funded_ratio_by_teacher_exp=fully_funded_ratio_by_teacher_exp.drop(-1)
fully_funded_ratio_by_teacher_exp.plot(kind='bar',title='Fully Funded Project Ratio by Experience',subplots=True,sharex=False,sharey=False,legend=False)
teacher_seq_fully_funded_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Teacher Project Posted Sequence','Fully Funded'].mean()
fully_funded_by_teacher_seq=teacher_seq_fully_funded_by_proj.groupby(['Teacher Project Posted Sequence'])['Fully Funded'].agg(['mean','count'])
fully_funded_by_teacher_seq=fully_funded_by_teacher_seq.drop(-1)
fully_funded_by_teacher_seq['mean'].head(30).plot(kind='line',title='Fully Funded Ratio by Teacher Project Posted Sqeuence',subplots=True,sharex=False,sharey=False,legend=False)
donations['Same State Donation']=(donations['Donor State']==donations['School State'])
print(str(round(donations['Same State Donation'].mean()*100,2))+'% of donors chose to donate to projects happening on the same state as their residence.')
by_state=donations.groupby(['Donor State'])['Same State Donation'].mean().sort_values(ascending=False)*100
by_state.head(20).plot(kind='bar',title='Percentage of donation to same state')
top_donors = donations['Donor ID'].value_counts().head(20)
top_donors=pd.DataFrame(top_donors)
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[np.logical_and(donations["Donor ID"]==donor,donations['Teacher Project Posted Sequence']!=-1)] \
                .groupby(['Teacher Project Posted Sequence'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor)    
plt.tight_layout()
plt.show()
top_donors = donations['Donor ID'].value_counts().head(20)
top_donors=pd.DataFrame(top_donors)
bars=pd.DataFrame(index=np.arange(len(fully_funded_by_teacher_seq))+1)
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[np.logical_and(donations["Donor ID"]==donor,donations['Teacher Project Posted Sequence']!=-1)] \
                .groupby(['Teacher Project Posted Sequence'])['Donation Amount'].sum()
    teacher_proj_seq=donations[np.logical_and(donations["Donor ID"]==donor,donations['Teacher Project Posted Sequence']!=-1)] \
                .groupby(['Project ID'])['Teacher Project Posted Sequence'].mean()
    teacher_proj_seq_count=teacher_proj_seq.value_counts()
    bars['donuts']=donuts
    bars['teacher_proj_seq']=teacher_proj_seq_count
    bars['donation_weighted_by_teacher_population']=np.divide(bars.donuts,bars.teacher_proj_seq)
    plt.subplot(5,4,i+1)
    bars['donation_weighted_by_teacher_population'].head(10).plot(kind='line',title=donor)    
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['Teacher ID'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['Project ID'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['Same State Donation'])['Donation Amount'].sum()
    plt.subplot(5,4,i+1)
    donuts.plot(kind='bar',title=donor)   
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['School State'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor)
    plt.xticks([])
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['School Zip'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor)
    plt.xticks([])
plt.tight_layout()
plt.show()
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['School ID'])['Donation Amount'].sum().sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
def returnself(x):
    return x.iloc[0]
proj_fully_funded=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Fully Funded'].mean()
proj_state=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['School Metro Type'].apply(returnself)
proj_fully_funded_state=pd.concat([proj_fully_funded,proj_state],axis=1)
fully_funded_ratio_by_state=proj_fully_funded_state.groupby(['School Metro Type'])['Fully Funded'].mean().sort_values(ascending=False)*100
fully_funded_ratio_by_state.plot(kind='bar',title='Fully Funded Project Percentage By School Metro Type',subplots=True,sharex=False,sharey=False,legend=False)
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['School Metro Type'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

don_amount_by_proj=donations.groupby(['Project ID'])['Donation Amount'].sum()
proj_state=donations.groupby(['Project ID'])['School State'].apply(returnself)
don_amount_school_state_by_proj=pd.concat([don_amount_by_proj,proj_state],axis=1)
don_amount_by_school_state=don_amount_school_state_by_proj.groupby(['School State'])['Donation Amount'].sum().sort_values(ascending=False)*100

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df=pd.DataFrame(index=don_amount_by_school_state.index)
df['Donation Amount']=don_amount_by_school_state/1000000
df['School States']=df.index

df['text'] = df.index + '<br>' +\
    'Donation Amount: $' + df['Donation Amount'].round(2).astype(str) + 'm'

state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

df['code'] = df['School States'].map(state_codes)
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['Donation Amount'],
        locationmode = 'USA-states',
        text=df.text,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = '$millions')
        ) ]

layout = dict(
        title = 'Donation Amount by School State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict(data=data,layout=layout)
iplot(fig)

del don_amount_by_proj
del proj_state
del don_amount_school_state_by_proj
proj_fully_funded=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Fully Funded'].mean()
proj_state=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['School State'].apply(returnself)
proj_fully_funded_state=pd.concat([proj_fully_funded,proj_state],axis=1)
fully_funded_ratio_by_state=proj_fully_funded_state.groupby(['School State'])['Fully Funded'].mean().sort_values(ascending=False)*100

df=pd.DataFrame(index=fully_funded_ratio_by_state.index)
df['Fully Funded Perc']=fully_funded_ratio_by_state
df['States']=df.index

df['text'] = df.index + '<br>' +\
    'Percentage of Full Fund: ' + df['Fully Funded Perc'].round(2).astype(str) + '%'

df['code'] = df['States'].map(state_codes)
    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['Fully Funded Perc'],
        locationmode = 'USA-states',
        text=df.text,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = 'Percentage (%)')
        ) ]

layout = dict(
        title = 'Fully Funded Project Percentage By State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict(data=data,layout=layout)
iplot(fig)

del proj_fully_funded
del proj_state
del proj_fully_funded_state

don_amount_by_type=donations.groupby(['Project Type'])['Donation Amount'].sum().sort_values(ascending=False)
don_amount_by_type.plot(kind='bar',title='Donation Amount by Project Type')
fully_funded_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Fully Funded'].mean()
res_type_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Project Resource Category'].apply(returnself)
fund_res_by_proj=pd.DataFrame(fully_funded_by_proj*100)
fund_res_by_proj['Project Resource Category']=res_type_by_proj
fund_by_res_type=fund_res_by_proj.groupby(['Project Resource Category'])['Fully Funded'].mean().sort_values(ascending=False)
bar_data=pd.DataFrame(fund_by_res_type)
bar_data.plot(kind='bar',title='Fully Funded Percentage by Project Resource Category')
del fund_res_by_proj
donation_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Donation Amount'].sum()
don_res_by_proj=pd.DataFrame(donation_by_proj)
don_res_by_proj['Project Resource Category']=res_type_by_proj
don_by_res_type=don_res_by_proj.groupby(['Project Resource Category'])['Donation Amount'].sum().sort_values(ascending=False)
bar_data=pd.DataFrame(don_by_res_type)
bar_data.plot(kind='bar',title='Donation by Project Resource Category')
del don_res_by_proj
del res_type_by_proj
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['Project Resource Category'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
grade_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Project Grade Level Category'].apply(returnself)
fund_grade_by_proj=pd.DataFrame(fully_funded_by_proj*100)
fund_grade_by_proj['Project Grade Level Category']=grade_by_proj
fund_by_grade=fund_grade_by_proj.groupby(['Project Grade Level Category'])['Fully Funded'].mean()
bar_data=pd.DataFrame(fund_by_grade)
bar_data.plot(kind='bar',title='Fully Funded Percentage by Project Grade Level')
del grade_by_proj
del fund_grade_by_proj
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    donuts = donations[donations["Donor ID"]==donor].groupby(['Project Grade Level Category'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts.head(10).plot(kind='bar',title=donor,label=None)
    plt.xticks([])
plt.tight_layout()
plt.show()
sub_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Project Subject Category Tree'].apply(returnself)
sub_by_proj_split=pd.DataFrame(sub_by_proj.str.split(', ').str.get(0))
sub_by_proj_split.columns=['First']
sub_by_proj_split['Second']=sub_by_proj.str.split(', ').str.get(1)
sub_by_proj_split['Third']=sub_by_proj.str.split(', ').str.get(2)
sub_by_proj_split['Fully Funded']=fully_funded_by_proj
cat_list=np.concatenate((sub_by_proj_split['First'].unique(),sub_by_proj_split['Second'].unique()),axis=0)
cat_list=np.concatenate((cat_list,sub_by_proj_split['Third'].unique()),axis=0)
cat_list=list(set(cat_list))
fully_fund_by_cat=pd.DataFrame(index=cat_list)
temp1=sub_by_proj_split.groupby(['First'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum1']=temp1['sum']
fully_fund_by_cat['Count1']=temp1['count']
temp2=sub_by_proj_split.groupby(['Second'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum2']=temp2['sum']
fully_fund_by_cat['Count2']=temp2['count']
temp3=sub_by_proj_split.groupby(['Third'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum3']=temp3['sum']
fully_fund_by_cat['Count3']=temp3['count']
fully_fund_by_cat=fully_fund_by_cat.fillna(0)
fully_fund_by_cat['Fully Funded Sum']=fully_fund_by_cat['Sum1']+fully_fund_by_cat['Sum2']+fully_fund_by_cat['Sum3']
fully_fund_by_cat['Fully Funded Count']=fully_fund_by_cat['Count1']+fully_fund_by_cat['Count2']+fully_fund_by_cat['Count3']
fully_fund_by_cat['Fully Funded']=fully_fund_by_cat['Fully Funded Sum']/fully_fund_by_cat['Fully Funded Count']*100
fully_fund_by_cat=fully_fund_by_cat.drop(['Warmth'])
fully_fund_by_cat=fully_fund_by_cat.drop([np.nan])
fully_fund_by_cat['Fully Funded'].sort_values(ascending=False).plot(kind='bar',title='Fully Funded Percentage by Project Subject Category Tree')
del sub_by_proj
sub_by_proj_split['Donation Amount']=donation_by_proj
donation_by_cat=pd.DataFrame(index=cat_list)
temp1=sub_by_proj_split.groupby(['First'])['Donation Amount'].sum()
donation_by_cat['Sum1']=temp1
temp2=sub_by_proj_split.groupby(['Second'])['Donation Amount'].sum()
donation_by_cat['Sum2']=temp2
temp3=sub_by_proj_split.groupby(['Third'])['Donation Amount'].sum()
donation_by_cat['Sum3']=temp3
donation_by_cat=donation_by_cat.fillna(0)
donation_by_cat['Donation']=donation_by_cat['Sum1']+donation_by_cat['Sum2']+donation_by_cat['Sum3']
donation_by_cat=donation_by_cat.drop(['Warmth'])
donation_by_cat=donation_by_cat.drop([np.nan])
donation_by_cat['Donation per Project']=np.divide(donation_by_cat['Donation'],fully_fund_by_cat['Fully Funded Count'])
donation_by_cat['Donation per Project'].sort_values(ascending=False).plot(kind='bar',title='Donation Amount per Project by Project Subject Category Tree')
del sub_by_proj_split
sub_by_proj=donations[donations['Fully Funded']!=-1].groupby(['Project ID'])['Project Subject Subcategory Tree'].apply(returnself)
sub_by_proj_split=pd.DataFrame(sub_by_proj.str.split(', ').str.get(0))
sub_by_proj_split.columns=['First']
sub_by_proj_split['Second']=sub_by_proj.str.split(', ').str.get(1)
sub_by_proj_split['Third']=sub_by_proj.str.split(', ').str.get(2)
sub_by_proj_split['Fully Funded']=fully_funded_by_proj
cat_list=np.concatenate((sub_by_proj_split['First'].unique(),sub_by_proj_split['Second'].unique()),axis=0)
cat_list=np.concatenate((cat_list,sub_by_proj_split['Third'].unique()),axis=0)
cat_list=list(set(cat_list))
fully_fund_by_cat=pd.DataFrame(index=cat_list)
temp1=sub_by_proj_split.groupby(['First'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum1']=temp1['sum']
fully_fund_by_cat['Count1']=temp1['count']
temp2=sub_by_proj_split.groupby(['Second'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum2']=temp2['sum']
fully_fund_by_cat['Count2']=temp2['count']
temp3=sub_by_proj_split.groupby(['Third'])['Fully Funded'].agg(['sum','count'])
fully_fund_by_cat['Sum3']=temp3['sum']
fully_fund_by_cat['Count3']=temp3['count']
fully_fund_by_cat=fully_fund_by_cat.fillna(0)
fully_fund_by_cat['Fully Funded Sum']=fully_fund_by_cat['Sum1']+fully_fund_by_cat['Sum2']+fully_fund_by_cat['Sum3']
fully_fund_by_cat['Fully Funded Count']=fully_fund_by_cat['Count1']+fully_fund_by_cat['Count2']+fully_fund_by_cat['Count3']
fully_fund_by_cat['Fully Funded']=fully_fund_by_cat['Fully Funded Sum']/fully_fund_by_cat['Fully Funded Count']*100
fully_fund_by_cat=fully_fund_by_cat.drop(['Warmth'])
fully_fund_by_cat=fully_fund_by_cat.drop([np.nan])
fully_fund_by_cat['Fully Funded'].sort_values(ascending=False).plot(kind='bar',title='Fully Funded Percentage by Project Subject Category Tree')
sub_by_proj_split['Donation Amount']=donation_by_proj
donation_by_cat=pd.DataFrame(index=cat_list)
temp1=sub_by_proj_split.groupby(['First'])['Donation Amount'].sum()
donation_by_cat['Sum1']=temp1
temp2=sub_by_proj_split.groupby(['Second'])['Donation Amount'].sum()
donation_by_cat['Sum2']=temp2
temp3=sub_by_proj_split.groupby(['Third'])['Donation Amount'].sum()
donation_by_cat['Sum3']=temp3
donation_by_cat=donation_by_cat.fillna(0)
donation_by_cat['Donation']=donation_by_cat['Sum1']+donation_by_cat['Sum2']+donation_by_cat['Sum3']
donation_by_cat=donation_by_cat.drop(['Warmth'])
donation_by_cat=donation_by_cat.drop([np.nan])
donation_by_cat['Donation per Project']=np.divide(donation_by_cat['Donation'],fully_fund_by_cat['Fully Funded Count'])
donation_by_cat['Donation per Project'].sort_values(ascending=False).plot(kind='bar',title='Donation Amount per Project by Project Subject Subcategory Tree')
del sub_by_proj
del sub_by_proj_split
from wordcloud import WordCloud, STOPWORDS
#title_by_proj=donations.groupby('Project ID')['Project Title'].apply(returnself)
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='white',
                     ).generate(donations['Project Title'].str.cat(sep=','))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
import re
import string
from itertools import chain
def preprocess(data):
    reviews_tokens = []
    for review in data:
        review=review.lower() #Convert to lower-case words
        review=re.sub('['+string.punctuation+'“”‘’'+']', '', review) #remove punctuation
        word_tokens = [w for w in review.split(' ') if not w in STOPWORDS] # do not add stop words
        reviews_tokens.append(word_tokens)
    return reviews_tokens #return all tokens

def construct_bag_of_words(data):
    corpus = preprocess(data)
    bag_of_words = {}
    word_count = 0
    for sentence in corpus:
        for word in sentence:
            if word not in bag_of_words: # do not allow repetitions
                bag_of_words[word] = word_count #set indexes
                word_count+=1
    return bag_of_words #index of letters

def get_words_list(data):
    corpus=preprocess(data)
    list_of_all_words=list(chain.from_iterable(corpus))
    return list(set(list_of_all_words))

def featurize(sentence_tokens,bag_of_words):
    sentence_features = [0 for x in range(len(bag_of_words))] 
    for word in sentence_tokens:
        index = bag_of_words[word]
        sentence_features[index] +=1
    return sentence_features

def get_batch_features(data,bag_of_words):
    batch_features = []
    reviews_text_tokens = preprocess(data)
    for review_text in reviews_text_tokens:
        feature_review_text = featurize(review_text,bag_of_words)
        batch_features.append(feature_review_text)
    return batch_features

title_by_proj=donations.groupby('Project ID')['Project Title'].apply(returnself).fillna('')
#bag_of_words = construct_bag_of_words(title_by_proj)
#invert_bag_of_words = {v: k for k, v in bag_of_words.items()}
def get_money_on_words(data,money,words_list):
    money_on_words=pd.DataFrame(index=words_list)
    money_on_words['Donation Amount']=0
    reviews_text_tokens = preprocess(data)
    for i in np.arange(len(data)):        
        for word in reviews_text_tokens[i]:
            money_on_words['Donation Amount'][word]+=money[i]
    return money_on_words

title_don_by_proj=pd.DataFrame(title_by_proj)
del title_by_proj
title_don_by_proj['Project Title']=title_don_by_proj['Project Title'].fillna('')
title_don_by_proj['Donation Amount']=donation_by_proj
title_don_by_proj['Donation Amount']=title_don_by_proj['Donation Amount'].fillna(0)
words_list=get_words_list(title_don_by_proj['Project Title'])
money_on_words=get_money_on_words(title_don_by_proj['Project Title'],title_don_by_proj['Donation Amount'],words_list)
money_on_words['Donation Amount'].sort_values(ascending=False).drop('').head(10).plot(kind='bar',title='Donation Amount on Project Title Words')
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    title_don_by_proj=pd.DataFrame(donations['Project Title'][donations['Donor ID']==donor])
    title_don_by_proj['Project Title']=title_don_by_proj['Project Title'].fillna('')
    title_don_by_proj['Donation Amount']=donations['Donation Amount'][donations['Donor ID']==donor]
    title_don_by_proj['Donation Amount']=title_don_by_proj['Donation Amount'].fillna(0)
    title_don_by_proj.index=np.arange(len(title_don_by_proj))
    donuts_on_words=get_money_on_words(title_don_by_proj['Project Title'],title_don_by_proj['Donation Amount'],words_list)
    #donuts = donations[donations["Donor ID"]==donor].groupby(['Project Grade Level Category'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts_on_words['Donation Amount'].sort_values(ascending=False).drop('').head(10).plot(kind='bar',title=donor,label=None)
    #plt.xticks([])
plt.tight_layout()
plt.show()
essay_by_proj=donations.groupby('Project ID')['Project Short Description'].apply(returnself).fillna('')
essay_don_by_proj=pd.DataFrame(essay_by_proj)
essay_don_by_proj['Project Short Description']=essay_don_by_proj['Project Short Description'].fillna('')
words_list=get_words_list(essay_don_by_proj['Project Short Description'])
essay_don_by_proj['Donation Amount']=donation_by_proj
essay_don_by_proj['Donation Amount']=essay_don_by_proj['Donation Amount'].fillna(0)
money_on_words=get_money_on_words(essay_don_by_proj['Project Short Description'],essay_don_by_proj['Donation Amount'],words_list)
money_on_words['Donation Amount'].sort_values(ascending=False).drop(['','students','school']).head(10).plot(kind='bar',title='Donation Amount on Project Short Description Words')
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    essay_don_by_proj=pd.DataFrame(donations['Project Short Description'][donations['Donor ID']==donor])
    essay_don_by_proj['Project Short Description']=essay_don_by_proj['Project Short Description'].fillna('')
    essay_don_by_proj['Donation Amount']=donations['Donation Amount'][donations['Donor ID']==donor]
    essay_don_by_proj['Donation Amount']=essay_don_by_proj['Donation Amount'].fillna(0)
    essay_don_by_proj.index=np.arange(len(essay_don_by_proj))
    donuts_on_words=get_money_on_words(essay_don_by_proj['Project Short Description'],essay_don_by_proj['Donation Amount'],words_list)
    #donuts = donations[donations["Donor ID"]==donor].groupby(['Project Grade Level Category'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts_on_words['Donation Amount'].sort_values(ascending=False).drop(['','students','school']).head(10).plot(kind='bar',title=donor,label=None)
    #plt.xticks([])
plt.tight_layout()
plt.show()
need_by_proj=donations.groupby('Project ID')['Project Need Statement'].apply(returnself).fillna('')
need_don_by_proj=pd.DataFrame(need_by_proj)
need_don_by_proj['Project Need Statement']=need_don_by_proj['Project Need Statement'].fillna('')
words_list=get_words_list(need_don_by_proj['Project Need Statement'])
need_don_by_proj['Donation Amount']=donation_by_proj
need_don_by_proj['Donation Amount']=need_don_by_proj['Donation Amount'].fillna(0)
money_on_words=get_money_on_words(need_don_by_proj['Project Need Statement'],need_don_by_proj['Donation Amount'],words_list)
money_on_words['Donation Amount'].sort_values(ascending=False).drop(['','need','students']).head(10).plot(kind='bar',title='Donation Amount on Project Need Statement Words')
plt.figure(6, figsize=(14,10))
for i in range(len(top_donors)):
    donor = top_donors.index[i]
    need_don_by_proj=pd.DataFrame(donations['Project Need Statement'][donations['Donor ID']==donor])
    need_don_by_proj['Project Need Statement']=need_don_by_proj['Project Need Statement'].fillna('')
    need_don_by_proj['Donation Amount']=donations['Donation Amount'][donations['Donor ID']==donor]
    need_don_by_proj['Donation Amount']=need_don_by_proj['Donation Amount'].fillna(0)
    need_don_by_proj.index=np.arange(len(need_don_by_proj))
    donuts_on_words=get_money_on_words(need_don_by_proj['Project Need Statement'],need_don_by_proj['Donation Amount'],words_list)
    #donuts = donations[donations["Donor ID"]==donor].groupby(['Project Grade Level Category'])['Donation Amount'].sum()#.sort_values(ascending=False)
    plt.subplot(5,4,i+1)
    donuts_on_words['Donation Amount'].sort_values(ascending=False).drop(['','need','students']).head(10).plot(kind='bar',title=donor,label=None)
    #plt.xticks([])
plt.tight_layout()
plt.show()