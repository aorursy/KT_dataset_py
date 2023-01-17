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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot

donationsfile = '../input/Donations.csv'
projectsfile = '../input/Projects.csv'
donorsfile = '../input/Donors.csv'
resourcesfile = '../input/Resources.csv'

donations = pd.read_csv(donationsfile)
projects = pd.read_csv(projectsfile)
donors = pd.read_csv( donorsfile )
resources = pd.read_csv( resourcesfile )

#Find unique single donation givers

#donors who have donated
t = donations['Donor ID'].value_counts() 
singledonors = pd.DataFrame(columns=['Donor ID'])
singledonors['Donor ID'] = t[t==1].index 

donorprojects = donations[['Donor ID', 'Project ID', 'Donation Amount']]
singledonations = pd.merge(singledonors, donorprojects, on='Donor ID')

#project cost, status and amount 
p = projects[['Project ID', 'Project Cost', 'Project Current Status']] 

# donors who donated to a single project, amount they donated, total project cost
singledonorprojects = pd.merge(singledonations, p, on='Project ID')

print(singledonorprojects.describe())
bins = [0, 5, 10, 50, 100, 500, 1000, 5000, 10000, 35000]
singledonorprojects['bins'] = pd.cut(singledonorprojects['Donation Amount'], bins=bins)

bins_count = pd.DataFrame( singledonorprojects['bins'].value_counts().sort_index())
bins_count.columns=['Bin Count']
donations_per_bin = pd.pivot_table(singledonorprojects,index=['bins'],values='Donation Amount',aggfunc=np.sum)

explode = (0,0,0.1,0,0,0,0,0,0)
fig = plot.figure()
fig.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set1(np.linspace(0, 1, 10))
patches, texts, autotexts = ax.pie(bins_count['Bin Count'], labels=bins_count['Bin Count'], 
                                   autopct='%1.1f%%', explode=explode, labeldistance=1.05,
                                   colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('First time donors and their contribution bracket', fontsize=20)
ax.legend(bins_count.index, loc = "lower center", ncol=3, fontsize='medium')
plot.show()
fig2 = plot.figure()
fig2.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set2(np.linspace(0, 1, 10))
patches, texts, autotexts = plot.pie(donations_per_bin['Donation Amount'], 
                                     labels = round(donations_per_bin['Donation Amount']),
                                     autopct='%1.1f%%', explode=explode, labeldistance=1.05,
                                     colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('Total contribution by first time donors in their contribution bracket', fontsize=20)
ax.legend(bins_count.index, loc = "lower center", ncol=3, fontsize='medium')
plot.show()
#let us work on the major donor bins
a = singledonorprojects.copy()
under50donors = a[(a['Donation Amount']>10.0) & (a['Donation Amount']<=50.0)]
under100donors = a[(a['Donation Amount']>50.0) & (a['Donation Amount']<=100.0)]
under500donors = a[(a['Donation Amount']>100.0) & (a['Donation Amount']<=500.0)]
  
#under $50 donor interests
x = pd.merge(under50donors, projects, on='Project ID') 
under50projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                        'Project Resource Category']]

temp = under50projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data50 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.PiYG(np.linspace(0, 1, 20))
ax = graph_data50.plot(kind='barh', stacked=True, color=colors, 
                  title='Resource Categories for Projects supported by First time Donors (\$10-\$50]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')
#under $100 donor interests
x = pd.merge(under100donors, projects, on='Project ID') 
under100projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                         'Project Resource Category']]

temp = under100projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data100 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.BrBG(np.linspace(0, 1, 20))
ax = graph_data100.plot(kind='barh', stacked=True, color=colors, 
                   title='Resource Categories for Projects supported by First time Donors (\$50-\$100]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')
#under $500 donor interests
x = pd.merge(under500donors, projects, on='Project ID') 
under500projectdata = x[['Project ID', 'Project Grade Level Category', 'Project Subject Category Tree',
                         'Project Resource Category']]

temp = under500projectdata.copy()
p = temp.groupby(['Project Grade Level Category', 'Project Resource Category']).size().reset_index()
p.columns = ['Grade Category','Resource Category','Count']
labels = pd.DataFrame(temp['Project Grade Level Category'].value_counts()).reset_index()
labels.columns = ['Grade Category','Count']
graph_data500 = pd.pivot_table(p,index=['Grade Category'],
                            columns=['Resource Category'],
                            values='Count',aggfunc=np.sum).fillna(0)

colors = plot.cm.bwr(np.linspace(0, 1, 20))
ax = graph_data500.plot(kind='barh', stacked=True, color=colors, 
                   title='Resource Categories for Projects supported by First time Donors (\$100-\$500]',
                  figsize=(17,10))

patches, labels = ax.get_legend_handles_labels()
ax.legend(patches, labels, loc='right')
#let us now work on donors who have donated to more than 1 project

newdonors = pd.DataFrame(donations['Donor ID'].value_counts()).reset_index()
newdonors.columns=['Donor ID', 'Count']
newdonors = newdonors[newdonors['Count'] >= 2 ]

#donors who have donated **only** to 2 projects
twicedonors = newdonors[newdonors['Count'] == 2 ]

#Get the donation information for these donors
tDonors = donations[donations['Donor ID'].isin(twicedonors['Donor ID'])]
temp_list = [1,2] * int(len(tDonors)/2)

tDonors = tDonors.sort_values(by=['Donor ID','Donation Received Date'])
tDonors['temp_label'] = temp_list

t1= pd.pivot_table(tDonors,index=['Donor ID'],
                            columns=['temp_label'],
                            values=['Project ID', 'Donation ID', 'Donation Received Date'],
                            aggfunc=lambda x: ' '.join(x) )
t1 = t1.reset_index()
t2 = pd.pivot_table(tDonors,index=['Donor ID'],
                            columns=['temp_label'],
                            values=['Donation Amount'])
t2 = t2.reset_index() 
df = pd.merge(t1, t2, on='Donor ID')

#Find the pattern in their donations
df['Donation Diff'] = df[('Donation Amount', 1)] - df[('Donation Amount', 2 )]
df['Donation Diff'] = df['Donation Diff'].abs()
print( df['Donation Diff'].describe() )
p = df['Donation Diff'].value_counts().reset_index()
top10count = p[0:10]
top10count.columns = ['Donation Difference', 'Count']
donationdiff = top10count.sort_values(by='Donation Difference')

fig3 = plot.figure()
fig3.set_size_inches(9,9)
ax = plot.subplot(111)
colors = plot.cm.Set2(np.linspace(0, 1, 10))
patches, texts, autotexts = ax.pie(donationdiff['Count'], 
                                   labels=donationdiff['Donation Difference'], 
                                   autopct='%1.1f%%', labeldistance=1.05,
                                   colors=colors)
for t in texts:
    t.set_size('medium')
for t in autotexts:
    t.set_size('small')
plot.title('Distribution of top 10 donation amount differences \n comparing the 2 donations made by second time donors.')
df['t1'] = pd.to_datetime(df[('Donation Received Date', 1)])
df['t2'] = pd.to_datetime(df[('Donation Received Date', 2)])

df['Date Diff'] = df['t2'] - df['t1']
print( df[('Date Diff')].describe() )
dp1 = pd.DataFrame(columns=['Donor ID', 'Project ID'])
dp1['Project ID'] = df[('Project ID', 1)]
dp1['Donor ID'] = df['Donor ID']
temp = pd.merge(dp1, projects, on='Project ID')
dp1_final = temp[['Donor ID', 'Project ID', 'Project Grade Level Category', 'Project Resource Category']]
dp1_final.columns=['Donor ID', 'Project ID1', 'Grade1', 'Resource1']
dp2 = pd.DataFrame(columns=['Donor ID', 'Project ID'])
dp2['Project ID'] = df[('Project ID', 2)]
dp2['Donor ID'] = df['Donor ID']
temp = pd.merge(dp2, projects, on='Project ID')
dp2_final = temp[['Donor ID', 'Project ID', 'Project Grade Level Category', 'Project Resource Category']]
dp2_final.columns=['Donor ID', 'Project ID2', 'Grade2', 'Resource2']

merged_data = pd.merge(dp1_final, dp2_final, on='Donor ID')
merged_data['Same Grade?'] = (merged_data['Grade1'] == merged_data['Grade2'])
merged_data['Same Resource?'] = (merged_data['Resource1'] == merged_data['Resource2'])
fig4 = plot.figure()
fig4.set_size_inches(9,9)
graph1 = merged_data['Same Grade?'].value_counts().reset_index()
graph1.columns = ['Same Grade?','Count']
plot.title('Did a second time donor contribute to \n a project in the same grade as the first time?')
plot.pie(graph1['Count'], labels=graph1['Same Grade?'], colors=['y','g'], autopct='%1.1f%%')
plot.show()
fig5 = plot.figure()
fig5.set_size_inches(9,9)
graph2 = merged_data['Same Resource?'].value_counts().reset_index()
graph2.columns = ['Same Resource?','Count']
plot.title('Did a second time donor contribute to \n a project with the same resource category as the first time?')
plot.pie(graph2['Count'], labels=graph2['Same Resource?'], autopct='%1.1f%%')
plot.show()
#Took data between 2013-2017
ts = donations.loc[:,['Donation Received Date', 'Donation Amount']]
ts.set_index('Donation Received Date', inplace=True)
ts = ts[(ts.index>='2013-01-01') & (ts.index<'2018-01-01') ]
ts['month'] = pd.to_datetime(ts.index, format='%Y-%m-%d %H:%M:%S.%f').month
ts.groupby('month').sum()['Donation Amount'].plot(kind='bar', color='darkorange', figsize=(10,5))
plot.xlabel('Donation Amount')
plot.show()
# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
#Number of donations per project
custom_bucket = [0, 1, 5, 10, 20, 1000000]
custom_bucket_label = ['Single Donor', '1-5 Donors', '6-10 Donors', '11-20 Donors', 'More than 20 Donors']
num_of_don = donations['Project ID'].value_counts().to_frame(name='Donation Count').reset_index()
num_of_don['Donation Cnt'] = pd.cut(num_of_don['Donation Count'], custom_bucket, labels=custom_bucket_label)
num_of_don = num_of_don['Donation Cnt'].value_counts().sort_index()

num_of_don.iplot(kind='bar', xTitle = 'Number of Donors', yTitle = 'Number of Projects', 
                title = 'Distribution on Number of Donors and Project Count')
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot

donations['Donation Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Year'] = donations['Donation Date'].dt.year
donations['Donation Month'] = donations['Donation Date'].dt.month

donations_g = donations.groupby(['Donation Month']).agg({'Donation Year' : 'count', 'Donation Amount' : 'mean'}).reset_index().rename(columns={'Donation Year' : 'Total Donations', 'Donation Amount' : 'Average Amount'})
x = donations_g['Donation Month']
y1 = donations_g['Total Donations']
y2 = donations_g['Average Amount']
trace1 = go.Scatter(x=x, y=y1, fill='tozeroy', fillcolor = '#kcc49f', mode= 'none')
trace2 = go.Scatter(x=x, y=y2, fill='tozeroy', fillcolor = "#a993f9", mode= 'none')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles = ["<b>Donations per Month</b>", "<b>Average Donation Amount per Month</b>"])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);

fig['layout'].update(height=300, showlegend=False, yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ), yaxis2=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ));


iplot(fig);

