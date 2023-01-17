
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import squarify

from scipy.stats import kurtosis
from scipy.stats import skew

from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
df_donation = pd.read_csv('../input/Donations.csv', low_memory=False)
df_donor = pd.read_csv('../input/Donors.csv', error_bad_lines=False, warn_bad_lines=False)

df_project = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False)
df_resource = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)

df_school = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
df_teacher = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
#Doing the merge of Donation 
df_merge_don = df_donation.merge(df_donor, on='Donor ID', how='inner')

#Merging all data
df_merge_proj = df_resource.merge(df_project, on='Project ID', how='inner')
df_merge_proj = df_merge_proj.merge(df_school, on='School ID', how='inner')
df_merge_proj = df_merge_proj.merge(df_teacher, on='Teacher ID', how='inner')

df_merge_proj = df_merge_proj.sample(10000, replace=True)
df_merge_don = df_merge_don.sample(10000, replace=True)
df_merge_proj.nunique()
df_merge_don.nunique()
print(df_merge_proj.shape)
print(df_merge_proj.isnull().sum())
df_merge_don.rename(columns={('Donation Included Optional Donation'): ('Optional'),
                             ('Donor Cart Sequence'):("Cart Sequence Number (Per Donor)")}, inplace=True)
print(df_merge_don.shape)
print(df_merge_don.isnull().sum())
df_merge_don['Donation_log'] = np.log(df_merge_don['Donation Amount'] + 1)
plt.figure(figsize = (16,6))

plt.subplot(1,2,1)
sns.distplot(df_merge_don['Donation_log'].dropna())
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Histogram of Donation Amount")

plt.subplot(1,2,2)
plt.scatter(range(df_merge_don.shape[0]), np.sort(df_merge_don['Donation Amount'].values))
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Distribution of Donation Amount")

plt.show()
print('Excess kurtosis of normal distribution (should be 0): {}'.format(
    kurtosis(df_merge_don[df_merge_don['Donation_log'] >0].Donation_log)))
print( 'Skewness of normal distribution (should be 0): {}'.format(
    skew((df_merge_don[df_merge_don['Donation_log'] >0].Donation_log))))
# calculating summary statistics
data_mean, data_std = np.mean(df_merge_don['Donation Amount']), np.std(df_merge_don['Donation Amount'])

# identify outliers
cut = data_std * 3
lower, upper = data_mean - cut, data_mean + cut

# identify outliers
outliers = [x for x in df_merge_don['Donation Amount'] if x < lower or x > upper]

# remove outliers
outliers_removed = [x for x in df_merge_don['Donation Amount'] if x > lower and x < upper]


print('Identified outliers: %d' % len(outliers))
print('Non-outlier observations: %d' % len(outliers_removed))
print("Percentiles of Amount: ")
print(df_merge_don['Donation Amount'].quantile([.05,.25,.5,.75,.95]))
optional = df_merge_don['Optional'].value_counts() / len(df_merge_don['Optional']) * 100
is_teacher = df_merge_don['Donor Is Teacher'].value_counts() / len(df_merge_don['Donor Is Teacher']) * 100
state_donor = df_merge_don['Donor State'].value_counts() / len(df_merge_don['Donor State']) * 100
city_donor = df_merge_don['Donor City'].value_counts() / len(df_merge_don['Donor City']) * 100
cart_seq = df_merge_don['Cart Sequence Number (Per Donor)'].value_counts() / len(df_merge_don['Cart Sequence Number (Per Donor)']) * 100
trace1 = go.Bar(x=optional.index[::-1],
                y=optional.values[::-1],
                      name='Optional Count')
                        

trace2 = go.Bar(y=is_teacher[:20].values,
                x=is_teacher[:20].index,
                name='Teacher Donors Distribuition')
                

trace3 = go.Bar(y=state_donor[:15].values,
                x=state_donor[:15].index, 
                name='State Donors Distribuition')


trace4 = go.Bar(y=city_donor[:15].values,
                x=city_donor[:15].index,
                name='Top 15 City')

trace5 = go.Bar(y=cart_seq[:15].values,
                x=cart_seq[:15].index,
                name='Top 15 Cart Sequence')

data = [trace1, trace2, trace3, trace4, trace5]


updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=list([  
             
            dict(
                label = 'Optional',
                 method = 'update',
                 args = [{'visible': [True, False, False, False,False]}, 
                     {'title': 'Optional Donors in % of Total'}]),
             
             dict(
                  label = 'Is Teacher',
                 method = 'update',
                 args = [{'visible': [False, True, False, False,False]},
                     {'title': 'Total of Donors Teachers in %'}]),

            dict(
                 label = 'States',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]},
                     {'title': 'TOP 15 State in % of Total'}]),

            dict(
                 label = 'City ',
                 method = 'update',
                 args = [{'visible': [False, False, False, True,False]},
                     {'title': 'Top 15 city in % of Total'}]),

            dict(
                 label = 'Cart Sequence',
                 method = 'update',
                 args = [{'visible': [False, False, False, False,True]},
                     {'title': ' Top 15 Cart Sequence in % of Total'}])
        ]),
    )
])


layout = dict(title='The percentual Distribuitions of Categorical Features (Select from Dropdown)', 
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)

iplot(fig)
#Creating the text to show in boxplots
hover_text = []

for index, row in df_merge_don.iterrows():
    hover_text.append(('Real Amount: {value}<br>' +
                       'Optional donation: {optional}<br>'+
                       'Is Teacher?: {teacher}<br>'+
                       'Cart Seq Num: {Cart_seq}<br>').format(value=row['Donation Amount'],
                                                          optional=row['Optional'],
                                                          teacher=row['Donor Is Teacher'],
                                                          Cart_seq=row['Cart Sequence Number (Per Donor)']))
df_merge_don['text'] = hover_text

trace1  = go.Box(
    x=df_merge_don['Optional'],
    y=df_merge_don['Donation_log'], 
    showlegend=False, text=df_merge_don['text'])
                        

trace2  = go.Box(
    x=df_merge_don['Donor Is Teacher'],
    y=df_merge_don['Donation_log'], 
    showlegend=False, text=df_merge_don['text']
)
                

trace3 = go.Box(
    x=df_merge_don[df_merge_don['Donor State'].isin(state_donor[:15].index.values)]['Donor State'],
    y=df_merge_don[df_merge_don['Donor State'].isin(state_donor[:15].index.values)]['Donation_log'],
    showlegend=False, text=df_merge_don[df_merge_don['Donor State'].isin(state_donor[:15].index.values)]['text']
)

trace4 = go.Box(
    x=df_merge_don[df_merge_don['Donor City'].isin(city_donor[:15].index.values)]['Donor City'],
    y=df_merge_don[df_merge_don['Donor City'].isin(city_donor[:15].index.values)]['Donation_log'],
    showlegend=False, text=df_merge_don[df_merge_don['Donor City'].isin(city_donor[:15].index.values)]['text']
)

trace5 = go.Box(
    x=df_merge_don[df_merge_don['Cart Sequence Number (Per Donor)'].isin(cart_seq[:15].index.values)]['Cart Sequence Number (Per Donor)'],
    y=df_merge_don[df_merge_don['Cart Sequence Number (Per Donor)'].isin(cart_seq[:15].index.values)]['Donation_log'],
    showlegend=False, text=df_merge_don[df_merge_don['Cart Sequence Number (Per Donor)'].isin(cart_seq[:15].index.values)]['text']
)


data = [trace1, trace2, trace3, trace4, trace5]


updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=list([  
             
            dict(
                label = 'Optional',
                 method = 'update',
                 args = [{'visible': [True, False, False, False, False]}, 
                     {'title': 'Donation Distribuition by Optional Donations'}]),
             
             dict(
                  label = 'Is Teacher?',
                 method = 'update',
                 args = [{'visible': [False, True, False, False, False]},
                     {'title': 'Donation Distribuition by Teachers or not'}]),

            dict(
                 label = 'States',
                 method = 'update',
                 args = [{'visible': [False, False, True, False, False]},
                     {'title': 'TOP 15 States - Donation Distribuition'}]),

            dict(
                 label =  'Citys',
                 method = 'update',
                 args = [{'visible': [False, False, False, True, False]},
                     {'title': 'TOP 15 Citys - Donation Distribuition'}]),

            dict(
                 label =  'Cart Sequence',
                 method = 'update',
                 args = [{'visible': [False, False, False, False, True]},
                     {'title': 'TOP 15 Cart Sequences - Donation Distribuition'}])
        ]),
    )
])

layout = dict(title='Donation Log Distribuitions Boxplots (Select from Dropdown)', 
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)

iplot(fig, filename='dropdown-donation')
plt.figure(figsize=(14,6))

plt.subplot(121)
g = sns.countplot(x='Optional', palette='hls',
                  data=df_merge_don)
g.set_title("Donor's Optional",fontsize=20)
g.set_xlabel("Optional",fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(122)
g1 = sns.boxplot(x='Optional', y='Donation Amount',
                 palette='hls', 
                 data=df_merge_don)
g1.set_title("Optional Dist",fontsize=20)
g1.set_xlabel("Optional",fontsize=15)
g1.set_ylabel("Donation Amount",fontsize=15)
g1.legend(loc=1)

plt.show()


plt.figure(figsize=(14,6))

plt.subplot(121)
g = sns.countplot(x='Donor Is Teacher', palette='hls', hue='Optional',
                  data=df_merge_don)
g.set_title("Donor's Teacher Count by Optional",fontsize=20)
g.set_xlabel("Teacher or Not",fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(122)
g1 = sns.boxplot(x='Donor Is Teacher', y='Donation Amount', hue='Optional',
                 palette='hls', 
                 data=df_merge_don)
g1.set_title("Teachers and Not Donations Dist",fontsize=20)
g1.set_xlabel("Teacher or Not",fontsize=15)
g1.set_ylabel("Donation Amount",fontsize=15)
g1.legend(loc=1)

plt.show()
plt.figure(figsize=(10,6))
g = sns.distplot(df_merge_don[df_merge_don['Optional'] == "Yes"]['Donation_log'].dropna(), 
                 label="Optional Donation")
g = sns.distplot(df_merge_don[df_merge_don['Optional'] == "No"]['Donation_log'].dropna(), 
                 label="Not Optional Donation")
g.set_title("Teachers and Not Donations Dist",fontsize=20)
g.set_xlabel("",fontsize=15)
g.set_ylabel("Donation Amount",fontsize=15)
g.legend()

plt.show()
state = df_merge_don['Donor State'].value_counts()

plt.figure(figsize=(13,9))

plt.subplot(2,1,1)
g = sns.countplot(x='Donor State', 
                  hue='Optional', 
                  palette='hls', 
                  data=df_merge_don[df_merge_don['Donor State'].isin(state[:15].index.values)])
g.set_title("TOP 15 State Donor's Count by Optional", fontsize=20)
g.set_xlabel("Donor States", fontsize=15)
g.set_ylabel("Count",fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(2,1,2)
g1 = sns.boxplot(x='Donor State', y='Donation_log', 
                hue='Optional',
                palette='hls',
                data=df_merge_don[df_merge_don['Donor State'].isin(state[:15].index.values)])
g1.set_title("Top 15 States Donation Log Distribuition", fontsize=20)
g1.set_xlabel("Donor States", fontsize=15)
g.set_ylabel("Donation(Log)",fontsize=15)
g1.set_xticklabels(g.get_xticklabels(),rotation=45)
g1.legend(loc=1)

plt.subplots_adjust(hspace = 0.7, top = 0.8)

plt.show()
plt.figure(figsize=(16,5))

g = sns.boxplot(x='Donor State', y='Donation_log', 
                hue='Optional', 
                palette='hls',
              data=df_merge_don[df_merge_don['Donor State'].isin(state[:15].index.values)])
g.set_title("Top 15 States Donation Log Distribuition", fontsize=20)
g.set_xlabel("Donor States", fontsize=15)
g.set_ylabel("Donation(Log)",fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)

plt.show()
city = df_merge_don['Donor City'].value_counts()

plt.figure(figsize=(14,10))

plt.subplot(2,1,1)
g = sns.countplot(x='Donor City', hue='Optional', palette='hls',
              data=df_merge_don[df_merge_don['Donor City'].isin(city[:15].index.values)])
g.set_title("TOP 15 City's'", fontsize=20)
g.set_xlabel("Donor City's", fontsize=15)
g.set_ylabel("Count",fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(2,1,2)
g1 = sns.lvplot(x='Donor City', y='Donation_log', hue='Optional', palette='hls',
              data=df_merge_don[df_merge_don['Donor City'].isin(city[:15].index.values)])
g1.set_title("TOP 15 City's Donation Log", fontsize=20)
g1.set_xlabel("Donor City", fontsize=15)
g1.set_ylabel("Donation(Log)",fontsize=15)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.legend(loc=1)

plt.subplots_adjust(hspace = 0.7, top = 0.8)

plt.show()
cart_seq = df_merge_don['Cart Sequence Number (Per Donor)'].value_counts()

plt.figure(figsize=(14,10))

plt.subplot(2,1,1)
g = sns.countplot(x='Cart Sequence Number (Per Donor)', 
                  hue='Donor Is Teacher', 
                  palette='hls',
              data=df_merge_don[df_merge_don['Cart Sequence Number (Per Donor)'].isin(cart_seq[:10].index.values)])
g.set_title("TOP 10 Cart Sequences Count by Is or Not Teacher", fontsize=20)
g.set_xlabel("Cart Sequences", fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(2,1,2)
g1 = sns.lvplot(x='Cart Sequence Number (Per Donor)', y='Donation_log', 
                hue='Donor Is Teacher', 
                palette='hls',
              data=df_merge_don[df_merge_don['Cart Sequence Number (Per Donor)'].isin(cart_seq[:10].index.values)])
g1.set_title("TOP 10 Cart Sequences Count by Is or Not Teacher", fontsize=20)
g1.set_xlabel("Cart Sequences", fontsize=15)
g1.set_ylabel("Donation(Log)",fontsize=15)
g1.legend(loc=1)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
df_merge_don['Donation Received Date'] = pd.to_datetime(df_merge_don['Donation Received Date'])
df_merge_don['Donation_year'] = df_merge_don['Donation Received Date'].dt.year
df_merge_don['Donation_month'] = df_merge_don['Donation Received Date'].dt.month
df_merge_don['Donation_day'] = df_merge_don['Donation Received Date'].dt.day 
df_merge_don['Donation_hour'] = df_merge_don['Donation Received Date'].dt.hour 
df_merge_don['Donation_year'].value_counts()
plt.figure(figsize=(16,20))

plt.subplot(3,1,1)
g = sns.countplot(x='Donation_year',
                  hue='Optional', 
                  palette='hls',
                  data=df_merge_don[df_merge_don['Donation_year'] > 2012])
g.set_title("Donation Count by Year", fontsize=20)
g.set_xlabel("Donation Months", fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(3,1,2)
g1 = sns.lvplot(x='Donation_year', y='Donation_log', 
                hue='Optional', 
                palette='hls',
              data=df_merge_don[df_merge_don['Donation_year'] > 2012])
g1.set_title("Donation Count - Optional or not by Year", fontsize=20)
g1.set_xlabel("Donation Years", fontsize=15)
g1.set_ylabel("Donation(Log)",fontsize=15)
g1.legend(loc=1)

plt.subplot(3,1,3)
g1 = sns.pointplot(x='Donation_year', y='Donation_log', 
                hue='Donor Is Teacher', 
                palette='hls',
              data=df_merge_don[df_merge_don['Donation_year'] > 2012])
g1.set_title("Looking the Donation Amount by Years", fontsize=20)
g1.set_xlabel("Donation Years", fontsize=15)
g1.set_ylabel("Donation(Log)",fontsize=15)
g1.legend(loc=2)


plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
plt.figure(figsize=(14,5))

g = sns.countplot(x='Donation_month',
                  hue='Optional', 
                  palette='hls',
                  data=df_merge_don[df_merge_don['Donation_year'] > 2012])
g.set_title("Donation Months Count by Optional", fontsize=20)
g.set_xlabel("Donation Months", fontsize=15)
g.set_ylabel("Count",fontsize=15)
g.legend(loc=1)

plt.show()

from random import sample

#Random subset the dataset to better work with graphics 
df_merge_proj = df_merge_proj.sample(n=1000000, replace=True)
df_merge_proj['Project Cost'].dtype
plt.figure(figsize = (16,6))

plt.subplot(1,2,1)
sns.distplot(np.log(df_merge_proj['Project Cost']+1).dropna())
plt.xlabel('Project Cost Amounts', fontsize=15)
plt.title("Histogram of Project Costs",fontsize=20)

plt.subplot(1,2,2)
plt.scatter(range(df_merge_proj.shape[0]), np.sort(df_merge_proj['Project Cost'].values))
plt.xlabel('Project Cost Amounts', fontsize=15)
plt.title("Distribution of Project Cost",fontsize=20)

plt.show()
# calculating summary statistics
data_mean, data_std = np.mean(df_merge_proj['Project Cost']), np.std(df_merge_proj['Project Cost'])

# identify outliers
cut = data_std * 3
lower, upper = data_mean - cut, data_mean + cut

# identify outliers
outliers = [x for x in df_merge_proj['Project Cost'] if x < lower or x > upper]

# remove outliers
outliers_removed = [x for x in df_merge_proj['Project Cost'] if x > lower and x < upper]
print('Identified outliers: %d' % len(outliers))
print('Non-outlier observations: %d' % len(outliers_removed))
print("Percentiles of Amount: ")
print(round(df_merge_proj['Project Cost'].quantile([.025,.25,.5,.75,.975]),2))
res_mask = df_merge_proj['Resource Quantity'] > 0

df_merge_proj["Total_Resource"] = df_merge_proj['Resource Quantity'] * df_merge_proj['Resource Unit Price']

df_merge_proj["Resource_Total_Log"] = np.log(df_merge_proj["Total_Resource"] + 1)
plt.figure(figsize = (16,6))

plt.subplot(1,2,1)
sns.distplot(np.log(df_merge_proj['Total_Resource']+1).dropna())
plt.xlabel('Project Cost Amounts', fontsize=15)
plt.title("Histogram of Project Costs",fontsize=20)

plt.subplot(1,2,2)
plt.scatter(range(df_merge_proj.shape[0]), np.sort(df_merge_proj['Total_Resource'].values))
plt.xlabel('Project Cost Amounts', fontsize=15)
plt.title("Distribution of Project Cost",fontsize=20)

plt.show()
# calculating summary statistics
data_mean, data_std = np.mean(df_merge_proj['Total_Resource']), np.std(df_merge_proj['Total_Resource'])

# identify outliers
cut = data_std * 3
lower, upper = data_mean - cut, data_mean + cut

# identify outliers
outliers = [x for x in df_merge_proj['Total_Resource'] if x < lower or x > upper]

# remove outliers
outliers_removed = [x for x in df_merge_proj['Total_Resource'] if x > lower and x < upper]

print("Outliers: ")
print('Identified outliers: %d' % len(outliers))
print('Non-outlier observations: %d' % len(outliers_removed))
print("")
print("Percentiles of Amount: ")
print(round(df_merge_proj['Total_Resource'].quantile([.025,.25,.5,.75,.975]),2))
df_merge_proj['Cost_log'] = np.log(df_merge_proj['Project Cost'] + 1)
plt.figure(figsize=(14,5))

plt.subplot(121)
g = sns.countplot(x='Project Current Status', palette='hls', 
                  data=df_merge_proj)
g.set_title("Project Status Count",fontsize=20)
g.set_xlabel("Project Status",fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(122)
g1 = sns.boxplot(x='Project Current Status', y='Project Cost',
                 palette='hls', 
                 data=df_merge_proj)
g1.set_title("Project Status Costs Distribuitions",fontsize=20)
g1.set_xlabel("Project Status",fontsize=15)
g1.set_ylabel("Project Costs",fontsize=15)

plt.show()
plt.figure(figsize=(14,5))

plt.subplot(121)
g = sns.boxplot(x='Project Current Status', y='Cost_log',
                 palette='hls', 
                 data=df_merge_proj)
g.set_title("Project Status Costs",fontsize=20)
g.set_xlabel("Project Status",fontsize=15)
g.set_ylabel("Project Costs(Log)",fontsize=15)

plt.subplot(122)
g1 = sns.boxplot(x='Project Current Status', y='Resource_Total_Log',
                 palette='hls', 
                 data=df_merge_proj)
g1.set_title("Project Status RESOURCE Costs",fontsize=20)
g1.set_xlabel("Project Status",fontsize=15)
g1.set_ylabel("Resource Costs",fontsize=15)

plt.show()
import squarify
plt.figure(figsize=(12,11))

plt.subplot(311)
g = sns.countplot(x='Project Grade Level Category', 
                  data=df_merge_proj, palette='hls')
g.set_title("Projects Grade Level Categorys",fontsize=20)
g.set_xlabel("",fontsize=15)
g.set_ylabel("Count",fontsize=15)

plt.subplot(312)
g1 = sns.violinplot(x='Project Grade Level Category', y='Cost_log',
                   data=df_merge_proj, palette='Set2')
g1.set_title("Projects Grade Level Categorys",fontsize=20)
g1.set_xlabel("",fontsize=15)
g1.set_ylabel("Project Costs(Log)",fontsize=15)

plt.subplot(313)
g2 = sns.boxplot(x='Project Grade Level Category', y='Resource_Total_Log',
                   data=df_merge_proj, palette='hls')
g2.set_title("Projects Grade Level Categorys",fontsize=20)
g2.set_xlabel("Grade Level Category",fontsize=15)
g2.set_ylabel("Resources Total(Log)",fontsize=15)

plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
plt.figure(figsize=(12,14))

plt.subplot(311)
g = sns.countplot(x='Teacher Prefix', 
                  data=df_merge_proj, palette='hls')
g.set_title("Project Resource Category Count",fontsize=20)
g.set_xlabel("",fontsize=15)
g.set_ylabel("Count",fontsize=15)
g.set_xticklabels(g.get_xticklabels(),rotation=45)

plt.subplot(312)
g1 = sns.boxplot(x='Teacher Prefix', y='Cost_log',
                   data=df_merge_proj, palette='hls')
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Project Resource Category",fontsize=20)
g1.set_xlabel("",fontsize=15)
g1.set_ylabel("Project Costs(Log)",fontsize=15)

plt.subplot(313)
g2 = sns.boxplot(x='Teacher Prefix', y='Resource_Total_Log',
                   data=df_merge_proj, palette='hls')
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Project Resource Category",fontsize=20)
g2.set_xlabel("Resource Category",fontsize=15)
g2.set_ylabel("Resources Total(Log)",fontsize=15)

plt.subplots_adjust(hspace = 0.85, top = 0.8)

plt.show()
import random

#To generate colors to Squarify
number_of_colors = 10
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
import squarify # pip install squarfy (algorithm for treemap)
teste = round(df_merge_proj['School Metro Type'].value_counts() / len(df_merge_proj['School Metro Type']) * 100,2)

plt.figure(figsize=(12,14))

plt.subplot(311)
g = squarify.plot(sizes=teste.values, label=teste.index, alpha=.4, value=teste.values,
                  color=["red","green","blue", "grey",'purple'])
g.set_title("'School Metro Type Squarify",fontsize=20)
g.set_axis_off()

plt.subplot(312)
g1 = sns.boxplot(x='School Metro Type', y='Cost_log',
                   data=df_merge_proj, palette='hls')

g1.set_title("Project Resource Category",fontsize=20)
g1.set_xlabel("",fontsize=15)
g1.set_ylabel("Project Costs(Log)",fontsize=15)

plt.subplot(313)
g2 = sns.boxplot(x='School Metro Type', y='Resource_Total_Log',
                   data=df_merge_proj, palette='hls')

g2.set_title("Project Resource Category",fontsize=20)
g2.set_xlabel("School Metro Type",fontsize=15)
g2.set_ylabel("Resources Total(Log)",fontsize=15)

plt.subplots_adjust(hspace = 0.85, top = 0.8)

plt.show()
category_tree = round((df_merge_proj['Project Subject Category Tree'].value_counts()[:12] \
                       / len(df_merge_proj['Project Subject Category Tree']) * 100),2)

plt.figure(figsize=(16,8))
g = squarify.plot(sizes=category_tree.values, label=category_tree.index, 
                  value=category_tree.values,
                  alpha=.4, color=color)
g.set_title("'Project Subject Category Tree Squarify in % size",fontsize=20)
g.set_axis_off()
plt.show()
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords


stopwords = set(STOPWORDS)
newStopWords = ["will", "students"]
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_merge_proj["Project Essay"][~pd.isnull(df_merge_proj["Project Essay"])].sample(100000)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("Wordcloud from Project Essay", fontsize=35)
plt.axis('off')
plt.show()
stopwords = set(STOPWORDS)

newStopWords = ["classroom"]
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_merge_proj["Project Title"][~pd.isnull(df_merge_proj["Project Title"])].sample(100000)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("Wordcloud from Project Title", fontsize=35)
plt.axis('off')
plt.show()
df_merge_proj['Teacher First Project Posted Date'] = pd.to_datetime(df_merge_proj['Teacher First Project Posted Date'])

df_merge_proj['year'] = df_merge_proj['Teacher First Project Posted Date'].dt.year
df_merge_proj['month'] = df_merge_proj['Teacher First Project Posted Date'].dt.month 
df_merge_proj['day'] = df_merge_proj['Teacher First Project Posted Date'].dt.day
ts_proj = round(df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                   (df_merge_proj['year'] > 2002)].groupby(['year','month']).agg({'Total_Resource': [np.var, np.mean, sum, np.std], # find the min, max, and sum of the duration column
                                                                                                  'Teacher ID' : ['count','nunique'], # find the number of Teacher ID entries
                                                                                                  'Project ID' : ['count','nunique'], # find the number of Project ID entries
                                                                                                  'School ID': ['count', 'nunique']}),2) # get the min, first, and number of unique dates per group
ts_proj.head()
g = np.log(ts_proj['Total_Resource']).plot(kind='line', figsize=(14,8))
g.set_xlabel('Years',fontsize=16)
g.set_ylabel('Statistics Distribuition',fontsize=16)
g.set_title('Some Statistics by Years',fontsize=20)
g.legend(fontsize='medium')

plt.show()

g = np.log(ts_proj[['Teacher ID', 'School ID', 'Project ID']]).plot(kind='line', figsize=(14,8))
g.set_xlabel('Time Distribuition 2003 to 2017',fontsize=16)
g.set_ylabel('Count and Uniques',fontsize=16)
g.set_title('Some Statistics by Years',fontsize=20)
g.legend(fontsize='medium')

plt.show()
plt.figure(figsize=(14,16))

plt.subplot(211)
g = sns.countplot(x='year',
                  data=df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                   (df_merge_proj['year'] > 2002)])
g.set_xlabel("YEAR", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_title("Projects by Year", fontsize=20)
g.legend()

plt.subplot(212)
g = sns.pointplot(x='year', y='Total_Resource', 
                  data=df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                   (df_merge_proj['year'] > 2002)])
g.set_xlabel("YEAR", fontsize=16)
g.set_ylabel("Resource Values Log", fontsize=16)
g.set_title("YEAR Resource_Total_Log", fontsize=20)

plt.subplots_adjust(hspace = 0.85, top = 0.8)

plt.show()
plt.figure(figsize=(12,5))

g = sns.countplot(x='year', hue='Project Grade Level Category',
                  data=df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                   (df_merge_proj['year'] > 2002)])
g.set_xlabel("YEAR", fontsize=16)
g.set_ylabel("Resource Values Log", fontsize=16)
g.set_title("YEAR Resource_Total_Log by Type of Projects", fontsize=20)
g.legend(loc=2)

plt.show()
plt.figure(figsize=(14,12))

plt.subplot(211)
g = sns.pointplot(x='year', y='Total_Resource', hue='School Metro Type',
                  data=df_merge_proj[(df_merge_proj['year'] > 2002) ])
g.set_xlabel("YEAR", fontsize=16)
g.set_ylabel("Resource Values Log", fontsize=16)
g.set_title("YEAR Resource_Total_Log by Type of Projects", fontsize=20)

plt.subplot(212)
g1 = sns.barplot(x='year', y='Project Cost', hue='School Metro Type',
                  data=df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                   (df_merge_proj['year'] > 2002)])
g1.set_xlabel("YEAR ", fontsize=16)
g1.set_ylabel("Project Cost", fontsize=16)
g1.set_title("YEAR Project Cost", fontsize=20)

plt.subplots_adjust(hspace = 0.85, top = 0.8)

plt.show()
vendor_name = df_merge_proj['Resource Vendor Name'].value_counts()

plt.figure(figsize=(12,5))

g = sns.countplot(x='Resource Vendor Name', 
                  data=df_merge_proj[(df_merge_proj['Resource Vendor Name'].isin(vendor_name[:10].index.values))])
g.set_xlabel("Month", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_title("TOP 10 Vendor Name", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.show()
print("The TOP Vendor Names Represents in % of Total: ")
print(vendor_name[:5] / len(df_merge_proj['Resource Vendor Name']) * 100)
print("")
print('Total representation of the 3 top Vendors: ')
print((vendor_name[:3] / len(df_merge_proj['Resource Vendor Name']) * 100).sum())
plt.figure(figsize=(14,5))

g = sns.countplot(x='year', hue='Resource Vendor Name',
                  data=df_merge_proj.loc[(df_merge_proj['year'] < 2018) & 
                                         (df_merge_proj['year'] > 2002) & 
                                         (df_merge_proj['Resource Vendor Name'].isin(vendor_name[:3].index.values))])
g.set_xlabel("year", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_title("Resource_Total_Log  - TOP 3 Vendor Names by Year", fontsize=20)
plt.show()
plt.figure(figsize=(12,5))

vendor_name = df_merge_proj['Resource Vendor Name'].value_counts()

g = sns.countplot(x='month', hue='Resource Vendor Name',
                  data=df_merge_proj[(df_merge_proj['year'] > 2002) &
                                     (df_merge_proj['Resource Vendor Name'].isin(vendor_name[:3].index.values))])
g.set_xlabel("month", fontsize=16)
g.set_ylabel("Total Resource Values Log", fontsize=16)
g.set_title("Month Resource_Total_Log by TOP 3 Vendor Names", fontsize=20)
plt.show()
school_states = df_merge_proj['School State'].value_counts()

plt.figure(figsize=(14,5))

g = sns.countplot(x='School State', 
                  data=df_merge_proj[(df_merge_proj['School State'].isin(school_states[:15].index.values))])
g.set_xlabel("School State", fontsize=16)
g.set_ylabel("Count", fontsize=16)
g.set_title("Top 15 School States in Projects", fontsize=20)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.show()
import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings

#First plot
trace0 = go.Histogram(
    x=df_merge_proj['School Percentage Free Lunch'],
    name="Free Lunch Distribuition ",
    histnorm='probability', showlegend=False,
    xbins=dict(
        start=0,
        end=100,
        size=5),
    autobiny=True)

#Second plot
trace1 = go.Histogram(
    x=np.log(df_merge_proj['School Percentage Free Lunch'] + 1),
    name="Free Lunch Distribuition Log",
    histnorm='probability', showlegend=False,
    xbins=dict(
        start=0,
        end=8,
        size=0.25),
    autobiny=True)

#Third plot
trace2 = go.Box(
    x=df_merge_proj[(df_merge_proj['School State'].isin(school_states[:10].index.values))]['School State'],
    y=df_merge_proj[(df_merge_proj['School State'].isin(school_states[:10].index.values))]['School Percentage Free Lunch'].sample(1500),
    name="All Category's Distribuition", showlegend=False
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=("Free Lunch Distribuition ",
                                          "Free Lunch Distribuition Log", 
                                          "Free Lunch by States"))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(showlegend=False)
iplot(fig)
districts_count = df_merge_proj['School District'].value_counts()
city_count = df_merge_proj['School City'].value_counts()
#First plot
trace0 = go.Bar(
    x=districts_count[:10].index,
    y=districts_count[:10].values,
    name="TOP 10 Districts", showlegend=False
)
#Second plot
trace1 =go.Bar(
    x=city_count[:10].index,
    y=city_count[:10].values,
    name="TOP 10 Citys", showlegend=False
)

#Third plot
trace2 = go.Box(
    x=df_merge_proj[(df_merge_proj['School District'].isin(districts_count[:10].index.values))]['School District'],
    y=df_merge_proj[(df_merge_proj['School District'].isin(districts_count[:10].index.values))]['School Percentage Free Lunch'],
    name="Top 10 Districts", showlegend=True
)
trace3 = go.Box(
    x=df_merge_proj[(df_merge_proj['School City'].isin(city_count[:10].index.values))]['School City'],
    y=df_merge_proj[(df_merge_proj['School City'].isin(city_count[:10].index.values))]['School Percentage Free Lunch'],
    name="Top 10 Citys", showlegend=True
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=('TOP 10 Districts','TOP 10 Citys', "Top 10 Districts and City by Percentage Free Lunch"))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 1)

fig['layout'].update(showlegend=True, title="Top Districts and Citys Schools Distribuitions", )
iplot(fig)
school_names = df_merge_proj['School Name'].value_counts()
#First plot
trace0 = go.Bar(
    x=school_names[:15].index,
    y=school_names[:15].values,
    name="Top 15 School Names", showlegend=False
)

#Third plot
trace1 = go.Box(
    x=df_merge_proj[(df_merge_proj['School Name'].isin(school_names[:15].index.values))]['School Name'],
    y=df_merge_proj[(df_merge_proj['School Name'].isin(school_names[:15].index.values))]['School Percentage Free Lunch'],
    name="School Name", showlegend=False
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{'colspan': 2}, None], [{'colspan': 2}, None]],
                          subplot_titles=('TOP School Names Count','TOP 15 School Names Freqncy by % of Free Lunch'))

#setting the figs
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)

fig['layout'].update(showlegend=True, title="Top 15 School Names Distribuitions")
iplot(fig)
