import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import  plotly
plotly.tools.set_credentials_file(username='RishiHazra', api_key='3WYShX1Rc0UlKTzCVggk')
from collections import Counter,OrderedDict
import plotly.offline as py
from plotly import tools
from plotly.graph_objs import *
py.init_notebook_mode(connected=True)
from sorted_months_weekdays import *
from sort_dataframeby_monthorweek import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud, STOPWORDS
import matplotlib.image as mpimg
from PIL import Image
img=mpimg.imread('../input/masks2/donorschoose.jpg')
plt.figure(figsize=[14,8])
imgplot = plt.imshow(img)
plt.axis('Off')
plt.show()
# A sample of 'Teachers' data
teachers=pd.read_csv('../input/io/Teachers.csv')
teachers.describe()
teachers['Teacher Prefix'].fillna('Mrs.', inplace=True)

# replacing 'Mx.' with 'Ms.' 
teachers['Teacher Prefix']=teachers['Teacher Prefix'].apply(lambda x: x.replace('Mx.','Ms.'))


count_prefix=Counter(teachers['Teacher Prefix'])
data = [Pie(labels=list(count_prefix.keys()),values=list(count_prefix.values()))]
#py.iplot(fig, filename='1')
print(count_prefix)
explode = ( 0.1, 0.05, 0.1, 0.05,0.1)
plt.figure(figsize=(8,8))
plt.rcParams['font.size'] = 12
plt.title('Teacher Prefix Information')
plt.pie(list(count_prefix.values()), explode=explode, labels=list(count_prefix.keys()), autopct='%1.1f%%',
        shadow=True)  

plt.show()
# separating the date into day, month and year
def date(df):
    df = df.copy()
    df['Year']=pd.DatetimeIndex(df['Teacher First Project Posted Date']).year
    df['Month']=pd.DatetimeIndex(df['Teacher First Project Posted Date']).month
    df['Day']=pd.to_datetime(df['Teacher First Project Posted Date']).dt.weekday_name
    return df
teachers_modified= date(teachers)
teachers_modified['Day']=pd.Categorical(teachers_modified['Day'], categories=['Sunday','Monday','Tuesday' ,'Wednesday',
                                                                              'Thursday', 'Friday','Saturday'],ordered=True)
months={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep',10:'Oct', 11:'Nov',12:'Dec'}
teachers=teachers_modified.replace({'Month':months})

# sorting the months serially
teachers=Sort_Dataframeby_Month(teachers,monthcolumnname='Month')
count=pd.DataFrame.from_dict(Counter(teachers['Month']), orient='index').reset_index()
count=count.rename(columns={'index':'Month', 0:'count'})
sort=Sort_Dataframeby_Month(count, monthcolumnname='Month')
x1 = list(teachers_modified['Day'].value_counts().sort_index().keys())
y1 = list(teachers_modified['Day'].value_counts().sort_index())

x2 = list(Counter(teachers['Year']).keys()) 
z1 = list(Counter(teachers['Year']).values())

x=list(sort['Month'])
y=list(sort['count'])


trace1 = Scatter(x=x1, y=y1, fill='tozeroy', mode= 'lines+markers')
#data1= Bar(x=list(Counter(teachers['Year']).keys()), y=list(Counter(teachers['Year']).values()),marker=dict(color='#A2D5F2'),
           #width=0.25,name='Teacher First Project Posted Date (Yearly count)')
trace2 = Bar(x=x2, y=z1, marker=dict(color='#e993f9'))
trace3 = Scatter(x=x, y=y, fill='tozeroy', fillcolor = '#fcc45f', mode= 'none')

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ["<b>Daily Count</b>", "<b>Monthly Count</b>", "<b>Yearly Count</b>"])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 3);
fig.append_trace(trace3, 1, 2);


fig['layout'].update(height=350, showlegend=False, yaxis1=dict(
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
    ),yaxis3=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    )
                    );

py.iplot(fig, filename='3')
import numpy as np
schools = pd.read_csv('../input/io/Schools.csv', error_bad_lines=False)
schools.head(3)
schools.apply(lambda x: sum(x.isnull()),axis=0)
#for schools without NCES data, a district average is used
grouped=schools.groupby('School District')
a=grouped['School Percentage Free Lunch'].agg(np.mean)
null_index= schools[schools['School Percentage Free Lunch'].isnull()].index.tolist()

for i in null_index:
   schools['School Percentage Free Lunch'][i]= a[schools['School District'][i]]
a[a.isnull()][:5]
# @credit:Sban
words = " ".join(schools['School Name']).split()
from collections import Counter

school_types = Counter(words).most_common(30)[1:9]
school_types = [list(x) for x in school_types]
labels = [x[0]+" School" for x in school_types]
values = [x[1] for x in school_types]

fig=plt.figure(figsize=(20,5))
plt.rcParams['font.size'] = 15
plt.title('School Types Distribution')
plt.bar(labels,values, color='#ff00bb',width=0.25)
plt.ylabel('Count')
plt.show()
donors = pd.read_csv('../input/io/Donors.csv')
donors.describe()
#index of null values in 'Donor City' and 'Donor Zip'
#c=donors[donors['Donor City'].isnull()].index.tolist()
d=donors[donors['Donor Zip'].isnull()].index.tolist()

donors.iloc[134,:].to_frame(name=None).style.highlight_null(null_color='red')
ind= donors.loc[(donors['Donor State']=='other') & (donors['Donor Zip']=='245')].index.tolist()
for i in ind:
    donors['Donor State'][i]='Virginia'
donations= pd.read_csv('../input/io/Donations.csv')
donations.head(3)
# optional donation frequency
from collections import Counter
count_opt_donations= Counter(donations['Donation Included Optional Donation'])
explode = ( 0.1, 0.05)
plt.figure(figsize=(5,5))
plt.rcParams['font.size'] = 15
plt.title('Optional donation information')
plt.pie(list(count_opt_donations.values()), explode=explode, labels=list(count_opt_donations.keys()), autopct='%1.1f%%',
        shadow=True)  

plt.show()
# separating the date into day, month and year
def date(df):
    df = df.copy()
    df['Year']=pd.DatetimeIndex(df['Donation Received Date']).year
    df['Month']=pd.DatetimeIndex(df['Donation Received Date']).month
    df['Day']=pd.to_datetime(df['Donation Received Date']).dt.weekday_name
    return df

donations_modified= date(donations)
months={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep',10:'Oct', 11:'Nov',12:'Dec'}
donations=donations_modified.replace({'Month':months})

# sorting the months serially
donations=Sort_Dataframeby_Month(donations,monthcolumnname='Month')
count=pd.DataFrame.from_dict(Counter(donations['Month']), orient='index').reset_index()
count=count.rename(columns={'index':'Month', 0:'count'})
sort=Sort_Dataframeby_Month(count, monthcolumnname='Month')
donations['Day']=pd.Categorical(donations['Day'], categories=['Sunday','Monday','Tuesday' ,'Wednesday',
                                                                              'Thursday', 'Friday','Saturday'],ordered=True)
t4 = donations.groupby(['Day']).agg({'Year' : 'count', 'Donation Amount' : 'mean'}).reset_index().rename(columns={'Year' : 'Total Donations', 'Donation Amount' : 'Average Amount'}).sort_index()
x3 = t4['Day']
q1 = t4['Total Donations']
q2 = t4['Average Amount']

t1 = donations.groupby(['Year']).agg({'Month' : 'count', 'Donation Amount' : 'mean'}).reset_index().rename(columns={'Month' : 'Total Donations', 'Donation Amount' : 'Average Amount'})
x1 = t1['Year']
y1 = t1['Total Donations']
y2 = t1['Average Amount']

t2 = donations.groupby(['Month']).agg({'Year' : 'count', 'Donation Amount' : 'mean'}).reset_index().rename(columns={'Year' : 'Total Donations', 'Donation Amount' : 'Average Amount'})
# sorting the months serially
t2=Sort_Dataframeby_Month(t2, monthcolumnname='Month')
x2 = t2['Month']
z1 = t2['Total Donations']
z2 = t2['Average Amount']


trace1 = Scatter(x=x1[:-1], y=y1[:-1], fill='tozeroy', fillcolor = '#fcc45f', mode= 'none')
trace2 = Scatter(x=x1[:-1], y=y2[:-1], fill='tozeroy', fillcolor = "#e993f9", mode= 'none')
trace3 = Scatter(x=x2, y=z1, fill='tozeroy', fillcolor = 'lightseagreen', mode= 'none')
trace4 = Scatter(x=x2, y=z2, fill='tozeroy', fillcolor = "tomato", mode= 'none')
trace5 = Scatter(x=x3, y=q1, fill='tozeroy', fillcolor = 'coral', mode= 'none')
trace6 = Scatter(x=x3, y=q2, fill='tozeroy', fillcolor = "palevioletred", mode= 'none')

fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles = ["<b>Total Donation Amount per Year</b>", "<b>Average Donation Amount per Year</b>", "<b>Total Donation Amount per Month</b>", "<b>Average Donation Amount per Month</b>","<b>Donations per Day</b>", "<b>Average Donation Amount per Day</b>"])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 2, 1);
fig.append_trace(trace4, 2, 2);
fig.append_trace(trace5, 3, 1);
fig.append_trace(trace6, 3, 2);


fig['layout'].update(height=900, showlegend=False, yaxis1=dict(
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
    ),yaxis3=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ), yaxis4=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
),yaxis5=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ), yaxis6=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False)                     
                    );

py.iplot(fig, filename='multiple-subplots3') 
# yearwise and monthwise total donation amount

z=donations[['Year','Month','Donation Amount']].groupby(['Year','Month'],as_index=False)['Donation Amount'].agg(np.sum)
z['Donation Amount']=z['Donation Amount'].apply(lambda x : int(x))
z =z.sort_values('Donation Amount', ascending=False)
st = 0

sizes = list(reversed([i for i in range(10,31)]))
intervals = int(len(z) / len(sizes))
size_array = [9]*len(z)

for i, size in enumerate(sizes):
    for j in range(st, st+intervals):
        size_array[j] = size 
    st = st+intervals
z['size_n'] = size_array
cols = list(z['size_n'])

# sorting the months serially
z=Sort_Dataframeby_Month(z, monthcolumnname='Month')

trace1 =Scatter( x=z['Month'], y=z['Year'], mode='markers', text=z['Donation Amount'],
        marker=dict( size=z.size_n, color=cols, colorscale='Picnic' ))
data = [trace1]
layout=dict(title='Total Donation Amount vs Donation Received Date')
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='jupyter-basic_pie1')
z=pd.merge(donors[['Donor State','Donor ID']],donations[['Donation Amount','Donor ID']],on='Donor ID')
#donor_donations= pd.merge(donors,donations,on='Donor ID')
# state_wise average donations
grouped = z.groupby('Donor State', as_index=False)
state_wise_donations=grouped['Donation Amount'].agg(np.mean)
pd.set_option('float_format', '{:f}'.format)

# state wise total donations
# arranging the state_donations in descending order
top_donation_states=grouped['Donation Amount'].agg(np.sum).sort_values(
        ['Donation Amount'], ascending=False)

colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]

top_donation_states['text'] = top_donation_states['Donor State'].astype(str) + '<br>' +\
    'Total donation amount:  $' + top_donation_states['Donation Amount'].astype(str)

    
state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
top_donation_states['State Code'] = top_donation_states['Donor State'].map(state_to_code)

data = [ dict(
        type = 'choropleth',
        locations = top_donation_states['State Code'],
        z = top_donation_states['Donation Amount'].astype(float),
        text = top_donation_states['text'],
        locationmode = 'USA-states',
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
        title = 'Donation amount in USD')) ]

layout = dict(
        title = 'State wise donations <br>(Hover for details) ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-usa-map' )

# city_wise average donation amount
z=pd.merge(donors[['Donor City','Donor ID']],donations[['Donation Amount','Donor ID']],on='Donor ID')
grouped= z.groupby('Donor City', as_index=False)
city_wise_donations=grouped['Donation Amount'].agg(np.mean)

# top cities in terms of total donations
top_donation_cities= grouped['Donation Amount'].agg(np.sum).sort_values(
        ['Donation Amount'], ascending=False)[:20]
trace1 = Bar(x=list(top_donation_cities['Donor City'][:20]), y=list(top_donation_cities['Donation Amount'][:20]),
              marker=dict(color='rgb(49,130,189)'), width=0.25, name='Top States in terms of donations')

trace2 = Bar(x=list(top_donation_states['Donor State'][:20]), y=list(top_donation_states['Donation Amount'][:20]),
              marker=dict(color='rgb(180,180,180)'), xaxis='x2',yaxis='y2', width=0.25, name='Top Cities in terms of donations')


data=[trace1, trace2]
layout = Layout(
    xaxis=dict(
        domain=[0, 1]
    ),
    yaxis=dict(
        domain=[0, 0.5]
    ),
    xaxis2=dict(
        domain=[0, 1],
        anchor='y2'
    ),
    yaxis2=dict(
        domain=[0.55, 1],
        anchor='x2'
    ),
    
    height=600,
)

fig=dict(data=data, layout=layout)
py.iplot(fig, filename='6')
count_donor= OrderedDict(Counter(donors['Donor State']).most_common(20))
plt.figure(figsize=(10,10))

plt.xticks(rotation=-90)
plt.rcParams['font.size'] = 10
plt.title('Total donors from each state',fontweight="bold")

plt.barh(list(count_donor.keys()),list(count_donor.values()), color='crimson')
plt.xlabel('Count')
plt.show()
count_donor_city= Counter(donors['Donor City'])
print('Cities with highest Donor Count:\n',count_donor_city.most_common()[1:10])
z=pd.merge(donors[['Donor Is Teacher','Donor ID']],donations[['Donor Cart Sequence','Donor ID']],on='Donor ID')
grouped=z.groupby(['Donor Is Teacher'], as_index=False)
grouped['Donor Cart Sequence'].agg(np.mean).style.set_properties(**{'background-color': 'black',
                           'color': 'aqua','border-color': 'white'})

del donations_modified, teachers_modified
resources=pd.read_csv('../input/io/Resources.csv')
resources.head(3)
z=resources['Resource Item Name'].value_counts()[:20]
z1 = pd.DataFrame()
z1['item'] = z.index 
z1['Count'] =z.values
z1['Items'] = list(reversed("""Genius Kit, Acer 11'6 Chromebook, Noise, Kids' Wobble Chair, Seat Storage Sack,Kids Stay & Play Ball, Apple Ipad, HP 11'6 Chromebook,Black write and wipe markers , Kids Stay N Play Ball, Soft Seats, Privacy Partition,Apple Ipad, Commercial Furniture, Apple Ipad, Apple Ipad, Noise, Apple Ipad, Noise, Trip""".split(",")))
z1.groupby('Items').agg({'Count' : 'sum'}).reset_index().sort_values('Count',ascending=False)
z1 = z1[z1['Items'] != " Noise"]

x=resources['Resource Vendor Name'].value_counts()[:20]
x1 = pd.DataFrame()
x1['Vendor'] = x.index 
x1['Count'] =x.values

fig=plt.figure(figsize=(20,8))
fig.patch.set_facecolor('lightgray')
plt.subplot(1,2,1)
plt.xticks(rotation=-90)
plt.rcParams['font.size'] = 15
plt.title('Most Requested Items')
plt.bar(list(z1['Items']),list(z1['Count']), color='dodgerblue',width=0.25)
plt.ylabel('Count')
plt.grid(True)

plt.subplot(1,2,2)
plt.xticks(rotation=-90)
plt.rcParams['font.size'] = 15
plt.title('Most Popular Vendors')
plt.bar(list(x1['Vendor']),list(x1['Count']), color='blueviolet',width=0.25)
plt.ylabel('Count')
plt.grid(True)
plt.show()
z=resources.sort_values(['Resource Quantity'],ascending=False)[:20]
z1 = pd.DataFrame()
z1['item'] = z['Resource Item Name']
z1['Quantity'] =z['Resource Quantity']
z1['Items'] = list(reversed("""Pencils, Dividers, Padlockcombinations, Spring Composition Book, Britannica School, Paw Print, patio stone, Paw Print, Paper, Bit Classroom Kit, Turintin Fbs, Eclipse Glasses, Math Subscription, Neck Ribbon, Floor Cover, Notebook, Rubber Flooring, Chair Glide, Renewable Glide, Crossling Blocks""".split(",")))
z1.groupby('Items')['Quantity'].agg(np.max).reset_index()

x=resources.sort_values(['Resource Unit Price'],ascending=False)[:20]
x1 = pd.DataFrame()
x1['item'] = x['Resource Item Name']
x1['Cost'] =x['Resource Unit Price']
x1['Items'] = list(reversed("""Ipad Cart, Home Theatre + Projector, Aromatic epdm, ddr Classroom Edition, Freeze Tag, Playground + Playstation, Fencing Supplies & Labor, Contrabassoon, Daktronic GS6, Leveled Bookroom, 10 Alpha, Playground + Playstation, Wood Playground Equipment, Commerical Structure, Sound System, Playstation, Telescoping Gym Bleachers, Commerical Structure, New Playground, Hadicapped Playground""".split(",")))
x1.groupby('Items')['Cost'].agg(np.max).reset_index()
plt.figure(figsize=(22,8))

plt.subplot(1,2,1)
plt.xticks(rotation=-90)
plt.rcParams['font.size'] = 15
plt.title('Most Expensive Items (per unit price)')
plt.barh(list(x1['Items']),list(x1['Cost']), color='mediumaquamarine')

plt.subplot(1,2,2)
plt.xticks(rotation=-90)
plt.rcParams['font.size'] = 15
plt.title('Items Requested In Bulk')
plt.barh(list(z1['Items']),list(z1['Quantity']), color='darkorange')

plt.show()
resources['Total Cost'] = resources['Resource Quantity'] * resources['Resource Unit Price']
z=resources[['Resource Vendor Name','Total Cost']]
z=z.dropna().groupby(['Resource Vendor Name'], as_index=False).agg(np.sum).sort_values('Total Cost', ascending=False)[:20]

z1=resources.sort_values('Resource Unit Price', ascending=False)[['Resource Vendor Name','Resource Unit Price']]
z1=z1.dropna().groupby(['Resource Vendor Name'], as_index=False).agg(np.mean).sort_values('Resource Unit Price', ascending=False)[:20]

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.rcParams['font.size'] = 15
plt.title('Highest Grossing Vendors')
plt.barh(list(z['Resource Vendor Name']),list(z['Total Cost']), color='yellow')

plt.subplot(1,2,2)
fig.patch.set_facecolor('lightgray')
plt.rcParams['font.size'] = 15
plt.title('Highest Grossing Vendors per unit price')
plt.barh(list(z1['Resource Vendor Name']),list(z1['Resource Unit Price']), color='darkgray')
plt.show()
projects=pd.read_csv('../input/io/Projects.csv')
projects.head(3)
plt.figure(figsize=(8,8))
colors= ['gold', 'lightskyblue', 'lightcoral']
plt.rcParams['font.size'] = 12
explode = ( 0.3, 0.3, 0.3)
plt.title('Percentage of projects of each project type')
plt.pie(list(Counter(projects['Project Type']).values()), explode=explode, labels=list(Counter(projects['Project Type']).keys()), autopct='%1.1f%%',
        shadow=True, colors=colors)  

plt.show()
grouped=projects[['Project Grade Level Category','Project Type']].groupby(['Project Type'], as_index=False)
grouped['Project Grade Level Category'].agg(lambda x: x.value_counts().index[0]).style.set_properties(**{'background-color': 'black',
                           'color': 'gold','border-color': 'white'})
grouped=projects[['Project Type','Project Grade Level Category']].groupby(['Project Grade Level Category'], as_index=False)
grouped['Project Type'].agg(lambda x: x.value_counts().index[0]).style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen','border-color': 'white'})
grouped=projects[['Teacher Project Posted Sequence','Project Type']].groupby('Project Type',as_index=False)
z=grouped['Teacher Project Posted Sequence'].agg(np.mean)
grouped1=projects[['Project Grade Level Category','Teacher Project Posted Sequence']].groupby(['Project Grade Level Category'], as_index=False)
z1=grouped1['Teacher Project Posted Sequence'].agg(np.mean).sort_values('Teacher Project Posted Sequence',ascending=False)

fig=plt.figure(figsize=[18,6])
plt.subplot(1,2,1)
plt.pie(z['Teacher Project Posted Sequence'],labels=z['Project Type'],autopct='%.0f%%', shadow=True)
plt.axis('equal')
plt.title('Teacher Project Posted Sequence \n vs \n Project Type',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.subplot(1,2,2)
plt.pie(z1['Teacher Project Posted Sequence'],labels=z1['Project Grade Level Category'],autopct='%.0f%%', shadow=True)
plt.axis('equal')
plt.title('Teacher Project Posted Sequence \n vs \n Project Grade Level Category',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()
projects['Day']=pd.to_datetime(projects['Project Posted Date']).dt.weekday_name
projects['Year']=pd.DatetimeIndex(projects['Project Posted Date']).year
projects['Month']=pd.DatetimeIndex(projects['Project Posted Date']).month
months={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep',10:'Oct', 11:'Nov',12:'Dec'}
projects=projects.replace({'Month':months})

# sorting the months serially
projects=Sort_Dataframeby_Month(projects,monthcolumnname='Month')
count=pd.DataFrame.from_dict(Counter(projects['Month']), orient='index').reset_index()
count=count.rename(columns={'index':'Month', 0:'count'})
sort1=Sort_Dataframeby_Month(count, monthcolumnname='Month')

projects['Day']=pd.Categorical(projects['Day'], categories=['Sunday','Monday','Tuesday' ,'Wednesday',
                                                                              'Thursday', 'Friday','Saturday'],ordered=True)
x3 = list(projects['Day'].value_counts().sort_index().keys())
y3 = list(projects['Day'].value_counts().sort_index())

x2 = list(Counter(projects['Year']).keys())
z1 = list(Counter(projects['Year']).values())

x=list(sort1['Month'])
y=list(sort1['count'])


trace1 = Bar(x=x3, y=y3)
trace2 = Bar(x=x2, y=z1, marker=dict(color='#e993f9'))
trace3 = Scatter(x=x, y=y, fill='tozeroy', fillcolor = '#fcc45f', mode= 'none')

fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = ["<b>Daily Count</b>", "<b>Monthly Count</b>", "<b>Yearly Count</b>"])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 3);
fig.append_trace(trace3, 1, 2);


fig['layout'].update(height=350, showlegend=False, yaxis1=dict(
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
    ),yaxis3=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    )
                    );

py.iplot(fig, filename='100')
# comparing School State and Project Posted Date
merged=pd.merge(projects[['Month','School ID']], schools[['School ID','School State']], on='School ID', how='inner')

grouped=merged[['Month','School State']].groupby(['School State'], as_index= False)
z=grouped.agg(lambda x: x.value_counts().index[0])
grouped=z[['Month','School State']].groupby(['Month'])
aug=grouped['School State'].apply(lambda x: sorted(set(x)))['Aug']
nov=grouped['School State'].apply(lambda x: sorted(set(x)))['Nov']
apr=grouped['School State'].apply(lambda x: sorted(set(x)))['Apr']
mar=grouped['School State'].apply(lambda x: sorted(set(x)))['Mar']
#Oct=grouped['School State'].apply(lambda x: sorted(set(x)))['Oct']
Sep=grouped['School State'].apply(lambda x: sorted(set(x)))['Sep']

z={'States':[apr,mar,aug,Sep,nov], 'Months': ['Apr','Mar','Aug','Sep','Nov']}
z=pd.DataFrame(z)
z.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen','border-color': 'white'})


del merged
#projects['Project Cost']= projects['Project Cost'].replace({'\$':'', ',':''},regex=True).astype(float)

# yearwise and monthwise project cost

z=projects[['Year','Month','Project Cost']].groupby(['Year','Month'],as_index=False)['Project Cost'].agg(np.sum)
z['Project Cost']=z['Project Cost'].apply(lambda x : int(x))
z =z.sort_values('Project Cost', ascending=False)
st = 0

sizes = list(reversed([i for i in range(10,31)]))
intervals = int(len(z) / len(sizes))
size_array = [9]*len(z)

for i, size in enumerate(sizes):
    for j in range(st, st+intervals):
        size_array[j] = size 
    st = st+intervals
z['size_n'] = size_array
cols = list(z['size_n'])

# sorting the months serially
z=Sort_Dataframeby_Month(z, monthcolumnname='Month')

trace1 =Scatter( x=z['Month'], y=z['Year'], mode='markers', text=z['Project Cost'],
        marker=dict( size=z.size_n, color=cols, colorscale='Rainbow' ))
data = [trace1]
layout=dict(title='Total Project Cost vs Project Posted Date')
fig = Figure(data=data, layout=layout)
py.iplot(fig, filename='9')
# project current status
# obtaining relation between 'Project Cost' and 'Project Current Status'
grouped=projects[['Project Cost','Project Current Status']].groupby(['Project Current Status'],as_index=False)
z=grouped['Project Cost'].agg(np.mean)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
labels='Fully Funded','Expired','Archived','Live'
plt.pie([241329, 821367, 98003, 47952] , labels=labels,autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('Project Current Status',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.subplot(1,2,2)
plt.title('Mean Project Cost   vs   Project Status')
barlist=plt.bar(z['Project Current Status'],z['Project Cost'], color='lightgray',width=0.15)
barlist[1].set_color('crimson')
plt.ylabel('Mean Project Cost')
plt.xlabel('Project Status')

plt.show()
grouped=projects[['Project Resource Category','Project Cost']].groupby(['Project Resource Category'], as_index=False)
z=grouped['Project Cost'].agg(np.mean).sort_values('Project Cost', ascending=False)
z=z.rename(columns={'Project Cost': 'Average Project Cost'})
z1=grouped['Project Cost'].agg(np.sum)
z1=z1.rename(columns={'Project Cost': 'Total Project Cost'})
z=pd.merge(z,z1,on='Project Resource Category')

fig, axes = plt.subplots(ncols=2, sharey=True)
fig.set_figheight(8)
fig.set_figwidth(15)
axes[0].barh(z['Project Resource Category'],z['Average Project Cost'], color='palegreen')
axes[0].set(title='Average Project Cost')

axes[1].barh(z['Project Resource Category'],z['Total Project Cost'], color='hotpink')
axes[1].set(title='Total Project Cost')

axes[0].invert_xaxis()
#fig.legend(loc='upper right ', frameon=False)

plt.show()
text = projects['Project Essay'] 
# read the mask image
mask = np.array(Image.open('../input/cloudmasks/kid3.png'))

stopwords = set(STOPWORDS)
plt.figure(figsize=(20,15))

# generate word cloud
wc = WordCloud(background_color="white", max_words=1000, mask=mask,
               stopwords=stopwords,colormap='cool',max_font_size=90)
wc.generate(str(text))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Word cloud of Projects (Essay)',fontweight="bold")
plt.show()
text = projects['Project Title'] + ' ' +projects['Project Short Description'] 
# read the mask image
mask = np.array(Image.open('../input/cloudmasks/kid2.jpg'))

stopwords = set(STOPWORDS)
plt.figure(figsize=(20,15))

# generate word cloud
wc = WordCloud(background_color="white", max_words=1000, mask=mask,
               stopwords=stopwords,colormap='cool',max_font_size=90)
wc.generate(str(text))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title('Word cloud of Projects (Essay)',fontweight="bold")
plt.show()
full_funded_projects = projects.loc[projects['Project Current Status'] == 'Fully Funded']
not_funded_projects  = projects[(projects['Project Current Status'] == 'Expired') ]

text = full_funded_projects['Project Title'] 
text1= not_funded_projects['Project Title'] 
# read the mask image
mask = np.array(Image.open('../input/cloudmasks/kid4.jpg'))

stopwords = set(STOPWORDS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20,15])

# generate word cloud
wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='magma',max_font_size=90)
wc.generate(str(text))
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Word cloud of Fully Funded Projects (Title)',fontweight="bold")

wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='nipy_spectral',max_font_size=90)
wc.generate(str(text1))
ax2.imshow(wc, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Word Cloud of Non-Funded Projects (Title)',fontweight="bold")
fig.show()
text = full_funded_projects['Project Essay'] 
text1= not_funded_projects['Project Essay'] 
# read the mask image
mask = np.array(Image.open('../input/cloudmasks/kid3.png'))

stopwords = set(STOPWORDS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20,15])

# generate word cloud
wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='magma',max_font_size=90)
wc.generate(str(text))
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Word cloud of Fully Funded Projects (Essay)',fontweight="bold")

wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='nipy_spectral',max_font_size=90)
wc.generate(str(text1))
ax2.imshow(wc, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Word Cloud of Non-Funded Projects (Essay)',fontweight="bold")
fig.show()
teacher_led=projects[projects['Project Type']=='Teacher-Led']
student_led=projects[projects['Project Type']=='Student-Led']

text = teacher_led['Project Short Description']+ ' ' + teacher_led['Project Title']
text1= student_led['Project Short Description']+ ' ' + student_led['Project Title']
# read the mask image
mask = np.array(Image.open('../input/cloudmasks/kid4.jpg'))

stopwords = set(STOPWORDS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20,15])

# generate word cloud
wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='magma',max_font_size=90)
wc.generate(str(text))
ax1.imshow(wc, interpolation='bilinear')
ax1.axis("off")
ax1.set_title('Teacher Led Projects (Description+Title)',fontweight="bold")

wc = WordCloud(background_color="white", max_words=200, mask=mask,
               stopwords=stopwords,colormap='nipy_spectral',max_font_size=90)
wc.generate(str(text1))
ax2.imshow(wc, interpolation='bilinear')
ax2.axis("off")
ax2.set_title('Student Led Projects (Description+Title)',fontweight="bold")
fig.show()
import string 
from nltk.corpus import stopwords
stopwords = stopwords.words("english")

def clean_text(txt):
    # lower case
    txt = txt.lower()

    # punctuation removal 
    txt = ''.join(x for x in txt if x not in string.punctuation)

    # stopwords and lemmatization
    clean_txt = ""
    for word in txt.split():
        if word in stopwords:
            continue 
        clean_txt += " "
        clean_txt += word 
        
    noise = ['title','students','would', 'will','donotremoveessaydivider']
    for ns in noise:
        clean_txt = clean_txt.replace(ns, "")
        
    return clean_txt

year_18=projects[(projects['Year']==2018) | (projects['Month']=='March')]['Project Essay'].apply(clean_text)
def generate_ngrams(txt, N):
    grams = [txt[i:i+N] for i in range(len(txt)-N+1)]
    grams = [" ".join(b) for b in grams]
    return grams 

nf_18=projects[(projects['Project Current Status']=='Expired') & (projects['Year']==2018) | (projects['Month']=='March')]['Project Essay'].apply(clean_text)
ff_18=projects[(projects['Project Current Status']=='Fully Funded') & (projects['Year']==2018) | (projects['Month']=='March')]['Project Essay'].apply(clean_text)

nf_18['bigrams'] = nf_18.apply(lambda x : generate_ngrams(str(x).split(), 2))
ff_18['bigrams'] = ff_18.apply(lambda x : generate_ngrams(str(x).split(), 2))
nf_18['trigrams'] = nf_18.apply(lambda x : generate_ngrams(str(x).split(), 3))
ff_18['trigrams'] = ff_18.apply(lambda x : generate_ngrams(str(x).split(), 3))

all_bigrams_nf = []
for each in nf_18['bigrams']:
    all_bigrams_nf.extend(each)    
all_bigrams_ff = []
for each in ff_18['bigrams']:
    all_bigrams_ff.extend(each)

all_trigrams_nf = []
for each in nf_18['trigrams']:
    all_trigrams_nf.extend(each)    
all_trigrams_ff = []
for each in ff_18['trigrams']:
    all_trigrams_ff.extend(each)

del nf_18['bigrams'],ff_18['bigrams'],nf_18['trigrams'],ff_18['trigrams']  
z=Counter(all_bigrams_nf).most_common(20)
x_nf_bi=[a[0] for a in z]
y_nf_bi=[a[1] for a in z]

z=Counter(all_bigrams_ff).most_common(20)
x_ff_bi=[a[0] for a in z]
y_ff_bi=[a[1] for a in z]

z=Counter(all_trigrams_nf).most_common(20)
x_nf_tri=[a[0] for a in z]
y_nf_tri=[a[1] for a in z]

z=Counter(all_trigrams_ff).most_common(20)
x_ff_tri=[a[0] for a in z]
y_ff_tri=[a[1] for a in z]

fig=plt.figure(figsize=[25,20])
plt.rcParams.update({'font.size': 18})
fig.patch.set_facecolor('azure')

plt.subplot(2,2,1)
plt.barh(x_nf_bi,y_nf_bi,color='palevioletred')
b=plt.title('Most Common bigrams used for Non Funded Projects')

plt.subplot(2,2,2)
plt.barh(x_ff_bi, y_ff_bi,color='palevioletred')
a=plt.title('Most Common bigrams used for Funded Projects')

plt.subplot(2,2,3)
plt.barh(x_nf_tri,y_nf_tri,color='salmon')
b=plt.title('Most Common trigrams used for Non Funded Projects')

plt.subplot(2,2,4)
plt.barh(x_ff_tri, y_ff_tri,color='salmon')
a=plt.title('Most Common trigrams used for Funded Projects')
# @ credit:sban
import operator
t = projects['Project Subject Category Tree'].value_counts()
x = list(t.index)
y = list(t.values)
r = {}
for i,val in enumerate(x):
    for each in val.split(","):
        x1 = each.strip()
        if x1 not in r:
            r[x1] = y[i]
        r[x1] += y[i]
sorted_x = list(sorted(r.items(), key=operator.itemgetter(1), reverse = True))[:10]
x1 = [a[0] for a in sorted_x][::-1]
y1 = [a[1] for a in sorted_x][::-1]



t1 = projects['Project Subject Subcategory Tree'].value_counts()
x2 = list(t1.index)
y2 = list(t1.values)
r = {}
for i,val in enumerate(x2):
    for each in val.split(","):
        x1_ = each.strip()
        if x1_ not in r:
            r[x1_] = y2[i]
        r[x1_] += y2[i]        
x2 = r.keys()
y2 = r.values()
sorted_x = list(sorted(r.items(), key=operator.itemgetter(1), reverse = True))[:10]
x2 = [a[0] for a in sorted_x][::-1]
y2 = [a[1] for a in sorted_x][::-1]

t = projects['Project Resource Category'].value_counts()
x3 = list(t.index[::-1])[:10]
y3 = list(t.values[::-1])[:10]

plt.figure(figsize=[20,8])
plt.subplot(1,3,1)
plt.title('Top Project Subject Category')
trace1 = plt.barh(x1, y1, color="#8cf2a9")
plt.subplot(1,3,2)
plt.title('Top Project Subject SubCategory')
trace2 = plt.barh(x2, y2,color="#8cf2a9")
plt.subplot(1,3,3)
trace3 = plt.barh(x3, y3, color="#8cf2a9")
plt.title('Top Project Resource Category')
plt.show()

#del resources
# mapping school states to donor states
z=pd.merge(projects[['Project ID','School ID']],schools[['School ID','School State']],on='School ID')
z=pd.merge(z[['School State','Project ID']],donations[['Donor ID','Project ID','Donation ID','Donation Amount']],on='Project ID')
z=pd.merge(z[['School State','Donor ID','Donation ID','Donation Amount']],donors[['Donor ID','Donor State']],on='Donor ID')

pivot2 = z.pivot_table(columns='School State',
                                      index='Donor State', 
                                      values='Donation Amount', 
                                      aggfunc='sum',
                                      fill_value=0)

# Scale again by the funds that state receives
sum_state_funds = z.groupby('School State')['Donation Amount'].sum()
pivot2 = pivot2 / sum_state_funds.transpose()
pivot2=pivot2[pivot2.index != 'other']
#@credit:shep312
state_lat_lon = {
    'Alabama': [32.806671,-86.791130],
    'Alaska': [61.370716,-152.404419],
    'Arizona': [33.729759,-111.431221],
    'Arkansas': [34.969704,-92.373123],
    'California': [36.116203,-119.681564],
    'Colorado': [39.059811,-105.311104],
    'Connecticut': [41.597782,-72.755371],
    'Delaware': [39.318523,-75.507141],
    'District of Columbia': [38.897438,-77.026817],
    'Florida': [27.766279,-81.686783],
    'Georgia': [33.040619,-83.643074],
    'Hawaii': [21.094318,-157.498337],
    'Idaho': [44.240459,-114.478828],
    'Illinois': [40.349457,-88.986137],
    'Indiana': [39.849426,-86.258278],
    'Iowa': [42.011539,-93.210526],
    'Kansas': [38.526600,-96.726486],
    'Kentucky': [37.668140,-84.670067],
    'Louisiana': [31.169546,-91.867805],
    'Maine': [44.693947,-69.381927],
    'Maryland': [39.063946,-76.802101],
    'Massachusetts': [42.230171,-71.530106],
    'Michigan': [43.326618,-84.536095],
    'Minnesota': [45.694454,-93.900192],
    'Mississippi': [32.741646,-89.678696],
    'Missouri': [38.456085,-92.288368],
    'Montana': [46.921925,-110.454353],
    'Nebraska': [41.125370,-98.268082],
    'Nevada': [38.313515, -117.055374],
    'New Hampshire': [43.452492,-71.563896],
    'New Jersey': [40.298904,-74.521011],
    'New Mexico': [34.840515,-106.248482],
    'New York': [42.165726,-74.948051],
    'North Carolina': [35.630066,-79.806419],
    'North Dakota': [47.528912,-99.784012],
    'Ohio': [40.388783,-82.764915],
    'Oklahoma': [35.565342,-96.928917],
    'Oregon': [44.572021,-122.070938],
    'Pennsylvania': [40.590752,-77.209755],
    'Rhode Island': [41.680893,-71.511780],
    'South Carolina': [33.856892,-80.945007],
    'South Dakota': [44.299782,-99.438828],
    'Tennessee': [35.747845,-86.692345],
    'Texas': [31.054487,-97.563461],
    'Utah': [40.150032,-111.862434],
    'Vermont': [44.045876,-72.710686],
    'Virginia': [37.769337,-78.169968],
    'Washington': [47.400902,-121.490494],
    'West Virginia': [38.491226,-80.954453],
    'Wisconsin': [44.268543,-89.616508],
    'Wyoming': [42.755966,-107.302490]
}

flight_paths = []
for i in pivot2.index:
    
    for j in pivot2.columns:
        
        # Only plot if significant
        if (pivot2.loc[i, j] > 0.02) * (i != j):
               
            flight_paths.append(
                dict(
                    type = 'scattergeo',
                    locationmode = 'USA-states',                           
                    lon = [state_lat_lon[i][1], state_lat_lon[j][1]],
                    lat = [state_lat_lon[i][0], state_lat_lon[j][0]],
                    mode = 'lines',
                    line = dict(
                        width = 10 * pivot2.loc[i, j],
                        color = 'red',                        
                    ),
                    text = '{:.2f}% of {}\'s donations come from {}'.format(100 * pivot2.loc[i, j], j, i),
                )
            )
    
layout = dict(
        title = 'Strongest out of state donation patterns (hover for details)',
        showlegend = False, 
        geo = dict(
            scope='usa',
            #projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            #subunitcolor = "rgb(217, 217, 217)",
            #countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
    
fig = dict(data=flight_paths, layout=layout)
py.iplot(fig,filename='d3-flight-paths')
merged=pd.merge(projects[['Project ID','Project Cost','School ID']], schools[['School ID','School Metro Type','School Percentage Free Lunch']],on='School ID',how='inner')
merged=pd.merge(merged[['Project Cost','School Metro Type','School Percentage Free Lunch','Project ID']], donations[['Donation Amount','Project ID']],on='Project ID', how='inner')

grouped=merged[['School Metro Type','Donation Amount']].groupby(['School Metro Type'], as_index=False)
z=grouped['Donation Amount'].agg(np.mean)

a=merged[merged['School Percentage Free Lunch']<20]['Donation Amount'].mean()
b=merged[merged['School Percentage Free Lunch'].between(20,40,inclusive=True)]['Donation Amount'].mean()
c=merged[merged['School Percentage Free Lunch'].between(40,60,inclusive=True)]['Donation Amount'].mean()
d=merged[merged['School Percentage Free Lunch'].between(60,80,inclusive=True)]['Donation Amount'].mean()
e=merged[merged['School Percentage Free Lunch']>80]['Donation Amount'].mean()

z1=[a,b,c,d,e]
z2=['below 20','20-40','40-60','60-80','80-100']

data1= Bar(x=list(z['School Metro Type']), y=list(z['Donation Amount']),marker=dict(color='#FDE724'), width=0.25)
data2 = Scatter(x = z2,y =z1,marker=dict(color='#35B778'))

grouped1=merged[['School Metro Type','Project Cost']].groupby(['School Metro Type'], as_index=False)
x1=grouped1['Project Cost'].agg(np.mean)

a=merged[merged['School Percentage Free Lunch']<20]['Project Cost'].mean()
b=merged[merged['School Percentage Free Lunch'].between(20,40,inclusive=True)]['Project Cost'].mean()
c=merged[merged['School Percentage Free Lunch'].between(40,60,inclusive=True)]['Project Cost'].mean()
d=merged[merged['School Percentage Free Lunch'].between(60,80,inclusive=True)]['Project Cost'].mean()
e=merged[merged['School Percentage Free Lunch']>80]['Project Cost'].mean()

z1=[a,b,c,d,e]

data3= Bar(x=list(x1['School Metro Type']), y=list(x1['Project Cost']),marker=dict(color='rgb(180,70,100)'), width=0.25)
data4= Scatter(x = z2,y =z1,marker=dict(color='rgb(100,70,120)'))

fig = tools.make_subplots(rows=2, cols=2, print_grid=False, subplot_titles = ['Donation Amount vs School Metro Type','Donation Amount vs School Percentage Free Lunch', 'Project Cost vs School Metro Type','Project Cost vs School Percentage Free Lunch'])
fig.append_trace(data1, 1, 1);
fig.append_trace(data2, 1, 2);
fig.append_trace(data3, 2, 1);
fig.append_trace(data4, 2, 2);

fig['layout'].update(height=700, showlegend=False, yaxis1=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ), yaxis2=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ),yaxis3=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ),yaxis4=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    )
                    );
py.iplot(fig, filename='11')
del merged
grouped=projects[['Project Cost','Project Type']].groupby(['Project Type'], as_index=False)
z=grouped['Project Cost'].agg(np.sum)
z1=grouped['Project Cost'].agg(np.mean)

#comparing 'Project Cost' and 'Teacher Prefix'
x1=pd.merge(projects[['Project Cost','Teacher ID']],teachers[['Teacher ID','Teacher Prefix']],on='Teacher ID',how='inner')
grouped1=x1[['Project Cost','Teacher Prefix']].groupby(['Teacher Prefix'], as_index=False)
x1=grouped1['Project Cost'].agg(np.sum)
x2=grouped1['Project Cost'].agg(np.mean)
trace1 = Bar(
    x=x1['Teacher Prefix'],
    y=x1['Project Cost'], width=0.15,
    marker=dict(color='#22A784'),
   
)
trace2 = Scatter(
    x=x2['Teacher Prefix'],
    y=x2['Project Cost'],
    marker=dict(color='#79D151'),
    
)

trace3 = Bar(
    x=z['Project Type'],
    y=z['Project Cost'], width=0.15,
    
)
trace4 = Scatter(
    x=z1['Project Type'],
    y=z1['Project Cost'],
    
)

fig = tools.make_subplots(rows=2, cols=2, print_grid=False, subplot_titles = ['Total Project Cost vs. Teacher Prefix','Average Project Cost vs. Teacher Prefix', 'Total Project Cost vs. Project Type','Average Project Cost vs. Project Type'])
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 2, 1);
fig.append_trace(trace4, 2, 2);

fig['layout'].update(height=700, showlegend=False, yaxis1=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ), yaxis2=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ),yaxis3=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    ),yaxis4=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=True
    )
                    );


py.iplot(fig, filename='multiple-subplots')
# comparing the project posted year of 'Teacher Prefix' with the number of projects posted per year
merged=pd.merge(projects[['Year','Project ID','Teacher ID']],teachers[['Teacher Prefix','Teacher ID']],on='Teacher ID',how='inner')
merged=pd.merge(merged[['Year','Teacher Prefix','Project ID']], donations[['Donor ID','Project ID']],on='Project ID',how='inner')
grouped=merged[['Year','Teacher Prefix','Donor ID']].groupby(['Teacher Prefix','Year'], as_index=False)
z=grouped.agg({'Donor ID':'count'})
z=z[z['Year']!=2018]
data = []
data.insert(0,{'Year':2012, 'Donor ID':0})
cal=pd.concat([pd.DataFrame(data),z[z['Teacher Prefix']=='Mr.'][['Year','Donor ID']]],ignore_index=True)
tex=pd.concat([pd.DataFrame(data),z[z['Teacher Prefix']=='Mrs.'][['Year','Donor ID']]],ignore_index=True)
flo=pd.concat([pd.DataFrame(data),z[z['Teacher Prefix']=='Ms.'][['Year','Donor ID']]],ignore_index=True)
illi=pd.concat([pd.DataFrame(data),z[z['Teacher Prefix']=='Teacher'][['Year','Donor ID']]],ignore_index=True)
norCar=pd.concat([pd.DataFrame(data),z[z['Teacher Prefix']=='Dr.'][['Year','Donor ID']]],ignore_index=True)

fig = plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 15
plt.title('Teacher Prefix Project Posted Year: Trend')
ax1 = fig.add_subplot(111)

ax1.plot(cal['Year'], cal['Donor ID'],marker='o',label='Mr.',linewidth=2, markersize=12)
ax1.plot(tex['Year'], tex['Donor ID'],marker='o', label='Mrs.',linewidth=2, markersize=12)
ax1.plot(flo['Year'], flo['Donor ID'],marker='o', label='Ms.',linewidth=2, markersize=12)
ax1.plot(illi['Year'], illi['Donor ID'],marker='o', label='Teacher',linewidth=2, markersize=12)
ax1.plot(norCar['Year'], norCar['Donor ID'],marker='o', label='Dr.',linewidth=2, markersize=12)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
# obtaining the relation between the 'Project Cost' and the number of days taken to fully fund the project
total_days=pd.to_datetime(projects['Project Fully Funded Date'])-pd.to_datetime(projects['Project Posted Date'])
x={'Project Cost':projects['Project Cost'], 'Total days': total_days}
x=pd.DataFrame(x)
grouped=x[['Project Cost','Total days']].groupby(['Total days'], as_index=False)
agg=grouped['Project Cost'].agg(np.mean)
ascend=agg.sort_values('Total days',ascending=False)[:50].mean()
descend=agg.sort_values('Total days',ascending=True)[:50].mean()
mean=agg.sort_values('Total days',ascending=True).mean()
z=agg.iloc[[1,10,30,50,70,100,130,150,170,180,200],:]

plt.figure(figsize=(20,8))
plt.rcParams['font.size'] = 15
plt.title('Project Cost   vs   Total days taken to fully fund the project')
plt.plot((z['Total days']/ np.timedelta64(1, 'D')).astype(int), z['Project Cost'], 'xb-')
#barlist[1].set_color('crimson')
plt.ylabel('Project Cost')
plt.xlabel('Total days taken to fund project')
plt.show()
# relation between 'Project Resource Category', 'Donor State', 'School State', 'Donor City', 'School City'
merged=pd.merge(projects[['Project Resource Category','Project ID','School ID']], schools[['School City','School State','School ID']],on='School ID',how='inner')
merged=pd.merge(merged[['Project Resource Category','Project ID','School City','School State']],donations[['Project ID','Donor ID']],on='Project ID',how='inner')
merged=pd.merge(merged[['Project Resource Category','Donor ID','School City','School State']], donors[['Donor State','Donor City','Donor ID']], on='Donor ID',how='inner')

grouped=merged[['Project Resource Category','School State', 'Donor State', 'School City','Donor City']].groupby(['Project Resource Category'], as_index=False)
z=grouped.agg(lambda x: x.value_counts().index[0])



table_trace2 = Table(
    type='table',
    header = dict(height = 50,
                  values = ['<b>Project Resource Category</b>','<b>School State</b>','<b>Donor State</b>','<b>School City</b>' ,'<b>Donor City</b>'], 
                  line = dict(color='rgb(50, 50, 50)'),
                  align = ['center'],
                  font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                  fill = dict(color='#d562be')),
    cells = dict(values =[list(z.iloc[:,0]),list(z.iloc[:,1]),list(z.iloc[:,2]), list(z.iloc[:,3]), list(z.iloc[:,4])],
                 line = dict(color='#506784'),
                 align = 'center',
                 font = dict(color=['rgb(40, 40, 40)'] * 5, size=12),
                 height = 27,
                 fill = dict(color=['rgb(235, 193, 238)', 'rgba(228, 222, 249, 0.65)','rgba(228, 222, 249, 0.65)', 
                                    'rgba(118, 202, 219, 0.35)','rgba(118, 202, 219, 0.35)']))
)
layout=dict(title='Leading states for Project Resource Category <br>(Scroll for more information)')
fig=dict(data=[table_trace2], layout=layout)
py.iplot(fig, filename='13')
merged=pd.merge(projects[['Project Current Status','Project ID']],donations[['Project ID','Donor ID']],on='Project ID',how='inner')
merged=pd.merge(merged[['Project Current Status','Donor ID']], donors[['Donor State','Donor ID']], on='Donor ID',how='inner')

grouped= merged[['Project Current Status','Donor State']].groupby(['Donor State'], as_index=False)
z=grouped['Project Current Status'].agg(lambda x: sum(x=='Fully Funded'))
z1=grouped['Project Current Status'].agg(lambda x: sum(x=='Expired'))
z2=z['Project Current Status'].div(z1['Project Current Status'])
z3={'Donor State':z['Donor State'], 'Fully Funded':z['Project Current Status'] ,
                'Expired':z1['Project Current Status'], 'Funded to Expired ratio':z2}
z3=pd.DataFrame(data=z3)
z3=z3.sort_values('Funded to Expired ratio', ascending=False)


table_trace2 = Table(domain=dict(x=[0, 1],
                y=[0,0.4]),
    type='table',
    header = dict(height = 50,
                  values = ['<b>Donor State</b>','<b>Fully Funded</b>','<b>Expired</b>','<b>Funded to Expired ratio</b>'], 
                  line = dict(color='rgb(0, 0, 0)'),
                  align = ['center'],
                  font = dict(color=['rgb(45, 45, 45)'] * 4, size=14),
                  fill = dict(color='#FDE724')),
    cells = dict(values =[list(z3.iloc[:,0]),list(z3.iloc[:,2]),list(z3.iloc[:,1]), list(z3.iloc[:,3].apply(lambda x: round(x,3)))],
                 line = dict(color='rgb(0, 0, 0)'),
                 align = 'center',
                 font = dict(color=['rgb(0, 0, 0)'] * 4, size=14),
                 height = 27,
                 fill = dict(color=['#FCFEA4', '#CAB969','#FD9F6C', 
                                    '#a1dab4']))
)

trace1 = Bar(
    xaxis='x1',
    yaxis='y1',
    x=z3['Donor State'][:20],
    y=z3['Funded to Expired ratio'][:20],
    width=0.20,
    name='Top States',
    marker=dict(
        color='#FFE945'
    )
)

data = [table_trace2,trace1]
layout1 = dict(     
    #width=950,
    height=600,
    autosize=False,
    title='Funded to Expired projects of states (Scroll for more information)',
    margin = dict(t=100),
    showlegend=True,   
    #xaxis1=dict(domain=[0, 1],anchor='y1'),
    yaxis1=dict(domain=[0.6,1],anchor='x1'),
    plot_bgcolor='rgba(249,249, 249, 0.65)'
)


#layout=dict(title='<b>Comparison of Funded to Expired projects of each state<b>\n (Scroll for more information)')
fig=dict(data=data, layout=layout1)
py.iplot(fig, filename='14')
del z1,z2,z3
merged=pd.merge(projects[['Project Current Status','School ID']], schools[['School ID','School Metro Type']], on='School ID',how='inner')

grouped= merged[['Project Current Status','School Metro Type']].groupby(['School Metro Type'], as_index=False)
z=grouped['Project Current Status'].agg(lambda x: sum(x=='Fully Funded'))
z1=grouped['Project Current Status'].agg(lambda x: sum(x=='Expired'))
z2=z['Project Current Status'].div(z1['Project Current Status'])
z3={'School Metro Type':z['School Metro Type'], 'Fully Funded':z['Project Current Status'] ,
                'Expired':z1['Project Current Status'], 'Funded to Expired ratio':z2}
z3=pd.DataFrame(data=z3)
z3=z3.sort_values('Funded to Expired ratio', ascending=False)

table_trace2 = Table(domain=dict(x=[0, 0.6],
                y=[0, 1.0]),
    type='table',
    header = dict(height = 30,
                  values = ['<b>School Metro Type</b>','<b>Funded to Expired ratio</b>'], 
                  line = dict(color='rgb(0, 0, 0)'),
                  align = ['center'],
                  font = dict(color=['#ffffff'] * 5, size=14),
                  fill = dict(color='#333333')),
    cells = dict(values =[list(z3['School Metro Type']), list(z3['Funded to Expired ratio'].apply(lambda x: round(x,4)))],
                 line = dict(color='rgb(0, 0, 0)'),
                 align = 'center',
                 font = dict(color=['rgb(0, 0, 0)'] * 5, size=12),
                 height = 27,
                 fill = dict(color=['#ffffff', '#cccccc']))
)


trace1 = Bar(
    xaxis='x1',
    yaxis='y1',
    x=z3['School Metro Type'],
    y=z3['Fully Funded'],
    name='Fully Funded projects', 
    marker=dict(
        color='rgb(49,130,189)'
    )
)
trace2 = Bar(
    xaxis='x1',
    yaxis='y1',
    x=z3['School Metro Type'],
    y=z3['Expired'],
    name='Expired projects',
    marker=dict(
        color='rgb(170,170,170)',
    )
)

axis=dict(
    showline=True,
    zeroline=False,
    showgrid=True,
    mirror=True,
    ticklen=4, 
    gridcolor='#ffffff',
    tickfont=dict(size=10)
)

fig=dict(data=[trace1 ,trace2], layout=layout)

data = [trace1, trace2,table_trace2]
layout1 = dict(
    width=950,
    height=500,
    autosize=False,
    margin = dict(t=100),
    showlegend=True,   
    xaxis1=dict(axis, **dict(domain=[0.65, 1], anchor='y1')),
    yaxis1=dict(axis, **dict(domain=[0.5, 1], anchor='x1', hoverformat='.2f')),  
    plot_bgcolor='rgba(228, 222, 249, 0.65)'
)

fig = Figure(data=data, layout=layout1)
py.iplot(fig, filename='2')
fig=plt.figure(figsize=[18,6])
plt.subplot(1,2,1)
merged=pd.merge(projects[['Project Current Status','Teacher ID']], teachers[['Teacher ID','Teacher Prefix']], on='Teacher ID',how='inner')

grouped= merged[['Project Current Status','Teacher Prefix']].groupby(['Teacher Prefix'], as_index=False)
z=grouped['Project Current Status'].agg(lambda x: sum(x=='Fully Funded'))
z1=grouped['Project Current Status'].agg(lambda x: sum(x=='Expired'))
z2=z['Project Current Status'].div(z1['Project Current Status'])
z3={'Teacher Prefix':z['Teacher Prefix'], 'Fully Funded':z['Project Current Status'] ,
                'Expired':z1['Project Current Status'], 'Funded to Expired ratio':z2}
z3=pd.DataFrame(data=z3)
z3=z3.sort_values('Funded to Expired ratio', ascending=False)
plt.pie(z3['Funded to Expired ratio'],labels=z3['Teacher Prefix'],autopct='%.0f%%', shadow=True)
plt.axis('equal')
plt.title('Project Current Status \n vs \n Teacher Prefix',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.subplot(1,2,2)
merged=pd.merge(projects[['Project Current Status','Project ID']], donations[['Project ID','Donation Included Optional Donation']], on='Project ID',how='inner')

grouped= merged[['Project Current Status','Donation Included Optional Donation']].groupby(['Donation Included Optional Donation'], as_index=False)
z=grouped['Project Current Status'].agg(lambda b: sum(b=='Fully Funded'))
z1=grouped['Project Current Status'].agg(lambda b: sum(b=='Expired'))
z2=z['Project Current Status'].div(z1['Project Current Status'])
z3={'Donation Included Optional Donation':z['Donation Included Optional Donation'], 'Fully Funded':z['Project Current Status'] ,
                'Expired':z1['Project Current Status'], 'Funded to Expired ratio':z2}
z3=pd.DataFrame(data=z3)
z3.sort_values('Funded to Expired ratio', ascending=False)
plt.pie(z3['Funded to Expired ratio'],labels=z3['Donation Included Optional Donation'],autopct='%.0f%%', shadow=True)
plt.axis('equal')
plt.title('Project Current Status \n vs \n Donation Included Option Donation',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.show()
#comparison of 'Donor Cart Sequence' with 'Donor State', 'Donor City' and 'School Name'.
merged=pd.merge(projects[['Project ID','School ID']], schools[['School Name','School ID']],on='School ID',how='inner')
merged=pd.merge(merged[['Project ID','School Name']],donations[['Project ID','Donor Cart Sequence','Donor ID']],on='Project ID',how='inner')
merged=pd.merge(merged[['Donor Cart Sequence','Donor ID','School Name']], donors[['Donor State','Donor City','Donor ID']], on='Donor ID',how='inner')

grouped=merged[['Donor Cart Sequence','Donor State','School Name','Donor City']].groupby(['Donor State','School Name','Donor City'], as_index=False)
z=grouped.agg(np.mean).sort_values('Donor Cart Sequence', ascending=False)[:30]
grouped1=merged[['Donor Cart Sequence','Donor State']].groupby(['Donor State'], as_index=False)
z1=grouped1.agg(np.mean).sort_values('Donor Cart Sequence', ascending=False)[:11]
z1=z1[z1['Donor State']!='other']
grouped2=merged[['Donor Cart Sequence','Donor City']].groupby(['Donor City'], as_index=False)
z2=grouped2.agg(np.mean).sort_values('Donor Cart Sequence', ascending=False)[:10]

table_trace2 = Table(domain=dict(x=[0, 1],
                y=[0, 0.7]),
    type='table',
    header = dict(height = 50,
                  values = ['<b>School Name<b>','<b>Donor City<b>','<b>Donor State</b>','<b>Mean Donor Cart Sequence</b>'], 
                  line = dict(color='rgb(0, 0, 0)'),
                  align = ['center'],
                  font = dict(color=['rgb(45, 45, 45)'] * 4, size=14),
                  fill = dict(color='#deebf7')),
    cells = dict(values =[list(z.iloc[:,1]),list(z.iloc[:,2]),list(z.iloc[:,0]),list(z.iloc[:,3].apply(lambda x: round(x,1)))],
                 line = dict(color='rgb(0, 0, 0)'),
                 align = 'center',
                 font = dict(color=['rgb(0, 0, 0)'] * 4, size=14),
                 height = 27,
                 fill = dict(color=['#4292c6', '#6baed6','#9ecae1', '#c6dbef']))
)


data = [table_trace2]
layout1 = dict(     
    #width=950,
    height=600,
    autosize=False,
    title='Mean Donor Cart Sequence \n (Scroll for more information)',
    margin = dict(t=100),
    showlegend=True,   
    plot_bgcolor='rgba(249,249, 249, 0.65)'
)


#layout=dict(title='<b>Comparison of Funded to Expired projects of each state<b>\n (Scroll for more information)')
fig=dict(data=data, layout=layout1)
py.iplot(fig, filename='1')
plt.figure(figsize=[20,8])
plt.rcParams['font.size'] = 15
plt.subplot(1,2,1)
plt.title('Mean Donor Cart Sequence for top 10 states')
plt.barh(z1.iloc[:,0],z1.iloc[:,1].apply(lambda x: round(x,1)), color='#084594')
plt.subplot(1,2,2)
#plt.rcParams['font.size'] = 15
plt.title('Mean Donor Cart Sequence for top 10 cities')
plt.barh(z2.iloc[:,0],z2.iloc[:,1].apply(lambda x: round(x,1)), color='#a1dab4')
plt.show()
del grouped1,grouped2
merged=pd.merge(projects[['Project Grade Level Category','Project ID']], donations[['Project ID','Donor Cart Sequence']], on='Project ID',how='inner')

grouped=merged[['Project Grade Level Category','Donor Cart Sequence']].groupby(['Project Grade Level Category'], as_index=False)
z=grouped['Donor Cart Sequence'].agg(np.mean).sort_values('Donor Cart Sequence',ascending=False)

fig=plt.figure(figsize=[6,6])
plt.axis('equal')
plt.pie(z['Donor Cart Sequence'] , labels=z['Project Grade Level Category'],autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.title('Average Donor Cart Sequence',fontweight='bold')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()
merged=pd.merge(projects[['Project Resource Category','Project ID']], donations[['Project ID','Donor ID','Donation Amount']], on='Project ID',how='inner')
merged=pd.merge(merged[['Project Resource Category','Donation Amount','Donor ID']], donors[['Donor State','Donor ID']], on='Donor ID',how='inner')

grouped= merged[['Donor State','Donation Amount','Project Resource Category']].groupby(['Project Resource Category','Donor State'], as_index=False)
df=grouped['Donation Amount'].agg(np.sum)
df_max = df.groupby('Project Resource Category').idxmax()
df_max['type'] = 'max'
df_min = df.groupby('Project Resource Category').idxmin()
df_min['type'] = 'min'

df2 = df_max.append(df_min).set_index('type',append=True).stack().rename('index')

df3 = pd.concat([ df2.reset_index().drop('Project Resource Category',axis=1).set_index('index'), 
                  df.loc[df2.values] ], axis=1 )

df3.set_index(['Project Resource Category','type']).sort_index().style.set_properties(**{'background-color': 'black',
                           'color': 'gold','border-color': 'white'})

del df_max,df_min,df2,df3
merged=pd.merge(projects[['Project Resource Category','Project ID']], donations[['Project ID','Donor ID']], on='Project ID',how='inner')
merged=pd.merge(merged[['Project Resource Category','Donor ID']], donors[['Donor State','Donor ID']], on='Donor ID',how='inner')
grouped= merged[['Donor State','Project Resource Category']].groupby(['Donor State'], as_index=False)
z=grouped['Project Resource Category'].agg(lambda x: x.value_counts().index[0])
z=z.groupby(['Project Resource Category'])['Donor State'].apply(lambda x: sorted(set(x)))
z1=z[0]
z2=z[1]
pd.DataFrame({'Project Resource Category':['Supplies','Technology'], 'Donor States':[z1,z2]}).style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen','border-color': 'white'})
# Comparing 'Teacher Project Posted Sequence'
merged=pd.merge(projects[['Teacher Project Posted Sequence','Project ID','School ID']], schools[['School Name','School ID']],on='School ID',how='inner')
merged=pd.merge(merged[['Teacher Project Posted Sequence','Project ID','School Name']],donations[['Project ID','Donor ID']],on='Project ID',how='inner')
merged=pd.merge(merged[['Teacher Project Posted Sequence','Donor ID','School Name']], donors[['Donor State','Donor City','Donor ID']], on='Donor ID',how='inner')

grouped=merged[['Teacher Project Posted Sequence','Donor State','School Name','Donor City']].groupby(['Donor State','School Name','Donor City'], as_index=False)
z=grouped.agg(np.mean).sort_values('Teacher Project Posted Sequence', ascending=False)[:30]
grouped1=merged[['Teacher Project Posted Sequence','Donor State']].groupby(['Donor State'], as_index=False)
z1=grouped1.agg(np.mean).sort_values('Teacher Project Posted Sequence', ascending=False)[:10]
grouped2=merged[['Teacher Project Posted Sequence','Donor City']].groupby(['Donor City'], as_index=False)
z2=grouped2.agg(np.mean).sort_values('Teacher Project Posted Sequence', ascending=False)[:10]

table_trace2 = Table(domain=dict(x=[0, 1],
                y=[0.55, 1]),
    type='table',
    header = dict(height = 50,
                  values = ['<b>School Name<b>','<b>Donor City<b>','<b>Donor State</b>','<b>Mean Teacher Project Posted Sequence</b>'], 
                  line = dict(color='rgb(0, 0, 0)'),
                  align = ['center'],
                  font = dict(color=['rgb(45, 45, 45)'] * 5, size=14),
                  fill = dict(color='#FBFCBF')),
    cells = dict(values =[list(z.iloc[:,1]),list(z.iloc[:,2]),list(z.iloc[:,0]),list(z.iloc[:,3].apply(lambda x: round(x,1)))],
                 line = dict(color='rgb(0, 0, 0)'),
                 align = 'center',
                 font = dict(color=['#ffffff'] * 5, size=12),
                 height = 27,
                 fill = dict(color=['#440154', '#30678D', '#35B778', '#30678D']))
)

trace1 = Bar(
    xaxis='x1',
    yaxis='y1',
    x=z1.iloc[:,0],
    y=z1.iloc[:,1].apply(lambda x: round(x,1)),
    name='Mean Teacher Project Posted Sequence for top 10 states', 
    width=0.25,
    marker=dict(
        color='#30678D'
    )
)

trace2 = Bar(
    xaxis='x2',
    yaxis='y2',
    x=z2.iloc[:,0],
    y=z2.iloc[:,1].apply(lambda x: round(x,1)),
    name='Mean Teacher Project Posted Sequence for top 10 cities', 
    width=0.25,
    marker=dict(
        color='#35B778'
    )
)

data = [table_trace2,trace1,trace2]
layout1 = dict(     
    #width=950,
    height=600,
    autosize=False,
    title='<b>Comparison of Mean Teacher Project Posted Sequence<b>\n (Scroll for more information)',
    margin = dict(t=100),
    showlegend=True,   
    xaxis1=dict(domain=[0, 0.5],anchor='y1'),
    yaxis1=dict(domain=[0,0.5],anchor='x1'),
    xaxis2=dict(domain=[0.55, 1],anchor='y2'),
    yaxis2=dict(domain=[0,0.5],anchor='x2'),
    plot_bgcolor='rgba(249,249, 249, 0.65)'
)


#layout=dict(title='<b>Comparison of Funded to Expired projects of each state<b>\n (Scroll for more information)')
fig=dict(data=data, layout=layout1)
py.iplot(fig, filename='4')
grouped=projects[['Teacher Project Posted Sequence','Project Resource Category']].groupby('Project Resource Category',as_index=False)
z=grouped['Teacher Project Posted Sequence'].agg(np.mean)

merged=pd.merge(projects[['Project Resource Category','Project ID']], donations[['Project ID','Donor Cart Sequence']], on='Project ID',how='inner')
grouped=merged[['Donor Cart Sequence','Project Resource Category']].groupby('Project Resource Category',as_index=False)
z1=grouped['Donor Cart Sequence'].agg(np.mean)

#cm = sns.light_palette("green", as_cmap=True)
a=pd.merge(z,z1,on='Project Resource Category').sort_values('Teacher Project Posted Sequence',ascending=True)
#.style.background_gradient(cmap=cm)
plt.figure(figsize=[20,10])
plt.title('Comparison of Teacher Project Posted Sequence and Donor Cart Sequence with Project Resource Category')
plt.barh(a['Project Resource Category'],-a['Teacher Project Posted Sequence'],color='darkgrey',label='Teacher Project Posted Sequence')
plt.barh(a['Project Resource Category'],a['Donor Cart Sequence'],color='gold',label='Donor Cart Sequence')
plt.legend(loc='upper right ', frameon=False)
plt.show()
# comparing the project posted year of states with the number of projects posted per year
merged=pd.merge(projects[['Year','Project ID']], donations[['Project ID','Donor ID']], on='Project ID',how='inner')
merged=pd.merge(merged[['Year','Donor ID']], donors[['Donor State','Donor ID']], on='Donor ID',how='inner')
grouped=merged[['Year','Donor State','Donor ID']].groupby(['Donor State','Year'], as_index=False)
z=grouped.agg({'Donor ID':'count'})
z=z[z['Year']!=2018]
data = []
data.insert(0,{'Year':2012, 'Donor ID':0})
cal=pd.concat([pd.DataFrame(data),z[z['Donor State']=='California'][['Year','Donor ID']]],ignore_index=True)
ny=pd.concat([pd.DataFrame(data),z[z['Donor State']=='New York'][['Year','Donor ID']]],ignore_index=True)
tex=pd.concat([pd.DataFrame(data),z[z['Donor State']=='Texas'][['Year','Donor ID']]],ignore_index=True)
flo=pd.concat([pd.DataFrame(data),z[z['Donor State']=='Florida'][['Year','Donor ID']]],ignore_index=True)
illi=pd.concat([pd.DataFrame(data),z[z['Donor State']=='Illinois'][['Year','Donor ID']]],ignore_index=True)

fig = plt.figure(figsize=(16,8))
plt.rcParams['font.size'] = 15
plt.title('Trend of top 10 Donor States')
ax1 = fig.add_subplot(111)

ax1.plot(cal['Year'], cal['Donor ID'],marker='o',label='California',linewidth=2, markersize=12)
ax1.plot(ny['Year'], ny['Donor ID'],marker='o', label='New York',linewidth=2, markersize=12)
ax1.plot(tex['Year'], tex['Donor ID'],marker='o', label='Texas',linewidth=2, markersize=12)
ax1.plot(flo['Year'], flo['Donor ID'],marker='o', label='Florida',linewidth=2, markersize=12)
ax1.plot(illi['Year'], illi['Donor ID'],marker='o', label='Illinois',linewidth=2, markersize=12)

plt.legend(loc='upper left')
plt.grid(True)
plt.show()
## Case 1: Recommend the most popular items
z=resources['Resource Item Name'].value_counts()[:20]
z1 = pd.DataFrame()
z1['item'] = z.index 
z1['Count'] =z.values
z1['Items'] = list(reversed("""Genius Kit, Acer 11'6 Chromebook, Noise, Kids' Wobble Chair, Seat Storage Sack,Kids Stay & Play Ball, Apple Ipad, HP 11'6 Chromebook,Black write and wipe markers , Kids Stay N Play Ball, Soft Seats, Privacy Partition,Apple Ipad, Commercial Furniture, Apple Ipad, Apple Ipad, Noise, Apple Ipad, Noise, Trip""".split(",")))
z1.groupby('Items').agg({'Count' : 'sum'}).reset_index().sort_values('Count',ascending=False)
z1 = z1[z1['Items'] != " Noise"]

z1[['Items','Count']]
# Most popular vendors
x=resources['Resource Vendor Name'].value_counts()[:20]
x1 = pd.DataFrame()
x1['Vendor'] = x.index 
x1['Count'] =x.values
x1[['Vendor','Count']]
del resources
# Short Description Based Recommender
projects['Project Short Description'].head()
# Term Frequency-Inverse Document Frequency (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
projects['Project Short Description'] = projects['Project Short Description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(projects['Project Short Description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
tfidf_matrix[0]
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix for that particular index
def get_recommendation(index):
    cosine_sim = linear_kernel(tfidf_matrix[index], tfidf_matrix).flatten()
    return cosine_sim
#Construct a reverse map of indices and project titles
indices = pd.Series(projects.index, index=projects['Project ID']).drop_duplicates()
# related projects to the project at index=0
cosine_sim=get_recommendation(0)
projects.loc[list(cosine_sim.argsort()[:-5:-1])]
# collaborative filtering based model
# finding the weighed score by Inverse Donor Frequency
grouped=donations[['Project ID','Donor ID','Donation Amount']]
#grouped['Donation Amount']=grouped['Donation Amount']/grouped['Donation Amount'].max()
grouped=grouped.groupby(['Project ID','Donor ID'],as_index=False)
# using donor frequency

score=grouped['Donation Amount'].agg(np.mean)

# normalized values
score['Donation Amount']=score['Donation Amount']/score['Donation Amount'].std()
score.head(5)
score[1:10].pivot(index='Project ID', columns='Donor ID', values='Donation Amount').fillna(0).as_matrix()
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

## SVD++
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise import evaluate

def SVDPP(data):
    # Use the famous SVD algorithm.
    algo = SVDpp()
    #data = Dataset.load_builtin(merged[['Project ID','Donor ID','score']][1:100])
    
    reader=Reader()
    data=Dataset.load_from_df(data, reader)    

    kf = KFold(n_splits=2)

    algo = SVDpp()
    evaluate(algo, data, measures=['RMSE', 'MAE'])
    # train and test algorithm.
    trainset = data.build_full_trainset()
    algo.train(testset)
    predictions = algo.test(trainset)

    top_n=get_top_n(predictions, n=2)


    for x,y in top_n.items():
        print(x,[z for (z,_) in y])


merged= pd.merge(projects[['Project Short Description','Project ID','Teacher ID','Project Title','Project Type']], donations[['Donor ID','Project ID']], on='Project ID')
merged=pd.merge(merged, donors[['Donor ID','Donor State','Donor City']], on='Donor ID')
merged=pd.merge(merged, teachers[['Teacher Prefix','Teacher ID']],on='Teacher ID')
# drop all rows with Nan values
merged=merged.dropna()
merged.head(5)
merged.shape
# using multiple features
merged=merged[1:10000]
def create_soup(x):
    return ' '.join([x['Donor State']]) + ' ' + ' '.join([x['Teacher Prefix']])+ ' ' + ' '.join([x['Project Type']])+ ' ' + ' '.join([x['Project Short Description']]) + ' ' + ' '.join([x['Project Title']]) + ' ' + ' '.join([x['Donor City']])
merged['soup'] = merged.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(merged['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendation(index):
    cosine_sim2 = cosine_similarity(count_matrix[index], count_matrix)
    return cosine_sim2
# Reset index of your main DataFrame and construct reverse mapping as before
merged = merged.reset_index()
indices = pd.Series(merged.index, index=merged['Project Title'])
#indices = pd.Series(merged.index, index=merged['Project ID']).drop_duplicates()
# related projects to the project at index=0
cosine_sim2=get_recommendation(0)
merged.loc[cosine_sim2.argsort()[0][:-5:-1]]
from surprise import Reader, Dataset, SVD, evaluate
def hybrid(userId, projectId):
    merged=pd.merge(merged,score[['Project ID','Donation Amount']],on='Project ID')
    #merged.rename(columns={'Donation Amount':'Score'})
    idx = indices[title]
        
    sim_scores = list(enumerate(cosine_sim2[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    project = merged.iloc[movie_indices]
    project['est'] = project['Project ID'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['Project ID']).est)
    project = project.sort_values('est', ascending=False)
    return project.head(10)