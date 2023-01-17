# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns #for plotting in seaborn



import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

regions=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
df.info()
df.head()
column_names=df.columns.tolist()

print('number of null values.... \n',df[column_names].isna().sum())
df['Medal'].fillna(value="No Medal",inplace=True)

#This is a sensible assumption

print('number of null values.... \n',df[column_names].isna().sum())
regions.head()

#We can merge on NOC but lets check if we have the same NOCS
regions['NOC'][regions['NOC'].isin(df['NOC'].unique().tolist())== False]

#Before you merge 2 files is always good to do a sanity check on it.... 

#Singapore isnt in my dataset....... seems odd, Lets search for Singapore in the other dataset just to double check
df[df['Team'].str.contains('.ingapore' ,regex=True)]
print("The different teams fron singapore:",df[df['Team'].str.contains('.ingapore' ,regex=True)]['Team'].unique(),'\n')

print('The current NOC code for these Teams:' ,df[df['Team'].str.contains('.ingapore' ,regex=True)]['NOC'].unique(),'\n')

#here we can see that there are 3 different teams with one NOC

#finally lets double check the regions dataset

print('number of SGP NOCs in our main dataset:' ,sum(regions.NOC=='SGP'),'(This will have to be replaced)','\n')

#None with this NOC

regions.NOC.replace({'SGP':'SIN'}, inplace=True)

#let replace this'

print("It has now been renamed: \n",regions[regions.NOC=='SIN'],'\n')

print(regions[regions.region.isna()],"\n", 'These ')

#these wont be an issue
dfm=df.merge(right=regions, how='left', on='NOC')

dfm.head()
#lets take a look at the 

e=dfm.notes.unique().tolist()

print(e)

#only want notes values not nan so drop nan wich is index 0 in this array

del e[0]

print(e)
f=dfm.Sport.unique().tolist()

print(f)

#A lot of sports are visible
todrop=['Aeronautics','Alpinism','Motorboating','Art Competitions',]

dfm=dfm[~dfm.Sport.isin(todrop)]

#thats all the cleaning done!!
tmp=dfm.groupby(by=['Year','City','Season'])['ID'].agg('count')

data=pd.DataFrame({'Number of Athletes':tmp}, index=tmp.index).reset_index()

datsum=data[data['Season']=='Summer']

datwint=data[data['Season']=='Winter']

data.head()
gsum=go.Scatter(

    x=datsum['Year'],y=datsum['Number of Athletes'],name="Summer Games",

    marker=dict(color="Red"),

    mode = "markers+lines",

    text=datsum['City']

    

)

gwint=go.Scatter(

    x=datwint['Year'],y=datsum['Number of Athletes'],name="Winter Games",

    marker=dict(color="Blue"),

    mode = "markers+lines",

    text=datwint['City']

)



Graphs=[gsum,gwint]

#PLoty is javascript and will only take args passed as dictonaries

layout = dict(title = 'Athlets per Olympic game',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Number of athlets'),

          hovermode = 'closest'

         )

fig=dict(data=Graphs,layout=layout)

g=iplot(fig,filename='events-athlets2')



#Notice how not assigning the graphs to graph objects makes it difficult to change thelayout of graphs and customise quickly this is something i will do in as i go on

#Atheletes in each summer olimpic game and number of sports in each olimpic game seasomn

dfsummer=dfm[dfm['Season']=='Summer']

dfwinter=dfm[dfm['Season']=='Winter']

tmp1=dfsummer[['Year','City']]

dfs=dfsummer.groupby(by=['Year','City'])['Sport'].nunique().reset_index()

dfw=dfwinter.groupby(by=['Year','City'])['Sport'].nunique().reset_index()

tmp2=go.Figure()



tmp2.add_trace(go.Scatter(

    x=dfs['Year'],y=dfs['Sport'],name="Summer Games",

    marker_color="Red",

    mode = "markers+lines",text=dfs['City'])

              )

tmp2.add_trace(go.Scatter(

    x=dfw['Year'],y=dfw['Sport'],name="Winter games",

    marker_color="Blue",

    mode = "markers+lines",

text=dfw['City'])

              )

tmp2.update_layout(

    dict(title_text='Number of sports in each games',hovermode='closest',

   xaxis_title='Year', yaxis_title='Number of sports'),

        title={'text': "Number of sports per Olimpics",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

tmp2.show()

#number of events in each sport

#lets ue our filtered dataframes from earlier

#tmps=dfsummer.groupby(by=['Sport','Year'])['Event'].nunique().reset_index()

tmps=pd.pivot_table(dfsummer,index=['Sport','Year'], fill_value=0,values='Event',aggfunc=lambda x: x.nunique()).reset_index()

indexeds=df[df['Season']=='Summer'].Year.unique().tolist()

#this is great an all but it leave us with missing o



indexeds=np.sort(indexeds)

indexeds=pd.DataFrame(indexeds, index=indexeds)

indexeds=indexeds.rename(columns={0:'Year'})

#notice how archery goes from 1912 to 1920, This will give us nan values in our ploty heatmap!
tmps3=pd.DataFrame()

tmps2=pd.DataFrame()

looper=tmps.Sport.unique()

for x in looper:

    tmps2=indexeds.merge(right=tmps[tmps['Sport']==x],how='outer',on='Year')

    #reassigns the sport column to the name of the sport

    tmps2['Sport']=x

    tmps3=tmps3.append(tmps2,sort=False)

#this  puts the data into a dataframe of length of the number of years for each sport and fills the 0s

tmps3=tmps3.fillna(0)

print(tmps3)



fig=go.Figure(

    go.Heatmap(x=tmps3['Year'],y=tmps3['Sport'],z=tmps3['Event'],

hovertemplate =

    '<br>Sport</b>: &nbsp; %{y}'+

    '<br><b>Year</b> </b>: &nbsp; %{x}<br>'+

    '<b>NO Event\'s</b></b>: &nbsp; %{z}<br>'))



fig.update_layout(title='Number of Events Per Sport',  width=700,

    height=1050,title_x=0.5)

fig.show()



#median age of medal winners

tmp=dfm.query('Season=="Summer" & Medal!="No Medal"')

tmp2=pd.pivot_table(tmp, index=['Year', 'Medal'], values='Age', aggfunc=np.median).reset_index()

tmp3=pd.pivot_table(tmp, index=['Year', 'Medal'], values='Age', aggfunc=np.mean).reset_index()

print(tmp2)

from plotly.subplots import make_subplots 



fig = make_subplots(rows=2, cols=1, subplot_titles=['Median Age of Medal Winners', 'Mean Age of Medal Winners'])

fig.add_trace(

    go.Heatmap(x=tmp2['Year'],y=tmp2['Medal'],z=tmp2['Age'],coloraxis = 'coloraxis1',

hovertemplate =

    '<br>Medal</b>: &nbsp; %{y}'+

    '<br><b>Year</b> </b>: &nbsp; %{x}<br>'+

    '<b>Median age\'s</b></b>: &nbsp; %{z}<br>'),row=1,col=1)

#coloraxis1 is an abitary colorscale thathas been made to allow the heatmaps to share a colorscale 

fig.add_trace(

    go.Heatmap(x=tmp3['Year'],y=tmp3['Medal'],z=tmp3['Age'],coloraxis = 'coloraxis1',

hovertemplate =

    '<br>Medal</b>: &nbsp; %{y}'+

    '<br><b>Year</b> </b>: &nbsp; %{x}<br>'+

    '<b>Mean age\'s</b></b>: &nbsp; %{z}<br>'),row=2,col=1)

fig.update_layout(title_x=0.1,title_text="Age Distribution of Atheletes",coloraxis1 = {'colorscale':'Electric'})

order=['Bronze', 'Silver','Gold']

fig.update_layout(yaxis={'categoryarray': order}

                 )

fig.show()



#This code gave my data all in the wrong order. This is a good oppotunity to make the medal row categorical, this will allow you to keep the pre defined order when the data is pivoted
tmp=dfm.query('Season=="Summer" & Medal!="No Medal"')

tmp2=pd.pivot_table(tmp, index=['Year', 'Medal'], values='Age', aggfunc=np.median).reset_index()

tmp3=pd.pivot_table(tmp, index=['Year', 'Medal'], values='Age', aggfunc=np.mean).reset_index()

tmp2['Medal']=tmp2['Medal'].astype('category')

tmp2.Medal.cat.reorder_categories(['Gold','Silver','Bronze'], inplace=True)



fig=go.Figure()

fig.add_trace(

    go.Heatmap(x=tmp2['Year'],y=tmp2['Medal'],z=tmp2['Age'],coloraxis = 'coloraxis1',

hovertemplate =

    '<br>Medal</b>: &nbsp; %{y}'+

    '<br><b>Year</b> </b>: &nbsp; %{x}<br>'+

    '<b>Median age\'s</b></b>: &nbsp; %{z}<br>') 

)

order=['Bronze', 'Silver','Gold']

fig.update_layout(yaxis={'categoryarray': order},width=800,height=300

                 )

fig.show()



#dfg=df[df['Year']==2000]

#fig=go.Figure(go.Violin(x=dfg['Year'][dfg['Sex'] == 'M' ],

                       # y=dfg['Age'][dfg['Sex'] == 'M' ],

                        #legendgroup='Yes', scalegroup='M', name='M',

                        #side='negative',

                        #line_color='blue'))



#fig.add_trace(go.Violin(x=dfg['Year'][dfg['Sex'] == 'F' ],

                       # y=dfg['Age'][dfg['Sex'] == 'F' ],

                        #legendgroup='Yes', scalegroup='F', name='F',

                        #side='positive',

                      #  line_color='Red'))

#fig.update_traces(meanline_visible=True)

#fig.update_layout(violinmode='overlay',violingap=0)

#fig.show()
summer=dfm[dfm['Season']=='Summer']

years=summer.Year.unique().tolist()

years=np.sort(years)

#print(years)

genders=['M','F']

gencol={"M":'Blue','F':'Red'}

genside={"M":'negative','F':'positive'}

fig=go.Figure()

for yr in years:

    tmp=summer[summer['Year']==yr]

    for gen in genders:

        fig.add_trace(go.Violin(x=tmp['Year'][tmp['Sex'] == gen ],

                        y=tmp['Age'][tmp['Sex'] == gen], scalegroup=gen, name=gen,legendgroup=gen,

                        side=genside.get(gen),

                        line_color=gencol.get(gen), showlegend=False))





fig.update_traces(meanline_visible=True)

fig.update_layout(violinmode='overlay',violingap=0)

fig.show()



        
print('Number of unique athlete names=', df[df['Year']==2012].Name.nunique())

print('Number of events=', df[df['Year']==2012]['ID'].count())

print('Number of Athletes who competed in a 2nd event: ~',df[df['Year']==2012]['Name'].duplicated(keep='first').sum(),

"\nThis doesnt account for athletes with identical names but the number of unique names can be seen as a good estimate of the number of athletes as the same first and last name would be unlikely...")

#Sanitiy check as you go 

smrratio=dfsummer.groupby(by=['Year','Sex'])['Name'].agg(lambda x: x.nunique()).reset_index()

print(smrratio.head())

check=dfsummer.groupby(by=['Year','Sex'])['Name'].agg('count').reset_index()

print(check.head())

#1900 male dataset shall be the chcck

print(dfm.query('Year==1900 & Sex=="M"')['Name'].nunique())

print(dfm.query('Year==1900 & Sex=="M"')['Name'].count())

smrratio=smrratio.pivot_table(smrratio, index=['Year', 'Sex'],aggfunc=sum).fillna(0)

smrratio=smrratio.unstack().fillna(0)

smrratio=smrratio.reset_index()
Female=smrratio['Name','F']

Male=smrratio['Name','M']

ratio=Male.div(Female,level=1)

ratio=ratio.replace([np.inf],np.nan)

ratio=ratio.fillna(Male)

ratio.index=smrratio.Year
fig=go.Figure()

fig.add_trace(go.Scatter(x=ratio.index, y=ratio,mode='markers'))

fig.update_layout(title='Gender Ratios for the Summer Olimpics',title_x=.5)

fig.update_yaxes(range=[-5, 180])

fig.update_yaxes(tick0=0, dtick=10,zeroline=True, zerolinewidth=2, zerolinecolor='LightPink',title="Gender ratio Male:Female")

fig.show()

              
#top 10 sports in uk

gbrdata=dfm[(dfm.NOC.str.contains('GBR') | dfm.region.str.contains('UK'))]

gbrdata=gbrdata[(gbrdata['Medal']!='No Medal')&(gbrdata['Season']=='Summer')]

gbrdata.head()

tmp1=gbrdata.groupby(by=['Year','Medal'])['Sport'].agg('count').fillna(0).reset_index()

print(tmp1)
fig=go.Figure(

    go.Heatmap(x=tmp1['Year'],y=tmp1['Medal'],z=tmp1['Sport']))

order=['Bronze', 'Silver','Gold']

fig.update_layout(yaxis={'categoryarray': order}

                 )

fig.show()