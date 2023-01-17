import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Plotly Libraris

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

# Minmax scaler

from sklearn.preprocessing import MinMaxScaler



#itertools

import itertools



#dataframe display settings

pd.set_option('display.max_columns', 5000000)

pd.set_option('display.max_rows', 50000000)



#to suppress un-necessary warnings

import warnings  

warnings.filterwarnings('ignore')
#Importing files

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding = "ISO-8859-1")

df.head()
desc = ['Static hyperlink for the startup on Crunchbase\'s website','name of the startup','Website address of the startup',

       'in which category the startups fall','which market the startup caters to','total funding received(in USD)',

        'current operating status','country of origin','state of origin','region','city of origin','total rounds of funding',

        'date of founding','month of founding','quarter of founding','year of founding','date of first funding','date of last funding',

        'seed funding received(in USD)','venture funding received(in USD)','funding received by diluting equity',

        'other undisclosed funding sources','funding received from convertible notes','funding received from debts',

        'funding received from angel investors','funding from grants','funding from private equity',

        'funding from equity dilution after IPO','funding from debts after IPO','funding from secondary markets',

        'funding from crowdfunding','round A funding','round B funding','round C funding','round D funding','round E funding',

       'round F funding']

df_details = pd.DataFrame(list(zip(df.columns, desc)), columns =['Column', 'Description'])

df_details
print(df.shape)

df.describe()
#deleting duplicate rows.

df = df.drop_duplicates()

print(df.shape)
#cleaning the dataframe by dropping uneccessary columns

df = df.drop(['permalink', 'homepage_url'], axis=1)

#Removing the row with no 'name'

df.dropna(how='any', subset=['name'], axis=0, inplace=True)

#Extracting year value from "first_funding_at" and changing to int

df['first_funding_at'] = df.first_funding_at.str.split("-").str[0]

df['first_funding_at'] = df['first_funding_at'].astype(int)

#Extracting year value from "last_funding_at" and changing to int

df['last_funding_at'] = df.last_funding_at.str.split("-").str[0]

df['last_funding_at'] = df['last_funding_at'].astype(int)

#Changing the values in column "funding_total_usd" from string to float

df[' funding_total_usd '] = df[' funding_total_usd '].str.strip().str.replace(",","")

df[' funding_total_usd '] = df[' funding_total_usd '].replace("-",0).astype("float")

#Replacing missing status with "unknown"

df['status'] = df['status'].replace(np.nan,"unknown")

print(df.shape)

df.head()
#cheking funding before 1902

print(df[df['first_funding_at']<1902][['name', 'founded_at', 'first_funding_at']])

df.drop(df[df['first_funding_at']<1902].index, inplace=True)

df.shape
fig_founded_funding = go.Figure()

fig_founded_funding.add_trace(go.Histogram(x=df['founded_year'], name="Founded year", marker=dict(opacity=0.9),

                                          hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                                          ))

fig_founded_funding.add_trace(go.Histogram(x=df['first_funding_at'], name="First Funding Year", marker=dict(opacity=0.5),

                                          hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                                          ))

fig_founded_funding.update_layout(barmode='overlay',

                                 title="Overall Relation between starting of Startups and starting of Funding.",

                                 xaxis_title='Year', yaxis_title="Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))



# Add range slider

fig_founded_funding.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=10,

                     label="10y",

                     step="year",

                     stepmode="backward"),

                dict(count=20,

                     label="20y",

                     step="year",

                     stepmode="backward"),

                dict(count=50,

                     label="50y",

                     step="year",

                     stepmode="todate"),

                dict(count=100,

                     label="100y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig_founded_funding.show()
#Making the list of some of the most famous unicorns

df_name_index = df.set_index('name', drop=True)

startup_unicorns = ["Uber","Amazon","Google","Dropbox","Facebook","Alibaba",

                     "Stripe","Airbnb","Robinhood","Reddit",

                    "DigitalOcean","Coursera"]

color = ['Black','Orange','Blue','Darkblue', 'lightblue', 'darkorange','teal',

         'red','lightgreen','orange','blue','lightblue']

unicorn_founding_year = []

total_funding=[]

seed = []

vc = []

#Extracting its details from the dataset

for i in startup_unicorns:

    unicorn_founding_year.append(int(df_name_index.loc[i]['founded_year']))

    total_funding.append(int(df_name_index.loc[i][' funding_total_usd ']))

    seed.append(int(df_name_index.loc[i]['seed']))

    vc.append(int(df_name_index.loc[i]['venture']))        

df_unicorns = pd.DataFrame(list(zip(startup_unicorns, unicorn_founding_year, total_funding, seed, vc, color)),

                           columns=['Unicorn name', 'Founding year','Total funding', 'Seed', 'Venture Capital', 'Color']).sort_values(by='Seed')

df_unicorns.head()
fig_unicorn_founded = go.Figure()

fig_unicorn_founded.add_trace(go.Histogram(x=df[df['founded_year']>1990]['founded_year'], name="Founded year", marker=dict(opacity=0.9),

                                          hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                                          ))



for i in df_unicorns['Unicorn name']:

    fig_unicorn_founded.add_shape(

            # Line Vertical

            dict(

                type="line",

                xref="x",

                yref="paper",

                x0=str(list(df_unicorns[df_unicorns['Unicorn name']==i]['Founding year'])[0]),

                y0=0,

                x1=str(list(df_unicorns[df_unicorns['Unicorn name']==i]['Founding year'])[0]),

                y1=1,

                line=dict(

                    color=str(list(df_unicorns[df_unicorns['Unicorn name']==i]['Color'])[0]),

                    width=1

                )

    ))

    

fig_unicorn_founded.add_trace(go.Scatter(x=df_unicorns['Founding year'],

                                        y=df_unicorns.index*700+300, mode="text", text=df_unicorns[['Unicorn name']], 

                                        textfont=dict(family="sans serif",size=15), showlegend=False,

                                        hovertemplate = '<br><b>Company</b>: %{text}'+'<br><i>Founding Year</i>: %{x}'

                                        ))



fig_unicorn_founded.update_layout(title="Years in which some Startup Unicorns were founded",

                             xaxis_title="Year", yaxis_title="Startup Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))

fig_unicorn_founded.show()
fig_country = go.Figure()

fig_country.add_trace(go.Bar(x=df['country_code'].value_counts().index[:20], y=df['country_code'].value_counts()[:20],

                           hovertemplate = '<br><b>Country</b>: %{x}'+'<br><i>Startup count</i>: %{y}',

                           marker=dict(color=list(range(20)), colorscale="Sunsetdark")))



fig_country.update_layout(title="Number of Startups in each Country",

                             xaxis_title="Country", yaxis_title="Startup Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_country.show()
fig_country_map = go.Figure()

fig_country_map.add_trace(go.Choropleth(locations=df['country_code'].value_counts().index,

                                       z=df['country_code'].value_counts(),

                                       colorscale='Peach',

                                       colorbar_title="Nos. of Startups founded",

                                       ))

fig_country_map.update_layout(title_text="Number of Startups Country wise.", title_x=0.5, title_font_size=20, paper_bgcolor="mintcream")

fig_country_map.show()
fig_USA_region = go.Figure()

fig_USA_region.add_trace(go.Bar(x=df[df.country_code == 'USA']['state_code'].value_counts().index[:20], y=df[df.country_code == 'USA']['state_code'].value_counts()[:20],

                            hovertemplate = '<br><b>Region</b>: %{x}'+'<br><i>Startup count</i>: %{y}',

                           marker=dict(color=list(range(20)), colorscale="Sunsetdark")))



fig_USA_region.update_layout(title="Number of Startups in each region in USA",

                             xaxis_title="Region", yaxis_title="Startup Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_USA_region.show()
fig_USA_region_map = go.Figure()

fig_USA_region_map.add_trace(go.Choropleth(locations=df[df.country_code == 'USA']['state_code'].value_counts().index,

                                       z=df[df.country_code == 'USA']['state_code'].value_counts(),

                                       locationmode="USA-states",

                                       colorscale='Peach',

                                       colorbar_title="Nos. of Startups founded"))

fig_USA_region_map.update_layout(title_text="Number of Startups State wise in USA.", title_x=0.5, title_font_size=20,

                                        geo = dict(

                                        scope='usa',

                                        projection=go.layout.geo.Projection(type = 'albers usa'),

                                        showlakes=True, # lakes

                                        lakecolor='rgb(255, 255, 255)'),

                                        paper_bgcolor="mintcream")

fig_USA_region_map.show()
fig_market = go.Figure()

fig_market.add_trace(go.Bar(x=df[' market '].value_counts().index[:30], y=df[' market '].value_counts()[:30],

                           hovertemplate = '<br><b>Market</b>: %{x}'+'<br><i>Startup count</i>: %{y}',

                           marker=dict(color=list(range(30)), colorscale="Sunsetdark")))



fig_market.update_layout(title="Number of Startups in each Market",

                             xaxis_title="Market", yaxis_title="Startup Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_market.show()
fig_closed = make_subplots(rows=1, cols=2, shared_yaxes=True,

                           subplot_titles=("Markets with Most closed Startups", "Markets with Least closed Startups"))

fig_closed.add_trace(go.Bar(x=df[df['status']=='closed'][' market '].value_counts()[:10].index,

                            y=df[df['status']=='closed'][' market '].value_counts()[:10], name="Market with most closed Startups",

                            marker=dict(color=list(range(20)), colorscale="reds_r"),

                           hovertemplate = '<br><b>Market</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                           ),

                            row=1, col=1)

fig_closed.add_trace(go.Bar(x=df[df['status']=='closed'][' market '].value_counts()[-10:].index,

                            y=df[df['status']=='closed'][' market '].value_counts()[-10:], name="Market with least closed Startups",

                            marker=dict(color=list(range(20)), colorscale="greens_r"),

                           hovertemplate = '<br><b>Market</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                           ),

                            row=1, col=2)



fig_closed.update_layout(showlegend=False, paper_bgcolor="mintcream")

fig_closed.show()
fig_acquired = make_subplots(rows=1, cols=2, shared_yaxes=True,

                           subplot_titles=("Markets with Most acquired Startups", "Markets with Least acquired Startups"), )

fig_acquired.add_trace(go.Bar(x=df[df['status']=='acquired'][' market '].value_counts()[:10].index,

                            y=df[df['status']=='acquired'][' market '].value_counts()[:10],

                            marker=dict(color=list(range(20)), colorscale="greens_r"),

                             hovertemplate = '<br><b>Market</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                             ),

                            row=1, col=1)

fig_acquired.add_trace(go.Bar(x=df[df['status']=='acquired'][' market '].value_counts()[-10:].index,

                            y=df[df['status']=='acquired'][' market '].value_counts()[-10:],

                            marker=dict(color=list(range(20)), colorscale="reds_r"),

                             hovertemplate = '<br><b>Market</b>: %{x}'+'<br><i>Startup count</i>: %{y}'

                             ),

                            row=1, col=2)



fig_acquired.update_layout(showlegend=False, paper_bgcolor="mintcream")

fig_acquired.show()
fig_funding_amt = px.scatter(df[:5000],x="name", y="funding_rounds", size=' funding_total_usd ', color='status')



fig_funding_amt.update_layout(title='Plot Showing the Funding and Total funding acquired by Startups',

                              xaxis_title="Startups",yaxis_title="Funding Rounds",

                            xaxis_showticklabels=False,paper_bgcolor="mintcream",

                             title_font_size=20, title_x=0.5,legend=dict(orientation='h',yanchor='top',y=1.08,xanchor='right',x=1))



#fig_funding_amt.update_traces(hovertemplate = '<br><b>Company</b>: %{x}'+'<br><i>Funding Rounds</i>: %{y}'+'<br><i>Funding(in USD)</i>: %{marker.size}')

fig_funding_amt.show()
fig_status = make_subplots(rows=2, cols=2, specs=[[{"type": "domain", "colspan": 2}, None],[{"type": "domain"}, {"type": "domain"}]],

                          subplot_titles = ("Current status of all Startups", "Status of Startups founded before 2000", 

                                            "Status of Startups founded after 2000"))



fig_status.add_trace(go.Pie(labels=df['status'].value_counts().index, values=df['status'].value_counts()), row=1, col=1)



fig_status.add_trace(go.Pie(labels=df[df['founded_year']<2000]['status'].value_counts().index, values=df[df['founded_year']<2000]['status'].value_counts()), row=2, col=1)



fig_status.add_trace(go.Pie(labels=df[df['founded_year']>=2000]['status'].value_counts().index, values=df[df['founded_year']>=2000]['status'].value_counts()), row=2, col=2)



fig_status.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=11,

                         insidetextorientation='horizontal', rotation=-45)

fig_status.update_layout(height=800, paper_bgcolor="mintcream")

fig_status.show()
############Test this while saving the first verison if this works fine#####################



# value = input("Enter ")

# value = int(value.split(" ")[0])

# fig_funding_amt = px.scatter(df[:value],x="name", y="funding_rounds", size=' funding_total_usd ', color='status')



# fig_funding_amt.update_layout(title='Plot Showing the Funding and Total funding acquired by Startups',

#                               xaxis_title="Startups",yaxis_title="Funding Rounds",

#                             xaxis_showticklabels=False)

# fig_funding_amt.show()
#creating dataframe with details of receiving seed and VC funding

df_seed_vc = pd.DataFrame(columns=['name','Founded','Seed','Venture Capital'])

df_seed_vc[['name','Founded','Seed','Venture Capital']] = df[['name','founded_year','seed','venture']]

              

df_seed_funding = df_seed_vc.dropna(subset=['Founded'])

df_seed_funding.reset_index(inplace=True, drop=True)

df_seed_funding = df_seed_funding.dropna(subset=['Seed'])

df_seed_funding.reset_index(inplace=True, drop=True)

bins = [1902, 1980, 2000, 2014]

labels = ['Founded Before 1980','Founded between 1980-2000','Founded after 2000']

df_seed_funding['Founded'] = pd.cut(df_seed_funding['Founded'], bins=bins, labels=labels)

df_seed_funding = df_seed_funding.dropna(subset=['Founded'])

df_seed_funding.reset_index(inplace=True, drop=True)

df_seed_funding['Seed'] = df_seed_funding['Seed'].apply(lambda x: "Got seed funding" if x!=0 else "Didn't get seed funding")

#######################

df_seed_funding['Venture Capital'] = df_seed_funding['Venture Capital'].apply(lambda x: "Got VC funding" if x!=0 else "Didn't get VC funding") 

#########################

df_seed_funding.head()
fig_seed = px.sunburst(df_seed_funding, path=['Founded','Seed'], color='Founded')

fig_seed.update_layout(title="Seed Funding distribution for Startups founded over the time", title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_seed.show()



#creating dataframe with avg amount of seed, VC and angel funding received by startups



df_yearly_seed = pd.DataFrame(columns=['Year','Avg seed','Avg Venture Capital','Avg Angel funding'])

df_yearly_seed['Year'] =df['first_funding_at'].unique()

df_yearly_seed.sort_values("Year",inplace = True, na_position ='first')

df_yearly_seed.reset_index(inplace=True, drop=True)

df_yearly_seed[['Avg seed','Avg Venture Capital','Avg Angel funding']] = df_yearly_seed[['Year',"Year",'Year']]

df_yearly_seed['Avg seed'] = df_yearly_seed['Avg seed'].apply(lambda x: df[df['first_funding_at']==x][df['seed']!=0]['seed'].mean() if x>0 else 0)

df_yearly_seed['Avg Venture Capital'] = df_yearly_seed['Avg Venture Capital'].apply(lambda x: df[df['first_funding_at']==x][df['venture']!=0]['venture'].mean() if x>0 else 0)

df_yearly_seed['Avg Angel funding'] = df_yearly_seed['Avg Angel funding'].apply(lambda x: df[df['first_funding_at']==x][df['angel']!=0]['angel'].mean() if x>0 else 0)

#df_yearly_seed.head()
fig_yearly_seed = go.Figure()

fig_yearly_seed.add_trace(go.Scatter(x=df_yearly_seed['Year'], y=df_yearly_seed['Avg seed'],

                                   mode='lines+markers', name="Seed funding per year",

                                    hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Seed Funding</i>: %{y}'

                                    ))

fig_yearly_seed.add_shape(

        # Horizontal Vertical

        dict(

            type="line",xref="paper",yref="y",

            x0=0,y0=df[df['seed']!=0]['seed'].mean(),

            x1=1,y1=df[df['seed']!=0]['seed'].mean(),

            line=dict(color="blue",width=1,dash="dashdot"

            )))

fig_yearly_seed.add_trace(go.Scatter(x=list(range(1930,2010,20)),y=list(itertools.repeat(df[df['seed']!=0]['seed'].mean()+50000, 5)),

                                    mode="text", text="Avg Seed Funding", showlegend=False, hoverinfo='skip'))



fig_yearly_seed.update_layout(title="Seed funding trend over the years",

                             xaxis_title="Years", yaxis_title="Seed Funding(in USD)",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))

fig_yearly_seed.show()

fig_seed_unicorns = go.Figure(go.Bar(x=df_unicorns['Unicorn name'], y=df_unicorns['Seed'], name="Seed Funding",

                                    hovertemplate = '<br><b>Company</b>: %{x}'+'<br><i>Seed Funding</i>: %{y}'

                                    ))

fig_seed_unicorns.add_shape(

        # Horizontal Vertical

        dict(

            type="line",xref="paper",yref="y",

            x0=0,y0=df[df['seed']!=0]['seed'].mean(),

            x1=1,y1=df[df['seed']!=0]['seed'].mean(),

            line=dict(color="blue",width=1,dash="dashdot"

            )))

fig_seed_unicorns.add_trace(go.Scatter(x=['Google','Coursera'],y=list(itertools.repeat(df[df['seed']!=0]['seed'].mean()+50000, 2)),

                                    mode="text", text="Avg Seed Funding", showlegend=False, hoverinfo='skip'))



fig_seed_unicorns.update_layout(title="Seed Funding Recieved by Unicorns",

                             xaxis_title="Year", yaxis_title="Funding(in USD)",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))

fig_seed_unicorns.show()
fig_vc = px.sunburst(df_seed_funding, path=['Founded','Venture Capital'], color='Founded')

fig_vc.update_layout(title="Venture Capital Funding distribution for Startups founded over the time",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_vc.show()

fig_yearly_vc = go.Figure()

fig_yearly_vc.add_trace(go.Scatter(x=df_yearly_seed['Year'].unique(), y=df_yearly_seed['Avg Venture Capital'],

                                   mode='lines+markers', name="VC funding per year",

                                  hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>VC Funding</i>: %{y}'

                                  ))



fig_yearly_vc.add_shape(

        # Horizontal Vertical

        dict(

            type="line",xref="paper",yref="y",

            x0=0,y0=df[df['venture']!=0]['venture'].mean(),

            x1=1,y1=df[df['venture']!=0]['venture'].mean(),

            line=dict(color="blue",width=1,dash="dashdot"

            )))

fig_yearly_vc.add_trace(go.Scatter(x=list(range(1930,2010,20)),y=list(itertools.repeat(df[df['venture']!=0]['venture'].mean()+300000, 5)),

                                    mode="text", text="Avg VC Funding", showlegend=False, hoverinfo="skip"))



fig_yearly_vc.update_layout(title="Venture Capital funding trend over the years",

                             xaxis_title="Year", yaxis_title="VC Funding(in USD)",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))

fig_yearly_vc.show()
df_unicorns = df_unicorns.sort_values(by='Venture Capital')

fig_vc_unicorns = go.Figure(go.Bar(x=df_unicorns['Unicorn name'], y=df_unicorns['Venture Capital'], name="Venture Capital Funding",

                                    hovertemplate = '<br><b>Company</b>: %{x}'+'<br><i>Total VC Funding</i>: %{y}'

                                  ))

fig_vc_unicorns.add_shape(

        # Horizontal Vertical

        dict(

            type="line",xref="paper",yref="y",

            x0=0,y0=df[df['venture']!=0]['venture'].mean(),

            x1=1,y1=df[df['venture']!=0]['venture'].mean(),

            line=dict(color="red",width=1,dash="dashdot"

            )))

fig_vc_unicorns.add_trace(go.Scatter(x=['Alibaba','Robinhood'],y=list(itertools.repeat(df[df['venture']!=0]['venture'].mean()+20000000, 2)),

                                    mode="text", text="Avg VC Funding", showlegend=False, hoverinfo="skip"))



fig_vc_unicorns.update_layout(title="VC Funding Recieved by Unicorns",

                             xaxis_title="Year", yaxis_title="Funding(in USD)",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))

fig_vc_unicorns.show()
fig_yearly_avs = go.Figure()

fig_yearly_avs.add_trace(go.Scatter(x=df_yearly_seed['Year'].unique(), y=df_yearly_seed['Avg seed'],

                                   mode='lines+markers', name="Seed funding per year",

                                     hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Total Seed Funding</i>: %{y}'

                                   ))

fig_yearly_avs.add_trace(go.Scatter(x=df_yearly_seed['Year'].unique(), y=df_yearly_seed['Avg Venture Capital'],

                                   mode='lines+markers', name="VC funding per year",

                                     hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Total VC Funding</i>: %{y}'

                                   ))

fig_yearly_avs.add_trace(go.Scatter(x=df_yearly_seed['Year'].unique(), y=df_yearly_seed['Avg Angel funding'],

                                   mode='lines+markers', name="Angel funding per year",

                                     hovertemplate = '<br><b>Year</b>: %{x}'+'<br><i>Total Angel Funding</i>: %{y}'

                                   ))

fig_yearly_avs.update_layout(title="Funding trend over the years",

                             xaxis_title="Year", yaxis_title="Funding(in USD)",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20,legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1))





fig_yearly_avs.show()
fig_round = go.Figure()

fig_round.add_trace(go.Bar(x=pd.DataFrame(df.columns[29:37]).apply(lambda x: df[x][df[x]!=0].mean()).index,

                          y=pd.DataFrame(df.columns[29:37]).apply(lambda x: df[x][df[x]!=0].mean())[0],

                          hovertemplate = '<br><b>Round</b>: %{x}'+'<br><i>Total Funding</i>: %{y}',

                          marker=dict(color=list(range(8)), colorscale="Sunsetdark")))



fig_round.update_layout(title="Funding in Each Round", title_x=0.5, title_font_size=20,

                        paper_bgcolor="mintcream")

fig_round.show()

#Total startups

total_india = df[df['country_code']=="IND"]['name'].shape[0]

total_us = df[df['country_code']=="USA"]['name'].shape[0]

#Total startups with seed funding

seed_funded_us = df[df['country_code']=="USA"][df['seed']!=0].shape[0]

seed_funded_ind = df[df['country_code']=="IND"][df['seed']!=0].shape[0]

#Total seed funding

total_seed_us = df[df['country_code']=="USA"]['seed'].sum()/1000000

total_seed_ind = df[df['country_code']=="IND"]['seed'].sum()/1000000

#Total startups with VC funding

vc_funded_us = df[df['country_code']=="USA"][df['venture']!=0].shape[0]

vc_funded_ind = df[df['country_code']=="IND"][df['venture']!=0].shape[0]

#Total VC funding

total_vc_us = df[df['country_code']=="USA"]['venture'].sum()/1000000

total_vc_ind = df[df['country_code']=="IND"]['venture'].sum()/1000000

#most famous market strength

market_us = df[df['country_code']=="USA"][' market '].value_counts()[0]

market_ind = df[df['country_code']=="IND"][' market '].value_counts()[0]

#most famous market name

market_us_name = df[df['country_code']=="USA"][' market '].value_counts().index[0]

market_ind_name = df[df['country_code']=="IND"][' market '].value_counts().index[0]
fig_ind_us = go.Figure()

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_india,

                                  title = {'text': "Total startups founded in India"},domain = {'row': 0, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_us,

                                  title = {'text': "Total startups founded in US"},domain = {'row': 0, 'column': 1}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = seed_funded_ind,

                                  title = {'text': "Total seed funded startups in India"},domain = {'row': 1, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = seed_funded_us,

                                  title = {'text': "Total seed funded startups in US"},domain = {'row': 1, 'column': 1}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_seed_ind,

                                  title = {'text': "Total seed fund in India (in million $)"},domain = {'row': 2, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_seed_us,

                                  title = {'text': "Total seed fund in US (in million $)"},domain = {'row': 2, 'column': 1}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = vc_funded_ind,

                                  title = {'text': "Total VC funded startups in India"},domain = {'row': 3, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = vc_funded_us,

                                  title = {'text': "Total VC funded startups in US"},domain = {'row': 3, 'column': 1}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_vc_ind,

                                  title = {'text': "Total VC fund in India (in million $)"},domain = {'row': 4, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = total_vc_us,

                                  title = {'text': "Total VC fund in US (in million $)"},domain = {'row': 4, 'column': 1}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = market_ind,

                                  title = {'text': "Most Famous Market segment in IND: "+market_ind_name},domain = {'row': 5, 'column': 0}))

fig_ind_us.add_trace(go.Indicator(mode = "number",value = market_us,

                                  title = {'text': "Most Famous Market segment in US: "+market_us_name},domain = {'row': 5, 'column': 1}))



fig_ind_us.update_layout(

    grid = {'rows': 6, 'columns': 2, 'pattern': "independent"},font=dict(family='Calibri'), paper_bgcolor="aliceblue",height=500, width=1000, )

fig_ind_us.show()
#initializing 2*2 subplot

fig_indvsus = make_subplots(rows=2, cols=2,

                           subplot_titles = ("Number of Startups","Seed funding Trend","VC funding Trend","Angel funding Trend"),

                           vertical_spacing=0.1, horizontal_spacing=0.04)



#line plot for number of startups in India vs USA

fig_indvsus.add_trace(go.Scatter(x=pd.DataFrame(df[df['country_code']=="IND"]['founded_year'])["founded_year"].value_counts().sort_index().index,

                        y=pd.DataFrame(df[df['country_code']=="IND"]['founded_year'])["founded_year"].value_counts().sort_index(),

                        mode="lines", line=dict(color="orange"), name="India",

                                hovertemplate = '<b>India</b>'+'<br><i>Startup count</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=1, col=1)

fig_indvsus.add_trace(go.Scatter(x=pd.DataFrame(df[df['country_code']=="USA"]['founded_year'])["founded_year"].value_counts().sort_index().index,

                        y=pd.DataFrame(df[df['country_code']=="USA"]['founded_year'])["founded_year"].value_counts().sort_index(),

                        mode="lines", line=dict(color="blue"), name="USA",

                                hovertemplate = '<b>USA</b>'+'<br><i>Startup count</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=1, col=1)



#line plot for seed funding trend in India vs USA

df_tmp = pd.DataFrame(columns=['Year','avg seed'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="IND"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg seed'] = df_tmp['Year']

df_tmp['avg seed'] = df_tmp['avg seed'].apply(lambda x:df[df['country_code']=="IND"][df['first_funding_at']==x][df['seed']!=0]['seed'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg seed'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg seed'],

                        mode="lines+markers", line=dict(color='orange'), name="India", showlegend=False,

                      hovertemplate = '<b>India</b>'+'<br><i>Seed funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=1, col=2)



df_tmp = pd.DataFrame(columns=['Year','avg seed'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="USA"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg seed'] = df_tmp['Year']

df_tmp['avg seed'] = df_tmp['avg seed'].apply(lambda x:df[df['country_code']=="USA"][df['first_funding_at']==x][df['seed']!=0]['seed'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg seed'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg seed'],

                        mode="lines+markers", line=dict(color='blue'), name="USA", showlegend=False,

                                hovertemplate = '<b>USA</b>'+'<br><i>Seed funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=1, col=2)



#line plot for VC funding trend in India vs USA

df_tmp = pd.DataFrame(columns=['Year','avg vc'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="IND"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg vc'] = df_tmp['Year']

df_tmp['avg vc'] = df_tmp['avg vc'].apply(lambda x:df[df['country_code']=="IND"][df['first_funding_at']==x][df['venture']!=0]['venture'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg vc'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg vc'],

                        mode="lines+markers", line=dict(color='orange'), name="India", showlegend=False,

                                hovertemplate = '<b>India</b>'+'<br><i>VC funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=2, col=1)



df_tmp = pd.DataFrame(columns=['Year','avg vc'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="USA"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg vc'] = df_tmp['Year']

df_tmp['avg vc'] = df_tmp['avg vc'].apply(lambda x:df[df['country_code']=="USA"][df['first_funding_at']==x][df['venture']!=0]['venture'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg vc'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg vc'],

                        mode="lines+markers", line=dict(color='blue'), name="USA", showlegend=False,

                                hovertemplate = '<b>USA</b>'+'<br><i>VC funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=2, col=1)



#line plot for angel funding trend in India vs USA

df_tmp = pd.DataFrame(columns=['Year','avg angel'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="IND"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg angel'] = df_tmp['Year']

df_tmp['avg angel'] = df_tmp['avg angel'].apply(lambda x:df[df['country_code']=="IND"][df['first_funding_at']==x][df['angel']!=0]['angel'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg angel'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg angel'],

                        mode="lines+markers", line=dict(color='orange'), name="India", showlegend=False,

                                hovertemplate = '<b>India</b>'+'<br><i>Angel funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=2, col=2)



df_tmp = pd.DataFrame(columns=['Year','avg angel'])

df_tmp['Year'] = pd.DataFrame(df[df['country_code']=="USA"]['first_funding_at'])['first_funding_at'].unique()

df_tmp['avg angel'] = df_tmp['Year']

df_tmp['avg angel'] = df_tmp['avg angel'].apply(lambda x:df[df['country_code']=="USA"][df['first_funding_at']==x][df['angel']!=0]['angel'].mean() if x>0 else 0)

df_tmp.sort_values(by='Year',inplace=True)

df_tmp.dropna(subset=['avg angel'])



fig_indvsus.add_trace(go.Scatter(x=df_tmp['Year'],

                         y=df_tmp['avg angel'],

                        mode="lines+markers", line=dict(color='blue'), name="USA", showlegend=False,

                                hovertemplate = '<b>USA</b>'+'<br><i>Angel funding</i>: %{y}'+'<br><b>Year</b>: %{x}<br>'

                                ), row=2, col=2)



#updating layout of subplot

fig_indvsus.update_layout(title="Comparision between India and USA", title_x=0.5, title_font_size=20, height=600,width=1150,

                              legend=dict(orientation='h',yanchor='top',y=1.1,xanchor='right',x=1), paper_bgcolor="mintcream")

fig_indvsus.update_traces(hoverinfo="all", textfont_size=11)

fig_indvsus.show()
fig_USA_region = go.Figure()

fig_USA_region.add_trace(go.Bar(x=df[df.country_code == 'IND']['city'].value_counts().index[:20], y=df[df.country_code == 'IND']['city'].value_counts()[:20],

                            hovertemplate = '<br><b>Region</b>: %{x}'+'<br><i>Startup count</i>: %{y}',

                           marker=dict(color=list(range(20)), colorscale="Sunsetdark")))



fig_USA_region.update_layout(title="Number of Startups founded in each city in India",

                             xaxis_title="Cities", yaxis_title="Startup Count",title_x=0.5, paper_bgcolor="mintcream",

                             title_font_size=20)

fig_USA_region.show()