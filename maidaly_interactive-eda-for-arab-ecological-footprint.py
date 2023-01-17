import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
NFA_df = pd.read_csv("../input/NFA 2018.csv")
NFA_df.info()
def extract_country_by_record(df,country_name,record):
    country_foot_print=df[df.country.isin([country_name])]
    country_by_record = country_foot_print [country_foot_print.record.isin([record])]
    return country_by_record

def extract_countries_feature_by_year (df,countries_list,feature,year,record="BiocapPerCap"):
    excluded_countries=[]
    feature_values=[]
    available_countries=[]
    for i in range (0,len(countries_list)):
        country_by_record = extract_country_by_record(df,countries_list[i],record)
        feature_value = country_by_record.loc[lambda df1: country_by_record.year == year][feature].values
        if  feature_value.size==0 or math.isnan(feature_value[0]) :
            excluded_countries.append(countries_list[i])
        else:
            feature_values.append(feature_value[0]) 
            available_countries.append(countries_list[i])
            
#  activate if you need to print the excluded countries in the year
#     if len(excluded_countries) != 0:
#         print("excluded countries in {0} are : ".format(year))
#         for i in excluded_countries:
#             print(i)
    return feature_values, available_countries, excluded_countries 
def print_excluded_countries (excluded_countries,year):
    if len(excluded_countries) != 0:
        print("excluded countries from dataset in {0} are : ".format(year))
        for i in excluded_countries:
            print(i)   
            
def calculate_growth_rate(present,past,period):
    #present : present year , past: past year , period: number of years between present and past
    percentage_growth_rate = ((present - past)/(past*period))*100
    return percentage_growth_rate
arab_countries = ['Egypt','Algeria','Bahrain','Libyan Arab Jamahiriya',
                 'Jordan','Iraq','Mauritania','Morocco',
                  'Saudi Arabia','Kuwait','Qatar','Sudan (former)',
                 'Oman','Tunisia','United Arab Emirates','Yemen',
                  'Lebanon','Syrian Arab Republic','Somalia','Comoros','Djibouti']

colors = ['blue','gray','red','green','pink',
          'steelblue','yellow','magenta','brown',
          'orange','tan','seagreen','olive',
          'turquoise','mintcream','yellowgreen',
          'darkkhaki','coral','chocolate','rosybrown',
          'dodgerblue','heather']

years=np.sort(NFA_df.year.unique())

arab_df = pd.DataFrame()
for country in arab_countries:
    arab_df=arab_df.append(NFA_df[NFA_df.country.isin([country])])
regoin=[]
sub_region=[]
for country in arab_countries:
    regoin.append(arab_df[arab_df.country.isin([country])]["UN_region"].unique()[0])
    sub_region.append(arab_df[arab_df.country.isin([country])]["UN_subregion"].unique()[0])
regoin_labels = pd.Series(regoin).value_counts().index
regoin_values = pd.Series(regoin).value_counts().values
sub_region_labels = pd.Series(sub_region).value_counts().index
sub_regoin_values = pd.Series(sub_region).value_counts().values

trace0  = go.Bar(x= regoin_labels,
                 y= regoin_values,
                 marker=dict(color='#f0000a',
                             line=dict(color='rgb(8,48,107)',
                                       width=1.5,)),
                 opacity=0.7,
                name = 'regoin',
                hoverinfo="x + y")
trace1  = go.Bar(x= sub_region_labels,
                 y= sub_regoin_values,
                 marker=dict(color='#fff000',
                             line=dict(color='rgb(8,48,107)',
                                       width=1.5,)),
                 opacity=0.6,
                name = 'sub_regoin',
               hoverinfo="x + y" )
go_plot = [trace0,trace1]
layout = go.Layout(
    title='Arab countries distrbution according to regions',)
fig = go.Figure(data=go_plot, layout=layout)

py.iplot(fig)
population,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'population',2014)
available_countries.append("Sudan")
population.append(37737900)
population_df = pd.DataFrame({'countery':available_countries,'population':population}).sort_values(by='population',ascending=True) 
population_list = list (population_df['population'])
countries = list (population_df['countery'])
annotations = []
y_nw = np.array(population_list)

for ydn,  xd in zip(y_nw, countries):
    # labeling the scatter savings
    annotations.append(dict(xref='x', yref='y',
                            y=xd, x=ydn + 5000000,
                            text='{:,} M'.format(np.round(ydn/10**6,2)),
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 0, 50)'),
                            showarrow=False))
fig  = {
  "data": [
    {
      "values": population_list,
      "labels": countries,
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie",
      'domain': {'x': [.4, 1],
                       'y': [0.2, .8]},
            'hoverinfo':'label+percent',
            'textinfo':'percent'
    },
      {
          "x": population_list,
          "y": countries,
          "type": "bar",
          "orientation" :'h',
          "hoverinfo":"x",
          "marker" : dict(color='rgba(128, 0, 128,0.7)',
                          line=dict(color='rgb(255,0,255)',
                                       width=2)),
          
          "opacity":0.7,
          "name":"Population",
          } 
  ],
  "layout": {
        "title":"Arab countries population 2014",
        'annotations': annotations,
        "yaxis":dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickfont=dict(family='Arial', size=12,color='rgb(50, 0, 50)')),
        "width": 1000,
        "height":700,
    "paper_bgcolor":'rgb(250, 240, 250)',
    "plot_bgcolor":'rgb(250, 240, 250)',
      
}}


py.iplot(fig)
traces = []
annotations = []
for i in range(len(arab_countries)):
    country_by_record = extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')
    traces.append(go.Scatter(
            x=country_by_record['year'],
            y=country_by_record['population'],
            mode='lines',
            line=dict(color=colors[i], width=1.5),
            text= arab_countries[i],
            hoverinfo="text + x + y",
            connectgaps=True,
            name =arab_countries[i],
            textfont=dict(family='Arial', size=12),
    ))
layout = go.Layout(
    title = "Arab countries population growth",
   xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(150, 150, 150)',
            linewidth=2,
            gridcolor='rgb(90, 90, 90)',
            ticks='outside',
            tickcolor='rgb(80, 80, 80)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)',
        ),
    ),
    yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            gridcolor='rgb(80, 80, 80)',
            showticklabels=True,
            tickcolor='rgb(150, 150, 150)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)')
    ),
   font=dict(family='Arial', size=12,
            color='rgb(180, 180, 180)'),
            showlegend=True, 
            width = 900,
            height = 700,
            paper_bgcolor='rgba(0, 0, 0,.9)',
            plot_bgcolor='rgba(0, 0, 0,0)'
)
    
fig = go.Figure(data=traces, layout= layout)
py.iplot(fig)
population_2000,available_countries,excluded_countries_2000=extract_countries_feature_by_year(arab_df,arab_countries,'population',2000)
population_2010,available_countries,excluded_countries_2010=extract_countries_feature_by_year(arab_df,arab_countries,'population',2010)
population_growth_rate = []
for i in range (0,len(population_2000)):
    growth_rate = np.round(calculate_growth_rate(population_2010[i],population_2000[i],10),2)
    population_growth_rate.append(growth_rate)
growth_rate_df = pd.DataFrame({"country":available_countries,"growth rate":population_growth_rate}).sort_values(by="growth rate",ascending=False)
print_excluded_countries(excluded_countries_2000,2000)  
print_excluded_countries(excluded_countries_2010,2010)
table = go.Table(header=dict(values=['Country', 'Growth rate']),
        cells=dict(values= [growth_rate_df['country'],growth_rate_df['growth rate'].astype(str)+"%"]))
py.iplot([table])
growth_rate_df=growth_rate_df.sort_values(by="growth rate",ascending=True)
trace0  = go.Bar(x= growth_rate_df["growth rate"],
                 y= growth_rate_df["country"],
                 orientation ='h',
                 marker=dict(color='rgba(255, 255, 0, 1.0)',
                             line=dict(color='rgba(250, 80, 0, 1.0)',
                                       width=4)),
                opacity=0.7,
                hoverinfo="x + y")

annotations = []
y_nw = np.array(growth_rate_df["growth rate"])
for ydn,  xd in zip(y_nw, growth_rate_df["country"]):
    # labeling the scatter savings
    annotations.append(dict(xref='x', yref='y',
                            y=xd, x=ydn+0.9,
                            text='{:,} %'.format(np.round(ydn,2)),
                            font=dict(family='Arial', size=12,
                                      color='rgba(250, 80, 0, 1.0)'),
                            showarrow=False))
layout = go.Layout(
                title='Arab countries annual growth rate of population [2000-2010]',
                margin=dict(
                        l=130,
                        r=20,
                        t=30,
                        b=30,
                    ),
               annotations = annotations,
               xaxis=dict(showgrid=True,
                       gridcolor="rgba(250, 80, 0, .2)",
                       showticklabels=True,
                       tickfont=dict(family='Arial', size=12,color='rgba(250, 80, 0, 1.0)')),
               yaxis=dict(showgrid=False,
                       showline=True,
                       linecolor='rgba(250, 80, 0, 1.0)',
                       showticklabels=True,
                       tickfont=dict(family='Arial', size=12,color='rgba(250, 80, 0, 1.0)')),
               font=dict(family='Arial', size=12,
                       color='rgba(250, 80, 0, 1.0)'),
               width = 1000,
               height = 600,
               paper_bgcolor='rgba(0, 0, 0,1)',
               plot_bgcolor='rgba(0, 0, 0,0)',
             )
# Adding labels
# Creating two subplots


fig = go.Figure(data=[trace0], layout=layout)

py.iplot(fig)
arab_countrs_population = []
for year in years:
    sum_population_per_year = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'population',year)[0]).sum()
    arab_countrs_population.append(sum_population_per_year)

arab_population_growth_rate = calculate_growth_rate(arab_countrs_population[49],arab_countrs_population[24],25)
trace0 = go.Scatter(
    x= years[24:49],
    y= arab_countrs_population[24:49],
    hoverinfo = 'name+x+y',
    name='Population',
    mode = "lines",
    line=dict(
        color='rgba(220,220,150,1)',
        width= 3)
    )
layout = go.Layout(
    title = "Arab countries total population growth",
    annotations = [dict(xref = 'x', yref = 'y',
                      x = 1990, y = arab_countrs_population[45],
                      text='growth rate = {0} %'.format(np.round(arab_population_growth_rate,2)),
                      font=dict(family='Arial', size=20,
                                color='rgba(200, 150, 0, 1.0)'),
                      showarrow=False)],

    xaxis=dict(
        showline=False,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(150, 150, 150)',
        linewidth=2,
        gridcolor='rgb(90, 90, 90)',
        ticks='outside',
        tickcolor='rgb(80, 80, 80)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=False,
        gridcolor='rgb(80, 80, 80)',
        showticklabels=True,
        tickcolor='rgb(180, 180, 180)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)')
    ),
   font=dict(family='Arial', size=12,
             color='rgb(180, 180, 180)'),
    showlegend=True, 
    width = 900,
    height = 700,
    paper_bgcolor='rgba(0, 0, 0,.95)',
    plot_bgcolor='rgba(0, 0, 0,0)',
)
    
fig = go.Figure(data=[trace0], layout= layout)
py.iplot(fig)

traces = []
annotations = []
for i in range(len(arab_countries)):
    country_by_record = extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')
    traces.append(go.Scatter(
        x=country_by_record['year'],
        y=country_by_record['Percapita GDP (2010 USD)'],
        mode='lines',
        line=dict(color=colors[i], width=1.5),
        text= arab_countries[i]+"<br>"+ country_by_record['Percapita GDP (2010 USD)'].dropna().apply(lambda x:int(x)).astype(str)+" $",
        hoverinfo="text + x ",
        connectgaps=True,
        name =arab_countries[i],
        textfont=dict(family='Arial', size=12),
    ))

world_by_record = extract_country_by_record(NFA_df,'World','BiocapPerCap')
traces.append(go.Scatter(
        x=world_by_record['year'],
        y=world_by_record['Percapita GDP (2010 USD)'],
        mode='lines',
        line=dict(color="rgb(255,0,0)", width=2.5, dash = 'dash'),
        text= "World"+"<br>"+ world_by_record['Percapita GDP (2010 USD)'].dropna().apply(lambda x:int(x)).astype(str)+" $",
        hoverinfo="text + x ",
        connectgaps=True,
        name ="World",
        textfont=dict(family='Arial', size=12),
    ))
    
layout = go.Layout(
    title = "Arab countries GDP per capita",
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(150, 150, 150)',
        linewidth=2,
        gridcolor='rgb(90, 90, 90)',
        ticks='outside',
        tickcolor='rgb(80, 80, 80)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=False,
        gridcolor='rgb(80, 80, 80)',
        showticklabels=True,
        tickcolor='rgb(150, 150, 150)',
        tickwidth=2,
        ticklen=5,
        tickfont=dict(
            family='Arial',
            size=13,
            color='rgb(180, 180, 180)')
    ),
   font=dict(family='Arial', size=12,
             color='rgb(180, 180, 180)'),
    showlegend=True, 
    width = 900,
    height = 700,
    paper_bgcolor='rgba(0, 0, 0,.95)',
    plot_bgcolor='rgba(0, 0, 0,0)',
)
    
fig = go.Figure(data=traces, layout= layout)
py.iplot(fig)

gdp,avilable_countries,excluded_countries = extract_countries_feature_by_year (arab_df,arab_countries,'Percapita GDP (2010 USD)',2014,record="BiocapPerCap")
locations = []
countries = avilable_countries + excluded_countries 
for c in countries:
    country_by_record = extract_country_by_record(arab_df,c,"BiocapPerCap")
    code = country_by_record.loc[lambda df1: country_by_record.year == 2014]['ISO alpha-3 code'].values
    if  not (code.size==0):
        locations.append (country_by_record.loc[lambda df1: country_by_record.year == 2014]['ISO alpha-3 code'].values[0])
    else:
        locations.append("SDN")    # only sudan dose not have and data after 2014 so it return empety array
    
for i in range(len(excluded_countries)):
    gdp.append("NAN")  
data = [ dict(
        type = 'choropleth',
        locations = locations,
        z = gdp,
        text = countries  ,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5 )
        ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'GDP per capita<br> US$'),
      ) ]

layout = dict(
    title = '2014 Arab countries GDP per capita',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'equirectangular'
        )
    )
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False )
GDP,available_countries, excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'Percapita GDP (2010 USD)',2014)
GDP_df = pd.DataFrame({'country':available_countries,'GDP':GDP}).sort_values(by='GDP',ascending=True)
trace0  = go.Bar(x= GDP_df["GDP"],
                 y= GDP_df["country"],
                 orientation ='h',
                 marker=dict(color='rgba(255, 255, 0, 1.0)',
                             line=dict(color='rgba(250, 80, 0, 1.0)',
                                       width=4)),
                 opacity=0.7,
                 name="GDP",
                hoverinfo="name+x + y")

annotations = []
y_nw = np.array(GDP_df["GDP"])
for ydn,  xd in zip(y_nw, GDP_df["country"]):
    # labeling the scatter savings
    annotations.append(dict(xref='x', yref='y',
                            y=xd, x=ydn+3000,
                            text='{:,} $'.format(np.int(ydn)),
                            font=dict(family='Arial', size=12,
                                      color='rgba(250, 80, 0, 1.0)'),
                            showarrow=False))
    
layout = go.Layout(
    title='2014 Arab countries GDP per cabita',
     margin=dict(
        l=130,
        r=20,
        t=30,
        b=30,
    ),
    annotations = annotations,
    xaxis=dict(showgrid=True,
               gridcolor="rgba(250, 80, 0, .2)",
               showticklabels=True,
               tickfont=dict(family='Arial', size=12,color='rgba(250, 80, 0, 1.0)')),
    yaxis=dict(showgrid=False,
               showline=False,
               linecolor='rgba(250, 80, 0, 1.0)',
               showticklabels=True,
               tickfont=dict(family='Arial', size=12,color='rgba(250, 80, 0, 1.0)')),
    font=dict(family='Arial', size=12,
             color='rgba(250, 80, 0, 1.0)'),
    width = 1000,
    height = 600,
    paper_bgcolor='rgba(0, 0, 0,1)',
    plot_bgcolor='rgba(0, 0, 0,0)',
             )
# Adding labels
# Creating two subplots


fig = go.Figure(data=[trace0], layout=layout)

py.iplot(fig)
cuntries = extract_countries_feature_by_year(arab_df,arab_countries,'Percapita GDP (2010 USD)',2014)[1]
y = extract_countries_feature_by_year(arab_df,cuntries,'Percapita GDP (2010 USD)',2014)[0]
x = extract_countries_feature_by_year(arab_df,cuntries,'population',2014)[0]

colors = np.random.rand(22)
text = []
for i in range (len(cuntries)):
    text.append(cuntries[i]+"<br>"+"GDP Percap: {0} K".format(np.round((y[i]/10**3),2))+"<br>"+"population: {0} M".format(np.round((x[i]/10**6),2)))
annotations = []
y_nw = np.array(y)
for ydn,  xd , c in zip(y_nw, x,cuntries):
    # labeling the scatter savings
    annotations.append(dict(xref='x', yref='y',
                            y=ydn, x=xd,
                            text= c,
                            font=dict(family='Raleway', size=12,
                                      color='rgba(50, 50, 50, 1.0)'),
                           showarrow=False))
# The marker size is proportional to population
trace = go.Scatter(x=x,
                y=y,
                text = text,
                mode='markers',
                hoverinfo = 'text ',
                marker={'size': x,        
                        'color': colors,
                        'opacity': 0.6,
                        'sizemode' : 'area',
                        'sizeref' : 40000,
                        'colorscale': 'Viridis'
                       });
layout = go.Layout(title = " GDP and Population",
                  yaxis=dict(title = "GDP per capita"),
                  xaxis=dict(title = "Population"),
                  height = 700)
fig = go.Figure(data=[trace],layout = layout)
py.iplot(fig)
arab_consumption_corr=arab_df[arab_df.record.isin(["EFConsPerCap"])].drop('year',axis=1).corr()
cons_heatmap = go.Heatmap(z=arab_consumption_corr.values,x=arab_consumption_corr.index,y=arab_consumption_corr.index)
layout = go.Layout(title = "Correlation between features according to ecological footprint (per capita)")
fig = go.Figure(data=[cons_heatmap], layout=layout)

py.iplot(fig)
biocapcity_corr=arab_df[arab_df.record.isin(["BiocapPerCap"])].drop(['year','carbon'],axis=1).corr()
biocap_heatmap = go.Heatmap(z=biocapcity_corr.values,x=biocapcity_corr.index,y=biocapcity_corr.index)
layout = go.Layout(title = "Correlation between features according to biocapcity")
fig = go.Figure(data=[biocap_heatmap], layout=layout)

py.iplot(fig)

record ={1:['BiocapPerCap','EFConsPerCap','rgba(0,255,0,1)','rgba(255,0,0,1)'],
         2:['BiocapTotGHA','EFConsTotGHA','rgba(0,140,0,1)','rgba(140,0,0,1)'],}

for c in range (0,len(arab_countries)):
    fig = tools.make_subplots(rows=1, cols=2, specs=[[{},{}]], horizontal_spacing=0.1,
                         subplot_titles=["BioCapacity vs Ecological footprint (per capita)","BioCapacity vs Ecological footprint (GHA)"])
    for r in record.keys():
            country_by_record_bio = extract_country_by_record(arab_df,arab_countries[c],record[r][0])
            country_by_record_cons = extract_country_by_record(arab_df,arab_countries[c],record[r][1])
            trace1 = go.Scatter(
            x=country_by_record_bio['year'],
            y=country_by_record_bio['total'],
            mode= 'lines',
            name = record[r][0],
            line=dict(color=record[r][2], width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )
            trace2 = go.Scatter(
            x=country_by_record_cons['year'],
            y=country_by_record_cons['total'],
            mode='lines',
            name = record[r][1],
            line=dict(color=record[r][3], width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )
            data= [trace1,trace2]

            fig.append_trace(trace1, 1, r)
            fig.append_trace(trace2, 1, r)
    fig['layout'].update(height=450, width=1000,
                     title=arab_countries[c])
    py.iplot(fig)



Arab_BiocapTotal = []
Arab_EFConsTotal = []
Arab_BiocapPerCap = []
Arab_EFConsPerCap = []
world_BiocapTotal = []
world_EFConsTotal = []
mean_BiocapPerCap = []
mean_EFConsPerCap = []
for year in years :
    sum_BiocapTotal_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record= 'BiocapTotGHA')[0]).sum()
    sum_EFConsTotal_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record='EFConsTotGHA')[0]).sum()
    sum_population_per_year = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'population',year)[0]).sum()
    world_BiocapTotal.append(np.array(extract_countries_feature_by_year(NFA_df,['World'],'total',year,record= 'BiocapTotGHA')[0]))
    world_EFConsTotal.append(np.array(extract_countries_feature_by_year(NFA_df,['World'],'total',year,record= 'EFConsTotGHA')[0]))
    Arab_BiocapTotal.append(sum_BiocapTotal_value)
    Arab_EFConsTotal.append(sum_EFConsTotal_value)
    Arab_BiocapPerCap.append(sum_BiocapTotal_value/sum_population_per_year)
    Arab_EFConsPerCap.append(sum_EFConsTotal_value/sum_population_per_year)
    mean_BiocapPerCap.append(np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year)[0]).mean())
    mean_EFConsPerCap.append(np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record='EFConsPerCap')[0]).mean())
fig = tools.make_subplots(rows=2, cols=2, specs=[[{},{}],[{'colspan': 2}, None]], horizontal_spacing=0.1, vertical_spacing = 0.1,
                         subplot_titles=["Per capita","Mean (per capita)","Total (GHA)"])

arab_biocapPerCap_plt = go.Scatter(
            x=years[19:],
            y=Arab_BiocapPerCap[19:],
            mode= 'lines',
            name = "Biocapcity",
            line=dict(color="green", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )
arab_EFperCap_plt = go.Scatter(
            x=years[19:],
            y=Arab_EFConsPerCap[19:],
            mode='lines',
            name = "Ecological footprint",
            line=dict(color="red", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )

fig.append_trace(arab_biocapPerCap_plt, 1, 1)
fig.append_trace(arab_EFperCap_plt, 1, 1)

arab_meanBiocapPerCap_plt = go.Scatter(
            x=years[19:],
            y=mean_BiocapPerCap[19:],
            mode= 'lines',
            showlegend = False,
            line=dict(color="green", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )
arab_meanEFperCap_plt = go.Scatter(
            x=years[19:],
            y=mean_EFConsPerCap[19:],
            mode='lines',
            showlegend = False,
            line=dict(color="red", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )

fig.append_trace(arab_meanBiocapPerCap_plt, 1, 2)
fig.append_trace(arab_meanEFperCap_plt, 1, 2)

arab_totalBiocap_plt = go.Scatter(
            x=years[19:],
            y=Arab_BiocapTotal[19:],
            mode= 'lines',
            showlegend = False,
            line=dict(color="green", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )
arab_totalEF_plt = go.Scatter(
            x=years[19:],
            y=Arab_EFConsTotal[19:],
            mode='lines',
            showlegend = False,
            line=dict(color="red", width=1.5),
            hoverinfo="y + x ",
            textfont=dict(family='Arial', size=12),
        )

fig.append_trace(arab_totalBiocap_plt, 2, 1)
fig.append_trace(arab_totalEF_plt, 2, 1)

            
fig['layout'].update(height=900, width=1000,
                     title= "Arab World BioCapacity vs Ecological footprint")
py.iplot(fig)
difference  = []
countries_list = []
deficit_or_reserve = []
for country in arab_countries:
    BiocapPerCap=np.array(extract_countries_feature_by_year(arab_df,[country],'total',2014)[0])
    EFConsPerCap=np.array(extract_countries_feature_by_year(arab_df,[country],'total',2014,record="EFConsPerCap")[0])
    difference_value = BiocapPerCap - EFConsPerCap
    if difference_value < 0 :
        deficit_or_reserve.append ("deficit")
        difference.append(np.abs(difference_value[0]))
    if difference_value > 0 :
        deficit_or_reserve.append("reserve")
        difference.append(difference_value[0])
    if difference_value.size==0:
        deficit_or_reserve.append("nan")
        difference.append(np.NAN)
    countries_list.append(country)
defict_reserve_df = pd.DataFrame({"country":countries_list,"deficit/reserve":deficit_or_reserve,"value":difference}).dropna().sort_values(by="value",ascending=False)

trace0 = go.Bar(
    y=defict_reserve_df[defict_reserve_df['deficit/reserve'].isin(['deficit'])]['country'],
    x=defict_reserve_df[defict_reserve_df['deficit/reserve'].isin(['deficit'])]['value'],
    orientation ='h',
    name='Deficit',
    marker=dict(
        color='rgb(180,0,0)'
    )
)
trace1 = go.Bar(
    y=defict_reserve_df[defict_reserve_df['deficit/reserve'].isin(['reserve'])]['country'],
    x=defict_reserve_df[defict_reserve_df['deficit/reserve'].isin(['reserve'])]['value'],
    orientation ='h',
    name='Reserve',
    marker=dict(
        color='rgb(0,180,0)',
    )
)
data = [trace0, trace1]
layout = go.Layout(title = "Arab countries Ecological footprint [Deficit/Reserve] (per capita) 2014",
                   yaxis = dict(showline = False,
                               zeroline = False),
                   width=900,height=500,
                   margin=dict(
                        l=140,
                        r=20,
                        t=30,
                        b=30)
                   )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
defict_reserve_df
def extract_defict_value (df,country_name):
    c =df[df.country.isin([country_name])]
    if c['deficit/reserve'].values.size !=0 :
        if (c['deficit/reserve'].values[0]=='deficit'):
            return c['value'].values[0]
         
deficit_value = []
for c in cuntries :
    if extract_defict_value(defict_reserve_df,c) != None:
        deficit_value.append(extract_defict_value(defict_reserve_df,c))
    else:
        continue

cuntries = extract_countries_feature_by_year(arab_df,arab_countries,'Percapita GDP (2010 USD)',2014)[1]
cuntries.remove('Mauritania')     # remove Mauritania as it has no deficit 
x = extract_countries_feature_by_year(arab_df,cuntries,'Percapita GDP (2010 USD)',2014)[0]
y = deficit_value
colors = np.random.rand(100)
sz = (np.array(y)*10000)
text = []
for i in range (len(cuntries)):
    text.append(cuntries[i]+"<br>"+"GDP: {0} K".format(np.round((x[i]/10**3),2))+"<br>"+"Deficit: {0}".format(np.round((y[i]),2)))
annotations = []
y_nw = np.array(y)
for ydn,  xd , c in zip(y_nw, x,cuntries):
    # labeling the scatter savings
    annotations.append(dict(xref='x', yref='y',
                            y=ydn, x=xd,
                            text= c,
                            font=dict(family='Raleway', size=11,
                                      color='rgba(50, 50, 50, 1.0)'),
                            showarrow=False))
trace = go.Scatter(x=x,
                y=y,
                text = text,
                mode='markers',
                hoverinfo = 'text ',
                marker={'size': sz,
                        'color': colors,
                        'opacity': 0.5,
                        'sizemode' : 'area',
                        'sizeref' : 80,
                        'colorscale': 'Viridis'
                       });
layout = go.Layout(title= " Ecological deficit and GDP",
                  yaxis=dict(title = "Ecological deficit (per capita)"),
                  xaxis=dict(title = "GDP per capita"),
                  annotations = annotations,
                  height = 700,
                  width = 1500)
fig = go.Figure(data=[trace],layout = layout)
py.iplot(fig)
import datetime
arab_eod_dates = []
eod_dates_world=[]
def calc_earth_overshot_day(biocap,ecofootp):
    eod = (np.array(biocap) / np.array(ecofootp))*365
    return eod
eod_arab = calc_earth_overshot_day(Arab_BiocapTotal,Arab_EFConsTotal)
eod_world = calc_earth_overshot_day(world_BiocapTotal,world_EFConsTotal)
eod_month_arab = []
eod_month_world = []
for i in range (0,len(eod_arab)):
    if eod_arab[i]>365:
        arab_eod_dates.append("no EOD")
        eod_month_arab.append("no EOD")
    if eod_world[i]>365:
        eod_dates_world.append("no EOD")
        eod_month_world.append("no EOD")
    if eod_arab[i] < 365:
        date_arab = datetime.datetime(years[i],1,1) + datetime.timedelta(days=eod_arab[i])
        eod_month_arab.append(date_arab.strftime('%b'))
        arab_eod_dates.append(date_arab.strftime('%b-%d'))
    if eod_world[i] < 365:
        date_world = datetime.datetime(years[i],1,1) + datetime.timedelta(days=int(eod_world[i]))
        eod_month_world.append(date_world.strftime('%b'))
        eod_dates_world.append(date_world.strftime('%b-%d'))
        #[19:] represents the year that the EOD begins to appear 
eod_df = pd.DataFrame({"year":years[19:],"Arab Earth Overshoot Day":arab_eod_dates[19:],"World Earth Overshoot Day":eod_dates_world[19:]})
eod_df
EOD_arab_plt = go.Scatter(
            x=eod_df['year'],
            y= eod_month_arab[19:],
            mode= 'lines',
            name = "Arab EOD",
            text = eod_df["Arab Earth Overshoot Day"],
            line=dict(color="green", width=1.5),
            hoverinfo="name + text + x ",
            textfont=dict(family='Arial', size=12),
        )
EOD_world_plt = go.Scatter(
            x=eod_df['year'],
            y= eod_month_world[19:],
            mode= 'lines',
            name = "World EOD",
            text = eod_df["World Earth Overshoot Day"],
            line=dict(color="red", width=1.5),
            hoverinfo="name + text + x ",
            textfont=dict(family='Arial', size=12),
        )
layout = go.Layout(title= "Earth Overshoot Day",
                  yaxis=dict(title = "Month"),
                  xaxis=dict(title = "Year"),
                  height = 500,
                  width = 800)

fig = go.Figure(data=[EOD_arab_plt,EOD_world_plt],layout=layout)
py.iplot(fig)
Arab_carbon,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'carbon',2014,record="EFConsPerCap")
carbon_df = pd.DataFrame({'country':available_countries,'carbon':Arab_carbon}).sort_values(by='carbon',ascending=False)
avail_countris = extract_countries_feature_by_year(arab_df,carbon_df['country'],'Percapita GDP (2010 USD)',2014)[1]
y = avail_countris
x = extract_countries_feature_by_year(arab_df,avail_countris,'Percapita GDP (2010 USD)',2014)[0]
size = extract_countries_feature_by_year(arab_df,avail_countris,'carbon',2014,record="EFConsPerCap")[0]
colors = size
text = []
for i in range (len(avail_countris)):
    text.append(y[i]+"<br>"+"GDP Percap: {0} K".format(np.round((x[i]/10**3),2))+"<br>"+"EFcarbon: {0} ".format(np.round((np.array(size)[i]),2)))

trace = go.Scatter(x=x,
                y=y,
                text = text,
                mode='markers',
                hoverinfo = 'text ',
                name = "EFCarbon",
                showlegend = False,
                marker={'size': size,        
                        'color': colors,
                        'opacity': 0.6,
                        'sizemode' : 'area',
                        'sizeref' : 0.005,
                        'colorscale': 'Portland',
                         'showscale' : True,
                         'cmax' : np.max(size),
                         'cmin' : np.min(size),
                         'colorbar' : dict( y= 0.52,
                                            len= .8,
                                            x = 1,
                                            title = "EF Carbon",
                                            titlefont = dict(size=15))
                       },
                  
                  );
layout = go.Layout(
                  title = "Ecological footprint of Carbon and GDP (per capita) [2014]",
                  xaxis=dict(title = "GDP (per capita)",
                             titlefont = dict (family = "Arial"),
                             zeroline=False),
                  yaxis=dict(
                        zeroline=True,
                        showticklabels=True,
                        tickfont=dict(family='Arial', size=12)),
                  margin=dict(
                        l=140,
                        r=20,
                        t=40,
                        b=45,),
                  width = 1000,
                  height = 700)
fig = go.Figure(data=[trace],layout = layout)
py.iplot(fig)
