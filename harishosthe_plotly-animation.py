import pandas as pd
import plotly.graph_objs as go

import numpy as np
import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

cf.go_offline()

init_notebook_mode(connected=True)
df=pd.read_csv("../input/deliveries.csv")
dg=pd.read_csv("../input/matches.csv")
print(df.columns)
print("\n",dg.columns)
print(df["batting_team"].unique())
print("\n",dg["team1"].unique())
df["batting_team"]=df["batting_team"].replace("Rising Pune Supergiants","Rising Pune Supergiant")
df["bowling_team"]=df["bowling_team"].replace("Rising Pune Supergiants","Rising Pune Supergiant")

dg["team1"]=dg["team1"].replace("Rising Pune Supergiants","Rising Pune Supergiant")
dg["team2"]=dg["team2"].replace("Rising Pune Supergiants","Rising Pune Supergiant")
years = ["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017"]
teams=df["batting_team"].unique()
teams
total_data=df.groupby(['batting_team']).sum()
total_data["total_balls"]=df["batting_team"].value_counts()
total_data["total_overs"]=(total_data["total_balls"]//6)
total_data=total_data.sort_values(by="total_runs",ascending=False)
total_data=total_data.drop(total_data.columns[[0,1,2,3,4,5,6,7,8,9,10,11]],axis=1) # removing unwanted columns..
total_data
runs_per_over=df.groupby(["match_id","batting_team" ,"over"])[["total_runs"]].sum().reset_index()
# runs_per_over
runs_per_over.head()
runs_per_over["Year"]=np.nan
for i in range(2008,2018):
    x=list(dg[dg["season"]==i]["id"]) # there are x matches in i year
    for k in x:
        for j in list(runs_per_over[runs_per_over["match_id"]==k].index):
            runs_per_over["Year"].loc[j]=str(i)
runs_per_over[(runs_per_over["batting_team"]=="Chennai Super Kings") & (runs_per_over["Year"]=="2008")].head()
total_runs_season=runs_per_over.groupby(["Year","batting_team"]).sum()
# n=list(osthe["batting_team"])
total_runs_season.head(3)
over=[] # it stores the sum of overs of each team by season
overs_year=[] # it stores the season_year of each over 
team_names=[] # it stores the team name
for i in years:
    x=list(runs_per_over[runs_per_over["Year"]==str(i)]["batting_team"].unique())
#     team_names.append(x)
    
    for j in x:
        over.append(sum(runs_per_over[(runs_per_over["Year"]==str(i)) & (runs_per_over["batting_team"]==j)]["over"].value_counts()))
        overs_year.append(i)
        team_names.append(j)
overs_df=pd.DataFrame(over,index=[overs_year,team_names],columns=["over_2"])
overs_df.head(10)
overs_df.shape
total_runs_season.shape
total_runs_season.head(2)
overs_df.head(2)
ipl_df=pd.concat([total_runs_season, overs_df], axis=1, join_axes=[total_runs_season.index])
ipl_df2=pd.concat([total_runs_season, overs_df], axis=1, join_axes=[total_runs_season.index])
ipl_df.head()
ipl_df.reset_index(inplace=True)
ipl_df.columns
ipl_df.head(3)
ipl_df.describe()
t=['Sunrisers Hyderabad', 'Royal Challengers Bangalore','Mumbai Indians', 'Rising Pune Supergiant', 'Gujarat Lions',
   'Kolkata Knight Riders', 'Kings XI Punjab', 'Delhi Daredevils','Rajasthan Royals', 'Deccan Chargers',
   'Kochi Tuskers Kerala', 'Pune Warriors']

QW=ipl_df[ipl_df["batting_team"]=="Chennai Super Kings"].cumsum()

for i in t:
    QW=QW.append(ipl_df[ipl_df["batting_team"]==i].cumsum())

QW=QW.rename(columns={"Year":"Match_Year","batting_team":"Team_Names","total_runs":"cum_Total","over_2":"cum_over"})
QW.head()
ipl_df2=pd.concat([ipl_df, QW], axis=1, join_axes=[ipl_df.index])
ipl_df2.head()
ipl_df3=ipl_df2.drop(ipl_df2.columns[[2,3,6,7,8,9]], axis=1) 
ipl_df3.head(10)
xc={
    "Year": ["2008","2008","2008","2008","2008",
             "2009","2009","2009","2009","2009",
            "2010","2010","2010","2010","2010",
            "2011","2011","2011",
            "2012","2012","2012","2012",
            "2013","2013","2013","2013",
            "2014","2014","2014","2014","2014",
            "2015","2015","2015","2015","2015",
            "2016","2016","2016","2016","2016",
            "2017","2017","2017","2017","2017"],
    "batting_team": ['Sunrisers Hyderabad','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala','Pune Warriors',
                    'Sunrisers Hyderabad','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala','Pune Warriors',
                    'Sunrisers Hyderabad','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala','Pune Warriors',
                    'Sunrisers Hyderabad','Rising Pune Supergiant','Gujarat Lions',
                    'Sunrisers Hyderabad','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala',
                    'Deccan Chargers','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala',
                    'Deccan Chargers','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala','Pune Warriors',
                    'Deccan Chargers','Rising Pune Supergiant','Gujarat Lions','Kochi Tuskers Kerala','Pune Warriors',
                    'Rajasthan Royals','Chennai Super Kings','Deccan Chargers','Kochi Tuskers Kerala','Pune Warriors',
                    'Rajasthan Royals','Chennai Super Kings','Deccan Chargers','Kochi Tuskers Kerala','Pune Warriors'],
    "total_runs": list(np.zeros(46)),
    "over_2": list(np.zeros(46)),
    "cum_Total": list(np.zeros(46)),
    "cum_over": list(np.zeros(46))
}
ipl_df3=ipl_df3.append(pd.DataFrame(xc))
len(xc["batting_team"])
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

# unique colors for each team...
colors_code=["#FF8000","#FF0000","#0040FF","#DF01D7","#FA5858","#4B088A","#FF0040","#0431B4","#FFFF00","#0000FF","#2E64FE","#FF8000","#02A2C3"]

figure['layout']['xaxis'] = {'range': [1000,26600], 'title': 'Total runs', "titlefont":dict(family='Arial, sans-serif',size=30,color='lightgrey')}
figure['layout']['yaxis'] = {"type": "category", "automargin": True}

    
figure["layout"]["colorway"] = colors_code
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 3000, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 3000, 'easing': 'quad'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 45},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 2000, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}



year = "2008"
for t in teams:
    dataset_by_year = ipl_df3[ipl_df3['Year'] == year]
    dataset_by_year_and_t = dataset_by_year[dataset_by_year['batting_team'] == t]

    data_dict = {
        'x': list(dataset_by_year_and_t['cum_Total']),
        'y': list(dataset_by_year_and_t['batting_team']),
        "mode": 'markers+text',
        'text': list(dataset_by_year_and_t['batting_team']),
        "textposition": "middle left",
        
        'xaxis': 'x1', 'yaxis': 'y1',
        
        'marker': {
            'opacity': 0.8,
            'symbol': 'star-triangle-up',
            'sizemode': 'area',
            'sizeref': 10,
            'size': list(dataset_by_year_and_t['cum_Total'])
        },
        'name': t
    }
    figure['data'].append(data_dict)


for year1 in years:
    frame = {'data': [], 'name': str(year1)}
    for t in teams:
        dataset_by_year = ipl_df3[ipl_df3['Year'] == str(year1)]
        dataset_by_year_and_t = dataset_by_year[dataset_by_year['batting_team'] == t]
        
        osthe1=[]
        for k in list(dataset_by_year_and_t['batting_team']):
            if k == "Sunrisers Hyderabad":
                osthe1.append("SRH"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Royal Challengers Bangalore':
                osthe1.append("RCB"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k == 'Mumbai Indians':
                osthe1.append("MI"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k == 'Rising Pune Supergiant':
                osthe1.append("RPS"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  "Gujarat Lions":
                osthe1.append("GL"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Kolkata Knight Riders':
                osthe1.append("KKR"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Kings XI Punjab':
                osthe1.append("KXIP"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Delhi Daredevils':
                osthe1.append("DD"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Kochi Tuskers Kerala':
                osthe1.append("KTK"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Chennai Super Kings':
                osthe1.append("CSK"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Rajasthan Royals':
                osthe1.append("RR"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Deccan Chargers':
                osthe1.append("DC"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
            elif k ==  'Pune Warriors':
                osthe1.append("PW"+":"+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["total_runs"]))+"Runs in "+str(list(dataset_by_year_and_t[dataset_by_year_and_t['batting_team']==k]["over_2"]))+" Overs")
                  
        
        data_dict = {
            'x': list(dataset_by_year_and_t['cum_Total']),
            'y': list(dataset_by_year_and_t['batting_team']),
            "mode": 'markers+text',
            'text': osthe1,
            "textfont": dict(
                family='sans serif',
                size=14,
            ),
            "textposition" : "middle center",
            "marker": {
            'opacity': 0.8,
            'symbol': 'star-triangle-up',
            'sizemode': 'area',
            'sizeref': 10,
            'size': list(dataset_by_year_and_t['cum_Total'])
        },
            'name': t
        }
        frame['data'].append(data_dict)
    
    figure['frames'].append(frame)

    slider_step = {'args': [
        [year1],
        {'frame': {'duration': 2000, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 2000}}
     ],
     'label': year1,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)
    figure['layout']['sliders'] = [sliders_dict]
    
iplot(figure) # use plot(figure) insted u get a better visualization in a new tab ...