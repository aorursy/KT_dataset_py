from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
from geopandas import GeoDataFrame

from shapely.geometry import Point
import pandas as pd, numpy as np 
import shutil, os, ast, folium
import geopandas as gpd

init_notebook_mode(connected=True)

def _read_shape_gdf(_base_dir, selected_dept):
    shape_pth = _base_dir + "/shapefiles/department.shp"
    shape_gdf = gpd.read_file(shape_pth)
    return shape_gdf

def _get_latlong_point(point):
    _ll = str(point)
    _ll = _ll.replace("POINT (","").replace(")", "")
    _ll = list(reversed([float(_) for _ in _ll.split()]))
    return _ll

depts_config = {
    'Dept_23-00089' : {'_rowid' : "DISTRICT", "ct_num" : "18"},  
    'Dept_49-00035' : {'_rowid' : "pol_dist", "ct_num" : "06"},  
    'Dept_24-00013' : {'_rowid' : "PRECINCT", "ct_num" : "27"},  
    'Dept_24-00098' : {'_rowid' : "gridnum",  "ct_num" : "27"},   
    'Dept_49-00033' : {'_rowid' : "number",   "ct_num" : "06"},    
    'Dept_11-00091' : {'_rowid' : "ID",       "ct_num" : "25"},         
    'Dept_49-00081' : {'_rowid' : "company",  "ct_num" : "06"},   
    'Dept_37-00049' : {'_rowid' : "Name",     "ct_num" : "48"},      
    'Dept_37-00027' : {'_rowid' : "CODE",     "ct_num" : "48"},     
    'Dept_49-00009' : {'_rowid' : "objectid", "ct_num" : "53"}, 
}

_identifier = "LOCATION_DISTRICT"

def _agebin(x):
    if str(x).lower() == "nan":
        return None
    
    ranges = [20, 24, 34, 44, 54, 59, 100]
    tags = ["<20", "20-24", "25-34", "35-44", "45-54", "55-59", "60+"]
    for i, rng in enumerate(ranges):
        if int(x) <= rng:
            return tags[i]

def _get_dfs(_dept):
    _base_dir = "../input/3-example-runs-of-automation-pipeline/CPE_ROOT/" + _dept

    enriched_df = pd.read_csv(_base_dir + "/enriched_df.csv")
    police_df = pd.read_csv(_base_dir + "/police_df.csv")
    shape_gdf = _read_shape_gdf(_base_dir, _dept)

    ## Convert Dictionary Columns 
    for c in police_df.columns:
        if c != _identifier:
            police_df[c] = police_df[c].apply(ast.literal_eval)

    shape_gdf = shape_gdf.rename(columns = {depts_config[_dept]['_rowid'] : _identifier})
    shape_gdf[_identifier] = shape_gdf[_identifier].astype(str)
    enriched_df[_identifier] = enriched_df[_identifier].astype(str)
    police_df[_identifier] = police_df[_identifier].astype(str)
    police_df = police_df.merge(shape_gdf[[_identifier, "geometry"]], on=_identifier)
    
    events_df = pd.read_csv(_base_dir + "/events/events_df.csv", low_memory=False, parse_dates = ["INCIDENT_DATE"])[1:]
    
    if "SUBJECT_AGE" in events_df.columns:
        events_df["agebin"] = events_df["SUBJECT_AGE"].apply(lambda x : _agebin(x))
    
    return enriched_df, police_df, shape_gdf, _base_dir, events_df
_dept = "Dept_49-00033"
enriched_df, police_df, shape_gdf, _base_dir, events_df = _get_dfs(_dept)

############# Also load the external dataset used #######################
vstops_df = pd.read_csv("../input/external-datasets-cpe/la_stops/vehicle-and-pedestrian-stop-data-2010-to-present.csv", low_memory=False)
vstops_df["Stop Date"] = pd.to_datetime(vstops_df["Stop Date"])
vstops_df["Year"] = vstops_df["Stop Date"].dt.year
vstops_df["month"] = vstops_df["Stop Date"].dt.month
vstops = vstops_df[(vstops_df['Year'] == 2015) & (vstops_df['Stop Type'] == "VEH")]

enriched_df.head()
police_df.head()
"""
About the Code in this Cell 

This code produces the folium map as the base map, on which we plot three items : district polygons, aggregated number of total incidents by district, recent incidents
"""

## Add some more features
police_df["total_incidents"] = police_df["arrest_sex"].apply(lambda x : sum(x.values()))
police_df["total_vstops"] = police_df["vstops_sex"].apply(lambda x : sum(x.values()))

## Plot the base map
center_pt = police_df.geometry[0].centroid
center_ll = _get_latlong_point(center_pt)
mapa = folium.Map(center_ll, zoom_start=10, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.2}).add_to(mapa)

## plot recent incidents 
for i, row in events_df[events_df["INCIDENT_REASON"].isin(["Disturbing the Peace", "Non-Criminal Detention"])].iterrows():
    folium.CircleMarker([float(row["LOCATION_LATITUDE"]), float(row["LOCATION_LONGITUDE"])], 
                        radius=1, color='red').add_to(mapa)

## plot aggregated number of total incidents
for i, row in police_df.iterrows():
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>UseOfForce:</b> "+str(row["total_incidents"]), 
                       radius=float(row["total_incidents"])*0.003, color='green', fill=True).add_to(mapa)
    
mapa
def month_to_quarter(x):
    if x <= 3:
        return "Q1"
    elif x <= 6:
        return "Q2"
    elif x <= 9:
        return "Q3"
    else:
        return "Q4"

events_df["INCIDENT_MONTH"] = events_df["INCIDENT_DATE"].dt.month
events_df["INCIDENT_QUARTER"] = events_df["INCIDENT_MONTH"].apply(lambda x : month_to_quarter(x))

# reasons = ["Moving Traffic Violations", "Driving Under Influence", "Disturbing the Peace","Miscellaneous Other Violations", "Other Assaults"]
reasons = ["Q1", "Q2", "Q3", "Q4"]
year_dic = {}
for reason in reasons:
    yeardf = events_df[events_df["INCIDENT_QUARTER"] == reason]
    t1 = dict(yeardf["SUBJECT_RACE"].value_counts())
    t2 = {}
    for k,v in t1.items():
        t2[k] = round(100*float(v) / sum(t1.values()), 2)
    if reason not in year_dic:
        year_dic[reason] = t2

## Generate Bar Plot
data = []
races = ["Black", "White", "Hispanic", "Other"]
colors = ["Black", "White", "orange", "Green"]
for i, rac in enumerate(races):
    trace1 = go.Bar(x=list(year_dic.keys()), y=[year_dic[_][rac] for _ in year_dic], name=rac, 
                    marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    data.append(trace1)     
layout = go.Layout(title='Police Arrests by Race : 2015', height=350, 
                   yaxis=dict(range=(0,100)), legend=dict(orientation="h", x=0.1, y=1.2))
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='style-bar')
## Add Some features
races = ["Hispanic", "Black", "White"]
colors = ["orange", "black", "white"]
for c in races:
    police_df[c.lower() + "_cnt"] = police_df["arrest_race"].apply(lambda x : x[c])
    police_df[c.lower() + "_per"] = police_df["arrest_race"].apply(lambda x : 100*float(x[c]) / sum(x.values()))
    enriched_df[c.lower() + "_per"] = 100*enriched_df[c.lower() + "_pop"] / enriched_df["total_pop"]

wp = sum(enriched_df["white_pop"])
bp = sum(enriched_df["black_pop"])
hp = sum(enriched_df["hispanic_pop"])
tp = wp + hp + bp

wp1 = sum(police_df["white_cnt"])
bp1 = sum(police_df["black_cnt"])
hp1 = sum(police_df["hispanic_cnt"])
tp1 = wp1 + hp1 + bp1

data = [go.Bar(x=[100*float(wp1)/tp1, 100*float(wp)/tp], y=["Arrests", "Population"], name="Whites", 
               marker=dict(color="white", opacity=1, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[100*float(bp1)/tp1, 100*float(bp)/tp], y=[ "Arrests", "Population"], name="Blacks", 
              marker=dict(color="black", opacity=1.0, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[100*float(hp1)/tp1, 100*float(hp)/tp], y=[ "Arrests", "Population"], name="Hispanic", 
              marker=dict(color="orange", opacity=0.5, line=dict(color='black',width=1)), orientation='h')]

layout = go.Layout(barmode='stack',height=400, title='Police Arrests and Population Proportion : By Race', 
                   legend = dict(orientation="h", x=0.1, y=1.15) , showlegend=True)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


########### Next Part : Breakdown by Districts 

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=["Population Breakdown", "Arrests BreakDown"])
for i, rac in enumerate(races):
    trace1 = go.Bar(y = "D"+enriched_df[_identifier], x = enriched_df[rac.lower() + "_per"], orientation = "h", name=rac, 
                    marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    fig.append_trace(trace1, 1, 1)
    
    trace2 = go.Bar(y = "D"+police_df[_identifier], x = police_df[rac.lower() + "_per"], orientation = "h", name=rac, 
                marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    fig.append_trace(trace2, 1, 2)
    
fig["layout"].update(barmode='stack', showlegend = False, 
                     hovermode='closest', title="Race Distribution in different Districts - Population and Use-of-Force", 
                     height=900)
iplot(fig)
wp = 100 * sum(enriched_df["white_pop"]) / sum(enriched_df["total_pop"])
bp = 100 * sum(enriched_df["black_pop"]) /  sum(enriched_df["total_pop"])
hp = 100 * sum(enriched_df["hispanic_pop"]) /  sum(enriched_df["total_pop"])
op = 100 - wp - bp
xx = ["Blacks (" + str(int(bp)) + "%)", "Whites (" + str(int(wp)) + "%)", "Other (" + str(int(op)) + "%)" , "Hispanic (" + str(int(hp)) + "%)"]
        
data = [go.Scatter(x = xx, y=["","","",""], mode='markers', name="",  
                   marker=dict(color=["black","white","green" ,"orange"], opacity=1.0, size= [bp, wp, op, hp]))]
layout = go.Layout(barmode='stack', height=300, margin=dict(l=100), title='Population Distribution by Race', 
                   legend = dict(orientation="h", x=0.1, y=1.15),plot_bgcolor='#d7e9f7', paper_bgcolor='#d7e9f7', showlegend=False)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Police Behaviour Graph by different categories 
traces = []
titles = []
for vi in events_df["INCIDENT_REASON"].value_counts().index[:20]:
    if vi == "0":
        continue
    tempdf = events_df[events_df['INCIDENT_REASON'] == vi]
    tdoc = dict(tempdf["SUBJECT_RACE"].value_counts()[:3])
    pdoc = {"H" : 100*float(tdoc["Hispanic"]) / sum(tdoc.values()), 
            "W" : 100*float(tdoc["White"]) / sum(tdoc.values()),
            "B" : 100*float(tdoc["Black"]) / sum(tdoc.values())}

    xx = ["Blacks", "White", "Hispanic"]
    yy = ["", "", ""]
    ss = [pdoc[_[0]] for _ in xx]
    xx = ["Blacks (" + str(int(ss[0])) + "%)", "Whites(" + str(int(ss[1])) + "%)", "Hispanic (" + str(int(ss[2])) + "%)"]
    trace0 = go.Scatter(x=xx, y=yy, mode='markers', name="",  marker=dict(color=["black", "white", "orange"], opacity=1.0, size=ss))
    traces.append(trace0)
    titles.append(vi)

fig = tools.make_subplots(rows=5, cols=3, print_grid=False, subplot_titles = titles[:15])

r, c = 1, 1
for trace in traces[:15]:
    fig.append_trace(trace, r, c)
    c += 1 
    if c == 4:
        r += 1
        c = 1

fig["layout"].update(showlegend = False, height = 1000, plot_bgcolor='#d7e9f7', paper_bgcolor='#d7e9f7',
                     title = "Different Police Arrests Reasons",
                     xaxis=dict(showgrid=False))
iplot(fig, filename='bubblechart-color')  
# Add hover text
def cbp(df, col1, col2, aggcol, func, title, cs, bottom_margin=None):
    tempdf = df.groupby([col1, col2]).agg({aggcol : func}).reset_index()
    tempdf[aggcol] = tempdf[aggcol].apply(lambda x : int(x))
    tempdf = tempdf.sort_values(aggcol, ascending=False)

    sizes = list(reversed([i for i in range(10,31)]))
    intervals = int(len(tempdf) / len(sizes))
    size_array = [9]*len(tempdf)
    
    st = 0
    for i, size in enumerate(sizes):
        for j in range(st, st+intervals):
            size_array[j] = size 
        st = st+intervals
    tempdf['size_n'] = size_array
    # tempdf = tempdf.sample(frac=1).reset_index(drop=True)

    cols = list(tempdf['size_n'])

    trace1 = go.Scatter( x=tempdf[col1], y=tempdf[col2], mode='markers', text=tempdf[aggcol],
        marker=dict( size=tempdf.size_n, color=cols, colorscale=cs ))
    data = [trace1]
    if bottom_margin:
        layout = go.Layout(title=title, margin=dict(b=150))
    else:
        layout = go.Layout(title=title)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
    
###

tmp = events_df.groupby("agebin").agg({'SUBJECT_GENDER' : "count"}).reset_index().rename(columns={"SUBJECT_GENDER" : "count"})
tmp = tmp[~tmp["agebin"].isin(["60+", "<20"])]
tmp["percentage"] = tmp["count"].apply(lambda x : 100*float(x) / sum(tmp["count"]))

keys = ["20_24_pop", "25_34_pop", "35_44_pop", "45_54_pop", "55_59_pop"]
tmp2 = pd.DataFrame()
tmp2["agebin"] = [x.replace("_","-").replace("-pop", "") for x in keys]
tmp2["cnt"] = [sum(enriched_df[k]) for k in keys]
tmp2["percentage"] = [100*y/sum(tmp2["cnt"]) for y in tmp2["cnt"]]

trace1 = go.Bar(x = tmp2["agebin"], y = tmp2["percentage"], marker=dict(color="purple", opacity=0.6))
trace2 = go.Bar(x = tmp["agebin"], y = tmp["percentage"], marker=dict(color="orange", opacity=0.6))

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=["Population % Breakdown by Age", "Use-of-force % BreakDown by Age"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig["layout"].update(barmode='group', showlegend = False, 
                     hovermode='closest', title="", 
                     height=450)
iplot(fig)

###

cbp(events_df[~events_df["SUBJECT_RACE"].isin(["-"])], 'SUBJECT_RACE', 'agebin', 'SUBJECT_GENDER', 'count', "Use of Force breakdown by Age and Race", 'Jet')
mapb = folium.Map(center_ll, zoom_start=10, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.5}).add_to(mapb)

## plot recent incidents 
small_df = events_df[(events_df["INCIDENT_REASON"].isin(["Moving Traffic Violations"])) & (events_df["INCIDENT_MONTH"] > 7)]
for i, row in small_df.iterrows():
    folium.CircleMarker([float(row["LOCATION_LATITUDE"]), float(row["LOCATION_LONGITUDE"])], 
                        radius=1, color='orange').add_to(mapb)

## plot aggregated number of total incidents
for i, row in police_df.iterrows():
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Vehicle Stops:</b> "+str(row["total_vstops"]), 
                       radius=float(row["total_vstops"])*0.001, color='white', fill=True).add_to(mapb)
mapb
races = ["Blacks", "Whites", "Hispanic"]
for c in races:
    police_df[c.lower() + "_vcnt"] = police_df["vstops_race"].apply(lambda x : x[c[0]])
    police_df[c.lower() + "_vper"] = police_df["vstops_race"].apply(lambda x : 100*float(x[c[0]]) / (x["B"] + x["W"] + x["H"]))

plt.figure(figsize=(16,16))

for i in range(1, 22):
    tmp = police_df[police_df[_identifier] == str(i)]
    plt.subplot(5, 5, i)
    docs = [tmp["blacks_vper"].iloc(0)[0], tmp["whites_vper"].iloc(0)[0], tmp["hispanic_vper"].iloc(0)[0]]
    plt.pie(docs, labels=["Blacks", "Whites", "Hispanic"],
            colors=["black", "white", "green"],wedgeprops={"edgecolor":"orange",'linewidth': 1})
    plt.title("Dist: " + str(i))
    
    
plt.rcParams['axes.facecolor']='red'
plt.rcParams['savefig.facecolor']='red'
t1  = enriched_df[[_identifier, 'black_per', 'white_per']]
t1  = t1.merge(police_df[[_identifier, 'blacks_vper', 'whites_vper']], on=_identifier)


data_radar = [
    go.Scatterpolar(
      r = list(t1.black_per.values),
      theta = ["D"+ str(_) for _ in t1[_identifier].values],
      fill = 'toself',
      name = 'Blacks Population Proportion'
    ),
    go.Scatterpolar(
      r = list(t1.blacks_vper.values),
      theta = ["D"+ str(_) for _ in t1[_identifier].values],
      fill = 'tonext',
      name = 'Blacks Vehicle Stops Proportion'
    )
]

layout = go.Layout(margin=dict(l=120), width=800, title="Vehicle Stops and Population Proportion of Blacks",
                   legend=dict(orientation="h"),  polar = dict(
    radialaxis = dict(visible = False )))

fig = go.Figure(data=data_radar, layout=layout)
iplot(fig)


data_radar = [
    go.Scatterpolar(
      r = list(t1.white_per.values),
      theta = ["D"+ str(_) for _ in t1[_identifier].values],
      fill = 'toself',
      name = 'Whites Population Proportion'
    ),
    go.Scatterpolar(
      r = list(t1.whites_vper.values),
      theta = ["D"+ str(_) for _ in t1[_identifier].values],
      fill = 'toself',
      name = 'Whites Vechile Stops Proportion'
    )
]

layout = go.Layout(margin=dict(l=120), width=800, title="Vehicle Stops and Population Proportion of Whites", legend=dict(orientation="h"),  polar = dict(
    radialaxis = dict(visible = False)))

fig = go.Figure(data=data_radar, layout=layout)
iplot(fig)
tempdf = police_df[["LOCATION_DISTRICT", "total_vstops", "total_incidents"]]
tempdf = tempdf.merge(enriched_df[['LOCATION_DISTRICT', 'blacks_income', 'whites_income']], on="LOCATION_DISTRICT")
tempdf

trace0 = go.Scatter(x=tempdf["blacks_income"], y=tempdf["whites_income"],
    mode='markers', marker=dict(size=tempdf["total_vstops"]*0.003, color="green", opacity=0.7))
data = [trace0]
layout = go.Layout(title="Vehicle Stops by Median Income", xaxis=dict(title="Median Income of Blacks"), yaxis=dict(title="Median Income of Whites"))
fig = go.Figure(data=data, layout = layout)
iplot(fig, filename='bubblechart-color')


####### 

tempdf = police_df[["LOCATION_DISTRICT", "total_vstops", "total_incidents"]]
tempdf = tempdf.merge(enriched_df[['LOCATION_DISTRICT', 'blacks_ep_ratio', 'whites_ep_ratio']], on="LOCATION_DISTRICT")

trace0 = go.Scatter(x=tempdf["blacks_ep_ratio"], y=tempdf["whites_ep_ratio"], mode='markers', 
                    marker=dict(opacity=1, size=tempdf["total_vstops"]*0.003, color="pink"))
data = [trace0]
layout = go.Layout(title="Vehicle Stops by Employment to Population Ratio", 
                   xaxis=dict(title="Blacks : Employment to Population Ratio"), yaxis=dict(title="Whites: Employment to Population Ratio"))
fig = go.Figure(data=data, layout = layout)
iplot(fig, filename='bubblechart-color')
enriched_cols = ['LOCATION_DISTRICT', 'whites_income',
       'blacks_income', 'hispanic_income', 'below_pov_pop',
       'whites_ep_ratio', 'blacks_ep_ratio', 'whites_unemp_ratio',
       'blacks_unemp_ratio', 'hispanic_per', 'black_per', 'white_per']

police_cols = ['LOCATION_DISTRICT', 'total_incidents' , 'total_vstops']


merged_df = police_df[police_cols].merge(enriched_df[enriched_cols], on=_identifier)

## give a high level overview summary as well
merged_df = merged_df.rename(columns={"whites_income" : "Median Income : Whites", 
                                      "blacks_income" : "Median Income : Blacks",
                                      "hispanic_income" : "Median Income : Hispanic",
                                      "below_pov_pop" : "Below Poverty Population",
                                      "whites_ep_ratio" : "Employment : Whites",
                                      "blacks_ep_ratio" : "Employment : Blacks",
                                      "whites_unemp_ratio" : "Unemployment : Whites",
                                      "blacks_unemp_ratio" : "Unemployment : Blacks",
                                      "hispanic_per" : "Hispanic Population %",
                                      "black_per" : "Black Population %",
                                      "white_per" : "White Population %",
                                      "total_vstops" : "Total Vehicle Stops", 
                                      "total_incidents" : "Total Arrests"
                                     })

import plotly.figure_factory as ff

corr = merged_df.corr(method='pearson').round(2)
xcols = list(merged_df.columns)[1:]
ycols = list(merged_df.columns)[1:]

layout = dict(
    title = 'Ordinal feature correlations',
    width = 900,
    height = 900,
    # margin=go.Margin(l=200, r=50, b=50, t=250, pad=4),
    margin=go.layout.Margin(l=200, r=50, b=50, t=250, pad=4),
)
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(xcols),
    y=list(ycols),
    colorscale='Reds',
    reversescale=False,
    showscale=True,
    )
fig['layout'].update(layout)
iplot(fig, filename='OrdinalCorrelations')
import statsmodels.api as sm_api
from sklearn import preprocessing

merged_df = police_df[police_cols].merge(enriched_df, on=_identifier)
cols = [c for c in merged_df.columns if c not in ["LOCATION_DISTRICT", "total_incidents"]]

x = merged_df[cols].values
min_max_scaler = preprocessing.StandardScaler()
t4 = min_max_scaler.fit_transform(x)
t4 = pd.DataFrame(t4, columns = cols)

t4["whites_to_blacks_ratio"] = t4["white_pop"] / t4["black_pop"]
cols.extend(["whites_to_blacks_ratio"])

target = "total_vstops"
features = [c for c in cols if target not in c]
features = [c for c in features if "per" not in c.lower()]
features = [c for c in features if "pop" not in c.lower()]
features = [c for c in features if c not in ["whites_ep_ratio"]]
X = t4[features]
Y = t4[target]
model = sm_api.OLS(Y, X)
results = model.fit()
results.summary()
# cols