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

def _agebin(x):
    if str(x).lower() == "nan":
        return None
    
    ranges = [20, 24, 34, 44, 54, 59, 100]
    tags = ["<20", "20-24", "25-34", "35-44", "45-54", "55-59", "60+"]
    for i, rng in enumerate(ranges):
        if int(x) <= rng:
            return tags[i]


_identifier = "LOCATION_DISTRICT"

def _get_dfs(_dept):
    _base_dir = "../input/3-example-runs-of-automation-pipeline/CPE_ROOT/" + _dept

    enriched_df = pd.read_csv(_base_dir + "/enriched_df.csv")
    police_df = pd.read_csv(_base_dir + "/police_df.csv")
    shape_gdf = _read_shape_gdf(_base_dir, _dept)

    ## Convert Dictionary Columns 
    for c in police_df.columns:
        if c != _identifier:
            police_df[c] = police_df[c].apply(ast.literal_eval)

    shape_gdf = shape_gdf.rename(columns = {"PRECINCT" : _identifier})
    shape_gdf[_identifier] = shape_gdf[_identifier].astype(str)
    enriched_df[_identifier] = enriched_df[_identifier].astype(str)
    police_df[_identifier] = police_df[_identifier].astype(str)
    police_df = police_df.merge(shape_gdf[[_identifier, "geometry"]], on=_identifier)
    
    events_df = pd.read_csv(_base_dir + "/events/events_df.csv", low_memory=False, parse_dates = ["INCIDENT_DATE"])[1:]
    
    if "SUBJECT_AGE" in events_df.columns:
        events_df["agebin"] = events_df["SUBJECT_AGE"].apply(lambda x : _agebin(x))
    
    return enriched_df, police_df, shape_gdf, _base_dir, events_df
_dept = "Dept_24-00013"
enriched_df, police_df, shape_gdf, _base_dir, events_df = _get_dfs(_dept)

############# Also load the external dataset used #######################
external_data_path = "../input/external-datasets-cpe/minneapolis_stops/Minneapolis_Stops.csv"
vstops = pd.read_csv(external_data_path, low_memory=False)
vstops["responseDate"] = pd.to_datetime(vstops["responseDate"])
vstops["year"] = vstops["responseDate"].dt.year
vstops["month"] = vstops["responseDate"].dt.month

enriched_df.head()
"""
About the Code in this Cell 

This code produces the folium map as the base map, on which we plot three items : district polygons, aggregated number of total incidents by district, recent incidents
"""

## Add some more features
police_df["total_incidents"] = police_df["uof_sex"].apply(lambda x : sum(x.values()))
police_df["total_vstops"] = police_df["vstops_sex"].apply(lambda x : sum(x.values()))

## Plot the base map
center_pt = police_df.geometry[0].centroid
center_ll = _get_latlong_point(center_pt)
mapa = folium.Map(center_ll, zoom_start=10.5, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.2}).add_to(mapa)

## plot recent incidents 
for i, row in events_df[events_df["INCIDENT_YEAR"] == 2015].iterrows():
    folium.CircleMarker([float(row["LOCATION_LATITUDE"]), float(row["LOCATION_LONGITUDE"])], 
                        radius=1, color='red').add_to(mapa)

## plot aggregated number of total incidents
for i, row in police_df.iterrows():
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>UseOfForce:</b> "+str(row["total_incidents"]), 
                       radius=float(row["total_incidents"])*0.01, color='green', fill=True).add_to(mapa)
    
mapa
## plot the race distribution by different year
years = [2012, 2013, 2014, 2015, 2016, 2017]
year_dic = {}
for year in years:
    yeardf = events_df[events_df["INCIDENT_YEAR"] == year]
    t1 = dict(yeardf["SUBJECT_RACE"].value_counts())
    t2 = {}
    for k,v in t1.items():
        t2[k] = round(100*float(v) / sum(t1.values()), 2)
    if year not in year_dic:
        year_dic[year] = t2

## Generate Bar Plot
data = []
races = ["White","Black", "Native American", "Other", "Asian"]
colors = ["White","Black", "#6599ed", "Green", "Orange"]
for i, rac in enumerate(races):
    trace1 = go.Bar(x=list(year_dic.keys()), y=[year_dic[_][rac] for _ in year_dic], name=rac, 
                    marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    data.append(trace1)     
layout = go.Layout(title='Use-of-Force by Race : 2012 - 2017', height=350, 
                   yaxis=dict(range=(0,100), title="Use of Force %"), legend=dict(orientation="h", x=0.1, y=1.2))
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='style-bar')
## Add Some features
races = ["Black", "White", "Asian"]
colors = ["black", "white", "orange"]
for c in races:
    police_df["total"] = police_df["uof_race"].apply(lambda x : sum(x.values()))
    police_df[c.lower() + "_cnt"] = police_df["uof_race"].apply(lambda x : x[c])
    police_df[c.lower() + "_per"] = police_df["uof_race"].apply(lambda x : 100*float(x[c]) / sum(x.values()))
    
    enriched_df[c.lower() + "_per"] = 100*enriched_df[c.lower() + "_pop"] / enriched_df["total_pop"]

enriched_df["other_per"] = 100 - enriched_df["white_per"] - enriched_df["black_per"] - enriched_df["asian_per"]

wp = 100 * sum(enriched_df["white_pop"]) / sum(enriched_df["total_pop"])
bp = 100 * sum(enriched_df["black_pop"]) / sum(enriched_df["total_pop"])
hp = 100 * sum(enriched_df["asian_pop"]) / sum(enriched_df["total_pop"])
op = 100 - wp - bp - hp

wp1 = 100*sum(police_df["white_cnt"]) / sum(police_df["total"])
bp1 = 100*sum(police_df["black_cnt"]) / sum(police_df["total"])
hp1 = 100*sum(police_df["asian_cnt"])/ sum(police_df["total"])
op1 = 100 - wp1 - hp1 - bp1 

data = [go.Bar(x=[wp1, wp], y=["UseOfForce", "Population"], name="Whites", 
               marker=dict(color="white", opacity=1, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[bp1, bp], y=[ "UseOfForce", "Population"], name="Blacks", 
              marker=dict(color="black", opacity=1.0, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[hp1, hp], y=[ "UseOfForce", "Population"], name="Asian", 
              marker=dict(color="orange", opacity=0.5, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[op1, op], y=[ "UseOfForce", "Population"], name="Others", 
              marker=dict(color="purple", opacity=0.5, line=dict(color='black',width=1)), orientation='h')]

layout = go.Layout(barmode='stack',height=400, title='Use-of-Force and Population Proportion : By Race', 
                   legend = dict(orientation="h", x=0.1, y=1.15), xaxis=dict(title="Use of Force %"), showlegend=True)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


########### Next Part : Breakdown by Districts 

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=["Population Breakdown", "Use Of Force BreakDown"])
for i, rac in enumerate(races):
    trace1 = go.Bar(y = "D"+enriched_df[_identifier], x = enriched_df[rac.lower() + "_per"], orientation = "h", name=rac, 
                    marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    fig.append_trace(trace1, 1, 1)
    
    trace2 = go.Bar(y = "D"+police_df[_identifier], x = police_df[rac.lower() + "_per"], orientation = "h", name=rac, 
                marker=dict(color=colors[i], opacity=1.0, line=dict(color='black',width=1)))
    fig.append_trace(trace2, 1, 2)
    
fig["layout"].update(barmode='stack', showlegend = False, 
                     hovermode='closest', title="Use-of-Force and Population Proportion : By Race, District", 
                     height=600)
iplot(fig)
events_df[_identifier] = events_df[_identifier].apply(lambda x : str(x).replace(".0",""))

tempdf = events_df[events_df["SUBJECT_INJURY"] == -1]
titles = []
fig = tools.make_subplots(rows=1, cols=5, print_grid=False, subplot_titles=["Dist:"+str(i) for i in range(1, 6)])
traces = []
cnt = 1

for idf in police_df[_identifier].values:
    sdf = dict(tempdf[tempdf[_identifier] == idf]["SUBJECT_RACE"].value_counts())
    _perdic = {}
    for x, y in sdf.items():
        _perdic[x] = 100*float(y) / sum(sdf.values())
    
    _perdic2 = {"Black" : _perdic["Black"], "White" : _perdic["White"], "Others" : 100 - _perdic["Black"] - _perdic["White"]}
    tr = go.Bar(x = list(_perdic2.keys()), y = list(_perdic2.values()), 
                name="Dist: "+str(idf), marker=dict(color=["Black", "White", "Orange"], 
                                                   line=dict(color='black',width=1)))
    titles.append("Dist: "+str(idf))
    fig.append_trace(tr, 1, cnt)
    cnt += 1

fig["layout"].update(barmode='stack', showlegend = False, 
                     hovermode='closest', title="% of Subjects Injured by Race in 5 Districts", height=300)
iplot(fig)
wp = sum(enriched_df["white_pop"]) / sum(enriched_df["total_pop"])
bp = sum(enriched_df["black_pop"]) / sum(enriched_df["total_pop"])
hp = sum(enriched_df["asian_pop"]) / sum(enriched_df["total_pop"])
tp = wp + hp + bp

wp = 100 * sum(enriched_df["white_pop"]) / sum(enriched_df["total_pop"])
bp = 100 * sum(enriched_df["black_pop"]) /  sum(enriched_df["total_pop"])
op = 100 - wp - bp
xx = ["Blacks (" + str(int(bp)) + "%)", "Whites (" + str(int(wp)) + "%)", "Others (" + str(int(op)) + "%)"]
        
data = [go.Scatter(x = xx, y=["","",""], mode='markers', name="",  
                   marker=dict(color=["black","white", "orange"], opacity=1.0, size= [bp, wp, op]))]
layout = go.Layout(barmode='stack', height=300, margin=dict(l=100), title='Population Distribution by Race', 
                   legend = dict(orientation="h", x=0.1, y=1.15),plot_bgcolor='#d7e9f7', paper_bgcolor='#d7e9f7', showlegend=False)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


## Police Behaviour Graph by different categories 
traces = []
titles = []
for vi in events_df["WEAPON_OR_TOOL_USED"].value_counts().index[:21]:
    if vi == "0":
        continue
    tempdf = events_df[events_df['WEAPON_OR_TOOL_USED'] == vi]
    tdoc = dict(tempdf["SUBJECT_RACE"].value_counts())
    if "Black" in tdoc:
        pdoc = {"B" : 100*float(tdoc["Black"]) / sum(tdoc.values()), 
                "W" : 100*float(tdoc["White"]) / sum(tdoc.values()),
                "O" : 100*float(sum(tdoc.values()) - tdoc["White"] - tdoc["Black"]) / sum(tdoc.values())}
        
        xx = ["Blacks", "White", "Others"]
        yy = ["", "", ""]
        ss = [pdoc[_[0]] for _ in xx]
        xx = ["Blacks (" + str(int(ss[0])) + "%)", "Whites(" + str(int(ss[1])) + "%)", "Others (" + str(int(ss[2])) + "%)"]
        trace0 = go.Scatter(x=xx, y=yy, mode='markers', name="",  marker=dict(color=["black","white", "orange"], opacity=1.0, size=ss))
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
                     title = "Different Use-of-Force used by Police",
                    xaxis=dict(showgrid=False))
iplot(fig, filename='bubblechart-color')  
## Add hover text
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
                     height=400)
fig["layout"].yaxis1.range = (0, 50)
fig["layout"].yaxis2.range = (0, 50)
iplot(fig)

###

cbp(events_df, 'SUBJECT_RACE', 'agebin', 'SUBJECT_GENDER', 'count', "Use of Force breakdown by Age and Race", 'Jet')
_vs = vstops[(vstops["year"] == 2017) & (vstops["month"] == 1)]
_vs = _vs[~_vs["race"].isin(["Unknown", "not recorded"])]
_vs = _vs[_vs["problem"] == "Traffic Law Enforcement (P)"]

mapb = folium.Map(center_ll, zoom_start=10.5, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.5}).add_to(mapb)
for i, row in _vs.iterrows():
    folium.CircleMarker([float(row["lat"]), float(row["long"])], radius=1, color='white').add_to(mapb)

for i, row in police_df.iterrows():    
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Vehicle Stops:</b> "+str(row["total_vstops"]), 
                       radius=float(row["total_vstops"])*0.02, color='orange', fill=True).add_to(mapb)

mapb
races = ["Black", "White", "Latino"]
for c in races:
    police_df[c.lower() + "_vcnt"] = police_df["vstops_race"].apply(lambda x : x[c])
    police_df[c.lower() + "_vper"] = police_df["vstops_race"].apply(lambda x : 100*float(x[c]) / (x["Black"] + x["White"] + x["Latino"]))

fig = plt.figure(figsize=(16,3))
for i in range(1, 6):
    tmp = police_df[police_df[_identifier] == str(i)]
    plt.subplot(1, 5, i)
    docs = [tmp["black_vper"].iloc(0)[0], tmp["white_vper"].iloc(0)[0], tmp["latino_vper"].iloc(0)[0]]
    plt.pie(docs, labels=["Black", "Whites", "Latino"],
            colors=["black", "white", "green"],
           wedgeprops={"edgecolor":"orange",'linewidth': 1})
    plt.title("Dist: " + str(i)) 
    
plt.rcParams['axes.facecolor']='red'
plt.rcParams['savefig.facecolor']='red'
t1  = enriched_df[[_identifier, 'black_per', 'white_per']]
t1  = t1.merge(police_df[[_identifier, 'black_vper', 'white_vper']])

data_radar = [
    go.Scatterpolar(
      r = list(t1.black_per.values),
      theta = ["D"+ str(_) for _ in t1[_identifier].values],
      fill = 'toself',
      name = 'Blacks Population Proportion'
    ),
    go.Scatterpolar(
      r = list(t1.black_vper.values),
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
      r = list(t1.white_vper.values),
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
    mode='markers', marker=dict(size=tempdf["total_vstops"]*0.05, color="green", opacity=0.7))
data = [trace0]
layout = go.Layout(title="Vehicle Stops by Median Income", xaxis=dict(title="Median Income of Blacks"), yaxis=dict(title="Median Income of Whites"))
fig = go.Figure(data=data, layout = layout)
iplot(fig, filename='bubblechart-color')


####### 

tempdf = police_df[["LOCATION_DISTRICT", "total_vstops", "total_incidents"]]
tempdf = tempdf.merge(enriched_df[['LOCATION_DISTRICT', 'blacks_ep_ratio', 'whites_ep_ratio']], on="LOCATION_DISTRICT")

trace0 = go.Scatter(x=tempdf["blacks_ep_ratio"], y=tempdf["whites_ep_ratio"], mode='markers', 
                    marker=dict(opacity=1, size=tempdf["total_vstops"]*0.05, color="pink"))
data = [trace0]
layout = go.Layout(title="Vehicle Stops by Employment to Population Ratio", 
                   xaxis=dict(title="Blacks : Employment to Population Percent"), yaxis=dict(title="Whites: Employment to Population Percent"))
fig = go.Figure(data=data, layout = layout)
iplot(fig, filename='bubblechart-color')
## Load an external dataset about crime in minneapolis
crime = pd.read_csv("../input/external-datasets-cpe/minneapolis_Police_Incidents_2016.csv")

cr = crime["Precinct"].value_counts().to_frame().reset_index().rename(columns={"Precinct" : "CrimeCount", "index" : "LOCATION_DISTRICT"})[:5]
cr["LOCATION_DISTRICT"] = cr["LOCATION_DISTRICT"].astype(str)
police_df = police_df.merge(cr, on="LOCATION_DISTRICT")

mapc = folium.Map(center_ll, zoom_start=10.5, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.2}).add_to(mapc)
for i, row in crime[crime['Description'] == "Asslt W/dngrs Weapon"].iterrows():
    folium.CircleMarker([float(row["Lat"]), float(row["Long"])], radius=1, color='white').add_to(mapc)

for i, row in police_df.iterrows():    
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Vehicle Stops:</b> "+str(row["total_vstops"]), 
                       radius=float(row["total_vstops"])*0.01, color='green', fill=True).add_to(mapc)

    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Use of Force:</b> "+str(row["total_incidents"]), 
                       radius=float(row["total_incidents"])*0.01, color='gray', fill=True).add_to(mapc)

    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Total Crime: </b> "+str(row["CrimeCount"]), 
                       radius=float(row["CrimeCount"])*0.005, color='red', fill=True).add_to(mapc)

mapc
police_df = police_df.merge(enriched_df[[_identifier, "black_pop"]], on=_identifier)
mapd = folium.Map(center_ll, zoom_start=10.5, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.2}).add_to(mapd)
for i, row in crime[crime['Description'] == "Asslt W/dngrs Weapon"].iterrows():
    folium.CircleMarker([float(row["Lat"]), float(row["Long"])], radius=1, color='gray').add_to(mapd)

for i, row in police_df.iterrows():    
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Total Crime: </b> "+str(row["CrimeCount"]), 
                       radius=float(row["CrimeCount"])*0.005, color='red', fill=True).add_to(mapd)

    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Total Black Population: </b> "+str(int(row["black_pop"])), 
                       radius=float(row["black_pop"])*0.0005, color='white', fill=True).add_to(mapd)

mapd
police_df = police_df.merge(enriched_df[[_identifier, "below_pov_pop"]], on=_identifier)

mape = folium.Map(center_ll, zoom_start=10.5, tiles='CartoDB dark_matter')
folium.GeoJson(shape_gdf, style_function = lambda feature: { 'fillColor': "blue", 'color' : "blue", 'weight' : 1, 'fillOpacity' : 0.2}).add_to(mape)

for i, row in police_df.iterrows():    
    dist_ll = _get_latlong_point(row["geometry"].centroid)
    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Total Crime: </b> "+str(row["CrimeCount"]), 
                       radius=float(row["CrimeCount"])*0.005, color='red', fill=True).add_to(mape)

    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Total Below Poverty Population: </b> "+str(int(row["below_pov_pop"])), 
                       radius=float(row["below_pov_pop"])*0.0005, color='yellow', fill=True).add_to(mape)

    folium.CircleMarker(dist_ll, popup="<b>District ID:</b>" + row["LOCATION_DISTRICT"] +"<br> <b>Use of Force:</b> "+str(row["total_incidents"]), 
                       radius=float(row["total_incidents"])*0.01, color='white', fill=True).add_to(mape)

mape