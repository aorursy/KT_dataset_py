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

def _get_dfs(_dept):
    _base_dir = "../input/3-example-runs-of-automation-pipeline/CPE_ROOT/" + _dept

    enriched_df = pd.read_csv(_base_dir + "/enriched_df.csv")
    police_df = pd.read_csv(_base_dir + "/police_df.csv")
    shape_gdf = _read_shape_gdf(_base_dir, _dept)

    ## Convert Dictionary Columns 
    for c in police_df.columns:
        if c != _identifier:
            police_df[c] = police_df[c].apply(ast.literal_eval)
            
    events_df = pd.read_csv(_base_dir + "/events/events_df.csv", low_memory=False, parse_dates = ["INCIDENT_DATE"])[1:]

    ## custom code - only for _dept : Dept_23-00089
    if _dept == "Dept_23-00089":
        police_df[_identifier] = police_df[_identifier].fillna("")
        events_df[_identifier] = events_df[_identifier].fillna("")
        
        police_df[_identifier] = police_df[_identifier].apply(lambda x : x.replace(" District", ""))
        events_df[_identifier] = events_df[_identifier].apply(lambda x : x.replace(" District", ""))

    shape_gdf = shape_gdf.rename(columns = {depts_config[_dept]['_rowid'] : _identifier})
    shape_gdf[_identifier] = shape_gdf[_identifier].astype(str)
    enriched_df[_identifier] = enriched_df[_identifier].astype(str)
    police_df[_identifier] = police_df[_identifier].astype(str)
    police_df = police_df.merge(shape_gdf[[_identifier, "geometry"]], on=_identifier)
    
    
    if "SUBJECT_AGE" in events_df.columns:
        events_df["agebin"] = events_df["SUBJECT_AGE"].apply(lambda x : _agebin(x))
    
    return enriched_df, police_df, shape_gdf, _base_dir, events_df
_dept = "Dept_23-00089"
enriched_df, police_df, shape_gdf, _base_dir, events_df = _get_dfs(_dept)

white_subjects = events_df[events_df["SUBJECT_RACE"] == "White"]
black_subjects = events_df[events_df["SUBJECT_RACE"] == "Black"]
races = ["Black", "White"]
colors = ["black", "white"]
for c in races:
    enriched_df[c.lower() + "_per"] = 100*enriched_df[c.lower() + "_pop"] / enriched_df["total_pop"]
enriched_df["other_per"] = 100 - enriched_df["white_per"] - enriched_df["black_per"]

wp = 100 * sum(enriched_df["white_pop"]) / sum(enriched_df["total_pop"])
bp = 100 * sum(enriched_df["black_pop"]) / sum(enriched_df["total_pop"])
op = 100 - wp - bp

data = [go.Bar(x=[wp], y=["Population"], name="Whites", 
               marker=dict(color="white", opacity=1, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[bp], y=["Population"], name="Blacks", 
              marker=dict(color="black", opacity=0.5, line=dict(color='black',width=1)), orientation='h'),
       go.Bar(x=[op], y=["Population"], name="Others", 
              marker=dict(color="orange", opacity=0.5, line=dict(color='black',width=1)), orientation='h')]

layout = go.Layout(barmode='stack',height=250, title='Population Proportion : By Race', 
                   legend = dict(orientation="h", x=0.1, y=1.35), xaxis=dict(title="Population Proportion %"), showlegend=True)
fig = go.Figure(data=data, layout=layout)
iplot(fig)




race_doc_w = dict(white_subjects["OFFICER_AGE"].value_counts())
race_doc_b = dict(black_subjects["OFFICER_AGE"].value_counts())

race_per_w = {}
for k, v in race_doc_w.items():
    race_per_w[k] = round(100 * float(v) / sum(race_doc_w.values()), 2)

race_per_b = {}
for k, v in race_doc_b.items():
    race_per_b[k] = round(100 * float(v) / sum(race_doc_b.values()), 2)


## plot
trace1 = go.Bar(x=list(race_per_w.keys()), y=list(race_per_w.values()), name="Use of Force on Whites", 
                marker=dict(color="white", line=dict(width=1, color="black")))
trace2 = go.Bar(x=list(race_per_b.keys()), y=list(race_per_b.values()), name="Use of Force on Blacks", 
                marker=dict(color="black", opacity=0.5, line=dict(width=1, color="black")))

layout = go.Layout(title='Use-of-Force on Blacks and Whites by Officer Age', height=450, 
                   xaxis=dict(range=(21, 52.5), title="Officer Age"), 
                   yaxis=dict(title="Percentage of Subjects", range=(0,10)), 
                   legend=dict(orientation="h", x=0.1, y=1.17))
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig, filename='style-bar')
race_doc_w = dict(white_subjects["OFFICER_YEARS_ON_FORCE"].value_counts())
race_doc_b = dict(black_subjects["OFFICER_YEARS_ON_FORCE"].value_counts())

race_per_w = {}
for k, v in race_doc_w.items():
    race_per_w[k] = round(100 * float(v) / sum(race_doc_w.values()), 2)

race_per_b = {}
for k, v in race_doc_b.items():
    race_per_b[k] = round(100 * float(v) / sum(race_doc_b.values()), 2)


## plot
trace1 = go.Bar(x=list(race_per_w.keys()), y=list(race_per_w.values()), name="Use of Force on Whites", 
                marker=dict(color="white", line=dict(width=1, color="black")))
trace2 = go.Bar(x=list(race_per_b.keys()), y=list(race_per_b.values()), name="Use of Force on Blacks", 
                marker=dict(color="black", opacity=0.5, line=dict(width=1, color="black")))

layout = go.Layout(title='Use-of-Force on Blacks and Whites by Officer Years on Force', height=450, 
                   xaxis=dict(title="Officer Years on Force", range=(0,20.5)), 
                   yaxis=dict(title="Percentage of Subjects", range=(0,20)), 
                   legend=dict(orientation="h", x=0.1, y=1.17))
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig, filename='style-bar')
small_df = events_df[events_df["SUBJECT_RACE"].isin(["Black", "White", "Hispanic"])]
small_df = small_df[small_df["OFFICER_RACE"].isin(["Black", "White", "Hispanic"])]

cols = ['SUBJECT_RACE', 'OFFICER_RACE']
colmap = sns.light_palette("gray", as_cmap=True)
tb = pd.crosstab(small_df[cols[1]], small_df[cols[0]]).apply(lambda r: round(100*r/r.sum(),2), axis=1)
tb = tb.style.background_gradient(cmap = colmap)
tb
from collections import Counter 
def find_percentage(row):
    races = row["SUBJECT_RACE"]
    per_dic = {}
    for race, cnt in races.items():
        per_dic[race] = round(100 * float(cnt) / row["uof_count"], 2)
    return per_dic 


officer_df = events_df.groupby('OFFICER_ID').agg({"SUBJECT_RACE" : lambda x : Counter("|".join(x).split("|")) ,
                                                 "SUBJECT_GENDER" : "count"}).reset_index().rename(\
                                            columns = {"SUBJECT_GENDER" : "uof_count"})
officer_df = officer_df.sort_values("uof_count", ascending = False)
officer_df = officer_df[officer_df["uof_count"] >= 3]

officer_df["race_percent"] = officer_df.apply(lambda row : find_percentage(row), axis = 1)
officer_df["black_subjects_per"] = officer_df["race_percent"].apply(lambda x : x["Black"] if "Black" in x else 0.0)
officer_df["white_subjects_per"] = officer_df["race_percent"].apply(lambda x : x["White"] if "White" in x else 0.0)
officer_df["hispanic_subjects_per"] = officer_df["race_percent"].apply(lambda x : x["Hispanic"] if "Hispanic" in x else 0.0)
officer_df["difference_b_w"] = officer_df.apply(lambda x : x["black_subjects_per"] - x["white_subjects_per"], axis = 1)
officer_df = officer_df.drop(["SUBJECT_RACE", "race_percent"], axis = 1)
officer_df.head()
trace1 = go.Histogram(x=officer_df.black_subjects_per, name = "Blacks - Count of Force used", marker=dict(color="black", opacity=0.5))
trace2 = go.Histogram(x=officer_df.white_subjects_per, name = "Whites - Count of Force used", marker=dict(color="white", opacity=1.0, 
                                                                                                         line = dict(width=1, color="black")))
layout = go.Layout(title='', height=400, 
                   xaxis=dict(title="Officer's percentage of Force Used on Black / White ", range=(2, 98)), 
                   yaxis=dict(title="Number of Officers", range=(0, 130)), 
                   legend=dict(orientation="h", x=0.1, y=1.17))
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig, filename='style-bar')
from IPython.core.display import display, HTML

display(HTML("<h3>Key Statistics : Extent of Racial Disparity</h3>"))

val = round(100 * officer_df[officer_df["black_subjects_per"] >= 40].shape[0] / officer_df.shape[0])
html = "<font size=7>" + str(val) + "% </font> percentage of officers who have <b>used force more on blacks than whites</b>"
display(HTML(html))

val = round(100 * officer_df[officer_df["black_subjects_per"] >= 90].shape[0] / officer_df.shape[0])
html = "<font size=7>" + str(val) + "% </font> percentage of officers who have <b>used force more only on Blacks</b>"
display(HTML(html))

val = round(100 * officer_df[officer_df["white_subjects_per"] >= 90].shape[0] / officer_df.shape[0])
html = "<font size=7>" + str(val) + "% </font> percentage of officers who have <b>used force more only on Whites</b>"
display(HTML(html))

val = round(100 * officer_df[officer_df["difference_b_w"] >= 50].shape[0] / officer_df.shape[0])
html = "<font size=7>" + str(val) + "% </font> officers with <b>large difference (>50%) in force used on blacks and whites</b>"
display(HTML(html))
severity_score = {}
for x in events_df["TYPE_OF_FORCE_USED"].value_counts().index:
    if x.startswith("Physical"):
        severity_score[x] = 1
    elif x.startswith("Canine"):
        severity_score[x] = 2
    elif "CS/OC" in x:
        severity_score[x] = 4.5
    elif x.startswith("Less Lethal"):
        severity_score[x] = 5
    elif x == "Lethal-Vehicle":
        severity_score[x] = 10
    elif x.startswith("Lethal"):
        severity_score[x] = 8

events_df = events_df[~events_df["TYPE_OF_FORCE_USED"].isna()]
events_df["uof_severity"] = events_df["TYPE_OF_FORCE_USED"].apply(lambda x : severity_score[x])
t1 = events_df[events_df['INCIDENT_YEAR'] == 2015].groupby("SUBJECT_RACE").agg({"uof_severity" : "mean" }).reset_index()

trace1 = go.Bar(x = t1.SUBJECT_RACE[2:], y=t1.uof_severity[2:], name="Average Severity", 
                marker=dict(color="gray", opacity = 0.4))

layout = go.Layout(title='Use-of-Force Average Severity by Race of the Subject', height=400,
                   legend=dict(orientation="h", x=0.1, y=1.17), 
                  yaxis=dict(range=(0.85, 2.3)))
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='style-bar')
import statsmodels.api as sm_api
from sklearn import preprocessing

merged_df = enriched_df.drop_duplicates().merge(events_df, on = "LOCATION_DISTRICT")

merged_df['is_same_race'] = merged_df.apply(lambda x : 1 if x["SUBJECT_RACE"] == x["OFFICER_RACE"] else 0, axis = 1)
merged_df['is_black_subj'] = merged_df.apply(lambda x : 1 if x["SUBJECT_RACE"] == "Black" else 0, axis = 1)
merged_df['is_white_subj'] = merged_df.apply(lambda x : 1 if x["SUBJECT_RACE"] == "White" else 0, axis = 1)
merged_df['is_hispanic_subj'] = merged_df.apply(lambda x : 1 if x["SUBJECT_RACE"] == "Hispanic" else 0, axis = 1)

merged_df["OFFICER_AGE"] = merged_df["OFFICER_AGE"].fillna(30.0)
merged_df["OFFICER_YEARS_ON_FORCE"] = merged_df["OFFICER_YEARS_ON_FORCE"].fillna(5.0)
# merged_df = merged_df.dropna()

merged_df["white_per"] = merged_df["white_pop"] / merged_df["total_pop"]
merged_df["black_per"] = merged_df["black_pop"] / merged_df["total_pop"]
merged_df["hispanic_per"] = merged_df["hispanic_pop"] / merged_df["total_pop"]
merged_df["w2b"] = merged_df["white_pop"] / merged_df["black_pop"]


cols = ["is_black_subj", "is_white_subj", "is_hispanic_subj", "w2b", "white_per", "black_per", "hispanic_per", 'whites_income',
       'blacks_income', 'below_pov_pop',
       'whites_ep_ratio', 'blacks_ep_ratio', 'whites_unemp_ratio',
       'blacks_unemp_ratio', "OFFICER_AGE", "OFFICER_YEARS_ON_FORCE", "is_same_race"]

x = merged_df[cols].values
min_max_scaler = preprocessing.StandardScaler()
t4 = min_max_scaler.fit_transform(x)
t4 = pd.DataFrame(t4, columns = cols)

target = "uof_severity"
X = t4[cols]
Y = np.log1p(merged_df[target])
model = sm_api.OLS(Y, X)
results = model.fit()
results.summary()