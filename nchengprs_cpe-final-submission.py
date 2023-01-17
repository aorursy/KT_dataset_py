import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
import os
import datetime
import dexplot as dxp
import math
from geopy.geocoders import Nominatim
%matplotlib inline
def embed_map(m):
    from IPython.display import IFrame

    m.save('index.html')
    return IFrame('index.html', width='100%', height='750px')
data_dir = "../input/data-science-for-good/cpe-data/"
census_shape_dir = "../input/census-tract-shapefiles/census_shape/census_shape/"

departments = list(os.listdir(data_dir))
if ".DS_Store" in departments:
    departments.remove('.DS_Store')
if 'ACS_variable_descriptions.csv' in departments:
    departments.remove('ACS_variable_descriptions.csv')
department_numbers = [d[-8:] for d in departments]

state_code_lookup = {'11-00091':'MA',
                     '23-00089':'IN',
                     '24-00013':'MN',
                     '24-00098':'MN',
                     '35-00016':'FL',
                     '35-00103':'NC',
                     '37-00027':'TX',
                     '37-00049':'TX',
                     '49-00009':'WA',
                     '49-00033':'CA',
                     '49-00035':'CA',
                     '49-00081':'CA'}

FIPS_state_codes = {
    'WA': '53', 'DE': '10', 'DC': '11', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'PR': '72', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}

police_district_id_column_lookup = {'11-00091':'ID',
                                    '23-00089': 'DISTRICT',
                                    '24-00013': 'PRECINCT',
                                    '24-00098': 'gridnum',
                                    '35-00016': 'DISTRICT',
                                    '37-00049': 'Name',
                                    '49-00009': 'beat',
                                    '49-00033': 'external_i',
                                    '49-00035': 'pol_dist',
                                    '49-00081': 'district'}

education_variables = {
"HC01_EST_VC02": "Total; Estimate; Population 18 to 24 years",
"HC02_EST_VC02": "Percent; Estimate; Population 18 to 24 years",
"HC01_EST_VC03": "Total; Estimate; Population 18 to 24 years - Less than high school graduate",
"HC02_EST_VC03": "Percent; Estimate; Population 18 to 24 years - Less than high school graduate",
"HC01_EST_VC04": "Total; Estimate; Population 18 to 24 years - High school graduate (includes equivalency)",
"HC02_EST_VC04": "Percent; Estimate; Population 18 to 24 years - High school graduate (includes equivalency)",
"HC01_EST_VC05": "Total; Estimate; Population 18 to 24 years - Some college or associate's degree",
"HC02_EST_VC05": "Percent; Estimate; Population 18 to 24 years - Some college or associate's degree",
"HC01_EST_VC08": "Total; Estimate; Population 25 years and over",
"HC02_EST_VC08": "Percent; Estimate; Population 25 years and over",
"HC01_EST_VC09": "Total; Estimate; Population 25 years and over - Less than 9th grade",
"HC02_EST_VC09": "Percent; Estimate; Population 25 years and over - Less than 9th grade",
"HC01_EST_VC10": "Total; Estimate; Population 25 years and over - 9th to 12th grade, no diploma",
"HC02_EST_VC10": "Percent; Estimate; Population 25 years and over - 9th to 12th grade, no diploma",
"HC01_EST_VC11": "Total; Estimate; Population 25 years and over - High school graduate (includes equivalency)",
"HC02_EST_VC11": "Percent; Estimate; Population 25 years and over - High school graduate (includes equivalency)",
"HC01_EST_VC12": "Total; Estimate; Population 25 years and over - Some college, no degree",
"HC02_EST_VC12": "Percent; Estimate; Population 25 years and over - Some college, no degree"
}

race_sex_age_variables = {
"HC01_VC03": "Estimate; SEX AND AGE - Total population",
"HC03_VC03": "Percent; SEX AND AGE - Total population",
"HC01_VC48": "Estimate; RACE - One race",
"HC03_VC48": "Percent; RACE - One race",
"HC01_VC49": "Estimate; RACE - One race - White",
"HC03_VC49": "Percent; RACE - One race - White",
"HC01_VC50": "Estimate; RACE - One race - Black or African American",
"HC03_VC50": "Percent; RACE - One race - Black or African American",
"HC01_VC56": "Estimate; RACE - One race - Asian",
"HC03_VC56": "Percent; RACE - One race - Asian",
"HC01_VC87": "Estimate; HISPANIC OR LATINO AND RACE - Total population",
"HC03_VC87": "Percent; HISPANIC OR LATINO AND RACE - Total population"
}

poverty_variables = {
"HC01_EST_VC01": "Total; Estimate; Population for whom poverty status is determined",
"HC02_EST_VC01": "Below poverty level; Estimate; Population for whom poverty status is determined",
"HC03_EST_VC01": "Percent below poverty level; Estimate; Population for whom poverty status is determined",
"HC02_EST_VC03": "Below poverty level; Estimate; AGE - Under 18 years",
"HC03_EST_VC03": "Percent below poverty level; Estimate; AGE - Under 18 years",
"HC02_EST_VC07": "Below poverty level; Estimate; AGE - 18 to 64 years",
"HC03_EST_VC07": "Percent below poverty level; Estimate; AGE - 18 to 64 years",
"HC02_EST_VC18": "Below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - White alone",
"HC03_EST_VC18": "Percent below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - White alone",
"HC02_EST_VC19": "Below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - Black or African American alone",
"HC03_EST_VC19": "Percent below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - Black or African American alone",
"HC02_EST_VC21": "Below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - Asian alone",
"HC03_EST_VC21": "Percent below poverty level; Estimate; RACE AND HISPANIC OR LATINO ORIGIN - Asian alone",
"HC02_EST_VC26": "Below poverty level; Estimate; Hispanic or Latino origin (of any race)",
"HC03_EST_VC26": "Percent below poverty level; Estimate; Hispanic or Latino origin (of any race)"
}

education_keys = dict(zip(["education_" + k for k in education_variables.keys()], education_variables.values()))
race_sex_age_keys = dict(zip(["race_sex_age_" + k for k in race_sex_age_variables.keys()], race_sex_age_variables.values()))
poverty_keys = dict(zip(["poverty_" + k for k in poverty_variables.keys()], poverty_variables.values()))
#CHOOSE DEP NUMBER HERE
dep_number = "23-00089"
state = state_code_lookup[dep_number]
def get_county_ACS_data(data_dir, census_shape_dir, dep_number, state):
    dep_dir = data_dir + "Dept_" + dep_number + "/"
    
    #GET EDUCATION, RACE_SEX_AGE, POVERTY ACS DATA
    acs_path = dep_dir + dep_number + "_ACS_data/"
    
    acs_education_path = acs_path + dep_number + "_ACS_education-attainment/"
    acs_race_sex_age_path = acs_path + dep_number + "_ACS_race-sex-age/"
    acs_poverty_path = acs_path + dep_number + "_ACS_poverty/"
    
    #account for special case naming convention department 49-00081
    if dep_number == "49-00081":
        acs_poverty_path = acs_path + dep_number + "_ACS_poverty-status/"

    for f in os.listdir(acs_education_path):
        if "S1501_with_ann" in f:
            acs_education_path += f
            break
    for f in os.listdir(acs_race_sex_age_path):
        if "DP05_with_ann" in f:
            acs_race_sex_age_path += f
            break
    for f in os.listdir(acs_poverty_path):
        if "S1701_with_ann" in f:
            acs_poverty_path += f
            break
    
    education_df = pd.read_csv(acs_education_path, header = 0, skiprows = [1])
    race_sex_age_df = pd.read_csv(acs_race_sex_age_path, header = 0, skiprows=[1])
    poverty_df = pd.read_csv(acs_poverty_path, header = 0, skiprows=[1])
    
    #DO SOME PROCESSING
    education_df["GEO.id2"] = pd.to_numeric(education_df["GEO.id2"])
    race_sex_age_df["GEO.id2"] = pd.to_numeric(race_sex_age_df["GEO.id2"])
    poverty_df["GEO.id2"] = pd.to_numeric(poverty_df["GEO.id2"])
    
    c = list(education_df.columns)
    c[3:] = ["education_" + c_name for c_name in c[3:]]
    education_df.columns = c
    
    c = list(race_sex_age_df.columns)
    c[3:] = ["race_sex_age_" + c_name for c_name in c[3:]]
    race_sex_age_df.columns = c
    
    c = list(poverty_df.columns)
    c[3:] = ["poverty_" + c_name for c_name in c[3:]]
    poverty_df.columns = c
    
    #JOIN INTO SINGLE TABLE
    master_df = education_df.merge(race_sex_age_df, left_on='GEO.id2', right_on='GEO.id2', how='left')
    master_df = master_df.merge(poverty_df, left_on='GEO.id2', right_on='GEO.id2',how='left')
    
    #GET CENSUS TRACT SHAPEFILES
    #SHAPEFILES DOWNLOADED FROM https://www2.census.gov/geo/tiger/TIGER2016/TRACT/
    state_code = FIPS_state_codes[state]
    state_shapefile_path = census_shape_dir + "tl_2016_" + state_code + "_tract/" + "tl_2016_" + state_code + "_tract.shp"
    state_shapes = gpd.read_file(state_shapefile_path)
    state_shapes["GEOID"] = pd.to_numeric(state_shapes["GEOID"])
    
    #JOIN EDUCATION AND GEOMETRIES
    master_df = master_df.merge(state_shapes, left_on='GEO.id2', right_on='GEOID', how='left')
    #CONVERT TO GEODATAFRAME
    geometry = master_df["geometry"]
    master_df.drop(["geometry"], axis = 1)
    master_df = gpd.GeoDataFrame(master_df, crs=state_shapes["geometry"].crs, geometry=geometry)
    
    return master_df
ACS_df = get_county_ACS_data(data_dir, census_shape_dir, dep_number, state)
ACS_df.head()
def ACS_summary_table(ACS_df):
    all_keys = list(education_keys.keys()) + list(race_sex_age_keys.keys()) + list(poverty_keys.keys())
    sub_ACS_df = ACS_df[all_keys]
    summary_table = sub_ACS_df.mean()

    variable_names = []
    avg_values = []

    for k in range(summary_table.shape[0]):
        if "education" in summary_table.keys()[k]:
            variable_names.append(education_keys[summary_table.keys()[k]])
            avg_values.append([summary_table.values[k]])
        elif "race_sex_age" in summary_table.keys()[k]:
            variable_names.append(race_sex_age_keys[summary_table.keys()[k]])
            avg_values.append([summary_table.values[k]])
        elif "poverty" in summary_table.keys()[k]:
            variable_names.append(poverty_keys[summary_table.keys()[k]])
            avg_values.append([summary_table.values[k]])
            
    pairs = list(zip(variable_names, avg_values))
    pairs.sort(key = lambda p: -p[1][0])
    variable_names, avg_values = zip(*pairs)
    
    avg_values = np.array(avg_values).round(2)

    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(variable_names)))

    table = plt.table(cellText=avg_values,
              rowLabels=variable_names,
              rowColours=colors,
              colLabels=["Average Value Over Censuses in County"],
              loc = "top")
    
    plt.show()
ACS_summary_table(ACS_df)
def ACS_cluster_heatmap(ACS_df):
    all_keys = list(education_keys.keys()) + list(race_sex_age_keys.keys()) + list(poverty_keys.keys())
    sub_ACS_df = ACS_df[all_keys]
    cluster = sns.clustermap(sub_ACS_df.corr(), center=0, cmap="vlag", linewidths=.75, figsize=(13, 13))
    
def display_variable_legend():
    all_keys = list(education_keys.keys()) + list(race_sex_age_keys.keys()) + list(poverty_keys.keys())

    education_var = []
    education_meaning = []
    
    race_sex_age_var = []
    race_sex_age_meaning = []
    
    poverty_var = []
    poverty_meaning = []

    for k in all_keys:
        if "education" in k:
            education_meaning.append([education_keys[k]])
            education_var.append(k)
        elif "race_sex_age" in k:
            race_sex_age_meaning.append([race_sex_age_keys[k]])
            race_sex_age_var.append(k)
        elif "poverty" in k:
            poverty_meaning.append([poverty_keys[k]])
            poverty_var.append(k)

    fig, ax = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    all_keys = education_var + race_sex_age_var + poverty_var
    all_meanings = education_meaning + race_sex_age_meaning + poverty_meaning
    colors = ["lightgray"] * len(all_keys)

    table = plt.table(cellText=all_meanings,
              rowLabels=all_keys,
              rowColours=colors,
              colLabels=["Variable Meaning"],
              loc = "top")

    table.scale(2, 2)
    
    plt.show()
ACS_cluster_heatmap(ACS_df)
display_variable_legend()
def circle_map(ACS_df, variable_to_plot):
    start_coord = [float(ACS_df["INTPTLAT"][0]), float(ACS_df["INTPTLON"][0])]
    data_to_plot = ACS_df[variable_to_plot].astype(float)
    scaled_radii = data_to_plot * (300 / data_to_plot.mean())
    
    m = folium.Map(location=start_coord, zoom_start=10, tiles='Stamen Toner', min_zoom=8)
    for i in range(0,len(ACS_df)):
        folium.Circle(
          location=[float(ACS_df.iloc[i]['INTPTLAT']), float(ACS_df.iloc[i]['INTPTLON'])],
          popup = "CENSUS TRACT:" + str(ACS_df.iloc[i]['GEO.display-label']) + " | VALUE: " + str(ACS_df.iloc[i][variable_to_plot]),
          radius=scaled_radii[i],
          color='crimson',
          fill=True,
          fill_color='crimson'
       ).add_to(m)
    return m
m = circle_map(ACS_df, "education_HC02_EST_VC03")
embed_map(m)
def convert_time(crime, dep_number):
    crime["INCIDENT_TIME"] = crime.INCIDENT_TIME.astype("str").str.lower()
    #consider case where time contains am/pm, convert to army time 
    if crime.INCIDENT_TIME.str.contains("pm").any():
        crime.loc[crime["INCIDENT_TIME"].str.contains("pm"), "INCIDENT_TIME"] = pd.to_datetime(crime.loc[crime["INCIDENT_TIME"].str.contains("pm"), "INCIDENT_TIME"]).dt.time
    #all but one of the datasets have their time formatted as ##:## or #:##
    time = crime.INCIDENT_TIME.astype("str").str.extract("(\d?\d:\d\d)")
    date = pd.to_datetime(crime.INCIDENT_DATE).dt.date.astype("str")
    crime["INCIDENT_DATE"] = pd.to_datetime(date + " " + time)
    return crime
def plot_by_hour(df_and_meta):
    if not df_and_meta[1]:
        print("Cannot generate plot due to lack of time data for this department.")
        return
    crime = df_and_meta[0]
    
    fig, ax = plt.subplots(figsize = (15, 5))
    ax = sns.countplot(pd.to_datetime(crime.INCIDENT_DATE).dt.hour[~crime.INCIDENT_DATE.isnull()].astype("int"))
    plt.title("Number of Crimes per Hour")
    plt.xlabel("Hour")
    #displaying counts for each hour bin
    for patch in ax.patches: 
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        ax.annotate(f'{int(y)}', (x.mean(), y), ha = "center", va = "bottom")

def plot_race_count(df_and_meta):
    if not df_and_meta[3]:
        print("Cannot generate plot due to lack of offender race data for this department.")
        return
    crime = df_and_meta[0]
    
    for i in crime.columns: 
        if "SUBJECT_RACE" in i:
            race = i
            break
    
    fig, ax = plt.subplots(figsize = (8, 5))
    ax = sns.countplot(crime[race])
    plt.title("Number of Arrests by Race")
    #displaying counts for each race bin 
    for patch in ax.patches: 
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        ax.annotate(f'{int(y)}', (x.mean(), y), ha = "center", va = "bottom")
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation = 90)
    
def plot_injuries_by_race(df_and_meta):
    if not df_and_meta[4]:
        print("Cannot generate plot due to lack of injury or offender race data for this department.")
        return
    crime = df_and_meta[0]
    
    for i in crime.columns: 
        if "SUBJECT_RACE" in i:
            race = i
            break
    for i in crime.columns:         
        if "SUBJECT_INJURY" in i:
            injury = i
            break
    
    #allows to plot percentage by race 
    ax = dxp.aggplot(agg = race, hue = injury, data = crime, normalize = race, figsize = (10, 5))
    plt.title("Percent of Injuries by Race")
    plt.xlabel("Race")
    plt.ylabel("Percent")
    #displaying counts for each injury tyipe bin
    #this took 1 and a half hours to figure out I hate plotting 
    for p in ax.patches:
        height = p.get_height()
        if math.isnan(float(height)):
            height = 0
        ax.text(p.get_x() + p.get_width()/2, 
               height + 0.01, f"{round(height, 2)}", ha = "center")
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation = 90)
    
def plot_police_district(df_and_meta):
    if not df_and_meta[2]:
        print("Cannot generate plot due to lack of location data for this department.")
        return
    crime = df_and_meta[0]
    
    fig, ax = plt.subplots(figsize = (15, 5))
    #Most datasets have less than 25 police districts but one of them had like 200
    if len(crime.LOCATION_DISTRICT.unique()) > 25:
        top_25_districts = crime.LOCATION_DISTRICT[crime.LOCATION_DISTRICT.isin(crime.LOCATION_DISTRICT.value_counts().iloc[0:25].index)]
        ax = sns.countplot(top_25_districts)
        plt.title("Number of Crimes in Top 25 Police Districts")
    else:
        ax = sns.countplot(crime.LOCATION_DISTRICT);
        plt.title("Number of Crimes per Police District")
    #displaying counts for each police district bin 
    for patch in ax.patches: 
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        ax.annotate(f'{int(y)}', (x.mean(), y), ha = "center", va = "bottom")
    plt.xlabel("Police District")
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation = 90);
#import, clean, and standardize
dep_num_to_crime_path = {
"37-00049":"Dept_37-00049/37-00049_UOF-P_2016_prepped.csv",
"24-00013":"Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv",
"23-00089":"Dept_23-00089/23-00089_UOF-P.csv",
"11-00091":"Dept_11-00091/11-00091_Field-Interviews_2011-2015.csv",
"24-00098":"Dept_24-00098/24-00098_Vehicle-Stops-data.csv",
"35-00016":"Dept_35-00016/35-00016_UOF-OIS-P.csv",
"35-00103":"Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv",
"37-00027":"Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv",
"49-00009":"Dept_49-00009/49-0009_UOF.csv",
"49-00033":"Dept_49-00033/49-00033_Arrests_2015.csv",
"49-00035":"Dept_49-00035/49-00035_Incidents_2016.csv",
"49-00081":"Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv"
}

def load_crime_data_and_prepare(dep_number):
    datapath = data_dir + dep_num_to_crime_path[dep_number]
    crime = pd.read_csv(datapath)
    crime = crime.iloc[1:, :]
    
    #CLEAN UP SOME TYPOS
    cols = list(crime.columns)
    for c in range(len(cols)):
        if "INCIDENT_DATE" in cols[c]:
            cols[c] = "INCIDENT_DATE"
        if "SUBJECT_RAC" in cols[c]:
            cols[c] = "SUBJECT_RACE"
    crime.columns = cols
    
    HAS_TIME = True
    HAS_RACE = True
    HAS_LOC = True
    HAS_INJURY = False
    
    #PREPARE TIME FORMAT
    time = pd.to_datetime(crime.INCIDENT_DATE).dt.time
    if dep_number == "49-00033":
        print("Dataset doesn't have information on time of incidence")
        HAS_TIME = False
    elif (time[~time.isnull()] != datetime.time(0, 0)).any():
        pass
    elif "INCIDENT_TIME" in crime.columns:
        crime = convert_time(crime, dep_number)
        time = crime.INCIDENT_DATE.dt.time
        if not (time[~time.isnull()] != datetime.time(0, 0)).any():
            print("Dataset doesn't have information on time of incidence")
            HAS_TIME = False
    else:
        print("Dataset doesn't have information on time of incidence")
        HAS_TIME = False
    
    #LOG WHAT ELSE WE'RE MISSING
    if not pd.Series(crime.columns).str.contains("SUBJECT_RAC").any():
        print("Dataset doesn't have information on offender race")
        HAS_RACE = False
        
    if not "LOCATION_DISTRICT" in crime.columns:
        print("Dataset doesn't have information on Police District")
        HAS_LOC = False
        
    if not HAS_RACE:
        HAS_INJURY = False
    else:
        for i in crime.columns:         
            if "SUBJECT_INJURY" in i:
                HAS_INJURY = True
                
    return crime, HAS_TIME, HAS_LOC, HAS_RACE, HAS_INJURY
df_and_meta = load_crime_data_and_prepare(dep_number)
crime, HAS_TIME, HAS_LOC, HAS_RACE, HAS_INJURY = df_and_meta
crime.head()
plot_police_district(df_and_meta)
plot_by_hour(df_and_meta)
plot_race_count(df_and_meta)
plot_injuries_by_race(df_and_meta)
df_and_meta = load_crime_data_and_prepare("37-00027")
crime, HAS_TIME, HAS_LOC, HAS_RACE, HAS_INJURY = df_and_meta
crime.head()
plot_police_district(df_and_meta)
plot_by_hour(df_and_meta)
plot_race_count(df_and_meta)
plot_injuries_by_race(df_and_meta)
df_and_meta = load_crime_data_and_prepare("49-00009")
crime, HAS_TIME, HAS_LOC, HAS_RACE, HAS_INJURY = df_and_meta
crime.head()
plot_police_district(df_and_meta)
plot_by_hour(df_and_meta)
plot_race_count(df_and_meta)
plot_injuries_by_race(df_and_meta)
def find_intersections(master_df, police_district_shapes, dep_number):
    census_police_district_id = []
    census_police_district_percent_overlap = []
    police_district_id_column_name = police_district_id_column_lookup[dep_number]

    for i in range(master_df.shape[0]):
        g = master_df["geometry"][i]
        list_of_matches = []
        perc_overlap = []
        for j in range(police_district_shapes.shape[0]):
            a = police_district_shapes["geometry"][j].intersection(g)
            if a.area != 0:
                percent_overlap = 100 * (a.area / g.area)
                list_of_matches.append(police_district_shapes[police_district_id_column_name][j])
                perc_overlap.append(percent_overlap)
        census_police_district_id.append(tuple(list_of_matches))
        census_police_district_percent_overlap.append(tuple(perc_overlap))   

    master_df["Police District ID"] = pd.Series(census_police_district_id)
    master_df["Percentage Overlap"] = pd.Series(census_police_district_percent_overlap)

def process_census_with_district(data_dir, dep_number, master_df):
    dep_dir = data_dir + "Dept_" + dep_number + "/"
    
    #GET DISTRICT SHAPEFILES
    district_shapefile_path = dep_dir + dep_number + "_Shapefiles/"
    for f in os.listdir(district_shapefile_path):
        if f[-4:] == ".shp":
            district_shapefile_path += f
    #ORLANDO SPECIAL CASE
    if dep_number == "35-00016":
        district_shapefile_path = dep_dir + dep_number + "_Shapefiles/OrlandoPoliceDistricts.shp"
    
    police_district_shapes = gpd.read_file(district_shapefile_path)
    police_district_shapes["geometry"] = police_district_shapes["geometry"].to_crs(master_df["geometry"].crs)
    police_district_shapes.crs = master_df.crs
    
    #COMPUTE INTERSECTIONS BETWEEN CENSUS TRACT AND POLICE DISTRICTS
    find_intersections(master_df, police_district_shapes, dep_number)
    
    #DROP NONOVERLAPPING CENSUS TRACTS
    master_df = master_df[master_df["Police District ID"] != ()]
    
    return master_df, police_district_shapes


def weighted_vals_to_district(ACS_df, police_district_shapes, dep_number):
    all_keys = list(education_keys.keys()) + list(race_sex_age_keys.keys()) + list(poverty_keys.keys())
    
    data_matrix = np.zeros((len(police_district_shapes), len(all_keys)))
    rand_lat_lon = np.zeros((len(police_district_shapes), 2))
    
    for d in range(police_district_shapes.shape[0]):
        g = police_district_shapes["geometry"].values[d]
        data_row_vec = np.zeros(len(all_keys))
        add_lat_lon = True
        for r in range(ACS_df.shape[0]):
            a = ACS_df["geometry"].values[r].intersection(g)
            if a.area != 0:
                if add_lat_lon:
                    rand_lat_lon[d] = np.array([float(ACS_df["INTPTLAT"].values[r]), float(ACS_df["INTPTLON"].values[r])])
                    add_lat_lon = False
                for k in range(len(all_keys)):
                    fraction_overlap = a.area / g.area
                    if "X" not in str(ACS_df[all_keys[k]].values[r]) and "-" not in str(ACS_df[all_keys[k]].values[r]):
                        data_row_vec[k] += np.array(ACS_df[all_keys[k]].values[r]).astype(np.float64) * fraction_overlap
        data_matrix[d] += data_row_vec
                    
    weighted_vals = pd.DataFrame(data_matrix, index=np.arange(len(police_district_shapes)), columns=all_keys)
        
    return weighted_vals, rand_lat_lon
ACS_df, police_district_shapes = process_census_with_district(data_dir, dep_number, ACS_df)
ACS_df.head()
police_district_shapes.head()
def plot_weighted_districts_circle(police_district_shapes, rand_lat_lon, weighted_vals, value_key, dep_number):
    data_to_plot = weighted_vals[value_key].astype(float)
    scaled_radii = (data_to_plot / data_to_plot.mean()) * 450
    
    start_coord = list(rand_lat_lon[0])
    
    m = folium.Map(location=start_coord, zoom_start=10, tiles='Stamen Toner', min_zoom=8)
    folium.GeoJson(police_district_shapes, style_function= lambda x:{'color':'blue', 'fillColor': 'lightblue'}).add_to(m)
    for i in range(0,len(police_district_shapes)):
        folium.Circle(
          location=rand_lat_lon[i],
          popup = "DISTRICT ID: " + str(police_district_shapes[police_district_id_column_lookup[dep_number]].values[i]) + " | VALUE: " + str(data_to_plot[i]),
          radius=scaled_radii[i],
          color='crimson',
          fill=True,
          fill_color='crimson'
       ).add_to(m)
    return m
weighted_vals, rand_lat_lon = weighted_vals_to_district(ACS_df, police_district_shapes, dep_number)
plot_weighted_districts_circle(police_district_shapes, rand_lat_lon, weighted_vals, "education_HC01_EST_VC02", dep_number)
