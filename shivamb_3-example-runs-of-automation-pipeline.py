import shutil, os, folium, warnings
from shapely.geometry import Point
import pandas as pd, numpy as np 
from collections import Counter
from statistics import median
import geopandas as gpd
warnings.filterwarnings('ignore')

## define base paths 
ct_base_path = "../input/census-tracts/cb_2017_<NUM>_tract_500k/cb_2017_<NUM>_tract_500k.shp"
_base_dir = "../input/data-science-for-good/cpe-data/"
external_datasets_path = "../input/external-datasets-cpe/"
_root_dir = "CPE_ROOT/"

## define the new directory names and mandatory shape files 
mandatory_shapefiles = ["shp", "shx", "dbf", "prj"]
new_dirs = ["shapefiles", "events", "metrics", "metrics_meta"]

## Utility function to cleanup the environment
def _cleanup_environment():
    if os.path.exists(_root_dir):
        !rm -r CPE_ROOT
        pass
    return None

## Function to create a new repository structure 
def _create_repository_structure():            
    ## refresh environment 
    _cleanup_environment()
    
    ## list of all departments whose raw data is available
    depts = [_ for _ in os.listdir(_base_dir) if "Dept" in _]
    
    ## master folder
    os.mkdir(_root_dir) 
    for dept in depts:

        ## every department folder 
        os.mkdir(_root_dir + "/" + dept)         
        for _dir in new_dirs:
        
            ## sub directories for - shapefiles, acsdata, metrics, metrics-meta
            os.mkdir(_root_dir + "/" + dept + "/" + _dir + "/")            
    print ("Status : Directory Structured Created")
    
dept_37_27_prj = 'PROJCS["NAD_1983_StatePlane_Texas_Central_FIPS_4203_Feet",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",2296583.333333333],PARAMETER["False_Northing",9842499.999999998],PARAMETER["Central_Meridian",-100.3333333333333],PARAMETER["Standard_Parallel_1",30.11666666666667],PARAMETER["Standard_Parallel_2",31.88333333333333],PARAMETER["Latitude_Of_Origin",29.66666666666667],UNIT["Foot_US",0.30480060960121924],AUTHORITY["EPSG","102739"]]'

## create a config to handle the errors in raw shape files
missing_shape_meta = {
    "Dept_37-00027" : {"prj" : dept_37_27_prj}
}

## Function to fix / cleanup the errors in shapefile types
def _fix_errors_shapefiles(_path, dept):
    """
    :params:
    _path : root path containig the shape files 
    dept : selected dept if it is called only for a particular department
    """
    
    if dept not in missing_shape_meta:
        return False
    
    ## Fix the errors in raw corresponding shape files
    for extension, content in missing_shape_meta[dept].items():
        if extension == "prj": 
            # Step1: Add missing prj file
            with open(_path + "department.prj", 'w') as outfile:
                outfile.write(content)
            
            # Step2: Fix CRS of shape file
            df = gpd.read_file(_path + 'department.shp')
            df.to_file(filename = _path + 'department.shp', 
                       driver='ESRI Shapefile', crs_wkt = content)

        elif extension == "shx":
            ## This function can be extended for other shape filetypes
            ## the corresponding logic can be added in these blocks 
            pass
    return True

## Function to standardize the shape files
def _standardize_shapefiles():
    depts = [_ for _ in os.listdir(_base_dir) if "Dept" in _]
    for dept in depts:    
        ## Step1: Configure the old and new path
        shp_dir = dept.replace("Dept_","") + "_Shapefiles/"
        old_pth = _base_dir + dept + "/" + shp_dir
        new_pth = _root_dir + dept + "/" + "shapefiles/"

        ## Step2: Standardize the file names and move to new path 
        _files = os.listdir(old_pth)
        for _file in _files:
            if _file[-3:].lower() not in mandatory_shapefiles:
                continue
            ext = ".".join(_file.split(".")[1:]).lower()
            new_name = "department." + ext
            shutil.copy(old_pth+_file, new_pth+new_name)

        ## Step3: Fix Erroroneus shapefiles
        fix_flag = _fix_errors_shapefiles(new_pth, dept)
        
    print ("Status : Shapefile Standardization Complete")
    return None


## cleaned names corresponding to given raw metric names
acs_metrics_dic = { 'owner-occupied-housing' : 'housing', 'education-attainment' : 'education', 'employment' : 'employment', 'education-attainment-over-25' : 'education25', 'race-sex-age' : 'race-sex-age', 'poverty' : 'poverty', 'income' : 'income' }
metrics_names = list(acs_metrics_dic.values())

## function to cleanup and move the ACS data
def _standardize_acs():
    depts = [_ for _ in os.listdir(_base_dir) if "Dept" in _]
    for dept in depts:  
        ## Step1: Configure the old and new path
        acs_dir = dept.replace("Dept_","") + "_ACS_data"
        old_dirs = os.listdir(_base_dir + dept +"/"+ acs_dir)
        new_dirs = [f.replace(dept.replace("Dept_",""),"") for f in old_dirs]
        new_dirs = [f.replace("_ACS_","") for f in new_dirs]
        
        ## Step2: Move all ACS datafiles
        for j, metric in enumerate(old_dirs):
            metric_files = os.listdir(_base_dir + dept +"/"+ acs_dir +"/"+ metric)
            _file = [f for f in metric_files if "metadata" not in f][0]
            _meta = [f for f in metric_files if "metadata" in f][0]

            ## Step3: Standardize / Cleanup the name 
            for name, clean_name in acs_metrics_dic.items():
                if "25" in metric:
                    cname = "education25"
                if name in metric:
                    cname = clean_name     

            ## Step4.1 : Move Metric File
            old_path = _base_dir + dept +"/"+ acs_dir +"/"+ metric +"/"+ _file
            new_path = _root_dir + dept +"/metrics/" + cname + ".csv"
            shutil.copy(old_path, new_path)

            ## Step4.2 : Move Metrics meta files
            old_path = _base_dir + dept +"/"+ acs_dir +"/"+ metric +"/"+ _meta
            new_path = _root_dir + dept +"/metrics_meta/" + cname + ".csv"
            shutil.copy(old_path, new_path)

    print ("Status : Standardization of Metrics complete")
    
    
def _run_standardization_pipeline():
    _create_repository_structure()
    _standardize_shapefiles()
    _standardize_acs()


## Provide the config file for the departments
depts_config = {
    'Dept_23-00089' : {'_rowid' : "DISTRICT", "ct_num" : "18"},  
    'Dept_49-00035' : {'_rowid' : "pol_dist", "ct_num" : "06"},  
    'Dept_24-00013' : {'_rowid' : "OBJECTID", "ct_num" : "27"},  
    'Dept_24-00098' : {'_rowid' : "gridnum",  "ct_num" : "27"},   
    'Dept_49-00033' : {'_rowid' : "number",   "ct_num" : "06", "center_ll" : [34.0883787,-118.37781]},    
    'Dept_11-00091' : {'_rowid' : "ID",       "ct_num" : "25"},         
    'Dept_49-00081' : {'_rowid' : "company",  "ct_num" : "06"},   
    'Dept_37-00049' : {'_rowid' : "Name",     "ct_num" : "48"},      
    'Dept_37-00027' : {'_rowid' : "CODE",     "ct_num" : "48"},     
    'Dept_49-00009' : {'_rowid' : "objectid", "ct_num" : "53"}, 
}


## Function to read a shapefile
def _read_shape_gdf(_dept):
    shape_pth = _root_dir + _dept + "/shapefiles/department.shp"
    ## ensure that CRS are consistent
    shape_gdf = gpd.read_file(shape_pth).to_crs(epsg=4326)
    return shape_gdf

## Read the CT File
def _read_ctfile(_dept):
    ## find the corresponding CT number from the config
    _ct = depts_config[_dept]["ct_num"]
    ## generate the base CT path 
    ct_path = ct_base_path.replace("<NUM>", _ct)
    ## load the geo data frame for CT 
    state_cts = gpd.read_file(ct_path).to_crs(epsg='4326')
    return state_cts

## Function to get the centroid of a polygon
def _get_latlong_point(point):
    _ll = str(point).replace("POINT (","").replace(")", "")
    _ll = list(reversed([float(_) for _ in _ll.split()]))
    return _ll

## Function to plot a shapefile
def _plot_shapefile_base(shape_gdf, _dept, overlapped_cts = {}):
    ## obtain the center most point of the map 
    
    if "center_ll" not in depts_config[_dept]:
        center_pt = shape_gdf.geometry.centroid[0]
        center_pt = _get_latlong_point(center_pt)
    else:
        center_pt = depts_config[_dept]["center_ll"]
    
    ## initialize the folium map 
    mapa = folium.Map(center_pt,  zoom_start=10, tiles='CartoDB dark_matter')
    if len(overlapped_cts) == 0:
        ## only the base map
        folium.GeoJson(shape_gdf).add_to(mapa)
    else:
        ## overlapped map
        ct_style = {'fillColor':"red",'color':"red",'weight':1,'fillOpacity':0.5}
        base_style = {'fillColor':"blue",'color':"blue",'weight':1,'fillOpacity':0.5}
        folium.GeoJson(overlapped_cts, style_function = lambda feature: ct_style).add_to(mapa)
        folium.GeoJson(shape_gdf, style_function = lambda feature: base_style).add_to(mapa)
    return mapa


## Find Overlapping Census Tracts
def find_overlapping_cts(dept_gdf, state_cts, _identifier, _threshold = 10.0):
    """
    :params:
    dept_gdf : the geo dataframe loaded from shape file for the department 
    state_cts : the geo dataframe of the corresponding ct file
    _identifier : the unique row identifier for the department 
    _threshold : the overlapping threshold percentage to consider 
    """
    
    
    ## Step 1: Initialize
    olaps_percentages, overlapped_idx = {}, []
    for i, row in dept_gdf.iterrows():
        if row[_identifier] not in olaps_percentages: 
            olaps_percentages[row[_identifier]] = {}

        ## Step 2: Find overlap bw district and ct layer
        layer1 = row["geometry"] # district layer
        for j, row2 in state_cts.iterrows():
            layer2 = row2["geometry"] # ct layer
            layer3 = layer1.intersection(layer2) # overlapping layer
            
            ## Step 3: Save overlapping percentage
            overlap_percent = layer3.area / layer2.area * 100
            if overlap_percent >= _threshold: 
                olaps_percentages[row[_identifier]][row2["GEOID"]] = overlap_percent
                overlapped_idx.append(j)
    
    ## Step 4: Find unique overlapping census tracts
    overlapped_idx = list(set(overlapped_idx))
    overlapped_cts = state_cts.iloc[overlapped_idx]
    # print ("Status : Overlapped CTs Found: ", len(overlapped_cts))
    return overlapped_cts, olaps_percentages

## function to convert overlapping percentages dictionary to a dataframe 
def _prepare_olaps_df(olaps_percentages):
    temp = pd.DataFrame()
    distid, ct, pers = [], [], []
    for k, vals in olaps_percentages.items():
        for v, per in vals.items():
            distid.append (k)
            ct.append(v)
            pers.append(round(per, 2))
    temp["DistId"] = distid
    temp["CensusTract"] = ct
    temp["Overlap %"] = pers
    return temp

## Specific Metrics and their measures 
metrics_config = {
            'race-sex-age': {'metrics':['race','age','sex'], "measure":"proportion"},
            'income':       {'metrics':['median_income'],    "measure":"median"},
            'poverty':      {'metrics':['below_poverty'],    "measure":"proportion"},
            'employment':   {'metrics':['ep_ratio', 'unemp_ratio'], "measure" : "mean"}
            }

## Cleaned Column Names 
_column_names = {"race" : { "HC01_VC43" : "total_pop",
                            "HC01_VC49" : "white_pop",
                            "HC01_VC50" : "black_pop",
                            "HC01_VC56" : "asian_pop",
                            "HC01_VC88" : "hispanic_pop"},
                "age" : {
                            "HC01_VC12" : "20_24_pop", 
                            "HC01_VC13" : "25_34_pop", 
                            "HC01_VC14" : "35_44_pop", 
                            "HC01_VC15" : "45_54_pop", 
                            "HC01_VC16" : "55_59_pop", 
                },
                "sex": {
                            "HC01_VC04" : "male_pop",
                            "HC01_VC05" : "female_pop",
                },
                "median_income" : {
                            "HC02_EST_VC02" : "pop_income",
                            "HC02_EST_VC04" : "whites_income",
                            "HC02_EST_VC05" : "blacks_income",
                            "HC02_EST_VC07" : "asian_income",
                            "HC02_EST_VC12" : "hispanic_income",
                },
                "below_poverty" : {
                            "HC02_EST_VC01" : "below_pov_pop"},
                 "ep_ratio" : {
                             "HC03_EST_VC15" : "whites_ep_ratio",
                             "HC03_EST_VC16" : "blacks_ep_ratio"
                  },
                 "unemp_ratio" : {
                             "HC04_EST_VC15" : "whites_unemp_ratio",
                             "HC04_EST_VC16" : "blacks_unemp_ratio"}
                }


## Function to perform basic pre-processing on metrics data 
def _cleanup_metrics_data(_dept):
    metrics_df = {}
    for _metric in metrics_names: ## metrics_name is deinfed in config 
        mpath = _root_dir + _dept + "/metrics/" + _metric + ".csv"
        mdf = pd.read_csv(mpath, low_memory=False).iloc[1:]
        mdf = mdf.reset_index(drop=True).rename(columns={'GEO.id2':'GEOID'})
        metrics_df[_metric] = mdf
    
    ## returns metrics_df that contains all the dataframe for ACS metrics 
    return metrics_df

## Function to Flatten the details
def _flatten_gdf(df, _identifier):
    relevant_cols = [_identifier]
    flatten_df = df[relevant_cols]
    for c in df.columns:
        if not c.startswith("_"):
            continue
        _new_cols = list(df[c].iloc(0)[0].keys())
        for _new_col in _new_cols:
            _clean_colname = _column_names[c[1:]][_new_col]
            flatten_df[_clean_colname] = df[c].apply(lambda x : x[_new_col]\
                                                if type(x) == dict else 0.0)
            relevant_cols.append(_clean_colname)
    return flatten_df[relevant_cols]


## Function that enriches the information using overlapped percentage
def _enrich_info(idf, percentages, m_df, columns, m_measure):
    """
    :params:
    idf : unique identifier for the police department information
    percentages : The overalapped CTs and their percentages
    m_df : the dataframe of the metric containing all the information
    columns : the corresponding column names of the metric, defined in config
    m_measure : the measure (mean, median, proportion) to perform
    """
    
    ## define the updated_metrics object that will store the estimated information
    updated_metrics = {}
    
    ## return None if no overlapping CTs
    if len(percentages[idf]) == 0:
        return ()
    
    ## Iterate in all Districts with the overlapped CTs and percentage
    for idd, percentage in percentages[idf].items(): 
        ## find the corresponding row for an overlapped CT in the metric data 
        ct_row = m_df[m_df["GEOID"] == idd]
        for rcol in columns:
            if rcol not in updated_metrics:
                updated_metrics[rcol] = []
            
            ## Perform the necessary calculation to find the estimated number 
            try:
                actual_value = ct_row[rcol].iloc(0)[0].replace("-","")
                actual_value = actual_value.replace(",","")
                actual_value = float(actual_value.replace("+",""))
                if m_measure == "proportion":
                    updated_value = actual_value * percentage / 100
                else:
                    updated_value = actual_value
                updated_metrics[rcol].append(updated_value)
            except Exception as E:
                pass
    
    ## Update the information in updated_metrics
    for rcol in columns:
        if len(updated_metrics[rcol]) == 0:
            updated_metrics[rcol] = 0
        else:
            if m_measure == "proportion":
                updated_metrics[rcol] = sum(updated_metrics[rcol])
            elif m_measure == "median":
                updated_metrics[rcol] = median(updated_metrics[rcol])
            elif m_measure == "mean":
                _mean = float(sum(updated_metrics[rcol])) / len(updated_metrics[rcol])
                updated_metrics[rcol] = _mean
    return updated_metrics


## Master Function to process the ACS info in dept df
def _process_metric(metrics_df, dept_df, _identifier, olaps_percentages, metric_name):
    """
    :params:
    metrics_df : the complete dataframe containing the metrics data
    dept_df : the geodataframe for police shape files 
    _identifier : the row identifier column corresponding to the police dept shape file 
    olaps_percentages : the overlapping percentage object calculated in previous step
    metric_name : Name of the metric, example - education / poverty / income 
    """
    
    m_df = metrics_df[metric_name]
    m_measure = metrics_config[metric_name]["measure"]
    for flag in metrics_config[metric_name]['metrics']:
        cols = list(_column_names[flag].keys())
        dept_df["_"+flag] = dept_df[_identifier].apply(lambda x : \
                            _enrich_info(x, olaps_percentages, m_df, cols, m_measure))
    return dept_df 


subject_race_csv_content = """W	White
W(White)	White
White	White
B	Black
B(Black)	Black
Black	Black
Black or African American	Black
Black, Black	Black
Unk	Unknown
Unknown	Unknown
UNKNOWN	Unknown
No Data	Unknown
NO DATA ENTERED	Unknown
not recorded	Unknown
Not Specified	Unknown
P	Pacific Islander
Pacific Islander	Pacific Islander
O	Other
Other	Other
Other / Mixed Race	Other
Native Am	Native American
Native Amer	Native American
Native American	Native American
Latino	Latino
H	Hispanic
H(Hispanic)	Hispanic
Hispanic	Hispanic
Hispanic or Latino	Hispanic
A	Asian
A(Asian or Pacific Islander)	Asian
Asian	Asian
Asian or Pacific islander	Asian
American Ind	American Indian
American Indian/Alaska Native	American Indian"""

subject_gender_csv_content = """F	Female
Female	Female
FEMALE	Female
M	Male
M, M	Male
Male	Male
MALE	Male
No Data	Unknown
not recorded	Unknown
Not Specified	Unknown
Unk	Unknown
Unknown	Unknown
UNKNOWN	Unknown
-	Unknown"""


## utility function to get the map of raw -> standardized
def _get_map(content):
    _map = {}
    for line in content.split("\n"):
        raw = line.split("	")[0]
        standardized = line.split("	")[1]
        _map[raw] = standardized
    return _map

## utility function to get the frequency count of elements 
def _get_count(x):
    return dict(Counter("|".join(x).split("|")))

## utility function to cleanup the name 
def _cleanup_dist(x):
    try:
        x = str(int(float(x)))
    except Exception as E:
        x = "NA"
    return x 

## Create the raw-standardized maps after reading the csv content as shown in image above 
subject_race_map = _get_map(subject_race_csv_content)
subject_gender_map = _get_map(subject_gender_csv_content)

column_config = {
    "SUBJECT_RACE" : { "variations": ["SUBJECT_RACT"],  "values_map" : subject_race_map },
    "SUBJECT_GENDER" : { "variations": [],  "values_map" : subject_gender_map },
    }


## master function to standardize the column names and values
def _standardize_columns(datadf):
    for col, col_dict in column_config.items():
        col_dict["variations"].append(col)
        _map = col_dict["values_map"]
        for colname in col_dict["variations"]:
            if colname in datadf.columns:
                datadf[col] = datadf[colname].apply(lambda x : _map[x] if x in _map else "-")
                
    ## Standardize Date Column, add Year and Month
    if "INCIDENT_DATE" in datadf.columns:
        datadf["INCIDENT_DATE"] = pd.to_datetime(datadf["INCIDENT_DATE"])
        datadf["INCIDENT_YEAR"] = datadf["INCIDENT_DATE"].dt.year
        datadf["INCIDENT_MONTH"] = datadf["INCIDENT_DATE"].dt.month
    
    if "LOCATION_DISTRICT" in datadf.columns:
        datadf["LOCATION_DISTRICT"] = datadf["LOCATION_DISTRICT"].astype(str)    

    return datadf

## Function to standardize the events data file
def _standardize_filename(_dept):
    _file = [f for f in os.listdir(_base_dir + _dept) if f.endswith(".csv")][0]
    old_path = _base_dir + _dept + "/" + _file
    new_path = _root_dir + _dept + "/events/" + _file
    shutil.copy(old_path, new_path)
    return _file

def _process_events(pol_config):
    ## load the given police incidents file and cleanup some missing info
    ppath = _root_dir + _dept + "/events/" + pol_config["police_file"]
    events_df = pd.read_csv(ppath, low_memory=False)[1:]
    events_df = _standardize_columns(events_df)

    ## Slice the data for the given years, if given by user
    years_to_process = pol_config["years_to_process"]
    if len(years_to_process) != 0: 
        events_df = events_df[events_df['INCIDENT_YEAR'].isin(years_to_process)]
    
    ## Aggregate the events by every district of the department
    police_df = events_df.groupby("LOCATION_DISTRICT")

    ## [Extendable] Obtain the distribution by gender, race etc
    police_df = police_df.agg({"SUBJECT_GENDER" : lambda x : _get_count(x),\
                               "SUBJECT_RACE"   : lambda x : _get_count(x)})
    police_df = police_df.reset_index()
    police_df = police_df.rename(columns={
                    "SUBJECT_GENDER" : pol_config['event_type'] + "_sex",\
                    "SUBJECT_RACE" : pol_config['event_type'] + "_race"})
    return police_df, events_df 


def _load_external_dataset(pol_config):
    ## load the dataset 
    _path = external_datasets_path + pol_config["path"]
    events2 = pd.read_csv(_path, parse_dates=[pol_config["date_col"]])

    ## basic standardization
    events2['year'] = events2[pol_config["date_col"]].dt.year
    years_to_process = pol_config["years_to_process"]
    events2 = events2[events2['year'].isin(years_to_process)]
    events2[pol_config["race_col"]] = events2[pol_config["race_col"]].fillna("")
    events2[pol_config["gender_col"]] = events2[pol_config["gender_col"]].fillna("")
    
    ## Aggregate and cleanup
    events2["LOCATION_DISTRICT"] = events2[pol_config['identifier']].apply(
                                                lambda x : _cleanup_dist(x))
    temp_df = events2.groupby("LOCATION_DISTRICT").agg({
                                pol_config['gender_col'] : lambda x : _get_count(x),\
                                pol_config['race_col'] : lambda x : _get_count(x)})
    
    ## cleanup the column names
    temp_df = temp_df.reset_index().rename(columns={
                                pol_config['gender_col'] : pol_config["event_type"]+"_sex", 
                                pol_config['race_col'] : pol_config["event_type"]+"_race"})
    return temp_df


def _save_final_data(enriched_df, police_df, events_df):
    enriched_df.to_csv(_root_dir +"/"+ _dept + "/enriched_df.csv", index = False)
    police_df.to_csv(_root_dir +"/"+ _dept + "/police_df.csv", index = False)
    events_df.to_csv(_root_dir +"/"+ _dept + "/events/events_df.csv", index = False)

def _execute_district_pipeline(_dept, _police_config1, _police_config2=None):
    print ("Selected Department: ", _dept)
    
    ## department shape file
    print (". Loading Shape File Data")
    dept_shape_gdf = _read_shape_gdf(_dept)
    base_plot = _plot_shapefile_base(dept_shape_gdf, _dept, overlapped_cts = {})    

    ## finding overlapped CTs percentages
    print (".. Finding Overlapping CTs")
    _identifier = depts_config[_dept]["_rowid"]
    state_cts = _read_ctfile(_dept)
    overlapped_cts, olaps_percentages = find_overlapping_cts(dept_shape_gdf, state_cts, _identifier)
    overlapped_plot = _plot_shapefile_base(dept_shape_gdf, _dept, overlapped_cts)
    
    ## Adding the Metrics Data
    print ("... Loading ACS Metrics Data")
    metrics_df = _cleanup_metrics_data(_dept)

    ## Add Metrics to the dept df
    print (".... Enrichment of ACS Metrics with Overlapped Data")
    dept_enriched_gdf = dept_shape_gdf.copy(deep=True)
    for metric_name in metrics_config.keys():
        dept_enriched_gdf = _process_metric(metrics_df, dept_enriched_gdf, _identifier, 
                                            olaps_percentages, metric_name=metric_name)
    
    ## Find Enriched DF
    enriched_df = _flatten_gdf(dept_enriched_gdf, _identifier)
    enriched_df = enriched_df.rename(columns={_identifier : "LOCATION_DISTRICT"})
    
    ## Processing Police DF
    if _police_config1 != None:
        print ("..... Standardizing the Police Events")
        police_file1 = _standardize_filename(_dept)
        _police_config1["police_file"] = police_file1
        police_df, events_df = _process_events(_police_config1)
    else:
        police_df, events_df = pd.DataFrame(), pd.DataFrame()
    
    ## Adding any other external Police Data 
    if _police_config2 != None:
        print ("..... Standardizing the External Data")
        external_df = _load_external_dataset(_police_config2)
        police_df = police_df.merge(external_df, on="LOCATION_DISTRICT")
    
    ## Save Final Data
    print ("...... Saving the Final Data in New Repository")
    _save_final_data(enriched_df, police_df, events_df)
    
    response = {
                "dept_shape_gdf" : dept_shape_gdf,
                "base_plot" : base_plot,
                "olaps_percentages" : _prepare_olaps_df(olaps_percentages),
                "overlapped_plot" : overlapped_plot,
                "dept_enriched_gdf" : dept_enriched_gdf,
                "enriched_df" : enriched_df,
                "police_df" : police_df,
                "events_df" : events_df
                }
    return response

from IPython.display import display, HTML
def _view_output(pipeline_resp, dpt):
    display(HTML("<h3>Pipeline Output: " + dpt + "</h3>"))
    display(HTML("<b>GeoDataframe: " + dpt+ "</b>"))
    display(pipeline_resp["dept_shape_gdf"].head())
    display(HTML("<b>Districts Map: "+ dpt+ "</b>"))
    display(pipeline_resp["base_plot"])
    display(HTML("<b>Overlapped Tracts & Percentages : "+ dpt+ " </b>"))
    display(pipeline_resp["olaps_percentages"].head(10))
    display(HTML("<b>Overlapped Tracts: "+ dpt+ "</b>"))
    display(pipeline_resp["overlapped_plot"])
    display(HTML("<b>Final Enriched Data : "+ dpt+ "</b>"))
    display(pipeline_resp["enriched_df"].head())
    if len(pipeline_resp["police_df"]) > 0:
        display(HTML("<b>Final Police Incidents : "+ dpt+ "</b>"))
        display(pipeline_resp["police_df"].head())
    display(HTML("<hr>"))
_run_standardization_pipeline()
## select department 
_dept = "Dept_49-00033"

## given police data config 
_police_config1 = { 'event_type' : 'arrest', "years_to_process" : []}

# ## external police data config
_police_config2 = {  'path' : "la_stops/vehicle-and-pedestrian-stop-data-2010-to-present.csv", 
                     'event_type' : 'vstops',
                     'identifier' : "Officer 1 Division Number" , 
                     'gender_col' : 'Sex Code', 
                     'race_col' : 'Descent Code', 
                     'date_col' : "Stop Date", 
                     'years_to_process' : [2015] }

## call the trigger for the given department and their configurations
pipeline_resp = _execute_district_pipeline(_dept, _police_config1, _police_config2)
_view_output(pipeline_resp, _dept)
_dept = "Dept_24-00013"
_police_config1 = { 'event_type' : 'uof', "years_to_process" : [2012, 2013, 2014, 2015, 2016, 2017]}
_police_config2 = { "path" : "minneapolis_stops/Minneapolis_Stops.csv", "years_to_process" : [2016] , 'identifier' : "policePrecinct" , 'gender_col' : 'gender', 'race_col' : 'race', 'date_col' : 'responseDate', 'event_type' : 'vstops'}

pipeline_resp = _execute_district_pipeline(_dept, _police_config1, _police_config2)
_view_output(pipeline_resp, _dept)
_dept = "Dept_49-00081" 
_police_config1 = { 'event_type' : 'uof', "years_to_process" : []}
pipeline_resp = _execute_district_pipeline(_dept, _police_config1 = None)
_view_output(pipeline_resp, _dept)
_dept = "Dept_49-00035" 
_police_config1 = { 'event_type' : 'uof', "years_to_process" : []}
pipeline_resp = _execute_district_pipeline(_dept, _police_config1 = None)
_view_output(pipeline_resp, _dept)
_dept = "Dept_23-00089"
_police_config1 = { 'event_type' : 'uof', "years_to_process" : []}
pipeline_resp = _execute_district_pipeline(_dept, _police_config1)
_view_output(pipeline_resp, _dept)
_dept = "Dept_37-00049"
_police_config1 = { 'event_type' : 'uof', "years_to_process" : [2016]}
pipeline_resp = _execute_district_pipeline(_dept, _police_config1)
_view_output(pipeline_resp, _dept)
_dept = "Dept_24-00098"
_police_config1 = { 'event_type' : 'uof', "years_to_process" : [2015, 2016, 2017]}
pipeline_resp = _execute_district_pipeline(_dept, _police_config1)
_view_output(pipeline_resp, _dept)