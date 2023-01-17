# libs
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import re
import math
import folium
import pickle
from collections import namedtuple
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from functools import reduce
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.distance import vincenty
from IPython.display import Image, display, HTML
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import ConvexHull
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import dill

pd.options.display.max_rows = 10

init_notebook_mode(connected=True)
%matplotlib inline
# Custom pipe Functions
def csnap(df, fn=lambda x: x.shape, msg=None):
    """ Custom Help function to print things in method chaining.
        Returns back the df to further use in chaining.
    """
    if msg:
        print(msg)
    display(fn(df))
    return df
    
def cfilter(df, fn, axis='rows'):
    """ Custom Filters based on a condition and returns the df.
        function - a lambda function that returns a binary vector
        thats similar in shape to the dataframe
        axis = rows or columns to be filtered.
        A single level indexing
    """
    if axis == 'rows':
        return df[fn(df)]
    elif axis == 'columns':
        return df.iloc[:, fn(df)]
    
def ccol(df, string, sep=';'):
    """ Custom column filtering that works in multi level indexing.
    """
    return df.iloc[:, reduce(lambda x, y: x & y,
                          [(df.columns.get_level_values(i).to_series()
                                 .str.contains(string, case=False)
                                 .fillna(True).values)
                            for i, string in enumerate(string.split(sep))])]

def setcols(original, fn=lambda x: x, cols=None):
    """Sets the column of the data frame to the passed column list.
    """
    df = original.copy()
    if cols:
        df.columns = cols
    elif isinstance(df.columns, pd.core.index.MultiIndex):
        df.columns = ['_'.join(map(fn, args)).strip('_') 
                         for args in zip(*[df.columns.get_level_values(i).fillna('') 
                                           for i in range(len(df.columns.levels))])]
    elif not isinstance(df.columns, pd.core.index.MultiIndex):
        df.columns = fn(df)
    return df
# data codes
dc = \
{
    'edu': "S1501",
    'edu_25': "B15003",
    'housing': "S2502",
    'poverty': "S1701",
    'rsa': "DP05",
    'employement': "S2301",
    'income': "S1903"
}

# city codes
cc = \
{
    'boston': "11-00091",
    'indianapolis': "23-00089",
    'minneapolis': "24-00013",
    'st_paul': "24-00098",
    'orlando': "35-00016",
    'charlotte': "35-00103",
    'austin': "37-00027",
    'dallas': "37-00049",
    'seattle': "49-00009",
    'los_angeles': "49-00033",
    'oakland': "49-00035",
    'san_francisco': "49-00081"
}

city_state = ["boston, Massachusetts", "indianapolis, indiana", "charlotte, north carolina",
              "austin, texas", "dallas, texas", "seattle, washington", "minneapolis, minnesota",
             "st paul, minnesota", "orlando, florida", "los angeles, california",
              "oakland, california", "san francisco, california"]

# data containers
data_list = list(dc.keys())

depts = dict()
uof = dict()

is_uof_present = dict()

dept = namedtuple('dept', data_list)
acs_data = namedtuple('acs_data', ['data', 'meta'])
# reading data
data_dir = '../input/data-science-for-good/cpe-data/'
dept_paths = [f.path for f in os.scandir(data_dir)
                         if f.is_dir()]
ndepts = len(dept_paths)
dept_list = [None] * len(dept_paths)
print(f"Processing {ndepts} PD's")
shape_folder = []

dept_cnt = 0
for path in tqdm(dept_paths):
    try:
        dept_folder = path.split('/')[-1]
        folder = ''
        dept_code = re.findall(r"_(\d{2}-\d{5})", dept_folder)[0]
        dept_list[dept_cnt] = dept_code

        acs_folder = [f.path for f in os.scandir(path)
                      if f.is_dir()
                      and (f.path).lower().find('acs') > 0][0]
        shape_folder.append([f.path for f in os.scandir(path)
                        if f.is_dir()
                        and (f.path).lower().find('shape') > 0][0])

        acs_dl = [None] * len(dc.keys())
        for folder in os.scandir(acs_folder):
            if folder.is_dir():
                file_code = ''
                for sub_folder in os.scandir(folder.path):
                    if (sub_folder.path).split('\\')[-1][0] != "_":
                        file_code = re.findall(r"5YR_([A-Z\d]*)_",
                                               sub_folder.path,
                                               flags=re.IGNORECASE)
                    if sub_folder.path.lower().find('meta') > 0:
                        meta = pd.read_csv(sub_folder.path,
                                           low_memory=False)
                    elif sub_folder.path.lower().find('ann') > 0:
                        data = pd.read_csv(sub_folder.path,
                                           skiprows=1,
                                           low_memory=False)
                        
                data_meta = acs_data(data, meta)

                if len(file_code) == 1:
                    for k,v in dc.items():
                        if v == file_code[0]:
                            acs_dl[data_list.index(k)] = data_meta
        
        # Checks for a file in the folder and sees if it has known headers.
        uof_path = [f.path for f in os.scandir(path)
                      if f.is_file()
                      and (f.path).lower().find('csv') > 0]
        
        #temp = pd.read_csv(uof_path[0])
        #if temp.columns.str.contains('incident|subject', case=False)
        
        if len(uof_path) == 1:
            uof[dept_code] = pd.read_csv(uof_path[0],
                                         low_memory=False)
            is_uof_present[dept_code] = True
        else:
            is_uof_present[dept_code] = False
                
        depts[dept_code] = dept(*acs_dl)
        dept_cnt += 1
    except:
        print(f"{path} - {folder} has unexpected structure")
depts[cc["dallas"]].poverty.data.head(2)
dept_shapes, has_req_files, is_consistent = dict(), dict(), dict()
# shape file load
for sub_folder in shape_folder:
    req_file = True
    consistent_flag = False
    try:
        dept_code = re.findall(r"_(\d{2}-\d{5})", sub_folder)[0]
        file = ''
        has_req_files[dept_code] = len([file.path for file in os.scandir(sub_folder)
                               for ext in '.shp|.shx|.dbf'.split('|')
                               if (file.path).endswith(ext)]) >= 3

        for file in os.scandir(sub_folder):
            if (file.path).endswith('.shp'):
                dept_shapes[dept_code] = gpd.read_file(file.path)   
                if (~((dept_shapes[dept_code].geometry.type == 'Polygon') |
                        (dept_shapes[dept_code].geometry.type == 'MultiPolygon'))).sum() != 0:
                    is_consistent[dept_code] = False
                    print(f"{file.path} has inconsistent geometry")
                else:
                    is_consistent[dept_code] = True
    except:
        print(f"{sub_folder} - {file} has incorrect structure")

dept_shapes[cc["charlotte"]]= gpd.read_file('../input/cpe-external-adiamaan/external/CMPD_Police_Divisions/CMPD_Police_Divisions.shp')
census = dict()
#read census
dest_crs = {'init': 'epsg:4326'}
for folder in os.scandir('../input/cpe-external-adiamaan/external/census'):
    if folder.path.split('/')[-1].startswith('cb'):
        state = re.findall(r'cb_2017_(.*)_tract', folder.path, flags=re.IGNORECASE)
        for file in os.scandir(folder):
            if (file.path).endswith('.shp'):
                sp = gpd.read_file(file.path)  
                print(f"{file.path} = {sp.crs}")
                if sp.crs:
                    sp = sp.to_crs(dest_crs)
                if (~((sp.geometry.type == 'Polygon') |
                        (sp.geometry.type == 'MultiPolygon'))).sum() != 0:
                    print(f"{files} has inconsistent geometry")
                else:
                    census[state[0]] = sp
census = \
(
    pd.concat(census.values())
        .pipe(setcols, lambda x: x.columns.str.lower())
        .pipe(csnap, lambda x: x.sample(5))
)
is_acs_data_consistent = dict()
for data in data_list:
    is_acs_data_consistent[cc['boston']] = True
    for x,y in zip(['boston']*len(cc), list(cc.keys())[1:]):
        consistent = True
        if (getattr(depts[cc[x]], data).meta != \
                getattr(depts[cc[y]], data).meta).sum().sum() == 0 :
            consistent = False
        is_acs_data_consistent[cc[y]] = True
data_dict = ()
data_dict = \
{data:pd.concat(
[(getattr(depts[cc[x]], data).data
              .assign(City = x)) for x in list(cc.keys())])
    for data in data_list}
for _, x in data_dict.items():
    x.columns = x.columns.str.split(r';\s|\s-\s', expand=True)
for _, df in data_dict.items():
    char_cols = (df.columns.get_level_values(0)
                     .str.contains('geo|id|city',
                                   case=False))
    df.iloc[:, ~char_cols] = (df.iloc[:, ~char_cols]
                                .apply(lambda x: 
                                       pd.to_numeric(x, errors="coerce"),
                                       axis=1))
for key, value in data_dict.items():
    data_dict[key] = \
    (value.assign(id2_str = lambda x: x.Id2.astype(str))
        .assign(statefp = lambda x: x.id2_str.str[:2],
                countyfp = lambda x: x.id2_str.str[2:5],
                tractce = lambda x: x.id2_str.str[5:]))
orig_crs = dict()
orig_crs = {city:dept_shapes[cc[city]].crs
                for city,_ in cc.items()}
[print(f"{k} = {dept_shapes[cc[k]].crs}")
    for k,_ in cc.items()];
[print(f"{k} = {dept_shapes[cc[k]].geometry.head(1)}")
     for k,_ in cc.items()];
epsg = \
(
    pd.read_csv('../input/cpe-external-adiamaan/external/epsg_code.csv')    
        .pipe(csnap, lambda x: x.sample(5))
)
(
    dept_shapes[cc["boston"]]
            .pipe(csnap, lambda x: x.head(2),
                  msg="Original")
            .to_crs(epsg='4326')
            .pipe(csnap, lambda x: x.head(2),
                  msg="In GCS")
            .to_crs(epsg='2249')
            .pipe(csnap, lambda x: x.head(2),
                  msg="Original with negligible loss")
);
city_cs = dict()
coord_system = 'gcs'
for city in cc.keys():
    sp = dept_shapes[cc[city]]
    ndigits = max([math.floor(math.log10(abs(n))) + 1
                     for n in sp.geometry.bounds.max().values])
    if ndigits in [2,3]:
        city_cs[city] = 'gcs'
    elif ndigits in [6,7,8]:
        city_cs[city] = 'pcs'
    else:
        print(f"{city} unrecognized coordinate system")
            
    if sp.crs:
        sp = sp.to_crs(dest_crs)
    elif not sp.crs and city_cs[city] == 'gcs':
        sp = sp.to_crs(dest_crs)

    dept_shapes[cc[city]] = sp
city_cs
locations = dict()
#uncomment code to automatically fetch the lat/long of the cities
# %%time
# geolocator = Nominatim(user_agent="cpe_mak_kaggle_kernel")
# for i, loc in enumerate(city_state):
#     city = loc.split(",")[0]
#     locations[city] = geolocator.geocode(loc)

#comment below
with open('../input/cpe-external-adiamaan/locations.pkl', 'rb') as handle:
    locations = pickle.load(handle)
locations
# logic demo
boston_bounds =  dept_shapes[cc["boston"]].bounds
minx, miny, maxx, maxy = \
boston_bounds.minx.min(), boston_bounds.miny.min(),\
boston_bounds.maxx.max(), boston_bounds.maxy.max()
corners = [(miny, minx), (miny, maxx),
           (maxy, minx), (maxy, maxx)]

ffig = folium.Figure(height=400)
fmap = folium.Map(location=[42.3248716,-71.102893],
                  zoom_start=10.45, tiles= "CartoDB positron").add_to(ffig)

(folium.GeoJson(dept_shapes[cc["boston"]],
               style_function=lambda x: {'fillColor': 'grey',
                                       'color': 'Black',
                                       'fillOpacity': 0.30,
                                       'weight':0.75},)
    .add_to(fmap))

folium.CircleMarker(location=[locations['boston'].latitude, locations['boston'].longitude],
                     radius=10,
                    popup='Boston City', 
                    fill_color='blue', fill_opacity= 1.0).add_to(fmap)

for corner in corners:
    folium.CircleMarker(location=[corner[0], corner[1]],
                     radius=10,
                    fill_color='red', fill_opacity= 1.0).add_to(fmap)
    folium.PolyLine([[corner[0], corner[1]], 
                    [locations['boston'].latitude,
                     locations['boston'].longitude]],
                   color='blue').add_to(fmap)

folium.PolyLine([[miny, minx], [maxy, minx],
                 [maxy, maxx], [miny, maxx],
                [miny, minx]], color="red").add_to(fmap)
    
fmap
%%time
for city in cc.keys():
    sp = dept_shapes[cc[city]].copy()
    if city_cs[city] == 'pcs' and not sp.crs:
            total_distance = [None] * epsg.shape[0]
            for i,code in enumerate(epsg.crs_code):
                sp = dept_shapes[cc[city]].copy()
                crs = {'init': 'epsg:' + str(code)}
                sp.crs = crs
                sp = sp.to_crs(dest_crs)
                bounds = sp.bounds
                minx, miny, maxx, maxy = \
                bounds.minx.min(), bounds.miny.min(),\
                bounds.maxx.max(), bounds.maxy.max()
                corners = [(miny, minx), (miny, maxx),
                           (maxy, minx), (maxy, maxx)]
                distance = 0
                for lat, long in corners:
                    distance += vincenty((locations[city].latitude,
                                         locations[city].longitude),
                                         (lat, long)).miles
                total_distance[i] = distance/4
            print(epsg.iloc[total_distance.index(min(total_distance)), 0], city)
            correct_crs_code = epsg.iloc[total_distance.index(min(total_distance)), 0]
            correct_crs = {'init': 'epsg:' + str(correct_crs_code)}
            orig_crs[city] = correct_crs
            dept_shapes[cc[city]].crs = correct_crs
            dept_shapes[cc[city]] = dept_shapes[cc[city]].to_crs(dest_crs)
[print(f"{k} = {dept_shapes[cc[k]].crs}")
     for k,_ in cc.items()];
# distance calc
for city in cc.keys():
    loc_city = city.replace('_', ' ')
    bounds = dept_shapes[cc[city]].bounds
    minx, miny, maxx, maxy = \
    bounds.minx.min(), bounds.miny.min(),\
    bounds.maxx.min(), bounds.maxy.min()
    corners = [(miny, minx), (miny, maxx),
               (maxy, minx), (maxy, maxx)]
    total_distance = 0
    for lat, long in corners:
        total_distance += vincenty((locations[loc_city].latitude,
                                    locations[loc_city].longitude),
                                   (lat, long)).miles
    print(f"{city} - {total_distance/4:.2f} miles")
# wrapping plotters
def folium_map(data, style_function, **kwargs):
    fig_pars = ['height', 'width']
    fig_args = {k:kwargs[k] for k in kwargs.keys() if k in fig_pars}
    kwargs = {k:kwargs[k] for k in kwargs.keys() if k not in fig_pars}
    
    ffig = folium.Figure(**fig_args)
    fmap = folium.Map(**kwargs).add_to(ffig)
    
    (folium.GeoJson(data,
               style_function=style_function)
        .add_to(fmap))
    
    display(fmap)
    return fmap

def cplotly(df, traces, **kwargs):
    config={'showLink': False}
    
    layout = go.Layout(**kwargs)
    fig = go.Figure(traces(df), layout)
    
    display(iplot(fig, config=config))
    return df

def cmatplot(df, **kwargs):
    display(df.plot(**kwargs))
    return df
# codes
map_latlong = dict(boston = [42.3248716,-71.102893],
                     indianapolis = [39.81, -86.15],
                     charlotte = [35.2030728,-80.8799136],
                     dallas = [32.8203525,-96.811731],
                     austin = [30.308179,-97.8184848],
                     seattle = [47.6129432,-122.3821475],
                     minneapolis = [44.9706756,-93.3315183],
                     st_paul = [44.9396219,-93.2461368],
                     orlando = [28.4810971,-81.5088354],
                     los_angeles = [34.0201613,-118.6919205],
                     oakland = [37.7583925,-122.3754124],
                     san_francisco = [37.7576171,-122.5776844])

map_color = dict(boston = 'red',
                     indianapolis = 'blue',
                     charlotte = 'green',
                     dallas = 'yellow',
                     austin = 'beige',
                     seattle = 'pink',
                     minneapolis = 'brown',
                     st_paul = 'teal',
                     orlando = 'purple',
                     los_angeles = 'orange', 
                     oakland = 'lightgreen',
                     san_francisco = 'lightblue')
ncities = len(cc.keys())
axis_list =  ['axis_' + str(i) for i in range(ncities)]
figure, axis_list = plt.subplots(nrows=ncities//3, ncols=3)

for i, city in enumerate(cc.keys()):
    j, k = i%3, i%4
    dept_shapes[cc[city]].plot(ax=axis_list[k][j],
                               color=map_color[city],
                               edgecolor='black')
    axis_list[k][j].set_title(city.upper().replace('_', ' '),
                              fontsize=15)
    axis_list[k][j].set_axis_off()

figure.set_size_inches(14, 14)
uof_present = [key for key, value in cc.items()
                       if is_uof_present[value]]
uof_present
[(uof[cc[city]]
      .pipe(csnap, msg=f"{city}")
      .pipe(csnap, lambda x: x.head(2)))
     for city in uof_present[:3]];
common_cols = ['INCIDENT_DATE',  'LOCATION_DISTRICT',
               'LOCATION_LATITUDE', 'LOCATION_LONGITUDE',
               'SUBJECT_RACE']
import warnings
warnings.filterwarnings('ignore')
# normalize uof_data
uof_list = []
for city in tqdm(uof_present):
    data = uof[cc[city]].iloc[1:,]
    data = data.loc[:, common_cols]
    
    match_cols = uof[cc[city]].iloc[1:,].columns.str.contains('^x|^y|coord',
                                                              case=False)
    filt_data = uof[cc[city]].iloc[1:,].iloc[:, match_cols]
    
    if sum(match_cols) == 2:
        data['COORDINATES'] = list(zip(pd.to_numeric(filt_data.iloc[:, 0],
                                                     errors="coerce"),
                                         pd.to_numeric(filt_data.iloc[:, 1],
                                                       errors="coerce")))
        crs = orig_crs[city]

    elif sum(match_cols) == 0:
        data['COORDINATES'] = list(zip(data.LOCATION_LONGITUDE.astype(np.float16),
                                       data.LOCATION_LATITUDE.astype(np.float16)))
        crs = dest_crs
    
    data['COORDINATES'] = data['COORDINATES'].apply(Point)
    data = gpd.GeoDataFrame(data, geometry='COORDINATES')
    data.crs = crs
    data = data.to_crs(dest_crs)

    data = data.reset_index().rename(columns={'index': 'ID'})
        
    data = data.assign(city = city,
                      INCIDENT_DATE = lambda x: pd.to_datetime(x.INCIDENT_DATE))
    
    uof_list.append(data.loc[:, common_cols+['COORDINATES', 'city']])
uof_agg = pd.concat(uof_list).reset_index(drop=True)
for city in cc.keys():
    dept_shapes[cc[city]] = (dept_shapes[cc[city]]
                                 .reset_index()
                                 .rename(columns={'index': 'UUID'}))
district_name = dict(orlando = 'Sector',
                    charlotte = 'DNAME')
do_point_in_polygon = False
if do_point_in_polygon:
    uof_agg['SHAPE_ID'] = np.nan
    for i in tqdm(range(uof_agg.shape[0])):
        lat = float(uof_agg.at[i, 'LOCATION_LATITUDE'])
        long = float(uof_agg.at[i, 'LOCATION_LONGITUDE'])
        if pd.notna(lat) and pd.notna(long) and lat != 0 and long != 0:
            city = uof_agg.at[i, 'city']
            PD = dept_shapes[cc[city]]
            for j in range(PD.shape[0]):
                if PD.at[j, 'geometry'].contains(uof_agg.at[i, 'COORDINATES']):
                    uof_agg.at[i, 'SHAPE_ID'] = PD.at[j, 'UUID']
    uof_agg.to_pickle('uof_agg_with_PD.pkl')
else:
    uof_agg = pd.read_pickle('../input/cpe-external-adiamaan/external/uof_agg_with_PD.pkl')
# find state and county code from data
state_city_data = dict()
county_city_data = dict()
for city in cc.keys():
    state_city_data[city] = list()
    county_city_data[city] = list()
    for key, value in data_dict.items():
        data = (data_dict['edu']
                     .pipe(setcols)
                     .query("City == @city"))
        state_city_data[city].append(data.statefp.unique())
        county_city_data[city].append(data.countyfp.unique())     
        
state_city_data = {key:(np.unique(value).astype(int))
                       for key, value in state_city_data.items()}
county_city_data = {key:(np.unique(value).astype(int))
                        for key, value in county_city_data.items()}        
census = \
(
    census
        .assign(statefp = lambda x: pd.to_numeric(x.statefp),
                countyfp = lambda x: pd.to_numeric(x.countyfp))
)
overlay, is_missing_overlay = dict(), dict()
overlay_working = False
if overlay_working:
    for city in tqdm(cc.keys()):
        statefp, countyfp = state_city_data[city], county_city_data[city]
        overlay[city] = gpd.overlay((census.query("statefp in @statefp \
                                                and countyfp in @countyfp")),
                                    dept_shapes[cc[city]], how='intersection')
        if overlay[city].shape[0] > 0:
            is_missing_overlay[city] = False
        else:
            is_missing_overlay[city] = True
        overlay[city].crs = dest_crs
else:
    with open('../input/cpe-external-adiamaan/overlay.pkl', 'rb') as handle:
        overlay = pickle.load(handle)
ncities = len(cc.keys())
axis_list =  ['axis_' + str(i) for i in range(ncities)]
figure, axis_list = plt.subplots(nrows=ncities//3, ncols=3)
figure.subplots_adjust(hspace=0.5)

for i, city in enumerate(cc.keys()):
    j, k = i%3, i%4
    axis_list[k][j].set_title(city.upper().replace('_', ' '),
                                  fontsize=15)
    axis_list[k][j].get_xaxis().set_ticklabels([])
    axis_list[k][j].get_yaxis().set_ticklabels([])
    if overlay[city].shape[0] > 0:
        overlay[city].plot(ax=axis_list[k][j],
                                   color=map_color[city],
                                   edgecolor='black')
        axis_list[k][j].set_axis_off()

figure.set_size_inches(12, 12)
#figure.set_size_inches(10.5, 10.5)
fields = ['statefp', 'countyfp', 'tractce']
for city in cc.keys():
    if overlay[city].shape[0] > 0:
        overlay[city].loc[:, fields] = \
        (overlay[city]
            .loc[:, fields]
            .apply(lambda x: pd.to_numeric(x, errors="coerce")))
    for key in data_dict.keys():
            data_dict[key].loc[:, fields] = \
            (data_dict[key]
                .loc[:, fields]
                .apply(lambda x: pd.to_numeric(x, errors="coerce")))
        
flag_dicts = [is_uof_present, has_req_files, is_consistent,
              is_acs_data_consistent]     
(
pd.concat([
(
pd.DataFrame
    .from_dict([is_uof_present, has_req_files,
                is_consistent,is_acs_data_consistent])
    .rename(columns={value:key
                     for key, value in cc.items()})
),
(
pd.DataFrame.from_dict([is_missing_overlay,
                        state_city_data,
                        county_city_data])
)]
).assign(flag = ['is_uof_preset', 'has_required_shape_files',
                 'is_shape_file_consistent', 'is_acs_data_consistent',
                 'is_not_overlaying', 'state_city_code', 'county_city_code'])
 .set_index('flag')
)
# For brevity
race_naming = ((r"American Indian (?:or|and) Alaska Native\s?",
                'Native'),
               (r"Native Hawaiian (?:or|and) Other Pacific Islander\s?",
                'Pacific'),
               (r"Black (?:or|and) African American\s?",
                "Black"))

edu_naming = ((r"Bachelor's degree or higher",
               "Bach"),
              (r"High school graduate or higher",
               "HS"))
races = ['Native', 'Asian', 'Black',
         'Pacific', 'White']
#edu
edu = data_dict["edu"].copy()
edu = edu.pipe(ccol, 'total|city|state|county|tract;'
                      'Estimate|^$;alone|^$;')
edu.columns = edu.columns.droplevel([1, -1])
edu = \
(edu.pipe(setcols)
     .pipe(setcols, lambda x: x.columns.str.replace('Total_|alone|, '
                                                    'not Hispanic or Latino', ''))
     .pipe(setcols, lambda x: [reduce(lambda a, kv: re.sub(*kv, a),
                                      race_naming, cname)
                                for cname in x.columns])
     .pipe(setcols, lambda x: [reduce(lambda a, kv: a.replace(*kv),
                                      edu_naming, cname)
                                for cname in x.columns])
     .pipe(cfilter, lambda x: (~x.columns.str
                                 .contains('Some', case = False)),
           axis='columns')
     .pipe(setcols, lambda x: x.columns.str.replace('\s+', ''))
)

for race in races:
    for education in ['Bach', 'HS']:
        edu[race + '_' + education] = \
            edu[race + '_' + education]/edu[race]
edu = (edu.pipe(cfilter, lambda x: ~x.columns.isin(races),
                 axis='columns'))
(
    edu
        .groupby('City', as_index=False).mean()
        .pipe(cfilter, lambda x: x.columns.str.contains('City|HS'), axis = 'columns')
        .pipe(setcols, lambda x: x.columns.str.replace('_HS', ''))
        .assign(City = lambda x: x.City.str.upper().str.replace('_', ' '))
        .set_index('City')
        .sort_index(axis=1)
        .style.format("{:.0%}").bar()
)
# poverty
poverty = \
(
 data_dict["poverty"]
     .pipe(ccol, 'city|state|county|tract|perc;'
                 'est|^$;^race|^$;alone|^$')
     .pipe(setcols)
     .pipe(setcols, lambda x: (x.columns
                                .str.replace('Percent below poverty level_Estimate_', '')
                                .str.replace('RACE AND HISPANIC OR LATINO ORIGIN_|'
                                              ', not Hispanic or Latino|'
                                              '\(of any race\)|alone', '')))
     .pipe(setcols, lambda x: [reduce(lambda a, kv: re.sub(*kv, a),
                                  race_naming, cname)
                            for cname in x.columns])
     .pipe(setcols, lambda x: x.columns.str.strip())
     .pipe(cfilter, lambda x: (~x.columns.str
                             .contains('Some', case = False)),
           axis='columns')

)
(
    poverty
        .pipe(cfilter, lambda x: ~x.columns.str.contains('state|county|tract'),
             axis = 'columns')
        .groupby('City', as_index=False).mean()
        .assign(City = lambda x: x.City.str.upper().str.replace('_', ' '))
        .set_index('City')
        .sort_index(axis=1)
        .style.format("{:.0f}%").bar()
)
# employment
employement = data_dict['employement']
employement = \
(
    employement
        .pipe(ccol, 'unempl|city|state|county|tract;'
                    'est|^$;^race|white|alone|^$')
        .pipe(setcols)
        .pipe(setcols,
              lambda x: (x.columns
                          .str.replace('Unemployment rate_Estimate_', '')
                          .str.replace('RACE AND HISPANIC OR LATINO ORIGIN_|'
                                       ', not Hispanic or Latino|'
                                       ' \(of any race\)|alone', '')
                          .str.strip()))
        .pipe(setcols, lambda x: [reduce(lambda a, kv: re.sub(*kv, a),
                                  race_naming, cname)
                            for cname in x.columns])
)
(
    employement
        .pipe(cfilter, lambda x: ~x.columns.str.contains('state|county|tract'),
         axis = 'columns')
        .groupby('City', as_index=False).mean()
        .assign(City = lambda x: x.City.str.upper().str.replace('_', ' '))
        .set_index('City')
        .pipe(cfilter, lambda x: x.columns.isin(races), axis='columns')
        .iloc[:, 1:]
        .sort_index(axis=1)
        .style.format("{:.0f}%").bar()
)
#rsa
rsa = data_dict["rsa"].copy()
rsa = \
(
    rsa.pipe(ccol, 'Percent$|city|state|county|tract;'
                   '^race|^$;one|^$;' +
                   '|'.join(races)+ '|^$' + ';^$')
        .pipe(setcols)
        .iloc[:,1:]
        .pipe(setcols,
              lambda x: (x.columns
                         .str.replace('Percent_RACE_One race_|\(of any race\)', '')
                         .str.replace('Percent_HISPANIC OR LATINO AND RACE_Total population_', '')))
        .pipe(setcols, lambda x: [reduce(lambda a, kv: re.sub(*kv, a),
                              race_naming, cname)
                            for cname in x.columns])
)
(
    rsa
        .pipe(cfilter, lambda x: ~x.columns.str.contains('state|county|tract'),
         axis = 'columns')
        .groupby('City', as_index=False).mean()
        .assign(City = lambda x: x.City.str.upper().str.replace('_', ' '))
        .set_index('City')
        .pipe(cfilter, lambda x: x.columns.isin(races), axis='columns')
        .sort_index(axis=1)
        .style.format("{:.0f}%").bar()
)
# income
income = data_dict["income"].copy()
income = \
(
    income.pipe(ccol, 'median|city|state|county|tract;'
                      'estim|^$;^hispanic|^Households$|^$')
        .pipe(setcols)
        .pipe(setcols,
              lambda x: (x.columns
                          .str.replace('Median income \(dollars\)_Estimate_Households_One race--_', '')
                          .str.replace('Median income \(dollars\)_Estimate_|\(of any race\)', '')))
        .pipe(setcols, lambda x: [reduce(lambda a, kv: re.sub(*kv, a),
                          race_naming, cname)
                        for cname in x.columns])
        .iloc[:, list(range(1, 6)) + list(range(8, 13))]
)

(
    income
        .pipe(cfilter, lambda x: ~x.columns.str.contains('state|county|tract'),
         axis = 'columns')
        .groupby('City', as_index=False).median()
        .assign(City = lambda x: x.City.str.upper().str.replace('_', ' '))
        .set_index('City')
        .pipe(cfilter, lambda x: x.columns.isin(races), axis='columns')
        .sort_index(axis=1)
        .apply(lambda x: x/1000)
        .style.format("${:.2f} K").bar()
)
austin_uof = \
(
    uof_agg.query("city == 'austin'").pipe(csnap)
        .pipe(cfilter, lambda x: x.SHAPE_ID.notnull())
        .pipe(csnap)
        .assign(SHAPE_ID = lambda x: x.SHAPE_ID.astype(int))
)
# austin with crime data
city = 'austin'
boston_bounds =  dept_shapes[cc[city]].bounds
minx, miny, maxx, maxy = \
boston_bounds.minx.min(), boston_bounds.miny.min(),\
boston_bounds.maxx.max(), boston_bounds.maxy.max()
corners = [(miny, minx), (miny, maxx),
           (maxy, minx), (maxy, maxx)]

ffig = folium.Figure(height=400)
fmap = folium.Map(location=map_latlong[city],
                  zoom_start=10.45, tiles= "CartoDB positron").add_to(ffig)

(folium.GeoJson(dept_shapes[cc[city]].assign(geometry = lambda x: x.geometry.simplify(0.0001, preserve_topology=True)),
               style_function=lambda x: {'fillColor': 'green',
                                       'color': 'Black',
                                       'fillOpacity': 0.75,
                                       'weight':0.75})
    .add_to(fmap))

for corner in austin_uof.COORDINATES:
    folium.CircleMarker(location=[corner.y, corner.x],
                         radius=0.125,
                        fill_color='red',
                        fill_opacity= .75,
                       color='red').add_to(fmap)
    
fmap
austin_uof = \
(
    pd.DataFrame(
        austin_uof
            .groupby(['SHAPE_ID', 'SUBJECT_RACE'],
                 as_index=False).size().reset_index()
            .rename(columns={0:'narrests'}))
        .pivot(index='SHAPE_ID',
              columns='SUBJECT_RACE',
            values='narrests').reset_index()
)
# combining all the data
city = 'austin'
austin_results = \
(
    dept_shapes[cc[city]]
        .loc[:, ['UUID', 'geometry']]
        .pipe(csnap)
        .merge(overlay[city]
                .loc[:, ['statefp', 'countyfp',
                 'tractce', 'UUID']]
                .merge(rsa,
                       on = ['statefp', 'countyfp', 'tractce'])
                .pipe(csnap)
                .merge(poverty,
                       on = ['statefp', 'countyfp', 'tractce'],
                       suffixes = ('_POP', '_POV'))
                .merge(austin_uof,
                       left_on = 'UUID',
                       right_on = 'SHAPE_ID')
                .fillna(0)
                .groupby('UUID', as_index=False).mean())
)
for race in races:
    if race in austin_results.columns:
        austin_results[race+'_ADJ_ARR'] = \
        (austin_results[race] *
            (1 - austin_results[race + '_POV']/100) *
            (1 - austin_results[race + '_POP']/100))
austin_races = [col for col in austin_results.columns
                   if col in races]
for i, race in enumerate(austin_races):
    ax = austin_results.plot(column = race+'_ADJ_ARR',legend = True,
                             figsize = (6,6), k=4, cmap = 'viridis')
    ax.set_title(race+' Adjusted Arrests',
                 fontsize=18)
    plt.axis('equal')
    ax.set_axis_off()
