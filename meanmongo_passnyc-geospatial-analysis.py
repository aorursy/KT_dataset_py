import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import folium

from IPython.display import Image
from folium.plugins import HeatMap, HeatMapWithTime

%pylab inline
from pylab import rcParams
rcParams['figure.figsize'] = 16,8
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
stats = pd.read_csv('../input/nyc-school-demographics-over-time/nyc_school_demographics.csv')
stats.query('dbn == "15K039"')
stats.query('DBN == "15K039"').reset_index()[['black_per','white_per']].plot()
plt.title('PS 39 White and Black Student Populations Over Time')
stats.query('DBN == "15K321"').reset_index()[['black_per','white_per']].plot()
plt.title('PS 321 White and Black Student Populations Over Time')
stats.query('DBN == "15K107"').reset_index()[['black_per','white_per']].plot()
plt.title('PS 107 White and Black Student Populations Over Time')
stats.query('DBN == "13K282"').reset_index()[['black_per','white_per']].plot()
plt.title('PS 282 White and Black Student Populations Over Time')
stats.query('DBN == "15K010"').reset_index()[['black_per','white_per']].plot()
plt.title('PS 10 White and Black Student Populations Over Time')
# housing builds
house = pd.read_csv('../input/nyc-housing-builds/housing_builds.csv')
house.head()
def Zones1(directory):
    map_osm = folium.Map(location=[40.665535, -73.969749],zoom_start=14)
    # add heat
    folium.GeoJson(directory,
                   style_function=lambda feature: {
                       'fillColor': 'blue' if '15K039' in feature['properties']['DBN'] else 'red' if '15K107' in \
                       
                       feature['properties']['DBN'] else 'yellow' if '15K321' in feature['properties']['DBN'] \
                       else 'green' if '15K010' in feature['properties']['DBN'] else 'white',
                       'weight': 1
    }).add_to(map_osm)
    
    return map_osm

def Zones2(directory):
    map_osm = folium.Map(location=[40.665535, -73.969749],zoom_start=14)
    # add heat
    folium.GeoJson(directory,
                   style_function=lambda feature: {
                       'fillColor': 'blue' if '15K039' in feature['properties']['dbn'] else 'red' if '15K107' in \
                       
                       feature['properties']['dbn'] else 'yellow' if '15K321' in feature['properties']['dbn'] \
                       else 'green' if '15K010' in feature['properties']['dbn'] else 'white',
                       'weight': 1
    }).add_to(map_osm)
    
    return map_osm
# Zones1('../input/20122013-nyc-school-zones/2012 - 2013 School Zones.geojson')
# nypass 1
Image('../input/passny-map1/nypass1.png')
# Zones2('../input/20152016-nyc-school-zones/2015-2016_school_zones_fixed.geojson')
# nypass 2
Image('../input/passny-map-2/nypass2.png')
def profileChange1(year):
    m = folium.Map([40.665535, -73.969749], zoom_start=14)
    m.choropleth(
        geo_data='../input/20122013-nyc-school-zones/2012 - 2013 School Zones.geojson',
        data=stats.query('year == @year'),
        columns=['DBN','white_per'],
        key_on='feature.properties.DBN',
        fill_color='YlOrRd',
        threshold_scale=[5,25,45,65,85,100],
        legend_name='Percent White Students (%)'
    )
    return m

def profileChange2(year):
    m = folium.Map([40.665535, -73.969749], zoom_start=14)
    m.choropleth(
        geo_data='../input/20152016-nyc-school-zones/2015-2016_school_zones_fixed.geojson',
        data=stats.query('year == @year'),
        columns=['dbn','white_per'],
        key_on='feature.properties.dbn',
        fill_color='YlOrRd',
        threshold_scale=[5,25,45,65,85,100],
        legend_name='Percent White Students (%)'
    )
    return m
#profileChange1("2006")
# nypass 3
Image('../input/passny-map-3/nypass3.png')
#profileChange2("2017")
# nypass 4
Image('../input/passny-map-4/nypass4.png')
# demographic change and building
demo_builds = pd.read_csv('../input/nyc-school-demographic-change-and-housing-builds/nyc_school_demographic_change_and_housing_builds.csv')
demo_builds.head()
# build a table of black_change take absolute values
demo_tmp = demo_builds.query('black_change <= 0')
demo_tmp['black_change'] = np.abs(demo_tmp['black_change'])
def spatialChange():
    m = folium.Map([40.665535, -73.969749], zoom_start=14)
    m.choropleth(
        geo_data='../input/20152016-nyc-school-zones/2015-2016_school_zones_fixed.geojson',
        data=demo_tmp,
        columns=['dbn','black_change'],
        key_on='feature.properties.dbn',
        fill_color= 'YlOrRd',
        #threshold_scale=[0,5,25,50,100],
        legend_name=''
    )
    return m
#spatialChange()
# nypass 5
Image('../input/passny-map-5/nypass5.png')
def Location(radius, min_opacity, max_val):
    map_osm = folium.Map(location=[40.665535, -73.969749],zoom_start=14)
    coordinates = [(a,b) for a,b in zip(house['Latitude'], house['Longitude'])]
    # add heat
    folium.GeoJson('../input/20122013-nyc-school-zones/2012 - 2013 School Zones.geojson',
                   style_function=lambda feature: {
                       'fillColor': 'blue' if '15K039' in feature['properties']['DBN'] else 'red' if '15K107' in \
                       
                       feature['properties']['DBN'] else 'yellow' if '15K321' in feature['properties']['DBN'] \
                       else 'green' if '15K010' in feature['properties']['DBN'] else 'white',
                       'weight': 1
    }).add_to(map_osm)
    
    map_osm = map_osm.add_child(HeatMap(coordinates, radius = radius, min_opacity=min_opacity, max_val=max_val))
    return map_osm
# Location(12, 0.9, 0.9)
# nypass 6
Image('../input/passny-map-6/nypass6.png')
def heatTime(radius, opacity):
    year_dict = {2014:'2014',2015:'2015',2016:'2016',2017:'2017',2018:'2018'}
    heat_index = [house[house['year']==house['year'].unique()[i]][['Latitude','Longitude']].values.tolist() 
              for i in range(len(house['year'].unique()))]
    year_index = [year_dict[i] for i in sorted(house['year'].unique())]
    
    map_osm = folium.Map([40.665535, -73.969749], zoom_start=14)
    
    heat_map = HeatMapWithTime(
        heat_index,
        index=year_index,
        auto_play=False,
        radius = radius,
        max_opacity=opacity)
    heat_map.add_to(map_osm)
    return map_osm
heatTime(30, 0.3)
# School Explorer data
se = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
se = se.rename(columns={'School Name':'school_name',
                        'Location Code':'dbn',
                        'Zip':'zip'})

se = se[['dbn','school_name','zip','Economic Need Index','School Income Estimate','Percent ELL','Student Attendance Rate',
         'Percent of Students Chronically Absent','Supportive Environment %',
         'Effective School Leadership %','Strong Family-Community Ties %','Trust %',
         'Average ELA Proficiency','Average Math Proficiency']]

def removeDollar(data):
    return data.replace('$','')

def removeComa(data):
    return data.replace(',','')

def removePercent(data):
    return data.replace('%','')
for col in ['Percent ELL','Student Attendance Rate',
            'Percent of Students Chronically Absent','Supportive Environment %',
            'Effective School Leadership %','Strong Family-Community Ties %','Trust %']:
    se[col] = [removePercent(str(b)) for b in se[col]]
    
se['School Income Estimate'] = [removeDollar(removeComa(str(b))) for b in se['School Income Estimate']]

for col in ['School Income Estimate','Percent ELL','Student Attendance Rate',
            'Percent of Students Chronically Absent','Supportive Environment %',
            'Effective School Leadership %','Strong Family-Community Ties %','Trust %']:
    se[col] = se[col].astype('float64')
dbn_builds = pd.read_csv('../input/nyc-housing-builds-and-school-zones/nyc_housing_builds_and_dbn.csv')
diff = pd.read_csv('../input/nyc-school-demographic-change-and-housing-builds/nyc_school_demographic_change_and_housing_builds.csv')
# merge housing with school stats

data = pd.merge(diff[['dbn','asian_change','black_change','hispanic_change','white_change','enrollment_change']],
         dbn_builds.groupby(['dbn','label']).sum()['Counted Rental Units'].reset_index(),
         how = 'inner',
         on='dbn')
data = pd.merge(data, dbn_builds[['dbn','esid_no',
                              'Project ID','NTA - Neighborhood Tabulation Area','boro']], how = 'inner', on = 'dbn')
data = data.drop(['Project ID','NTA - Neighborhood Tabulation Area'], axis = 1)
data = data.drop_duplicates()
data = pd.merge(data, se, how = 'left', on = 'dbn')
data.query('dbn == "13K282"')
sns.regplot(x="Counted Rental Units", y="Trust %", data=data)
plt.title('Trust and Housing Development')
sns.regplot(x="Counted Rental Units", y="Average ELA Proficiency", data=data)
plt.title('ELA Proficiency and Housing Development')
sns.regplot(x="Counted Rental Units", y="School Income Estimate", data=data)
plt.title('School Income and Housing Development')
def explorerChange(feature):
    m = folium.Map([40.665535, -73.969749], zoom_start=14)
    m.choropleth(
        geo_data='../input/20152016-nyc-school-zones/2015-2016_school_zones_fixed.geojson',
        data=data,
        columns=['dbn',feature],
        key_on='feature.properties.dbn',
        fill_color= 'YlOrRd',
        #threshold_scale=[0,5,25,50,100],
        legend_name=''
    )
    return m
# explorerChange('Trust %')
# nypass 7
Image('../input/passny-map-7/nypass7.png')
# explorerChange('Average ELA Proficiency')
# nypass 8
Image('../input/passny-map-8/nypass8.png')
sat = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
sat = sat.rename(columns={'DBN':'dbn',
                          'School name':'school_name', 
                          'Year of SHST':'year', 
                          'Grade level':'grade_level',
                          'Enrollment on 10/31':'enrollment',
                          'Number of students who registered for the SHSAT':'registered',
                          'Number of students who took the SHSAT':'took'})
sat = pd.merge(sat, stats, how = 'inner', on = ['dbn','year'] )
sat.head()
sns.regplot(x="asian_per", y="took", data=sat)
sns.regplot(x="black_per", y="took", data=sat)
