import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

from geopy.geocoders import Nominatim
import folium
import ast

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
pd.options.display.max_seq_items = 2000
#import data and show diminsions
data = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
data.info()
fig, ax = plt.subplots(figsize=(10,8)) 
sns.heatmap(data = data[data.columns[15:41]].corr(), cmap = 'coolwarm', ax=ax)
# Columns 15-41 can be converted to floating numbers to increase the size of our heatmap. 
# I will try to move left to right in this section to make it as straightforward as possible. 
# Choosing columns manually also helps me begin I understanding what each one means.

clean_data = data.copy()
clean_data['Community School?'] = pd.Series(np.where(clean_data['Community School?'].values == 'Yes', 1, 0),
                                  clean_data.index)
clean_data = clean_data.rename(columns = {'School Income Estimate': 'School Income Estimate ($)'})
clean_data['School Income Estimate ($)'].isnull().sum()
clean_data['School Income Estimate ($)'] = clean_data['School Income Estimate ($)']\
                                        .apply(lambda x: x if type(x) == float else x.replace('$',''))\
                                        .apply(lambda x: x if type(x) == float else x.replace(',',''))\
                                        .astype(float)
#Converting the % Amounts to decimals

percent_cols = ['Percent ELL', 'Percent Asian',
                'Percent Black', 'Percent Hispanic', 'Percent Black / Hispanic',
                'Percent White', 'Student Attendance Rate',
                'Percent of Students Chronically Absent', 'Rigorous Instruction %',
                'Collaborative Teachers %', 'Supportive Environment %', 'Effective School Leadership %',
                'Strong Family-Community Ties %', 'Trust %']
clean_data[percent_cols] = clean_data[percent_cols]\
                            .apply(lambda x: x.apply(lambda x: x if type(x) == float else x.replace('%',''))\
                            .astype(float)/100)
# Commented out due to relation to Percentage columns

#likert_columns = ['Rigorous Instruction Rating', 'Collaborative Teachers Rating',
#                 'Supportive Environment Rating', 'Effective School Leadership Rating',
#                 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']
#clean_data[likert_columns].apply(pd.Series.value_counts)

#These columns can be encoded to view the correlation as well. 
#pd.unique(clean_data[likert_columns].values.ravel())

#cat_type = pd.api.types.CategoricalDtype(categories=['Not Meeting Target','Approaching Target', 'Meeting Target', 'Exceeding Target'], 
#                            ordered=True)

#temp = pd.DataFrame()
#temp = data[likert_columns]

#for col in likert_columns:
#    temp[col] = temp[col].astype(cat_type)
    
#clean_data[likert_columns] = temp[likert_columns].apply(lambda x: x.cat.codes).replace(-1, np.nan)
clean_data[clean_data.columns[15:44]].head(2)
fig, ax = plt.subplots(figsize=(14,12)) 
sns.heatmap(data = clean_data[clean_data.columns[15:41]].corr(), cmap = 'coolwarm', ax=ax)
#Used to grab specific correlation numbers referenced in the section overview.
#clean_data[clean_data.columns[15:41]].corr()
#Import data and show dimensions

shsat = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
shsat.info()
shsat_clean = shsat[shsat['Grade level'] == 8]
shsat_locs = {
    '05M046' : (40.831629, -73.936006), '05M123' : (40.820165, -73.944486), '05M129' : (40.815000, -73.952222),
    '05M148' : (40.817322, -73.947338), '05M161' : (40.817755, -73.952468), '05M286' : (40.815478, -73.955556),
    '05M302' : (40.817458, -73.947372), '05M362' : (40.810687, -73.956061), '05M367' : (40.815478, -73.955556), 
    '05M410' : (40.815681, -73.955774), '05M469' : (40.807063, -73.938829), '05M499' : (40.824398, -73.936545),
    '05M514' : (40.819702, -73.956747), '05M670' : (40.815225, -73.944321), '84M065' : (40.810745, -73.949076),
    '84M284' : (40.812433, -73.948153), '84M336' : (40.820126, -73.956664), '84M341' : (40.808695, -73.936839),
    '84M350' : (40.814584, -73.944991), '84M384' : (40.805584, -73.935484), '84M388' : (40.815042, -73.945689),
    '84M481' : (40.805976, -73.951846), '84M709' : (40.821182, -73.940665), '84M726' : (40.819764, -73.95724) 
    }
print('2016: ' + str(shsat_clean[shsat_clean['Year of SHST'] == 2016]['School name'].unique().size))
print('2015: ' + str(shsat_clean[shsat_clean['Year of SHST'] == 2015]['School name'].unique().size))
print('2014: ' + str(shsat_clean[shsat_clean['Year of SHST'] == 2014]['School name'].unique().size))
print('2013: ' + str(shsat_clean[shsat_clean['Year of SHST'] == 2013]['School name'].unique().size))
this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')

def plotDot(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    x = point['School name'] + '<br>' + "Enrollment on 10/31: " + str(point['Enrollment on 10/31']) + '<br>' + \
    "Students who took SHSAT: " + str(point['Number of students who took the SHSAT'])
    iframe = folium.IFrame(html=x, width=500, height=90)
    popup = folium.Popup(iframe)
    folium.CircleMarker(location=[shsat_locs[point.DBN][0], shsat_locs[point.DBN][1]],
                        radius=3, weight=5, color='red', popup=popup).add_to(this_map)

#The red schools are schools contained in the SHSAT dataset. 
shsat_clean.apply(plotDot, axis = 1)

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

this_map
shsat_clean_2016 = shsat_clean[shsat_clean['Year of SHST'] == 2016].reset_index(drop=True)
schools_shsat_data = clean_data[clean_data['Location Code'].isin(shsat_clean['DBN'].unique())].reset_index(drop=True)
schools_shsat_data = schools_shsat_data.merge(shsat_clean_2016, left_on='Location Code', right_on='DBN')
# I choose not to drop these extra columns because it is a nice sanity check to make sure the merge worked correctly. 
#schools_shsat_data.drop(columns=['School name', 'DBN', 'Year of SHST', 'Grade level'])
fig, ax = plt.subplots(figsize=(14,12)) 
sns.heatmap(data = schools_shsat_data[list(schools_shsat_data.columns[15:41]) + 
                                      list(schools_shsat_data.columns[-3:])].corr(), 
            cmap = 'coolwarm', ax=ax)
#Used to grab specific correlation numbers referenced in the section overview.
#schools_shsat_data[list(schools_shsat_data.columns[15:41]) + list(schools_shsat_data.columns[-3:])].corr()
train = schools_shsat_data[list(schools_shsat_data.columns[15:41]) + list(schools_shsat_data.columns[-3:])]
train = train[train.select_dtypes(include='number').columns]
train.dropna(inplace=True)
train.info()
X = train[train.columns[:-10]]
y1 = train[train.columns[-2]]
y2 = train[train.columns[-1]]

X2 = sm.add_constant(X)
est = sm.OLS(y2, X2)
est2 = est.fit()
print(est2.summary())
highschools = pd.read_csv("../input/nyc-high-school-directory/2014-2015-doe-high-school-directory.csv")
highschools.info(verbose = False)
highschools.head(2)
#hs['school_type'].unique()
#hs[hs['school_type'].isnull()]
specialized_highschool_list = ['Bronx High School of Science', 'Brooklyn Latin School, The', 
                           'Brooklyn Technical High School', 'High School for Mathematics, Science and Engineering at City College',
                           'High School of American Studies at Lehman College', 'Queens High School for the Sciences at York College', 
                           'Staten Island Technical High School', 'Stuyvesant High School', 
                           ]

#Audition only schools = 'Fiorello H. LaGuardia High School of Music & Art and Performing Arts'

specialized_hs = highschools[highschools['school_name'].isin(specialized_highschool_list)].reset_index(drop=True)
schools_shsat_data.head(2)
this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')

def plotBlueDots(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    x = point['School Name'] + '<br>' + "Grade 8 Students: "
    iframe = folium.IFrame(html=x, width=300, height=40)
    popup = folium.Popup(iframe)
    folium.CircleMarker(location=[point.Latitude, point.Longitude],
                        radius=2, weight=5, popup=popup,
                       color = 'blue').add_to(this_map)
    
def plotRedDots(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    x = point['School name'] + '<br>' + "Enrollment on 10/31: " + str(point['Enrollment on 10/31']) + '<br>' + \
    "Students who took SHSAT: " + str(point['Number of students who took the SHSAT'])
    iframe = folium.IFrame(html=x, width=500, height=90)
    popup = folium.Popup(iframe)
    folium.CircleMarker(location=[point.Latitude, point.Longitude],
                        radius=2, weight=5, popup=popup, 
                       color = 'red').add_to(this_map)
    
def plotMarker(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.Marker(location=[float(ast.literal_eval(point['Location 1'])['latitude']), 
                          float(ast.literal_eval(point['Location 1'])['longitude'])],
                          popup=point['school_name']).add_to(this_map)

    
    
    
clean_data.apply(plotBlueDots, axis = 1)
#The red schools are schools contained in the SHSAT dataset. 
schools_shsat_data.apply(plotRedDots, axis = 1)
specialized_hs.apply(plotMarker, axis = 1)

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

this_map
district_demographics = pd.read_csv('../input/nyc-school-district-breakdowns/school-district-breakdowns.csv')
district_demographics['COUNT PARTICIPANTS'].sum()
students = list([])
temp_cols = ['COUNT BLACK NON HISPANIC', 'COUNT HISPANIC LATINO', 
        'COUNT WHITE NON HISPANIC', 'COUNT ASIAN NON HISPANIC']
for col in temp_cols:
    students.append(district_demographics[col].sum())

remainder = district_demographics['COUNT PARTICIPANTS'].sum() - sum(students)
students.append(remainder)
labels = ['Black', 'Hispanic', 'White', 'Asian', 'Other/Unspecified']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'grey']
tested = np.array([5770, 6514, 5125, 8732, 2192])
offered = np.array([207, 319, 1342, 2619, 580])
pass_rates = offered/tested

title_size = 25
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)

plt.figure(0, figsize = (20, 20))
plt.subplot(2,2,1)
plt.title('2018 SHSAT Testing', 
          fontdict = {'fontsize': title_size})
plt.pie(tested, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle = 90)
plt.axis('square')
 
#plt.figure(1, figsize = (10, 10))
plt.subplot(2,2,2)
plt.title('2018 Specialized Schools Acceptances',
         fontdict = {'fontsize': title_size})
plt.pie(offered, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle = 90)
plt.axis('square')

#plt.figure(1, figsize = (10, 10))
plt.subplot(2,2,3)
plt.title('2018 Specialized Schools Pass Rates',
         fontdict = {'fontsize': title_size})
plt.barh(width = pass_rates, y=labels, align='center',
        color='green', ecolor='black')
plt.axis('tight')

#plt.figure(2, figsize = (10, 10))
plt.subplot(2,2,4)
plt.title('2018 Student Survey',
         fontdict = {'fontsize': title_size})
plt.pie(students, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle = 90)

plt.axis('square')
plt.show()
this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')

def plotBlueDots(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.CircleMarker(location=[point.Latitude, point.Longitude],
                        radius=2, weight=5,
                       color = 'blue').add_to(this_map)
    
def plotMarker(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.Marker(location=[float(ast.literal_eval(point['Location 1'])['latitude']), 
                          float(ast.literal_eval(point['Location 1'])['longitude'])],
                          popup=point['school_name']).add_to(this_map)

clean_data[(clean_data['Grade 8 Math - All Students Tested'] > 0) &
          (clean_data['Percent Black / Hispanic'] > .60)].apply(plotBlueDots, axis = 1)



#use df.apply(,axis=1) to "iterate" through every row in your dataframe
specialized_hs.apply(plotMarker, axis = 1)


#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

this_map
high_qualifiers = ['Economic Need Index', 'Percent Black / Hispanic', 'Student Attendance Rate', 
                   'Average ELA Proficiency', 'Average Math Proficiency',
                   'Grade 8 Math - All Students Tested', 'Grade 8 ELA - All Students Tested']

low_qualifiers = ['School Income Estimate ($)', 'Effective School Leadership %', 
                  'Collaborative Teachers %', 'Strong Family-Community Ties %', 'Trust %']

target_cols = clean_data[['School Name', 'Latitude', 'Longitude', 'Grade High'] + high_qualifiers + low_qualifiers]
target_cols = target_cols[target_cols['Grade 8 Math - All Students Tested'] > 0]
#target_cols = target_cols[target_cols['Grade High'] == '08']
#plt.figure(0, figsize = (20, 20))

target_cols.hist(figsize=(15,25),layout=(5,3))
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

target_cols.select_dtypes(include=numerics).apply(np.log).drop(['Latitude', 'Longitude'], axis=1).describe()
target_cols.select_dtypes(include=numerics).apply(np.log).drop(['Latitude', 'Longitude'], axis=1).describe()
target_cols.select_dtypes(include=numerics).apply(np.log).drop(['Latitude', 'Longitude'], axis=1).info()
target_cols.quantile(q=0.75, axis='rows', numeric_only=True)
target_cols.quantile(q=0.25, axis='rows', numeric_only=True)
high_qualifiers = ['Economic Need Index', 'Percent Black / Hispanic', 'Student Attendance Rate', 
                   'Average ELA Proficiency', 'Average Math Proficiency',
                   'Grade 8 Math - All Students Tested', 'Grade 8 ELA - All Students Tested']

low_qualifiers = ['School Income Estimate ($)', 'Effective School Leadership %', 
                  'Collaborative Teachers %', 'Strong Family-Community Ties %', 'Trust %']

def filter_df_col(df, col, greater_flag):
    if greater_flag == True:
        df = df.loc[df[col] > target_cols.quantile(q=0.4, axis='rows', numeric_only=True)[col]]
    else:
        df = df.loc[df[col] < target_cols.quantile(q=0.6, axis='rows', numeric_only=True)[col]]
    return df

filtered_targets = target_cols.copy()

for col in high_qualifiers:
    filtered_targets = filter_df_col(filtered_targets, col, True)
    
for col in low_qualifiers:
    filtered_targets = filter_df_col(filtered_targets, col, False)
    
filtered_targets.info()
high_qualifiers = ['Economic Need Index', ]
low_qualifiers = ['School Income Estimate ($)']

high_percentile = .5
low_percentile = .2

def filter_df_col(df, col, greater_flag):
    if greater_flag == True:
        df = df.loc[df[col] > target_cols.quantile(q=high_percentile, axis='rows', numeric_only=True)[col]]
    else:
        df = df.loc[df[col] < target_cols.quantile(q=low_percentile, axis='rows', numeric_only=True)[col]]
    return df

filtered_targets1 = target_cols.copy()

for col in high_qualifiers:
    filtered_targets1 = filter_df_col(filtered_targets1, col, True)
    
for col in low_qualifiers:
    filtered_targets1 = filter_df_col(filtered_targets1, col, False)
    
filtered_targets1.info()
color_threshold = .7

this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')
    
def plotTargets(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    percent_black_hispanic = point['Percent Black / Hispanic']
    x = point['School Name'] + '<br>' + \
        "Highest Grade: " + str(point['Grade High']) + '<br>' + \
        "Grade 8 Students: " + str(point['Grade 8 Math - All Students Tested']) + '<br>' + \
        "Percent Black / Hispanic: " + str(percent_black_hispanic) + '<br>'
    iframe = folium.IFrame(html=x, width=400, height=90)
    popup = folium.Popup(iframe)
    folium.Circle(location=[point.Latitude, point.Longitude],
                  radius=400, fill = True, popup=popup, 
                  color = 'red' if percent_black_hispanic > color_threshold 
                                  else 'blue').add_to(this_map)

def plotMarker(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.Marker(location=[float(ast.literal_eval(point['Location 1'])['latitude']), 
                          float(ast.literal_eval(point['Location 1'])['longitude'])],
                          popup=point['school_name']).add_to(this_map)

filtered_targets1.apply(plotTargets, axis=1)
specialized_hs.apply(plotMarker, axis = 1)

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

this_map
high_qualifiers = ['Grade 8 Math - All Students Tested']

high_percentile = .9
low_percentile = 0

def filter_df_col(df, col, greater_flag):
    if greater_flag == True:
        df = df.loc[df[col] > target_cols.quantile(q=high_percentile, axis='rows', numeric_only=True)[col]]
    else:
        df = df.loc[df[col] < target_cols.quantile(q=low_percentile, axis='rows', numeric_only=True)[col]]
    return df

filtered_targets2 = target_cols.copy()

for col in high_qualifiers:
    filtered_targets2 = filter_df_col(filtered_targets2, col, True)

    
filtered_targets2.info()
color_threshold = .7

this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')
    
def plotTargets(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    percent_black_hispanic = point['Percent Black / Hispanic']
    x = point['School Name'] + '<br>' + \
        "Highest Grade: " + str(point['Grade High']) + '<br>' + \
        "Grade 8 Students: " + str(point['Grade 8 Math - All Students Tested']) + '<br>' + \
        "Percent Black / Hispanic: " + str(percent_black_hispanic) + '<br>'
    iframe = folium.IFrame(html=x, width=400, height=90)
    popup = folium.Popup(iframe)
    folium.Circle(location=[point.Latitude, point.Longitude],
                  radius=400, fill = True, popup=popup, 
                  color = 'red' if percent_black_hispanic > color_threshold 
                                  else 'blue').add_to(this_map)

def plotMarker(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.Marker(location=[float(ast.literal_eval(point['Location 1'])['latitude']), 
                          float(ast.literal_eval(point['Location 1'])['longitude'])],
                          popup=point['school_name']).add_to(this_map)

filtered_targets2.apply(plotTargets, axis=1)
specialized_hs.apply(plotMarker, axis = 1)

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())


this_map
filtered_targets3 = target_cols.copy()
color_threshold = .7

this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')
    
def plotTargets(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    percent_black_hispanic = point['Percent Black / Hispanic']
    x = point['School Name'] + '<br>' + \
        "Highest Grade: " + str(point['Grade High']) + '<br>' + \
        "Grade 8 Students: " + str(point['Grade 8 Math - All Students Tested']) + '<br>' + \
        "Percent Black / Hispanic: " + str(percent_black_hispanic) + '<br>'
    iframe = folium.IFrame(html=x, width=400, height=90)
    popup = folium.Popup(iframe)
    folium.Circle(location=[point.Latitude, point.Longitude],
                  radius=400, fill = True, popup=popup, 
                  color = 'red' if percent_black_hispanic > color_threshold 
                                  else 'blue').add_to(this_map)

def plotMarker(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    folium.Marker(location=[float(ast.literal_eval(point['Location 1'])['latitude']), 
                          float(ast.literal_eval(point['Location 1'])['longitude'])],
                          popup=point['school_name']).add_to(this_map)

filtered_targets3.apply(plotTargets, axis = 1)
specialized_hs.apply(plotMarker, axis = 1)
this_map.fit_bounds(this_map.get_bounds())


this_map
schools1 = ['P.S. 161 PEDRO ALBIZU CAMPOS']
schools2 = ['THE NEW SCHOOL FOR LEADERSHIP AND JOURNALISM']

schools3 = ['INTERNATIONAL SCHOOL FOR LIBERAL ARTS',
           'P.S./M.S. 280 MOSHOLU PARKWAY', 'J.H.S. 080 THE MOSHOLU PARKWAY', 
            'P.S./M.S. 20 P.O.GEORGE J. WERDANN, III', 'P.S. 095 SHEILA MENCHER'
           ]

specialized_target_schools = ['High School for Mathematics, Science and Engineering at City College', 
                              'High School of American Studies at Lehman College',
                              'Bronx High School of Science']


this_map = folium.Map(prefer_canvas=True, tiles='Stamen Toner')
    
def plotTargets(point, color):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    percent_black_hispanic = point['Percent Black / Hispanic']
    x = point['School Name'] + '<br>' + \
        "Highest Grade: " + str(point['Grade High']) + '<br>' + \
        "Grade 8 Students: " + str(point['Grade 8 Math - All Students Tested']) + '<br>' + \
        "Percent Black / Hispanic: " + str(percent_black_hispanic) + '<br>'
    iframe = folium.IFrame(html=x, width=500, height=90)
    popup = folium.Popup(iframe)
    folium.Circle(location=[point.Latitude, point.Longitude],
                  radius=400, fill = True, popup=popup, 
                  color = color).add_to(this_map)


#use df.apply(,axis=1) to "iterate" through every row in your dataframe
specialized_hs[specialized_hs['school_name'].isin(specialized_target_schools)].apply(plotMarker, axis = 1)
target_cols[target_cols['School Name'].isin(schools1)].apply(lambda x: plotTargets(x, 'blue'), axis=1)
target_cols[target_cols['School Name'].isin(schools2)].apply(lambda x: plotTargets(x, 'red'), axis=1)
target_cols[target_cols['School Name'].isin(schools3)].apply(lambda x: plotTargets(x, 'green'), axis=1)


#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())


this_map
