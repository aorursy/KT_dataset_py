%matplotlib inline

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import folium
import sklearn
import seaborn as sns
from IPython.core.display import display, HTML
import ipywidgets as widgets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import clear_output
import itertools
import warnings
import base64

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)
# Function to convert Percent fields to integers
def percent_to_int(df_in):
    for col in df_in.columns.values:
        if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
            df_in[col] = df_in[col].astype(np.object).str.replace('%', '').astype(float)
    return df_in
# Dataset 1: School Information
df_schools_raw = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')

# Preprocessing
# If school just has 0k, might as well drop
df_schools_raw = df_schools_raw[df_schools_raw['Grade High'] != '0K'] 
df_schools_raw = percent_to_int(df_schools_raw)

# Convert dollars to int 
df_schools_raw['School Income Estimate'] = df_schools_raw['School Income Estimate'].astype(np.object).str.replace('$', '').str.replace(',', '').str.replace('.', '').astype(float)
df_schools_raw.replace(np.NaN,0, inplace=True)

df_schools = df_schools_raw.copy()

# Dataset 2: SHSAT offers information
df_shsat_offers = pd.read_csv('../input/2017-2018-shsat-admissions-test-offers-by-schools/2017-2018 SHSAT Admissions Test Offers By Sending School.csv')

# Preprocessing - convert to int, remove NaNs
df_shsat_offers = percent_to_int(df_shsat_offers)
df_shsat_offers.replace(np.NaN,0, inplace=True)

print("After the pre-processing process, ")
print("We have school information data for {} schools".format(df_schools_raw.shape[0]))
print("We have admission information for {} schools".format(df_shsat_offers.shape[0]))

# Dataset 3: Information regarding PASSNYC resource centers
df_passnyc_centers = pd.read_csv('../input/passnyc-resource-centers/passnyc-resource-centers.csv')
# Drop list for merge - fields which are common in both
df_shsat_offers.drop(['Borough','School Category','School Name'], axis=1, inplace=True)

df_merged = pd.merge(df_schools_raw, df_shsat_offers, how='outer', left_on='Location Code' ,right_on='School DBN')
df_merged.dropna(inplace=True)
print("We have {} schools in the merged dataset out of the {} schools in the admissions dataset, implying that we do not have school information for {} schools."\
      .format(df_merged.shape[0], df_shsat_offers.shape[0], df_shsat_offers.shape[0] - df_merged.shape[0]))

# For SHSAT, if school has just grades below 5, no point in using those 
df_incorrect = df_merged[df_merged['Grade High'].astype(int) <= 5]
print("We have %d schools with Grade High field 5 and SHSAT results"%(df_incorrect.shape[0]))

# 4 types aids available in PASSNYC resource centers 
employment_centers = df_passnyc_centers[df_passnyc_centers['Crime']==1]
test_prep_centers = df_passnyc_centers[df_passnyc_centers['Test Prep']==1]
after_school_centers = df_passnyc_centers[df_passnyc_centers['After School Program']==1]
economic_help_centers = df_passnyc_centers[df_passnyc_centers['Economic Help']==1]

# print(len(set(df_schools_raw[df_schools_raw['Grade High'].astype(int)>7]['Location Code'].values)))
# print(len(set(df['School DBN'].values) - set(df_schools_raw[df_schools_raw['Grade High'].astype(int)>7]['Location Code'].values))) 
# Create a trace
layout = go.Layout(
        title='Number of Students taking test v/s Percent Black Hispanic',
        xaxis=dict(
            title='Percentage of Black/Hispanic students',
            titlefont=dict(size=18),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(size=10)
        ),  
        yaxis=dict(
        title='Number of Students taking test',
        titlefont=dict(size=18)
        )         
   )

trace = go.Scatter(
    x = df_merged['Percentage of Black/Hispanic students'],
    y =  df_merged['Number of students who took test'],
    mode = 'markers'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
iplot(fig)
def grade_minority_total_4s_append(df, grade):
    out_field = ('Grade %d Total Minority 4s')%(grade)
    num1 = ('Grade %d ELA 4s - Black or African American')%(grade)
    num2 = ('Grade %d ELA 4s - Hispanic or Latino')%(grade)
    num3 = ('Grade %d Math 4s - Black or African American')%(grade)
    num4 = ('Grade %d Math 4s - Hispanic or Latino')%(grade)
    df[out_field] = df[num1] + df[num2] + df[num3] + df[num4]

    return df
def grade_minority_percent_4s_append(df, grade, subject):
    out_field = ('Grade %d %s Minority 4s')%(grade, subject)
    num1 = ('Grade %d %s 4s - Black or African American')%(grade, subject)
    num2 = ('Grade %d %s 4s - Hispanic or Latino')%(grade, subject)
    den = ('Grade %d %s 4s - All Students')%(grade, subject)
    
    df[out_field] = (df[num1] + df[num2])/(df[den])*100
    df.fillna(0, inplace = True)
    return df
grades = range(5,9)
subjects = ['ELA','Math']

for grade in grades:
    for subject in subjects:
        df_merged = grade_minority_percent_4s_append(df_merged, grade, subject) 
    df_merged = grade_minority_total_4s_append(df_merged, grade) 

df_merged['5_6_minority_added'] = df_merged['Grade 5 Total Minority 4s'] + df_merged['Grade 6 Total Minority 4s'] 
df_merged['7_8_minority_added'] = df_merged['Grade 7 Total Minority 4s'] + df_merged['Grade 8 Total Minority 4s']
def download_link(df, filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    title = "Download CSV file"
    html = '<button type="button" style="font-size: larger;  background-color: #FFFFFF; border: 0pt;"><a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a></button>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
registered_students_thresh = 50
min_minority_students_thresh = 80
# Keep the schools where number of registrations are low and the percentage of Blank and Hispanic students is high
condition_registration = df_merged['Number of students who took test'] < registered_students_thresh
condition_minority = df_merged['Percentage of Black/Hispanic students'] > min_minority_students_thresh
condition = np.logical_and(condition_registration, condition_minority)

df_low_reg_high_minority = df_merged[condition]

# display(download_link(df_low_reg_high_minority, "schools_low_registrations_high_minority.csv"))
df_low_reg_high_minority.head()
print("There are {} schools with  Number of students who took test is less than {} and  percentage black / hispanic students are greater than {}%".format(
    df_low_reg_high_minority.shape[0], registered_students_thresh, min_minority_students_thresh))
percent_minority_4s_thresh = 70
total_4s_thresh = 7
def find_minority_high_score_schools(df, percent_minority_4s_thresh, total_4s_thresh, grade='7'):
    df_high_minority = df[condition_minority]
    condition_high_scoring_math = np.logical_and(df_high_minority["Grade {} Math Minority 4s".format(grade)] > percent_minority_4s_thresh,
                                        df_high_minority["Grade {} Math 4s - All Students".format(grade)] > total_4s_thresh)
    condition_high_scoring_ELA = np.logical_and(df_high_minority["Grade {} ELA Minority 4s".format(grade)] > percent_minority_4s_thresh,
                                                   df_high_minority["Grade {} ELA 4s - All Students".format(grade)] > total_4s_thresh)
    condition = np.logical_or(condition_high_scoring_math, condition_high_scoring_ELA)
    df_minority_highscore = df_high_minority[condition]
    return df_minority_highscore

for i in range(5,9):
    globals()['df_minority_highscore_grade'+str(i)] = find_minority_high_score_schools(df_merged, percent_minority_4s_thresh, total_4s_thresh, str(i)).sort_values(by='Grade ' + str(i) +' Total Minority 4s', ascending = False)
    globals()['relevant_fields_grade'+str(i)] = \
    ['School Name',  'Grade {} Math Minority 4s'.format(i), 'Grade {} Math 4s - All Students'.format(i), 
     'Grade {} ELA Minority 4s'.format(i), 'Grade {} ELA 4s - All Students'.format(i)] 

print("Below we show the schools with maximum number of students scoring 4s in grades 5 through 8: ")
display(HTML(df_minority_highscore_grade5[relevant_fields_grade5].head().to_html()))
display(HTML(df_minority_highscore_grade6[relevant_fields_grade6].head().to_html()))
display(HTML(df_minority_highscore_grade7[relevant_fields_grade7].head().to_html()))
display(HTML(df_minority_highscore_grade8[relevant_fields_grade8].head().to_html()))
# Function to find the nearest PASSNYC center of required type to the schools in need

def find_nearest_test_center(row, center_type):
    school_latitude = row['Latitude']
    school_longitude = row['Longitude']
    
    passnyc_centers = df_passnyc_centers[df_passnyc_centers[center_type]==1].copy(deep=True)
    test_center_lat = passnyc_centers['Lat'].astype(float)
    test_center_long = passnyc_centers['Long'].astype(float)
    test_prep_centers_nearest = passnyc_centers.copy(deep=True)
    
    test_prep_centers_nearest["Distance to Test Center"] = np.sqrt(np.square(test_center_lat-school_latitude)+np.square(test_center_long-school_longitude))
    nearest_test_center_argmin = test_prep_centers_nearest["Distance to Test Center"].argmin()
    nearest_test_center_series = (test_prep_centers_nearest.loc[nearest_test_center_argmin,["Resource Center Name","Address","Distance to Test Center","Lat","Long"]])
    nearest_test_center_series.rename({"Resource Center Name":"Nearest Test Center Name","Address":"Nearest Test Center Address","Distance to Test Center":"Nearest Test Center Distance","Lat":"Nearest Test Center Latitude","Long":"Nearest Test Center Longitude"}, inplace=True)
    return nearest_test_center_series
colors = [
    'red',
    'blue',
    'gray',
    'darkred',
    'lightred',
    'orange',
    'beige',
    'green',
    'darkgreen',
    'lightgreen',
    'darkblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black', 'darkgray','darkpink','lightpink','lightpurple','lightred'
]

# Function to plot schools and the PASSNYC help centers
def plot_school_test_centers_map(grouped_df):
    school_map = folium.Map([test_prep_centers['Lat'].mean(), test_prep_centers['Long'].mean()], 
                        zoom_start=10.50,
                        tiles='Stamen Terrain')
    for i, group in enumerate(grouped_df):
        group = group[1]
        for index in group.index:
            row = group.loc[index]
            folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color=colors[i], icon='university', prefix="fa")).add_to(school_map)

        for index in group.index:
            row = group.loc[index]
            folium.Marker([row['Nearest Test Center Latitude'], row['Nearest Test Center Longitude']], icon=folium.Icon(color=colors[i],  icon ='info-sign')).add_to(school_map)
    return school_map
# Merged Grade 7 and 8 for analysis
df_minority_highscore_grade7_grade8 = pd.concat([df_minority_highscore_grade7, df_minority_highscore_grade8])
print("The number of schools where either grade 7 or grade 8 minority students are performing well is "+ str(len(df_minority_highscore_grade7_grade8)))

test_prep_nearest_test_centers = (df_minority_highscore_grade7_grade8[["Latitude","Longitude"]].apply(find_nearest_test_center, center_type="Test Prep", axis=1))
df_minority_highscore_grade7_grade8_test_centers_merged = pd.merge(df_minority_highscore_grade7_grade8, test_prep_nearest_test_centers, left_index=True, right_index=True)
required_fields = ['School Name', 'Nearest Test Center Name', 'Number of students who took test'] + relevant_fields_grade7 + relevant_fields_grade8 + ['7_8_minority_added', 'Percent of Students Chronically Absent']

df_minority_highscore_grade7_grade8_test_centers_merged.drop_duplicates(inplace=True)
df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields = df_minority_highscore_grade7_grade8_test_centers_merged[required_fields].sort_values(['7_8_minority_added'], ascending = False)
display(download_link(df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields, "link_schools_after_school_centers.csv"))
print ("List of the schools with high minority students performing well in Grade 7 and Grade 8")
df_minority_highscore_grade7_grade8_test_centers_merged_relevant_fields.head()
nearest_after_school_centers_df_groupby = df_minority_highscore_grade7_grade8_test_centers_merged.groupby(by='Nearest Test Center Address', as_index=False)

# AFTER SCHOOL CENTERS - GRADE 7,8
print("Scatter map of schools with high minority and good performance of Grade-7 and 8 Students")
print ("")
print ("The university symbol on the marker represents a school belonging to a cluster (defined by a particular color) and the marker with the same color having 'i' symbol represents passnyc help center")
school_map = plot_school_test_centers_map(nearest_after_school_centers_df_groupby)
school_map
# Merged Grade 5 and 6 for analysis
required_fields = ['School Name', 'Nearest Test Center Name', 'Number of students who took test'] + relevant_fields_grade5 + relevant_fields_grade6 + ['5_6_minority_added', 'Percent of Students Chronically Absent']
df_minority_highscore_grade5_grade6 = pd.concat([df_minority_highscore_grade5, df_minority_highscore_grade6])
print("The number of schools where both grade 5 and grade 6 minority students are performing well is "+ str(len(df_minority_highscore_grade5_grade6)))

after_school_nearest_test_centers = (df_minority_highscore_grade5_grade6[["Latitude","Longitude"]].apply(find_nearest_test_center, center_type="After School Program", axis=1))
df_minority_highscore_grade5_grade6_test_centers_merged = pd.merge(df_minority_highscore_grade5_grade6, after_school_nearest_test_centers, left_index=True, right_index=True)

df_minority_highscore_grade5_grade6_test_centers_merged.drop_duplicates(inplace=True)
df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields = df_minority_highscore_grade5_grade6_test_centers_merged[required_fields].sort_values(['Grade 5 Math 4s - All Students', 'Grade 5 ELA 4s - All Students'], ascending = False)
display(download_link(df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields, "link_schools_after_school_centers.csv"))
print ("List of the schools with high minority students performing well in Grade 5 and Grade 6")
df_minority_highscore_grade5_grade6_test_centers_merged_relevant_fields.head()
nearest_after_school_centers_df_groupby = df_minority_highscore_grade5_grade6_test_centers_merged.groupby(by='Nearest Test Center Address', as_index=False)

# AFTER SCHOOL CENTERS - GRADE 5,6
print("Scatter map of schools with high minority and good performance of Grade-5 and 6 Students")
print ("")
print ("The university symbol on the marker represents a school belonging to a cluster (defined by a particular color) and the marker with the same color having 'i' symbol represents passnyc help center")
school_map = plot_school_test_centers_map(nearest_after_school_centers_df_groupby)
school_map
# Adding a field called 'Selection Ratio' which is the ratio of students who received an offer to the number of students who took SHSAT
df_merged['Selection_ratio'] = df_merged['Number of students who received offer']/df_merged['Number of students who took test']
# Filter out schools based on threshold of number of students who received an offer
df_high_offers = df_merged[df_merged['Number of students who received offer']>10]
# Fit a regression on Percentage of students who received a 4 and percent of students who received an offer
num_students_4s = df_high_offers['Grade 7 Math 4s - All Students']+df_high_offers['Grade 7 ELA 4s - All Students']+df_high_offers['Grade 6 Math 4s - All Students']+df_high_offers['Grade 6 ELA 4s - All Students']
num_studetns_tested = df_high_offers['Grade 7 Math - All Students Tested']+df_high_offers['Grade 7 ELA - All Students Tested']+df_high_offers['Grade 6 Math - All Students Tested']+df_high_offers['Grade 6 ELA - All Students Tested']
x = (num_students_4s/num_studetns_tested)*100
y =  df_high_offers['Selection_ratio']*100

sns.set(rc={'figure.figsize':(12,10)})
regression = sns.regplot(x, y)
regression = regression.set(xlabel='Percentage of students scoring 4s', ylabel='SHSAT Selection Percentage')
model  = sklearn.linear_model.LinearRegression()
model.fit(x.values.reshape(-1, 1),y)
print ("The coefficient of the regression model is %f."%model.coef_)
layout = go.Layout(
        title='Histogram of Economic Need Index (ENI)',
        xaxis=dict(
            title='Economic Need Index (ENI)',
            titlefont=dict(size=18),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(size=10)
        ),  
        yaxis=dict(
        title='Number of Schools',
        titlefont=dict(size=18)
        )         
   )

data = [go.Histogram(x = df_low_reg_high_minority['Economic Need Index'].values)]
fig = go.Figure(data=data, layout=layout);
iplot(fig);
layout = go.Layout(
        autosize=False,
        width=800,
        height=600,
        title='ENI Heatmap over the geographic layout',
        xaxis=dict(
            title='Longitude',
            titlefont=dict(size=18)
        ),  
        yaxis=dict(
        title='Latitude',
        titlefont=dict(size=18)
        )         
   )

trace1 = go.Scatter(
    x = df_low_reg_high_minority['Longitude'],
    y = df_low_reg_high_minority['Latitude'],
    mode='markers',
    marker=dict(
        symbol='circle',
        sizemode='area',
        sizeref = 2.*max(df_low_reg_high_minority['School Income Estimate'])/(100**1.75),
        size = df_low_reg_high_minority['School Income Estimate'].values,
        color = df_low_reg_high_minority['Economic Need Index'].values, #set color equal to a variable
        colorscale='Viridis',
        showscale=True
        
    )
)
data = [trace1]

fig = go.Figure(data=data, layout=layout);
iplot(fig);
#Divide schools based on Economic Need Index
df_schools_high_eni = df_low_reg_high_minority[df_low_reg_high_minority['Economic Need Index'] > df_low_reg_high_minority['Economic Need Index'].quantile(0.75)]
df_schools_low_eni = df_low_reg_high_minority[df_low_reg_high_minority['Economic Need Index'] < df_low_reg_high_minority['Economic Need Index'].quantile(0.25)]
df_schools_high_eni_relevant_fields = df_schools_high_eni[['School Name', 'Economic Need Index', 'Percent Black / Hispanic']].sort_values('Economic Need Index', ascending = False)
display(download_link(df_schools_high_eni_relevant_fields, 'schools_high_eni.csv'))
print ("There are %d schools with high ENI (where students' economic need is high)"%df_schools_high_eni_relevant_fields.shape[0])
df_schools_high_eni_relevant_fields.head()
def display_barplot(comparison_dict, axis = 0):
        
    list_traces = []
    for _trace in comparison_dict.keys():
        _data_trace = comparison_dict[_trace]
        _data_trace_name = _data_trace['name']
        del(_data_trace['name'])
    
        _trace_features = []
        _names = []
#         for ix in _data_trace.keys():
#             _trace_features.append(_data_trace[ix])
        _features = _data_trace['features']

        for i, cols in enumerate(_features):
            _names.append(_features.index[i])
            _trace_features.append(cols)
        
        if not axis:
            _trace = go.Bar(
                y = _names,
                x = _trace_features,
                name = _data_trace_name,
                orientation = 'h'
            )
        else:
            _trace = go.Bar(
                x = _names,
                y = _trace_features,
                name = _data_trace_name
            )
        list_traces.append(_trace)

    #print (list_traces)
    data = list_traces
    layout = go.Layout(
        height = 500,
        width = 800,
        barmode='group',
#         yaxis=dict(
#             tickangle=270
#         ),
#         xaxis = dict(
#             tickangle = 350
#         )
    )
    
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='grouped-bar')
print ("Average ELA / Math performance for the high ENI schools")
print (str(df_schools_high_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()))
print ("")
print ("Average ELA / Math performance for the low ENI schools")
print (str(df_schools_low_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()))

comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_eni[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'High ENI schools'
    },
    
    'trace2':
    {
        'features': df_schools_low_eni[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Low ENI schools'
    }
}


display_barplot(comparison_dict, axis=1)
print ("Click the link below to get details of the economic help centers")
display(download_link(economic_help_centers, 'economic_help_centers_details.csv'))
features_list = ['Rigorous Instruction %',
'Collaborative Teachers %',
'Supportive Environment %',
'Effective School Leadership %',
'Strong Family-Community Ties %',
'Trust %']

df_schools[['School Name'] + features_list ].head()
# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
# for i in range(2):
#     for j in range(3):
#         ax[i, j].set_title(features_list[i*3 + j])
#         sns.distplot(a=df_schools[features_list[i*3 + j]].dropna().values, kde_kws={"color": "red"}, color='darkblue', ax=ax[i, j])

# # fig.tight_layout()
# temp = fig.suptitle('School Performance features', fontsize=15)
df_schools[features_list].corr()
corr = df_schools[features_list].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='YlGnBu')
temp = plt.xticks(rotation=75, fontsize=8) 
temp = plt.yticks(fontsize=8) 
correlated_features_list = ["Effective School Leadership %","Collaborative Teachers %","Trust %"]
corr_features_values = df_schools[correlated_features_list].values

pca = PCA(n_components=1)
combined_feature_value = pca.fit_transform(corr_features_values)
df_schools['PCA Combined Feature'] = combined_feature_value
#df_schools[correlated_features_list + ['PCA Combined Feature']].corr()
scaler = sklearn.preprocessing.MinMaxScaler()
scale_factor = 2*(df_schools['PCA Combined Feature'].corr(df_schools["Effective School Leadership %"])>0) -1 
df_schools['PCA Combined Feature'] =  scaler.fit_transform(scale_factor * df_schools['PCA Combined Feature'].values.reshape(-1,1))*100

print ("The correlation between the three correlated features and their PCA is shown below:")
df_schools[correlated_features_list + ['PCA Combined Feature']].corr()
weights = [ 1, 0.8, 0.7, 0.5]
features = ['Supportive Environment %','Rigorous Instruction %','PCA Combined Feature',
            'Strong Family-Community Ties %']


df_schools['SPI'] = df_schools[features].dot(weights)


print ("A glimpse of the School Performance Index (SPI) :")
df_schools[features+['SPI']].head(5)
df_low_spi_schools = df_schools[df_schools['SPI'] < df_schools['SPI'].quantile(.25)]
df_high_spi_schools = df_schools[df_schools['SPI'] > df_schools['SPI'].quantile(.25)]
print ("Average ELA / Math performance for the high SPI schools")
print(df_high_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean())

print ("Average ELA / Math performance for the low SPI schools")
print(df_low_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean())

comparison_dict = {
    'trace1':
    {
        'features': df_high_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'High SPI schools'
    },
    
    'trace2':
    {
        'features': df_low_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Low SPI schools'
    }
}


display_barplot(comparison_dict, axis=1)
df_schools_clustering = df_schools.copy()
df_schools_clustering = df_schools_clustering.dropna(subset=['Longitude', 'Latitude'])
df_schools_clustering = df_schools_clustering[df_schools_clustering['SPI'] < df_schools_clustering['SPI'].quantile(0.15)]
df_schools_clustering = df_schools_clustering[df_schools_clustering['Economic Need Index'] > 0.8]
model = KMeans(n_clusters=7)
model.fit(df_schools_clustering[['Longitude', 'Latitude']].values)
model.cluster_centers_
color = 'blue'
school_map = folium.Map([model.cluster_centers_[:, 1].mean(), model.cluster_centers_[:, 0].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain') 

for index in df_schools_clustering.index:
    row = df_schools_clustering.loc[index]
    popup_text = "Economic Need Index : " + str(round(row['Economic Need Index'], 3)) + ' , SPI : ' + str(round(row['SPI'], 3))
    folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text, icon=folium.Icon(color='blue', icon="university", prefix="fa")).add_to(school_map)
    
for row in model.cluster_centers_:
    folium.Marker([row[1], row[0]], icon=folium.Icon(color='red')).add_to(school_map)

print ("Scatter plot of schools with high ENI, low SPI with recommended cluster(help) center")
school_map
school_map = folium.Map([model.cluster_centers_[:, 1].mean(), model.cluster_centers_[:, 0].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain') 
for row in model.cluster_centers_:
    folium.Marker([row[1], row[0]], icon=folium.Icon(color='red')).add_to(school_map)
    
for index in df_passnyc_centers.index:
        row = df_passnyc_centers.loc[index]
        folium.Marker([row['Lat'], row['Long']], icon=folium.Icon(color=colors[i],  icon ='info-sign')).add_to(school_map)

print ("Scatter plot of PASSNYC help centers and recommended help centers")
school_map
field_name = "SCHOOLWIDE PUPIL-TEACHER RATIO"
df_school_detail = pd.read_csv('../input/ny-2010-2011-class-size-school-level-detail/2010-2011-class-size-school-level-detail.csv')
df_school_detail["CSD"] =  df_school_detail['CSD'].astype('str').astype(np.object_).str.zfill(2)
df_school_detail["DBN_manual"] = df_school_detail["CSD"] + df_school_detail["SCHOOL CODE"] 
df_school_detail.dropna(subset=[field_name], inplace=True)
merged_str_df = pd.merge(df_school_detail, df_schools, how='inner', left_on=['DBN_manual'], right_on=['Location Code'])
data = [go.Histogram(x=merged_str_df['SCHOOLWIDE PUPIL-TEACHER RATIO'])]
iplot(data,filename='Number of Students per teacher')
higher_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)>18]
lower_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)<12.5]
feature_columns1 = ['Average ELA Proficiency','Average Math Proficiency', 'Economic Need Index']
feature_columns2 = ['Collaborative Teachers %', 'SPI','Percent Black / Hispanic']

print ("Average statistics for the schools with high STR")
print (higher_ratio_str_df[feature_columns1 + feature_columns2].mean())
print ("")
print ("Average statistics for the schools with low STR")
print (lower_ratio_str_df[feature_columns1 + feature_columns2].mean())
# Dataset 4: Information regarding criminal activities in school
df_school_crimes = pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv')
df_school_crimes = df_school_crimes[df_school_crimes["School Year"]=="2015-16"]
crimes_col = ['Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N']
crimes = df_school_crimes.groupby(['DBN'], as_index=False)[crimes_col].sum()

merged_safety_df = pd.merge(crimes[crimes_col + ['DBN']], df_merged, how='inner', left_on=['DBN'], right_on=['Location Code'])
merged_safety_df.dropna(subset=crimes_col, inplace=True,how='all')
features_pca = merged_safety_df[crimes_col].values

from sklearn.decomposition import PCA

school_crime_pca = PCA(n_components=1)
school_crime_pca.fit(features_pca)
reduced_crime_features = school_crime_pca.transform(features_pca)
# print school_crime_pca.explained_variance_ratio_
import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()
adjusted_reduced_crime_features = scaler.fit_transform(reduced_crime_features)

merged_safety_df['SRI'] = adjusted_reduced_crime_features
data = [go.Histogram(x=merged_safety_df['SRI'], nbinsx=20)]
print ("Histogram of School Risk Index")
iplot(data, filename='SRI histogram')
merged_safety_df.sort_values("SRI", inplace=True)

low_risk_schools_df = merged_safety_df[:30]
high_risk_schools_df = merged_safety_df[-30:]
feature_columns1 = ['Average ELA Proficiency','Average Math Proficiency', 'Economic Need Index']
feature_columns2 = ['Percent of Students Chronically Absent','Student Attendance Rate','Percent Black / Hispanic']
feature_columns = feature_columns1 + feature_columns2

print ("Average statistics for the schools with low risk index")
print (low_risk_schools_df[feature_columns].mean())
print ("")
print ("Average statistics for the schools with high risk index")
print (high_risk_schools_df[feature_columns].mean())


comparison_dict1 = {
    'trace1':
    {
        'features': low_risk_schools_df[feature_columns1].mean(),
        'name':'Low Risk schools'
    },
    
    'trace2':
    {
        'features': high_risk_schools_df[feature_columns1].mean(),
        'name':'High Risk schools'
    }
}

comparison_dict2 = {
    'trace1':
    {
        'features': low_risk_schools_df[feature_columns2].mean(),
        'name':'Low Risk schools'
    },
    
    'trace2':
    {
        'features': high_risk_schools_df[feature_columns2].mean(),
        'name':'High Risk schools'
    }
}

display_barplot(comparison_dict1, axis=1)
display_barplot(comparison_dict2, axis=1)
elite_schools_df = pd.read_csv('../input/elite-8-school-data/elite_eight_data.csv')
elite_schools_df = elite_schools_df.iloc[:8]
more_registered_students_df = df_merged[df_merged['Number of students who took test'] > 50]
less_registered_students_df = df_merged[df_merged['Number of students who took test'] < 10]
fig, ax = plt.subplots(figsize=(16,9))
less_registered_students_df.plot(kind="scatter", x="Longitude", y="Latitude", 
                   c=less_registered_students_df['Number of students who took test'], s=200, cmap=plt.get_cmap("jet"), 
                   label='Schools', title='SHSAT Registrations', 
                   colorbar=True, alpha=0.6, ax=ax)

elite_schools_df.plot(kind="scatter", x=" Long", y="Lat", 
                   c=elite_schools_df['Enrollment'], s=1000, cmap=plt.get_cmap("jet"), 
                   label='Elite School', title='SHSAT Schools', 
                   colorbar=False, alpha=0.6, marker='^', ax=ax)

ax.legend(markerscale=0.5)

#change the marker size manually for both lines
# legend = ax.legend(frameon=True)
# for legend_handle in legend.legendHandles:
#     legend_handle.set_markersize(9)

f = ax.set_ylabel('Latitude')
f = ax.set_xlabel('Longitude')
f = ax.set_xlim(-74.2, -73.75)
f = ax.set_ylim(40.5, 40.95)
print ("Schools with Low test takers")
print ("Mean Percent Black / Hispanic ratio : " + str(less_registered_students_df['Percent Black / Hispanic'].mean()))
print ("Standard Deviation Percent Black / Hispanic ratio : " + str(less_registered_students_df['Percent Black / Hispanic'].std()))
print ("")
print ("Schools with High test takers")
print ("Mean Percent Black / Hispanic ratio : " + str(more_registered_students_df['Percent Black / Hispanic'].mean()))
print ("Standard Deviation Percent Black / Hispanic ratio : " + str(more_registered_students_df['Percent Black / Hispanic'].std()))
df_schools_high_ell = df_merged[df_merged['Percent ELL'] > df_merged['Percent ELL'].quantile(0.90)]
df_schools_low_ell = df_merged[df_merged['Percent ELL'] < df_merged['Percent ELL'].quantile(0.10)]

print("Average performance of schools with high ELL")
print(df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean())
print()
print("Average performance of schools with low ELL")
print(df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean())
print()
print("Difference between average performance of schools with high and low number of ELL")
print(abs(df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean() - df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean()))

comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_ell[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Schools with High ELL'
    },
    
    'trace2':
    {
        'features': df_schools_low_ell[['Average ELA Proficiency','Average Math Proficiency']].mean(),
        'name':'Schools with Low ELL'
    }
    
}


display_barplot(comparison_dict, axis=1)
print("The mean selection ratio of students in schools with high ELL is %f."%(df_schools_high_ell['Selection_ratio'].mean()*100))
print("The mean selection ratio of students in schools with low ELL is %f."%(df_schools_low_ell['Selection_ratio'].mean()*100))


comparison_dict = {
    'trace1':
    {
        'features': df_schools_high_ell[['Selection_ratio']].mean()*100,
        'name':'Mean Selection Ratio of Schools with High ELL'
    },
    
    'trace2':
    {
        'features': df_schools_low_ell[['Selection_ratio']].mean()*100,
        'name':'Mean Selection Ratio of Schools with Low ELL'
    }
    
}


display_barplot(comparison_dict, axis=1)

