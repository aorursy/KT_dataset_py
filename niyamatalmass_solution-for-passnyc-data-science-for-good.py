import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting 
plt.style.use('ggplot') # use ggplot style for beautify
import seaborn as sns # useful library for many visualization and finding relation
import geojson # load jeojson file in python
import folium # folium for plotting asesome map
from folium.plugins import MarkerCluster # marker for clustering school marker
%matplotlib inline 
df_school_exlorer = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv') # load 2016 school explorer dataset
df_shsat_reg_test = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv') # load district5 schools SHSAT dataset
df_shsat_nyt = pd.read_csv('../input/parsing-nyt-shsat-table/nytdf.csv')
df_school_quality = pd.read_csv('../input/school-quality-dataset-report-201516/summery_school_quality_result.csv')
df_school_details = pd.read_csv('../input/nyc-schools-details/elem_schools_infowindow_master_file_with_ratios_12_11.csv')
df_school_demographics = pd.read_csv('../input/nyc-school-demographics/nyc_school_demographics.csv', encoding = "ISO-8859-1")
df_school_class_size = pd.read_csv('../input/ny-2010-2011-class-size-school-level-detail/2010-2011-class-size-school-level-detail.csv')
print("The shape of df_school_explorer " + str(df_school_exlorer.shape)) # print the shape of dataset
print("The shape of df_shsat_reg_test " + str(df_shsat_reg_test.shape)) # print the shape of dataset
print("The shape of df_shsat_nyt " + str(df_shsat_nyt.shape))
print("The shape of df_school_quality " + str(df_school_quality.shape))
print("The shape of df_school_details " + str(df_school_details.shape))
print("The shape of df_school_demographics " + str(df_school_demographics.shape))
print("The shape of df_school_class_size " + str(df_school_class_size.shape))
df_school_exlorer.head(2) # print the first two rwos of dataset
df_shsat_reg_test.head() # print the second datast
df_shsat_nyt.head()

with open('../input/minority-majority/nyc_scl_destrict.geojson') as f: # open the jeojson file
    gj = geojson.load(f) # load the jeojson file 

kw = {
    'prefix': 'fa',
    'color': 'green',
    'icon': 'adn'
}

folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_school_exlorer.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), icon=folium.Icon(**kw)).add_to(marker_cluster) # plot a marker on the map for a school
    
folium_map   # show the folium map
# some preprocessing
df_school_exlorer = df_school_exlorer.rename(columns={'Location Code': 'DBN'}) # change key column name so that it matches with each other 
df_school_details = df_school_details.rename(columns={'dbn': 'DBN'}) # change key column name so that it matches with each other 
df_school_class_size = df_school_class_size.rename(columns={'SCHOOL NAME': 'School Name'}) # change key column name so that it matches with each other 
df_school_demographics_new = df_school_demographics.loc[df_school_demographics['Year'] == '2017-18'] # we only need data on recent years 
df_school_class_size_new = df_school_class_size.loc[df_school_class_size['GRADE '] == '08'] # we only need data on 8 grade


# these are the columns name that will be used from School explorer 
needed_columns_df_school_explorer = ['School Name', 'SED Code', 'DBN', 'District', 
                                     'Latitude', 'Longitude', 'Address (Full)',
                                     'City', 'Zip', 'Grades', 
                                     'Grade Low','Grade High','Community School?',
                                     'Economic Need Index', 'School Income Estimate',
                                     'Percent ELL', 'Percent Asian', 'Percent Black', 
                                     'Percent Hispanic', 
                                     'Percent Black / Hispanic', 'Percent White',
                                     'Student Attendance Rate', 
                                     'Percent of Students Chronically Absent', 
                                     'Rigorous Instruction %', 'Collaborative Teachers %',
                                     'Supportive Environment %', 'Effective School Leadership %', 
                                     'Strong Family-Community Ties %',
                                     'Trust %', 'Student Achievement Rating', 
                                     'Average ELA Proficiency', 
                                     'Average Math Proficiency',
                                     'Grade 8 ELA - All Students Tested', 
                                     'Grade 8 ELA 4s - All Students', 
                                     'Grade 8 ELA 4s - American Indian or Alaska Native',
                                     'Grade 8 ELA 4s - Black or African American',
                                     'Grade 8 ELA 4s - Hispanic or Latino', 
                                     'Grade 8 ELA 4s - Asian or Pacific Islander',
                                     'Grade 8 ELA 4s - White', 'Grade 8 ELA 4s - Multiracial', 
                                     'Grade 8 ELA 4s - Limited English Proficient',
                                     'Grade 8 ELA 4s - Economically Disadvantaged', 
                                     'Grade 8 Math - All Students Tested', 
                                     'Grade 8 Math 4s - All Students',
                                     'Grade 8 Math 4s - American Indian or Alaska Native', 
                                     'Grade 8 Math 4s - Black or African American',
                                     'Grade 8 Math 4s - Hispanic or Latino', 
                                     'Grade 8 Math 4s - Asian or Pacific Islander', 
                                     'Grade 8 Math 4s - White',
                                     'Grade 8 Math 4s - Multiracial', 
                                     'Grade 8 Math 4s - Limited English Proficient', 
                                     'Grade 8 Math 4s - Economically Disadvantaged']

# these are the columns that will be used from School Quality dataset
needed_columns_df_school_quality = ['DBN','Enrollment',
                                    'Percent Students with Disabilities',
                                    'Percent Self-Contained',
                                    'Percent in Temp Housing', 
                                    'Percent HRA Eligible', 
                                    'Years of principal experience at this school',
                                    'Percent of teachers with 3 or more years of experience', 
                                    'Teacher Attendance Rate']

# these are the columns that will be used from School Demographic dataset
needed_columns_df_school_demographic = ['DBN', 'Total Enrollment','# Female', '# Male', '# Poverty', '% Poverty']
# # these are the columns that will be used from School Size dataset
needed_columns_df_school_size = ['School Name','AVERAGE CLASS SIZE']


punctuation_contains_columns = ['School Income Estimate', 'Percent ELL', 
                                'Percent Asian', 'Percent Black', 
                                'Percent Hispanic', 'Percent Black / Hispanic', 
                                'Percent White', 'Student Attendance Rate',
                                'Percent of Students Chronically Absent', 
                                'Rigorous Instruction %', 
                                'Collaborative Teachers %',
                                'Supportive Environment %', 
                                'Effective School Leadership %', 
                                'Strong Family-Community Ties %',
                                'Trust %', 'Percent Students with Disabilities',
                                'Percent Self-Contained',
                                'Percent in Temp Housing', 
                                'Percent HRA Eligible', 
                                'Percent of teachers with 3 or more years of experience',
                                'Teacher Attendance Rate'] # list of column names that contains punctuation 
# let's keep only those columns that we will be using 
df_school_exlorer_new = df_school_exlorer[needed_columns_df_school_explorer]
df_school_quality_new = df_school_quality[needed_columns_df_school_quality]
df_school_demographics_new = df_school_demographics_new[needed_columns_df_school_demographic]
df_school_class_size_new = df_school_class_size_new[needed_columns_df_school_size]

# lower case our school name because in this dataset we will be using School Name column as key
df_school_class_size_new['School Name'] = df_school_class_size_new['School Name'].str.lower()
df_school_exlorer_new['School Name'] = df_school_exlorer['School Name'].str.lower()
# let's merge our alll datasets
df_shsat_nyt = pd.merge(df_school_exlorer_new, df_shsat_nyt, on='DBN')
df_shsat_nyt = pd.merge(df_school_quality_new, df_shsat_nyt, on='DBN')
df_shsat_nyt = pd.merge(df_shsat_nyt, df_school_demographics_new, on='DBN')
df_shsat_nyt = pd.merge(df_shsat_nyt, df_school_class_size_new, on='School Name')
# some preprocessing
for col in punctuation_contains_columns: 
    df_shsat_nyt[col] = df_shsat_nyt[col].str.replace('[^\w\s]','') # remove punctuation from each column
    
df_shsat_nyt = df_shsat_nyt.apply(pd.to_numeric, errors='ignore') # some of our percentage column values data type is str. change it to int
df_shsat_nyt.head() # print the head of our mereged dataset of two previous dataset
# let's print the shape
df_shsat_nyt.shape
fig, axs = plt.subplots(figsize=(20,8),ncols=3)
sns.regplot(x="Economic Need Index", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="School Income Estimate", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="Percent ELL", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=5)
sns.regplot(x="Percent Asian", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="Percent Black", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])
sns.regplot(x="Percent Hispanic", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="Percent Black / Hispanic", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[3])
sns.regplot(x="Percent White", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[4])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=5)
sns.regplot(x="Grade 8 Math 4s - All Students", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="Grade 8 ELA 4s - All Students", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])
sns.regplot(x="Average Math Proficiency", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="Average ELA Proficiency", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[3])
sns.regplot(x="Strong Family-Community Ties %", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[4])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=5)
sns.regplot(x="Enrollment", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="Percent Students with Disabilities", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])
sns.regplot(x="Percent Self-Contained", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="Percent in Temp Housing", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[3])
sns.regplot(x="Percent HRA Eligible", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[4])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=6)
sns.regplot(x="Years of principal experience at this school", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="Percent of teachers with 3 or more years of experience", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])
sns.regplot(x="Teacher Attendance Rate", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="# Poverty", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[3])
sns.regplot(x="# Female", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[4])
sns.regplot(x="# Male", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[5])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=6)
sns.regplot(x="Grade 8 ELA 4s - American Indian or Alaska Native", y="NumSHSATTestTakers", data=df_shsat_nyt,  ax=axs[0])
sns.regplot(x="Grade 8 ELA 4s - Black or African American", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[1])
sns.regplot(x="Grade 8 ELA 4s - Hispanic or Latino", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[2])
sns.regplot(x="Grade 8 ELA 4s - Asian or Pacific Islander", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[3])
sns.regplot(x="Grade 8 ELA 4s - White", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[4])
sns.regplot(x="AVERAGE CLASS SIZE", y="NumSHSATTestTakers", data=df_shsat_nyt, ax=axs[5])

plt.show()
# these columns will be used to calculate teh ethnicity score
ethnicity_score_column = ['Percent Asian', 
                          'Percent Black', 
                          'Percent Hispanic', 
                          'Percent Black / Hispanic', 
                          'Percent White']
# these columns will be used to calculate education score
education_score_column = ['Percent ELL',
                          'Average ELA Proficiency', 
                          'Average Math Proficiency']
# these columns will be used to calculate education system score
education_system_column = ['Percent of Students Chronically Absent', 
                          'Rigorous Instruction %',
                          'Collaborative Teachers %',
                          'Supportive Environment %',
                          'Effective School Leadership %', 
                          'Strong Family-Community Ties %',
                          'Trust %']
# these columns will be used to calculate school quality score
school_quality_column = ['Student Attendance Rate',
                         'Years of principal experience at this school',
                         'Percent of teachers with 3 or more years of experience', 
                         'Teacher Attendance Rate',
                         'Total Enrollment']
# these columns will be used to calculate student situation score
student_situation_column = ['Percent Students with Disabilities',
                            'Percent Self-Contained',
                            'Percent in Temp Housing', 
                            'Percent HRA Eligible', 
                            '# Poverty']

# these columns will be used to calculate student gender score
student_gender_column = ['# Female', '# Male']

df_shsat_nyt_new = df_shsat_nyt.copy() # create new copy of merged dataset
# calculate ethnicity score by weighted sum approces
df_shsat_nyt_new['score_ethnicity'] = df_shsat_nyt_new['Percent Asian'] * 0.2 + \
                                      df_shsat_nyt_new['Percent Black / Hispanic'] * 0.5 + \
                                      df_shsat_nyt_new['Percent White'] * 0.3
# calculate education score by weighted sum approces        
df_shsat_nyt_new['score_education'] = df_shsat_nyt_new['Percent ELL'] * 0.4 + \
                                      df_shsat_nyt_new['Average ELA Proficiency'] * 0. + \
                                      df_shsat_nyt_new['Average Math Proficiency'] * 0.2

# calculate education score by weighted sum approces    
df_shsat_nyt_new['score_educaton_system'] = df_shsat_nyt_new['Percent of Students Chronically Absent'] * 0.1 + \
                                      df_shsat_nyt_new['Rigorous Instruction %'] * 0.1 + \
                                      df_shsat_nyt_new['Collaborative Teachers %'] * 0.1 + \
                                      df_shsat_nyt_new['Supportive Environment %'] * 0.2 + \
                                      df_shsat_nyt_new['Effective School Leadership %'] * 0.2 + \
                                      df_shsat_nyt_new['Strong Family-Community Ties %'] * 0.2 + \
                                      df_shsat_nyt_new['Trust %'] * 0.1

# calculate school quality score by weighted sum approces
df_shsat_nyt_new['score_school_quality'] = df_shsat_nyt_new['Student Attendance Rate'] * 0.2 + \
                                      df_shsat_nyt_new['Years of principal experience at this school'] * 0.2 + \
                                      df_shsat_nyt_new['Percent of teachers with 3 or more years of experience'] * 0.2 + \
                                      df_shsat_nyt_new['Teacher Attendance Rate'] * 0.2 + \
                                      df_shsat_nyt_new['Total Enrollment'] * 0.2

# calculate student situation score by weighted sum approces                        
df_shsat_nyt_new['score_student_situation'] = df_shsat_nyt_new['Percent Students with Disabilities'] * 0.6 + \
                                      df_shsat_nyt_new['Percent Self-Contained'] *0.1 + \
                                      df_shsat_nyt_new['Percent in Temp Housing'] * 0.1 + \
                                      df_shsat_nyt_new['Percent HRA Eligible'] * 0.1 + \
                                      df_shsat_nyt_new['# Poverty'] * 0.1
# calculate gender score by weighted sum approces                
df_shsat_nyt_new['score_gender'] = df_shsat_nyt_new['# Female'] * 0.5 + \
                                      df_shsat_nyt_new['# Male'] *0.5
    

# let's combined those score score with weighted sum approaches    
df_shsat_nyt_new['score_combined'] = df_shsat_nyt_new['score_ethnicity'] * 0.1 + \
                                      df_shsat_nyt_new['score_education'] * 0.2 + \
                                      df_shsat_nyt_new['score_educaton_system'] * 0.1 + \
                                      df_shsat_nyt_new['score_school_quality'] * 0.1 + \
                                      df_shsat_nyt_new['score_student_situation'] * 0.4 + \
                                      df_shsat_nyt_new['score_gender'] * 0.1
fig, axs = plt.subplots(figsize=(20,8),ncols=6)
sns.regplot(x="score_ethnicity", y="NumSHSATTestTakers", data=df_shsat_nyt_new,  ax=axs[0])
sns.regplot(x="score_education", y="NumSHSATTestTakers", data=df_shsat_nyt_new, ax=axs[1])
sns.regplot(x="score_educaton_system", y="NumSHSATTestTakers", data=df_shsat_nyt_new, ax=axs[2])
sns.regplot(x="score_school_quality", y="NumSHSATTestTakers", data=df_shsat_nyt_new, ax=axs[3])
sns.regplot(x="score_student_situation", y="NumSHSATTestTakers", data=df_shsat_nyt_new, ax=axs[4])
sns.regplot(x="score_gender", y="NumSHSATTestTakers", data=df_shsat_nyt_new, ax=axs[5])

plt.show()
fig, axs = plt.subplots(figsize=(20,8),ncols=1)
sns.regplot(x="score_combined", y="NumSHSATTestTakers", data=df_shsat_nyt_new,  ax=axs)

plt.show()
df_school_exlorer = df_school_exlorer.rename(columns={'Location Code': 'DBN'}) # change key column name so that it matches with each other 
df_school_details = df_school_details.rename(columns={'dbn': 'DBN'}) # change key column name so that it matches with each other 
df_school_class_size = df_school_class_size.rename(columns={'SCHOOL NAME': 'School Name'}) # change key column name so that it matches with each other 
df_school_demographics_new = df_school_demographics.loc[df_school_demographics['Year'] == '2017-18']
df_school_class_size_new = df_school_class_size.loc[df_school_class_size['GRADE '] == '08']


# let's use pandas merge function to merge our two datasets
df_school_exlorer_new = df_school_exlorer[needed_columns_df_school_explorer]
df_school_quality_new = df_school_quality[needed_columns_df_school_quality]
df_school_demographics_new = df_school_demographics_new[needed_columns_df_school_demographic]
df_school_class_size_new = df_school_class_size_new[needed_columns_df_school_size]

df_school_class_size_new['School Name'] = df_school_class_size_new['School Name'].str.lower()
df_school_exlorer_new['School Name'] = df_school_exlorer['School Name'].str.lower()

df_model = pd.merge(df_school_exlorer_new, df_school_quality_new, on='DBN')
df_model = pd.merge(df_model, df_school_demographics_new, on='DBN')
# df_model = pd.merge(df_model, df_school_class_size_new, on='School Name')


for col in punctuation_contains_columns: 
    df_model[col] = df_model[col].str.replace('[^\w\s]','') # remove punctuation from each column
    
df_model = df_model.apply(pd.to_numeric, errors='ignore') # some of our percentage column values data type is str. change it to int
df_model.head() # print the head of our mereged dataset of two previous dataset
df_model.shape
df_model['score_ethnicity'] = df_model['Percent Asian'] * 0.2 + \
                                      df_model['Percent Black / Hispanic'] * 0.5 + \
                                      df_model['Percent White'] * 0.3
        
df_model['score_education'] = df_model['Percent ELL'] * 0.4 + \
                                      df_model['Average ELA Proficiency'] * 0. + \
                                      df_model['Average Math Proficiency'] * 0.2

    
df_model['score_educaton_system'] = df_model['Percent of Students Chronically Absent'] * 0.1 + \
                                      df_model['Rigorous Instruction %'] * 0.1 + \
                                      df_model['Collaborative Teachers %'] * 0.1 + \
                                      df_model['Supportive Environment %'] * 0.2 + \
                                      df_model['Effective School Leadership %'] * 0.2 + \
                                      df_model['Strong Family-Community Ties %'] * 0.2 + \
                                      df_model['Trust %'] * 0.1


df_model['score_school_quality'] = df_model['Student Attendance Rate'] * 0.2 + \
                                      df_model['Years of principal experience at this school'] * 0.2 + \
                                      df_model['Percent of teachers with 3 or more years of experience'] * 0.2 + \
                                      df_model['Teacher Attendance Rate'] * 0.2 + \
                                      df_model['Total Enrollment'] * 0.2

                        
df_model['score_student_situation'] = df_model['Percent Students with Disabilities'] * 0.6 + \
                                      df_model['Percent Self-Contained'] *0.1 + \
                                      df_model['Percent in Temp Housing'] * 0.1 + \
                                      df_model['Percent HRA Eligible'] * 0.1 + \
                                      df_model['# Poverty'] * 0.1
                
df_model['score_gender'] = df_model['# Female'] * 0.5 + \
                                      df_model['# Male'] *0.5
    

    
df_model['score_combined'] = df_model['score_ethnicity'] * 0.1 + \
                                      df_model['score_education'] * 0.2 + \
                                      df_model['score_educaton_system'] * 0.1 + \
                                      df_model['score_school_quality'] * 0.1 + \
                                      df_model['score_student_situation'] * 0.4 + \
                                      df_model['score_gender'] * 0.1
# columns that will be used in Kmeans
kmeans_column = ['Economic Need Index',
                                     'Percent ELL', 'Percent Asian', 'Percent Black', 
                                     'Percent Hispanic', 
                                     'Percent Black / Hispanic', 'Percent White',
                                     'Student Attendance Rate', 
                                     'Percent of Students Chronically Absent', 
                                     'Rigorous Instruction %', 'Collaborative Teachers %',
                                     'Supportive Environment %', 'Effective School Leadership %', 
                                     'Strong Family-Community Ties %',
                                     'Trust %', 
                                     'Average ELA Proficiency', 
                                     'Average Math Proficiency',
                                     'Grade 8 ELA - All Students Tested', 
                                     'Grade 8 ELA 4s - All Students', 
                                     'Grade 8 ELA 4s - American Indian or Alaska Native',
                                     'Grade 8 ELA 4s - Black or African American',
                                     'Grade 8 ELA 4s - Hispanic or Latino', 
                                     'Grade 8 ELA 4s - Asian or Pacific Islander',
                                     'Grade 8 ELA 4s - White', 'Grade 8 ELA 4s - Multiracial', 
                                     'Grade 8 ELA 4s - Limited English Proficient',
                                     'Grade 8 ELA 4s - Economically Disadvantaged', 
                                     'Grade 8 Math - All Students Tested', 
                                     'Grade 8 Math 4s - All Students',
                                     'Grade 8 Math 4s - American Indian or Alaska Native', 
                                     'Grade 8 Math 4s - Black or African American',
                                     'Grade 8 Math 4s - Hispanic or Latino', 
                                     'Grade 8 Math 4s - Asian or Pacific Islander', 
                                     'Grade 8 Math 4s - White',
                                     'Grade 8 Math 4s - Multiracial', 
                                     'Grade 8 Math 4s - Limited English Proficient', 
                                     'Grade 8 Math 4s - Economically Disadvantaged', 'Enrollment',
                                    'Percent Students with Disabilities',
                                    'Percent Self-Contained',
                                    'Percent in Temp Housing', 
                                    'Percent HRA Eligible','Total Enrollment','# Female', '# Male', '# Poverty']

# some preprocessing
df_test = df_model[kmeans_column]
df_test = df_test.apply(pd.to_numeric)
from sklearn.preprocessing import Imputer
fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(df_test))
imputed_DF.columns = df_test.columns
imputed_DF.index = df_test.index
df_test = imputed_DF
from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import hdbscan
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from matplotlib.ticker import NullFormatter
def get_cluster_colors(clusterer, palette='Paired'):
    """Create cluster colors based on labels and probability assignments"""
    n_clusters = len(np.unique(clusterer.labels_))
    color_palette = sns.color_palette(palette, n_clusters)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    if hasattr(clusterer, 'probabilities_'):
        cluster_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_colors

# Prepare figure
_, ax = plt.subplots(1, 2, figsize=(20, 5))
settings = {'s':50, 'linewidth':0, 'alpha':0.2}

print(">> Calculating elbow plot for KMeans")
kmeans = KMeans(random_state=42)
skplt.cluster.plot_elbow_curve(kmeans, df_test, cluster_ranges=[1, 5, 6,10, 20], ax=ax[0])

print(">> Dimensionality reduction using TSNE")
projection = manifold.TSNE(init='pca', random_state=42).fit_transform(df_test)

print(">> Clustering using K-Means")
kmeans = KMeans(n_clusters=5).fit(projection)

# PLot on figure
ax[1].scatter(*projection.T, c=get_cluster_colors(kmeans), **settings)
ax[1].set_title('K-Means Clusters')
plt.show()
# Get number of clusters identified by HDBSCAN
unique_clusters = [c for c in np.unique(kmeans.labels_) if c > -1]

# Placeholder for our plotting
_, axes = plt.subplots(len(unique_clusters), 1, figsize=(15, 25))

# Go through clusters identified by HDBSCAN
for i, label in enumerate(unique_clusters):
    
    # Get index of this cluster
    idx = kmeans.labels_ == label
    
    # Identify feature where the median differs significantly
    median_diff = (df_test.median() - df_test[idx].median()).abs().sort_values(ascending=False)
    
    # Create boxplot of these features for all vs cluster
    top = median_diff.index[0:20]
    temp_concat = pd.concat([df_test.loc[:, top], df_test.loc[idx, top]], axis=0).reset_index(drop=True)
    temp_concat['Cluster'] = 'Cluster {}'.format(i+1)
    temp_concat.loc[0:len(df_test),'Cluster'] = 'All respondees'
    temp_long = pd.melt(temp_concat, id_vars='Cluster')
    
    sns.boxplot(x='variable', y='value', hue='Cluster', data=temp_long, ax=axes[i])
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(90)
    axes[i].set_title(f'Cluster #{i+1} - {idx.sum()} respondees')
df_model['c'] = kmeans.labels_
df_model.groupby('c')['score_education', 'score_educaton_system', 'score_school_quality','score_student_situation', 'score_gender'].mean()
   # show the folium map
df_cluster_0 = df_model.loc[df_model['c'] == 0]
folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_cluster_0.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), popup=folium.Popup(row['School Name'] + ' ' + row['DBN'], parse_html=True)).add_to(folium_map) # plot a marker on the map for a school

folium_map
   # show the folium map
df_cluster_1 = df_model.loc[df_model['c'] == 1]
folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_cluster_1.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), popup=folium.Popup(row['School Name'] + ' ' + row['DBN'], parse_html=True)).add_to(folium_map) # plot a marker on the map for a school

folium_map
   # show the folium map
df_cluster_2 = df_model.loc[df_model['c'] == 2]
folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_cluster_2.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), popup=folium.Popup(row['School Name'] + ' ' + row['DBN'], parse_html=True)).add_to(folium_map) # plot a marker on the map for a school

folium_map
   # show the folium map
df_cluster_3 = df_model.loc[df_model['c'] == 3]
folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_cluster_3.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), popup=folium.Popup(row['School Name'] + ' ' + row['DBN'], parse_html=True)).add_to(folium_map) # plot a marker on the map for a school

folium_map
   # show the folium map
df_cluster_4 = df_model.loc[df_model['c'] == 4]
folium_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="Stamen Terrain") # create the Map object 
marker_cluster = MarkerCluster().add_to(folium_map) # marker cluster for clustering marker for easy understanding
folium.GeoJson(gj).add_to(folium_map) # add jeojson to the map. So that NYC school border should show on the map

for index, row in df_cluster_4.iterrows(): # iter over every row in the dataset
    folium.Marker(location=(row["Latitude"],row["Longitude"]), popup=folium.Popup(row['School Name'] + ' ' + row['DBN'], parse_html=True)).add_to(folium_map).add_to(folium_map) # plot a marker on the map for a school

folium_map
