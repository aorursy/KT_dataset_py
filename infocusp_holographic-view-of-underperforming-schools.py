%matplotlib inline

# import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import folium
import sklearn
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn.decomposition import PCA
import sklearn
from sklearn.cluster import KMeans
import warnings
import itertools
import base64
from IPython.display import HTML

warnings.filterwarnings('ignore')
init_notebook_mode(connected=True)
# Function converting string % values to int
def percent_to_int(df_in):
    for col in df_in.columns.values:
        if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
            df_in[col] = df_in[col].astype(np.object).str.replace('%', '').astype(float)
    return df_in
df_schools_raw = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
df_schools_raw = df_schools_raw[df_schools_raw['Grade High'] != '0K'] 
df_schools_raw = percent_to_int(df_schools_raw)
df_schools_raw['School Income Estimate'] = df_schools_raw['School Income Estimate'].astype(np.object).str.replace('$', '').str.replace(',', '').str.replace('.', '').astype(float)

df_schools_relevant_grade = df_schools_raw[df_schools_raw['Grade High'].astype(int) > 5]

high_nan_columns = df_schools_raw.columns[df_schools_raw.isnull().mean()>0.95]
# print("Here are the fields having >95% NaNs which we can drop: \n")
# print(list(high_nan_columns))

df_schools = df_schools_relevant_grade.drop(high_nan_columns, axis=1)
print("We have %d relevant schools and %d fields describing the school/ students"%(df_schools.shape))
def plot_city_hist(df_schools, title_str):
    layout = go.Layout(
        title=title_str,
        xaxis=dict(
            title='City',
            titlefont=dict(
                family='Arial, sans-serif',
                size=12,
                color='black'
            ),
            showticklabels=True,
            tickangle=315,
            tickfont=dict(
                size=10,
                color='grey'
            )
        )
    )
    data = [go.Histogram(x=df_schools['City'])]
    fig = go.Figure(data=data, layout=layout)
    return fig
fig = plot_city_hist(df_schools, 'City wise School Distribution')
iplot(fig)
plt.figure(figsize = [10,8])
df_schools = df_schools.dropna(subset = ['Percent of Students Chronically Absent'])
temp = sns.distplot(df_schools[['Percent of Students Chronically Absent']].values, kde=False)
temp = plt.title('Distribution of schools based on chronically absent students')
temp = plt.xlabel("Percent of students")
temp = plt.ylabel("Count")
chronically_absent_foi = ['School Name', 'Percent of Students Chronically Absent','Percent Black / Hispanic','Economic Need Index']

df_schools_ca = df_schools[df_schools['Percent of Students Chronically Absent'] > 40]
df_schools_nca = df_schools[df_schools['Percent of Students Chronically Absent'] < 5]

df_schools_ca[chronically_absent_foi].sort_values('Percent of Students Chronically Absent', ascending = False)
plt.figure(figsize = [16,7])
plt.suptitle('Statistics for schools with high Chronically absent percentage', fontsize=15)
plt.subplot(1,2,1)
temp = sns.distplot(df_schools_ca[['Economic Need Index']].values, kde=False)
temp = plt.xlabel("Economic Need Index", fontsize=15)
temp = plt.ylabel("School count", fontsize=15)
plt.subplot(1,2,2)
temp = sns.distplot(df_schools_ca[['Percent Black / Hispanic']].values, kde=False)
temp = plt.xlabel("Percent Black / Hispanic", fontsize=15)
temp = plt.ylabel("School count", fontsize=15)
print ('1) ENI Statistics for schools with high % of chronically absent students')
print ('\t * Mean value : %f'%df_schools_ca['Economic Need Index'].mean())
print ('\t * Median value : %f'%df_schools_ca['Economic Need Index'].median())
print('2) %d schools have > 40%% students chronically absent'%(df_schools_ca.shape[0]))
features_list = ['Rigorous Instruction %',
'Collaborative Teachers %',
'Supportive Environment %',
'Effective School Leadership %',
'Strong Family-Community Ties %',
'Trust %']

df_schools[['School Name'] + features_list ].head()
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
for i in range(3):
    for j in range(2):
        ax[i, j].set_title(features_list[i*2 + j])
        sns.distplot(a=df_schools[features_list[i*2 + j]].dropna().values, kde_kws={"color": "red"}, color='darkblue', ax=ax[i, j])

# fig.tight_layout()
temp = fig.suptitle('School Performance features', fontsize=15)
df_schools[features_list].corr()
corr = df_schools[features_list].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='YlGnBu')
temp = plt.xticks(rotation=75, fontsize=15) 
temp = plt.yticks(fontsize=15) 
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
features = ['Rigorous Instruction %','Supportive Environment %','PCA Combined Feature',
            'Strong Family-Community Ties %']
weights = [0.8, 1, 0.7, 0.5]

df_schools['SPI'] = df_schools[features].dot(weights)


print ("A glimpse of the School Performance Index (SPI) :")
df_schools[features+['SPI']].head(5)
df_low_spi_schools = df_schools[df_schools['SPI'] < df_schools['SPI'].quantile(.25)]
df_high_spi_schools = df_schools[df_schools['SPI'] > df_schools['SPI'].quantile(.25)]
fig = plot_city_hist(df_low_spi_schools, 'Distribution of low SPI schools by city')
iplot(fig)
print ("Average ELA / Math performance for the high SPI schools")
df_high_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean()
print ("Average ELA / Math performance for the low SPI schools")
df_low_spi_schools[['Average ELA Proficiency','Average Math Proficiency']].mean()
plt.figure(figsize=(12,7))
temp = sns.distplot(df_schools[['Economic Need Index']].values, kde=False)
temp= plt.title("ENI distribution", fontsize=15)
temp = plt.xlabel("ENI", fontsize=15)
temp = plt.ylabel("School count", fontsize=15)
fig, ax = plt.subplots(figsize=(16,9))
fig.suptitle('New York School Population Map', fontsize=15)
ax = df_schools.plot(kind="scatter", x="Longitude", y="Latitude", 
                   s=df_schools['School Income Estimate']/10000, c="Economic Need Index", cmap=plt.get_cmap("jet"), 
                   label='Schools', colorbar=True, alpha=0.6, ax=ax)
temp = ax.set_ylabel('Latitude',fontsize=15)
temp = ax.set_xlabel('Longitude',fontsize=15)
fig, ax = plt.subplots(figsize=(16,9))
fig.suptitle('Percent Black / Hispanic and ENI', fontsize=15)
ax = df_schools.plot(kind="scatter", x="Longitude", y="Latitude", 
                   c='Economic Need Index', s=df_schools["Percent Black / Hispanic"]*3.5, cmap=plt.get_cmap("jet"), 
                   label='Schools', colorbar=True, alpha=0.6, ax=ax)
temp = ax.set_ylabel('Latitude', fontsize=15)
temp = ax.set_xlabel('Longitude', fontsize=15)
lower_bound = df_schools['Economic Need Index'].quantile(0.75)
df_schools_high_eni = df_schools[df_schools['Economic Need Index'] > lower_bound]
df_schools_low_eni = df_schools[df_schools['Economic Need Index'] < 0.25]
df_schools_high_eni[['School Name', 'Economic Need Index', 'Percent Black / Hispanic']].sort_values('Economic Need Index', ascending = False).head()
print ("Average ELA / Math performance for the high ENI schools")
df_schools_high_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()
print ("Average ELA / Math performance for the low ENI schools")
df_schools_low_eni[['Average ELA Proficiency','Average Math Proficiency']].mean()
def find_schools_with_black_high_4s_low(df, grade = 5, black_per_threshold = 90, all_students_threshold = 5):
    math_black_students = "Grade {} Math 4s - Black or African American".format(grade)
    math_hispanic_students = "Grade {} Math 4s - Hispanic or Latino".format(grade)
    math_all_students = "Grade {} Math 4s - All Students".format(grade)
    
    df_schools_copy = df.copy()
    df_schools_copy['4s ratio black / hispanic'] = ((df_schools_copy[math_black_students] + df_schools_copy[math_hispanic_students])/ df_schools_copy[math_all_students]).values
    df_schools_black_4s = df_schools_copy[np.logical_and(df_schools_copy['Percent Black / Hispanic'] > black_per_threshold, df_schools_copy[math_all_students] > all_students_threshold)]
    df_schools_black_4s.dropna(inplace=True)
    
    df_schools_black_4s_low = df_schools_black_4s[df_schools_black_4s['4s ratio black / hispanic'] < 0.5]
    num_schools = len(df_schools_black_4s_low)                      
    
    return df_schools_black_4s_low
schools_with_black_high_4s_low = []

for grade in range(5,9):
    df_schools_with_black_high_4s_low = find_schools_with_black_high_4s_low(df_schools, grade=grade)
    schools_with_black_high_4s_low.extend(np.unique(df_schools_with_black_high_4s_low['School Name']))

schools_with_black_high_4s_low = list(set(schools_with_black_high_4s_low))
schools_with_black_high_4s_low
df_filtered_schools = df_schools[df_schools['School Name'].isin(schools_with_black_high_4s_low)]
df_filtered_schools[['School Name', 'City', 'Economic Need Index', 'SPI']]
school_map = folium.Map([df_filtered_schools['Latitude'].mean(), df_filtered_schools['Longitude'].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain')

for index in df_filtered_schools.index:
    row = df_filtered_schools.loc[index]
    school_name = str(row['School Name'])
    
    color = 'blue'
    folium.Marker([row['Latitude'], row['Longitude']], popup=school_name, icon=folium.Icon(color=color)).add_to(school_map)

school_map
df_crime = pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv')

intersection = set(df_crime['DBN']).intersection(set(df_schools['Location Code']))
print('%d schools found in the supplementary dataset out of %d in original dataset'%(len(intersection), len(df_schools)))

crimes_col = ['Major N', 'Oth N', 'NoCrim N', 'Prop N', 'Vio N']
crimes = df_crime.groupby(['DBN'], as_index=False)[crimes_col].sum()

merged_safety_df = pd.merge(crimes[crimes_col + ['DBN']], df_schools, how='inner', left_on=['DBN'], right_on=['Location Code'])
merged_safety_df.dropna(subset=crimes_col, inplace=True,how='all')
def plot_crime_histogram(column_name, ax):
    sns.distplot(merged_safety_df[column_name][merged_safety_df[column_name]!=0], ax=ax, kde=False)
    ax.set_title(column_name)
    ax.set_xlabel('')
    
fig, ax = plt.subplots(2, 3, figsize = [16, 10])    
fig.suptitle("Distributions of various crimes")
for i, column_name in enumerate(crimes_col):
    plot_crime_histogram(column_name, ax[int(i/3), i%3])
ax[1, 2].remove()
'''
Filtering out the data-frame where Violent Crimes are greater than 4, and Major Crimes are greater than 2.
'''
violent_df = merged_safety_df[merged_safety_df['Vio N'] > 4]
major_crime_df = merged_safety_df[merged_safety_df['Major N'] > 2]
corr = crimes[crimes_col].corr()
corr
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt='0.2g',  annot_kws={"size": 15})
temp = plt.xticks(rotation=75, fontsize=15) 
temp = plt.yticks(fontsize=15) 
features_pca = merged_safety_df[crimes_col].values

school_crime_pca = PCA(n_components=1)
school_crime_pca.fit(features_pca)
reduced_crime_features = school_crime_pca.transform(features_pca)
print("The explained variance ratio (the amount of data covered by the PCA feature) is : " + str(school_crime_pca.explained_variance_ratio_[0]))
scaler = sklearn.preprocessing.MinMaxScaler()
adjusted_reduced_crime_features = scaler.fit_transform(reduced_crime_features)

merged_safety_df['SRI'] = adjusted_reduced_crime_features

print('Correlation of risk measure with different features: ')
print('School Performance Index: '  + str(merged_safety_df['SRI'].corr(merged_safety_df['SPI'])))
print('Percent of Students Chronically Absent: '  + str(merged_safety_df['SRI'].corr(merged_safety_df['Percent of Students Chronically Absent'])))
risky_schools_df = merged_safety_df[merged_safety_df['SRI']>merged_safety_df['SRI'].quantile(0.9)]
print ("The number of risky schools in top 10 percentile are " + str(risky_schools_df.shape[0]))
risky_schools_df[['School Name','SRI']].sort_values('SRI', ascending=False)
field_name = "SCHOOLWIDE PUPIL-TEACHER RATIO"
df_school_detail = pd.read_csv('../input/ny-2010-2011-class-size-school-level-detail/2010-2011-class-size-school-level-detail.csv')
df_school_detail["CSD"] =  df_school_detail['CSD'].astype('str').astype(np.object_).str.zfill(2)
df_school_detail["DBN_manual"] = df_school_detail["CSD"] + df_school_detail["SCHOOL CODE"] 
df_school_detail.dropna(subset=[field_name], inplace=True)
merged_str_df = pd.merge(df_school_detail, df_schools, how='inner', left_on=['DBN_manual'], right_on=['Location Code'])
plt.figure(figsize=(10,6))
sns.distplot(merged_str_df[field_name], kde=False)
# merged_str_df[field_name].hist()
temp = plt.title("Number of Students per Teacher", fontsize=15)
temp = plt.xlabel(field_name, fontsize=15)
higher_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)>18]
lower_ratio_str_df = merged_str_df[merged_str_df[field_name].astype(float)<12.5]
print ("Average statistics for the schools with high STR")
higher_ratio_str_df[['Average ELA Proficiency','Average Math Proficiency', 'Collaborative Teachers %', 'SPI','Percent Black / Hispanic','Economic Need Index']].mean()
print ("Average statistics for the schools with low STR")
lower_ratio_str_df[['Average ELA Proficiency','Average Math Proficiency', 'Collaborative Teachers %', 'SPI','Percent Black / Hispanic','Economic Need Index']].mean()
df_schools_clustering = df_schools.copy()
df_schools_clustering = df_schools_clustering.dropna(subset=['Longitude', 'Latitude'])
df_schools_clustering = df_schools_clustering[df_schools_clustering['SPI'] < df_schools_clustering['SPI'].quantile(0.25)]
df_schools_clustering = df_schools_clustering[df_schools_clustering['Economic Need Index'] > 0.8]
print ("The number of schools with low SPI and high ENI is : " + str(len(df_schools_clustering)))
model = KMeans(n_clusters=4)
model.fit(df_schools_clustering[['Longitude', 'Latitude']].values)
model.cluster_centers_
color = 'blue'
school_map = folium.Map([model.cluster_centers_[:, 1].mean(), model.cluster_centers_[:, 0].mean()], 
                        zoom_start=11,
                        tiles='Stamen Terrain') 
for row in model.cluster_centers_:
    folium.Marker([row[1], row[0]], icon=folium.Icon(color='red')).add_to(school_map)
for index in df_schools_clustering.index:
    row = df_schools_clustering.loc[index]
#     if row['Economic Need Index'] > 0.8 and row['SPI'] < 243.55:
    popup_text = "Economic Need Index : " + str(round(row['Economic Need Index'], 3)) + ' , SPI : ' + str(round(row['SPI'], 3))
    folium.Marker([row['Latitude'], row['Longitude']], popup=popup_text, icon=folium.Icon(color='blue')).add_to(school_map)

school_map
shsat_df = pd.read_csv('../input/data-science-for-good//D5 SHSAT Registrations and Testers.csv')
shsat_df.head()
shsat_df['Took Percentage'] = shsat_df['Number of students who took the SHSAT'] / shsat_df['Number of students who registered for the SHSAT']
shsat_sorted_df = shsat_df.groupby(['DBN','School name'], as_index=False).sum()
shsat_sorted_df.head()
shsat_sorted_df = shsat_df.groupby(['DBN','School name'], as_index=False).sum()
shsat_sorted_df = shsat_sorted_df[['DBN','School name', 
                                 'Enrollment on 10/31', 
                                 'Number of students who registered for the SHSAT', 
                                 'Number of students who took the SHSAT']].join(shsat_df[['Year of SHST', 
                                                                                          'Grade level']], how='inner')
shsat_sorted_df = shsat_sorted_df.sort_values('Number of students who registered for the SHSAT',ascending=False)
shsat_sorted_df.head()
shsat_sorted_df['Took Percentage'] = shsat_sorted_df['Number of students who took the SHSAT'] / shsat_sorted_df['Number of students who registered for the SHSAT']
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="Number of students who registered for the SHSAT", y='School name', data=shsat_sorted_df,
            label="# of SHSAT Registrations", color="b", ci=None)

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="Number of students who took the SHSAT", y="School name", data=shsat_sorted_df,
            label="# of Students who Took SHSAT", color="b",ci=None)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 300), ylabel="School Name", title='SHSAT School Registration Distribution',
       xlabel="# of Registrations")
sns.despine(left=True, bottom=True)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
sns.barplot(y='School name', x='Took Percentage', data=shsat_sorted_df.sort_values('Took Percentage', ascending = False), ax=ax, orient='h', ci=None, color='b')
shsat_low_appearance_df = shsat_sorted_df[shsat_sorted_df['Took Percentage'] <= 0.4]
shsat_low_appearance_df.sort_values('Took Percentage')
poor_schools_shsat_performance = list(np.unique(shsat_sorted_df['DBN']))
# poor_schools_shsat_performance = [i.strip().lower() for i in poor_schools_shsat_performance]

total_schools = list(set(df_schools['Location Code']))
# total_schools = [i.strip().lower() for i in total_schools]

intersection = list(set(poor_schools_shsat_performance).intersection(set(total_schools)))

print('Number of Intersecting schools = {}'.format(len(intersection)))
percent_black_hist = []
for school in intersection:
    percent_black_hist.append(list(df_schools[df_schools['Location Code'] == school]['Percent Black / Hispanic'])[0])
  
plt.figure(figsize=(10,6))
sns.distplot(percent_black_hist, kde=False)

temp = plt.title('Percent Black / Hispanic', fontsize=15)
temp = plt.xlabel(field_name, fontsize=15)
df_schools_minor = df_schools_raw[df_schools_raw['Percent Black / Hispanic'].astype(int)>70]
print(df_schools_minor.shape)
def grade_minority_percent_4s (df, grade, subject):
    out_field = ('Grade %d %s Minority 4s')%(grade, subject)
    num1 = ('Grade %d %s 4s - Black or African American')%(grade, subject)
    num2 = ('Grade %d %s 4s - Hispanic or Latino')%(grade, subject)
    den = ('Grade %d %s 4s - All Students')%(grade, subject)
    df = df[df[den].astype(int)>5]
    df[out_field] = (df[num1] + df[num2])/(df[den])
    grade_minority_df = df[df[out_field]>0.7]
    print(grade, subject, df.shape, grade_minority_df.shape, (grade_minority_df[num1] + grade_minority_df[num2]).mean(), df[num1].mean(), df[num2].mean())
    return df

grades = [5,6,7]
subjects = ['ELA', 'Math']
i=1
plt.figure(figsize=[19,10])
df_all_schools = pd.DataFrame(columns=df_schools_minor.columns)
for grade,subject in itertools.product(grades, subjects):
    df_schools_tmp = grade_minority_percent_4s(df_schools_minor, grade, subject)
    
#     df_schools_tmp[]
    fname = ('Grade %d %s Minority 4s')%(grade, subject)
    plt.subplot(3,2,i)
    df_schools_tmp[fname].hist()
    df_schools_tmp.drop([fname], axis=1, inplace=True)
    plt.title(fname)
    df_all_schools = pd.concat([df_all_schools, df_schools_tmp])
    i += 1
def download_link(df, filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    title = "Download CSV file"
    html = '<button type="button" style="font-size: larger;  background-color: #FFFFFF; border: 0pt;"><a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a></button>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
dfn = df_all_schools.drop_duplicates()
print(dfn.shape)
download_link(dfn, 'top_performing_students.csv')