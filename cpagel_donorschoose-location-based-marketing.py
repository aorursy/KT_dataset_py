import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
#from wordcloud import WordCloud, STOPWORDS
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
#plt.rcParams["figure.figsize"] = [16, 12]
#import chardet
from subprocess import check_output
data_directory = '../input/'

#print(check_output(["ls", "../input/"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", data_directory]).decode("utf8").strip().split('\r\n')
# helpful character encoding module
#filenames = filenames.split('\n')
#dfs = dict()
#for f in  filenames:
#    dfs[f[:-4]] = pd.read_csv(data_directory+ f)
for filename in filenames:
    print(filename)
donations = pd.read_csv(data_directory + 'Donations.csv')
donors = pd.read_csv(data_directory + 'Donors.csv')
projects = pd.read_csv(data_directory + 'Projects.csv', error_bad_lines=False, warn_bad_lines=False)
#Issue with project as of 5/7/2018 - ParserError: Error tokenizing data. C error: Expected 15 fields in line 10, saw 18
#May have some strange characters within the free text.
#Does not affect most of the rows
resources = pd.read_csv(data_directory + 'Resources.csv', error_bad_lines=False, warn_bad_lines=False)
#Error in rsources - ParserError: Error tokenizing data. C error: Expected 5 fields in line 1172, saw 8
schools = pd.read_csv(data_directory + 'Schools.csv', error_bad_lines=False, warn_bad_lines=False)
#ParserError: Error tokenizing data. C error: Expected 9 fields in line 59988, saw 10
#teachers = pd.read_csv(data_directory + 'Teachers.csv')
plt.hist(projects['Project ID'].map(str).apply(len))
projects_cleansed = projects.loc[(projects['Project ID'].map(str).apply(len) == 32)
                                & (projects['School ID'].map(str).apply(len) == 32)
                                & (projects['Teacher ID'].map(str).apply(len) == 32)
                                ]
#projects_cleansed.count
teacher_projects = pd.DataFrame(projects_cleansed[['Teacher ID','Project ID']].groupby(['Teacher ID']).count())
teacher_projects = teacher_projects.rename(columns={'Project ID':'project_count'})
teacher_projects[['project_count']] = teacher_projects[['project_count']].apply(pd.to_numeric)
frq, edges = np.histogram(teacher_projects['project_count'],bins = 250)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xlabel('Number of Projects')
plt.ylabel('Number of Teachers')
plt.title('Do teachers have more than one project?')
plt.show()
frq, edges = np.histogram(teacher_projects.query('project_count <= 5'),bins = 5)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xticks(range(1, 6))
plt.xlabel('Number of Projects')
plt.ylabel('Number of Teachers')
plt.title('Zoomed teachers have more than one project?')
plt.show()
donors.sample
schools.sample
donations.sample
donor_teacher = donations.merge(projects_cleansed,left_on='Project ID',right_on='Project ID', how='inner')
donor_teacher.sample
donors_same_teacher = pd.DataFrame(donor_teacher[['Teacher ID','Donor ID','Project ID']].groupby(['Teacher ID','Donor ID']).count())
donors_same_teacher = donors_same_teacher.rename(columns={'Project ID':'donor_teacher_count'})
donors_same_teacher[['donor_teacher_count']] = donors_same_teacher[['donor_teacher_count']].apply(pd.to_numeric)
#donors_same_teacher.sample
frq, edges = np.histogram(donors_same_teacher.query('donor_teacher_count <= 5'),bins = 5)

fig, ax = plt.subplots()
ax.bar(edges[:-1], frq, width=np.diff(edges), ec="k", align="edge")
plt.xticks(range(1, 6))
plt.xlabel('Donations to Same Teacher')
plt.ylabel('Donations')
plt.title('Do donors sponsor the same teacher twice?')
plt.show()
donations_teacher_donor = donor_teacher.merge(donors,left_on='Donor ID',right_on='Donor ID', how='inner')
#donations_teacher_donor.sample
donations_teacher_donor_school = donations_teacher_donor.merge(schools,left_on='School ID',right_on='School ID', how='inner')
#donations_teacher_donor_school.sample
donations_teacher_donor_school = donations_teacher_donor_school.rename(columns={'Donor State':'donor_state'})
donations_teacher_donor_school = donations_teacher_donor_school.rename(columns={'School State':'school_state'})
#donations_teacher_donor_school[donations_teacher_donor_school.donor_state == donations_teacher_donor_school.school_state].sample
#def same_state (row):
#    if [donations_teacher_donor_school.donor_state == donations_teacher_donor_school.school_state]:
#        return 1
#    return 0
#donations_teacher_donor_school['same_state'] = donations_teacher_donor_school.apply (lambda row: same_state (row), axis=1)#
states_grouped = donations_teacher_donor_school.groupby(['donor_state','school_state'],as_index=False)['Donor ID'].count()
states_grouped.sample
pivoted_states = pd.pivot_table(states_grouped, index='donor_state', columns='school_state', values='Donor ID')
pivoted_states.sample
fig, ax = plt.subplots(figsize=(15,15))
#mask = np.zeros_like(pivoted_states)
sns.heatmap(pivoted_states, cbar_kws={'label': 'Number of Donations'}, cmap="Greens", linewidth=.02)
ax.set_title('Donations from Donor States and School States')
plt.show()
location_outliers = pivoted_states.loc[:, ['North Dakota']]
location_outliers.sample
#This takes too long to run!!!
#import seaborn as sns 
#correlation = donations_state['donor_state'].corr(donations_state['school_state'])
#sns.heatmap(correlation)
#plt.matshow(donations_state.corr())
#import plotly.plotly as py
from mpl_toolkits.basemap import Basemap

map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
resolution = 'i', area_thresh = 0.1,
llcrnrlon=-136.25, llcrnrlat=56.0,
urcrnrlon=-134.25, urcrnrlat=58)


map.drawcoastlines()
map.drawcountries()
map.drawmapboundary()

lon = -135.3318
lat = 57.0799
x,y = map(lon, lat)
x2, y2 = map(lon+0.5,lat+0.5)

plt.arrow(x,y,x2-x,y2-y,fc="k", ec="k", linewidth = 4, head_width=10, head_length=10)
plt.show()
