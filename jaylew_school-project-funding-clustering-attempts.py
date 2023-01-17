# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
resources_table = pd.read_csv('../input/Resources.csv', low_memory=False, error_bad_lines=False, warn_bad_lines=False)
schools_table = pd.read_csv('../input/Schools.csv', low_memory=False, error_bad_lines=False, warn_bad_lines=False)
donors_table = pd.read_csv('../input/Donors.csv', low_memory=False)
donations_table = pd.read_csv('../input/Donations.csv', low_memory=False)
teachers_table = pd.read_csv('../input/Teachers.csv', low_memory=False)
projects_table = pd.read_csv('../input/Projects.csv', low_memory=False, error_bad_lines=False, warn_bad_lines=False)
resources_table.info()
schools_table.info()
donors_table.info()
donations_table.info()
teachers_table.info()
projects_table.info()
#Could attempt to use clustering to attribute each donor to a cluster, and then investigate relationships between clusters of donors and 
#different projects based on some feature selection work on the different projects.
print(donors_table['Donor State'].nunique())
print(donors_table['Donor Zip'].nunique())
print(donors_table['Donor City'].nunique())
donors_table['Donor Is Teacher'] = pd.get_dummies(donors_table['Donor Is Teacher'], drop_first=True)

#Realistically city/state and zip will likely be heavily collinear, may just be useful to use state and zip.
#Looks like 'other' and 'District of Columbia' make up the extra two. Check how many times each state is listed.
donors_table['Donor State'].value_counts().plot.bar(width=0.7, color=sns.color_palette('plasma', 53), figsize=(16,8))
plt.xlabel('States')
plt.ylabel('Number of Donors')
plt.title('Donors per State');
#Looks like they're all represented.
#Replace state/zip entries with placeholder integer values for the clustering 
states = donors_table['Donor State'].unique()
zips = donors_table['Donor Zip'].unique()

states_dict = {}
state_count = 1
for state in states:
    states_dict[state] = state_count
    state_count += 1

zips_dict = {}
zip_count = 1
for zipp in zips:
    zips_dict[zipp] = zip_count
    zip_count += 1
    
donors_table['Donor State'].replace(states_dict, inplace=True)
donors_table['Donor Zip'].replace(zips_dict, inplace=True)
#Check and confirm our columns have been changed as expected
donors_table.sample(10)
#Use the elbow method to look for the optimal number of clusters, began with i up to 10 to see if we could see an optimum 
donors_X = donors_table[['Donor State', 'Donor Is Teacher', 'Donor Zip']]
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(donors_X)
    wcss.append(kmeans.inertia_)

#This took several minutes, ran on the Kaggle online kernel
#Plot the elbow method to see if we found the optimal number of clusters
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15,8))
ax1.plot(range(1,11), wcss)
plt.xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method')
ax2.plot(range(1,11), wcss)
ax2.set_ylabel('WCSS - Zoomed')
ax2.set_ylim(0, 0.1e11)
#Going to try with 6 clusters to start
donors_kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10)
donors_cluster = donors_kmeans.fit_predict(donors_X)
donors_table['Cluster'] = donors_cluster
#Alright, we have our donors put into different clusters. 
#Time to look at who donated to what
donations_table.head()
#Going to include the donor's cluster, as well as state/zip/is teacher columns from our donor table.
donations_donors = donations_table.merge(donors_table[['Donor ID', 'Donor State', 'Donor Zip', 'Donor Is Teacher', 'Cluster']], how='left', on='Donor ID')
donations_donors['Donation Included Optional Donation'] = pd.get_dummies(donations_donors['Donation Included Optional Donation'], drop_first=True)
#Now have our donor information matched up to different projects. Now need some analysis of the projects themselves.
donations_donors.head()
projects_table.head(5)
#Changing the project costs into floats, and the dates to datetimes
import regex
projects_table['Project Cost'] = projects_table['Project Cost'].replace('[^.0-9]','', regex=True).astype(float)
projects_table['Project Posted Date'] = pd.to_datetime(projects_table['Project Posted Date'], format='%Y-%m-%d')
projects_table['Project Fully Funded Date'] = pd.to_datetime(projects_table['Project Posted Date'], format='%Y-%m-%d')
#Take a copy of the df to use for visualization 
sample_projects = projects_table
#Look at a few distributions
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,6))
sns.countplot(x='Project Current Status', data=sample_projects, ax=ax1)
ax1.set_title('Distribution of Project Statuses');
sns.countplot(x='Project Type', data=sample_projects, ax=ax2)
ax2.set_title('Distribution of Project Types');
plt.figure(figsize=(15,6))
sns.countplot(x='Project Resource Category', data=sample_projects)
plt.xticks(rotation=90)
plt.title('Distribution of Project Resource Categories');
#Look at the number of projects posted over each of the last 5 years
years = sample_projects['Project Posted Date'].apply(lambda x: x.year).value_counts().reset_index()
plt.figure(figsize=(12,6))
sns.barplot(y='Project Posted Date', x='index', data=years)
plt.xlabel('Year')
plt.title('Projects Posted Each Year')
#Taking a look at the titles and the essays for each project.
import re

def txtclean(txt):
    cleaned = ' '.join(word for word in (re.sub('r^[\w]', '', txt).lower().split()))
    return cleaned

#Cleaning the columns to only include words and numbers
projects_table['Project Title'].fillna(value='No Title', inplace=True)
projects_table['Project Essay'].fillna(value='No Essay', inplace=True)
projects_table['Project Title'] = projects_table['Project Title'].apply(lambda t_txt: txtclean(t_txt))
projects_table['Project Essay'] = projects_table['Project Essay'].apply(lambda ess_txt: txtclean(ess_txt))
#Making word counts for title and essays
projects_table['Title Len'] = projects_table['Project Title'].apply(len)
projects_table['Essay Len'] = projects_table['Project Essay'].apply(len)
#Creating dummies for the different project statuses
funded_dummies = pd.get_dummies(projects_table['Project Current Status'], drop_first=True)
projects_table = pd.concat([projects_table, funded_dummies], axis=1)
#Checking for any large correlations. Of course there will be some between the project status columns.
projects_corr = projects_table[['Project Cost', 'Title Len', 'Essay Len', 'Expired', 'Live', 'Fully Funded']].corr()
plt.figure(figsize=(12,6))
sns.heatmap(projects_corr, annot=True, cbar=False, cmap='coolwarm')

#Nothing very insightful other than a clear negative correlation between higher project costs and fully funded.
#Going to want to change some categories to numeric placeholders, similar to the donors section

#Grade groups
g_groups = projects_table['Project Grade Level Category'].unique()
grades_dict = {}
grade_count = 1
for grade in g_groups:
    grades_dict[grade] = grade_count
    grade_count += 1
    
#Project resource category
project_resources = projects_table['Project Resource Category'].unique()
project_resources_dict = {}
project_resource_count = 1
for pr in project_resources:
    project_resources_dict[pr] = project_resource_count
    project_resource_count += 1
    
#Project type
project_type = projects_table['Project Type'].unique()
project_type_dict = {}
project_type_count = 1
for pt in project_type:
    project_type_dict[pt] = project_type_count
    project_type_count += 1
    
#Project subject category
project_subject = projects_table['Project Subject Category Tree'].unique()
project_subject_dict = {}
project_subject_count = 1
for ps in project_subject:
    project_subject_dict[ps] = project_subject_count
    project_subject_count += 1

#Making the replacements
projects_table['Project Grade Level Category'].replace(grades_dict, inplace=True)
projects_table['Project Resource Category'].replace(project_resources_dict, inplace=True)
projects_table['Project Type'].replace(project_type_dict, inplace=True)
projects_table['Project Subject Category Tree'].replace(project_subject_dict, inplace=True)
projects_table.head()
donations_donors_projects = donations_donors.merge(projects_table[['Project ID','Project Subject Category Tree', 'Project Grade Level Category','Project Resource Category', 'Project Cost', 'Expired', 'Fully Funded', 'Live', 'School ID']], how='left', on='Project ID')

donations_donors_projects.head(5)
plt.figure(figsize=(15,8))
sns.countplot(x='Resource Vendor Name', data=resources_table, log=True,palette=sns.color_palette('viridis', resources_table['Resource Vendor Name'].nunique()))
plt.xticks(rotation=90)
plt.ylabel('Log(Count)')
plt.title('Distribution of Resource Vendors');
#Checking distributions of the number of requested resources
fig, ax = plt.subplots(ncols=5, sharey=True, figsize=(18,6))

fig.add_subplot(111, frameon=False);
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off');
plt.grid(False);
plt.xlabel('Number of Resources');

plt.suptitle('Number of Times Each Amount of Resources was Requested');

resources_table['Resource Quantity'].hist(bins=100, ax=ax[0]);
resources_table[resources_table['Resource Quantity'] < 1000]['Resource Quantity'].hist(bins=100, ax=ax[1]);
resources_table[resources_table['Resource Quantity'] < 100]['Resource Quantity'].hist(bins=100, ax=ax[2]);
resources_table[resources_table['Resource Quantity'] < 20]['Resource Quantity'].hist(bins=100, ax=ax[3]);
sns.barplot(x=['1', '>1', '>5'], y=[len(resources_table[resources_table['Resource Quantity'] == 1]), len(resources_table[resources_table['Resource Quantity'] > 1]), len(resources_table[resources_table['Resource Quantity'] > 5])], ax=ax[4]);
#Could potentially look at the different resource item names, and see if different resource key words correlate to more donations. Come back later.

print('Number of districts: ', schools_table['School District'].nunique())
print('Number of counties: ', schools_table['School County'].nunique())
print('Number of cities: ', schools_table['School City'].nunique())
    
#Replace the states with the same dictionary as used earlier
schools_table['School State'].replace(states_dict, inplace=True)
schools_table.head()

#Make a new column with only the first 3 digits of the zip code.
schools_table['School Zip - First 3'] = schools_table['School Zip'].apply(lambda n: str(n)[:3])
schools_table['School Zip - First 3'].replace(zips_dict, inplace=True)
schools_table.head()
plt.figure(figsize=(8,8))
schools_table['School Metro Type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('School Metro Types')
plt.ylabel('');
#Replace the metro types with numeric placeholders
metro_type = schools_table['School Metro Type'].unique()
metro_type_dict = {}
metro_type_count = 1
for metro in metro_type:
    metro_type_dict[metro] = metro_type_count
    metro_type_count += 1
schools_table['School Metro Type'].replace(metro_type_dict, inplace=True)
#Combine the school info with the donation/project/donor dataframe being built
donations_donors_projects_schools = donations_donors_projects.merge(schools_table[['School ID', 'School State', 'School Zip - First 3', 'School Percentage Free Lunch']], how='left', on='School ID')
donations_donors_projects_schools.head()
#Found that the school zip column was not a numeric column. Need to change that to compare to donor zips.
donations_donors_projects_schools['School Zip - First 3'] = pd.to_numeric(donations_donors_projects_schools['School Zip - First 3'])
projects_X = donations_donors_projects_schools[['Project Subject Category Tree', 'Project Grade Level Category', 'Project Resource Category', 'Project Cost', 'School State', 'School Zip - First 3']]
import time
#Attempting to perform similar clustering on the projects in the new dataframe
kmeans_start = time.time()
projects_wcss = []
for i in range(1,11):
    projects_kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    projects_kmeans.fit(projects_X)
    projects_wcss.append(projects_kmeans.inertia_)
print('Runtime: ', time.time() - kmeans_start)

#Runtime with 1-10: 1200-1400 seconds
print(projects_wcss)
#These values were extracted from a run of the above loop. The loop takes ~25 minutes to run. Wanted to store these for future use instead of running it again.
projects_wcss = [56738911547246.445, 17852990484691.875, 8926008180869.674, 5175261329923.946, 3527219179295.5073, 2255893077764.033, 1703757453632.341, 1309405008159.0378, 1062613550194.3402, 842506624394.2274]
#Plot the elbow method to see if we found the optimal number of clusters
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(16,10))
ax1.plot(range(1,11), projects_wcss)
plt.xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method')
ax2.plot(range(1,11), projects_wcss)
ax2.set_ylabel('WCSS - Zoomed')
ax2.set_ylim(0, 0.5e13)
#Looked like 8 clusters may have been optimal in the 1-10 range, but going to check 11-20 as well. Still seeing relatively high drops in WCSS.
kmeans_start = time.time()
for i in range(11,21):
    projects_kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    projects_kmeans.fit(projects_X)
    projects_wcss.append(projects_kmeans.inertia_)
print('Runtime: ', time.time() - kmeans_start)
#Run time with 11-20: 4000 seconds
projects_wcss = [56738911547246.445,
 17852990484691.875,
 8926008180869.674,
 5175261329923.946,
 3527219179295.5073,
 2255893077764.033,
 1703757453632.341,
 1309405008159.0378,
 1062613550194.3402,
 842506624394.2274,
 715401322475.9066,
 610572258514.3153,
 522607926942.7026,
 455400318429.33746,
 396530116672.5825,
 361653975503.06696,
 333126873840.38727,
 314647433228.24744,
 286748074483.87573,
 267010640600.80746]
fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(18,10))
ax[0].plot(range(1,21), projects_wcss)
plt.xlabel('Number of Clusters')
ax[0].set_ylabel('WCSS')
ax[0].set_title('Elbow Method')
ax[1].plot(range(1,21), projects_wcss)
ax[1].set_ylabel('WCSS - Zoomed')
ax[1].set_ylim(0, 0.5e13)
ax[2].plot(range(1,21), projects_wcss)
ax[2].set_ylabel('WCSS - Zoomed on Y and X')
ax[2].set_ylim(0, 1e12)
ax[2].set_xlim(10,20);
#Between 10 clusters and 15 the wcss value looks to drop to ~1/2. Drops after that are much slower. Going to say 15 is optimal for the projects.
k_start=time.time()
projects_kmeans = KMeans(n_clusters=15, init='k-means++', max_iter=300, n_init=10)
print('KMeans took: ', time.time() - k_start, ' seconds')
fit_start=time.time()
projects_cluster = projects_kmeans.fit_predict(projects_X)
print('KMeans fit and predict took: ', time.time() - fit_start, ' seconds')
donations_donors_projects_schools['Projects - Cluster'] = projects_cluster

donors_grouping = donations_donors_projects_schools.groupby('Donor ID')
donors_grouping.describe()


import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#Since we already cleaned the data up a bit, do not need to worry about upper/lower/punctuations
stemmer = SnowballStemmer(language='english', ignore_stopwords=True)
lemmer = WordNetLemmatizer() #This could be cleaner, but harder to implement.

stop_word_set = set(stopwords.words('english'))

def corpus_create(txt):
    cleaned = ' '.join([stemmer.stem(word) for word in txt.split() if not word in stop_word_set])
    return cleaned

#Titles took ~70 seconds
start= time.time()
projects_table['Title Corpus'] = projects_table['Project Title'].apply(lambda title: corpus_create(title))
print('Runtime: ', time.time()-start)
#Essays took ~50 minutes
start= time.time()
projects_table['Essay Corpus'] = projects_table['Project Essay'].apply(lambda essay: corpus_create(essay))
print('Runtime: ', time.time()-start)
#####################Come back to the text processing########################
