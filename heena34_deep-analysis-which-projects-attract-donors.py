import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from numpy import array
from matplotlib import cm
import missingno as msno
from wordcloud import WordCloud, STOPWORDS

from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False,\
                       parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False,\
                        warn_bad_lines=False)
# Merging projects and donations dataframe
project_donations = pd.merge(projects, donations, on='Project ID')
print('Missing values in Projects DataFrame')
msno.matrix(projects)
print('Missing values in Donors DataFrame')
msno.matrix(donors)
print('Missing values in Resources DataFrame')
msno.matrix(resources)
print('Missing values in Schools DataFrame')
msno.matrix(schools)
print('Missing values in donations DataFrame')
msno.matrix(donations)
print('Missing values in teachers DataFrame')
msno.matrix(teachers)
fig = plt.Figure(figsize=(12,12))
sns.distplot(donations['Donation Amount'].dropna())
plt.xlabel('Amount', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Amount Spent')
plt.show()
teachers_donor_id = donors[donors['Donor Is Teacher'] == 'Yes']['Donor ID']
amount_teachers_donated = donations[donations['Donor ID']\
                                    .isin(teachers_donor_id)]['Donation Amount']\
                                    .sum()
    
total_donation_amount = donations['Donation Amount'].dropna().sum()
teacher_or_not = ['Teachers', 'All']
amount = [amount_teachers_donated, total_donation_amount]

sns.barplot(teacher_or_not, amount)
        
plt.xlabel('People', fontsize=18)
plt.ylabel('Amount', fontsize=18)
plt.title('Amount Donated by Teachers v/s Total', fontsize=20)
plt.show()
fig, ax = plt.subplots(1,2, figsize=(25,12))

donor_states = donors['Donor State'].dropna().value_counts()\
               .sort_values(ascending=False).head(20)
    
sns.barplot(donor_states.values, donor_states.index, ax=ax[0])
for index, value in enumerate(donor_states.values):
        ax[0].text(0.8, index, value, color='k', fontsize=12)
        
ax[0].set_xlabel('Donors', fontsize=18)
ax[0].set_ylabel('States', fontsize=18)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=15)
ax[0].set_title('Donors from different States', fontsize=20)



donor_cities = donors['Donor City'].dropna().value_counts()\
               .sort_values(ascending=False).head(20)
    
sns.barplot(donor_cities.values, donor_cities.index, ax=ax[1])
for index, value in enumerate(donor_cities.values):
        ax[1].text(0.8, index, value, color='k', fontsize=12)
        
ax[1].set_xlabel('Donors', fontsize=18)
ax[1].set_ylabel('City', fontsize=18)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=15)
ax[1].set_title('Donors from dfferent cities', fontsize=20)
plt.show()
donations['Donation Included Optional Donation'].value_counts().plot.bar()
plt.title('Optional Donation Yes/No ?', fontsize=15)
plt.xlabel('Yes/No', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.show()
project_posted_date = pd.to_datetime(project_donations['Project Posted Date'])
donation_receive_date = pd.to_datetime(project_donations['Donation Received Date'])
days = (donation_receive_date - project_posted_date).dt.days
days.describe()
sns.boxplot(days)
plt.xlabel('Days', fontsize=14)
plt.title('Days taken to receive first donation', fontsize=16)
plt.show()
project_current_status = projects['Project Current Status'].value_counts()\
                        [['Fully Funded', 'Expired', 'Live']]
    
plt.pie(project_current_status.values, labels=list(project_current_status.index),\
        autopct='%1.1f%%', shadow=True)

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Current Status of Projects', fontsize=20)

plt.show()
categories = ['Literacy & Language', 'Literacy & Language, Math & Science', 'Math & Science',
              'Music & The Arts', 'Literacy & Language, Special Needs', 'Applied Learning',
              'Applied Learning, Literacy & Language']

donors = ['Donor1', 'Donor2', 'Donor3', 'Donor4', 'Donor5', 'Donor6', 'Donor7', 'Donor8',
          'Donor9', 'Donor10', 'Donor11', 'Donor12', 'Donor13', 'Donor14', 'Donor15',
          'Donor16', 'Donor17', 'Donor18', 'Donor19', 'Donor20']

donation_ids = donations['Donor ID'].value_counts().head(20)
only_donation_ids = donation_ids.index
df = {}
df['categories'] = []
df['donors'] = []
df['count'] = []
for i, id in enumerate(only_donation_ids):
    project_category = project_donations[project_donations['Donor ID'] == id]\
                        ['Project Subject Category Tree']
    for category in categories:
        try:
            df['count'].append(project_category.str.replace('"', '')\
                               .str.replace("'", '')\
                               .str.lstrip().value_counts()[category])
            
            df['categories'].append(category)
            df['donors'].append(donors[i])
        except:
            df['count'].append(0)
            df['categories'].append(category)
            df['donors'].append(donors[i])
    
fig, ax = plt.subplots(1, 2, figsize=(25, 14))

donation_ids = donations['Donor ID'].value_counts().head(20)
sns.barplot(donation_ids.values, donors, ax=ax[1])

ax[1].set_ylabel('Donors', fontsize=20)
ax[1].set_xlabel('Number of times Donor Donated', fontsize=20)
ax[1].set_title('Number of times Donations made by Donor', fontsize=25)
ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=20)
ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=20)

for index, value in enumerate(donation_ids):
        ax[1].text(0.8, index, str(value).strip("[]"), color='k', fontsize=12)


        
df = pd.DataFrame(df)
data_df = df.pivot('categories', 'donors', 'count')
sns.heatmap(data_df, cmap='YlGnBu', fmt='2.0f', linewidths=.5, ax=ax[0])
ax[0].set_ylabel('Categories', fontsize=20)

ax[0].set_xlabel('Donors', fontsize=20)
ax[0].set_title('Subject Categories for which Donors donated mostly!', fontsize=25)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=20)
ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=20)


plt.show()
print('Mapping of Donors with their Donor ID')
pd.DataFrame({'Donors': donors, 'Donor ID': donation_ids.index})
list(projects['Project Subject Category Tree'].unique()[0:10])
project_category = projects['Project Subject Category Tree'].str.replace('"', '')\
                    .str.replace("'", '')\
                    .str.lstrip().value_counts().head(30)
        
project_subcategory = projects['Project Subject Subcategory Tree']\
                    .value_counts().head(30)
    
project_grade_level_category = projects['Project Grade Level Category']\
                                .value_counts().head(5)

f1 = plt.figure(1, figsize=(25,25))
ax1 = plt.subplot2grid((2, 2), (0, 0))
sns.barplot(project_category.values, project_category.index, ax=ax1)
ax1.set_xlabel('Count', fontsize=23)
ax1.set_ylabel('Category', fontsize=23)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=20)
ax1.set_title('What are Project Categories?', fontsize=25)


f2 = plt.figure(1, figsize=(25,25))
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.barplot(project_subcategory.values, project_subcategory.index, ax=ax2)
ax2.set_xlabel('Count', fontsize=23)
ax2.set_ylabel('Subcategory', fontsize=23)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=20)
ax2.set_title('What are Project Sub Categories?', fontsize=25)


f3 = plt.figure(1, figsize=(25,25))
ax3 = plt.subplot2grid((2, 2), (0, 1))
wc = WordCloud(background_color="white", max_words=500, 
               stopwords=STOPWORDS, width=1000, height=1000)
wc.generate(" ".join(projects['Project Resource Category'].dropna()))

ax3.imshow(wc)
ax3.axis('off')
ax3.set_title('Resource Category', fontsize=25)


f2 = plt.figure(1, figsize=(25,25))
ax4 = plt.subplot2grid((2, 2), (1, 1))
patches, texts, autotexts = ax4.pie(project_grade_level_category.values,\
                                    labels=list(project_grade_level_category.index),\
                                    autopct='%1.1f%%', shadow=True)

[ _.set_fontsize(20) for _ in texts]

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Grade Level Category', fontsize=25)

plt.show()
project_types = projects['Project Type'].value_counts().head(3)
fig = plt.figure(figsize=(8, 5))
sns.barplot(project_types.index, project_types.values, palette="Blues_d")
plt.xlabel('Project Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Types of Projects', fontsize=16)
plt.show()
resource_vendors = resources['Resource Vendor Name'].value_counts().head(15)

fig = plt.figure(figsize=(8,8))
sns.barplot(resource_vendors.values, resource_vendors.index)
plt.xlabel('Count', fontsize=15)
plt.ylabel('Vendors', fontsize=15)
plt.title('Who are Resource Vendors?', fontsize=18)
plt.show()
school_metro_type = schools['School Metro Type'].value_counts().drop('unknown')
fig = plt.figure(figsize=(8,8))
plt.pie(school_metro_type.values, labels = school_metro_type.index, autopct='%1.1f%%')

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('Metero Type of Schools', fontsize=20)
plt.show()
schools_give_lunch = schools[['School Percentage Free Lunch', 'School State']]
schools_give_lunch.groupby('School State')['School Percentage Free Lunch']\
                  .describe().sort_values(by='mean', ascending=False).head(5)
f, ax = plt.subplots(1, 2, figsize=(15, 8))


# Projects posted every year
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
num_projects_posted = teachers.groupby(teachers['Teacher First Project Posted Date']\
                                       .dt.year)\
                                       .count()['Teacher First Project Posted Date']

ax[0].plot(num_projects_posted)
ax[0].set_xlabel('Years', fontsize=14)
ax[0].set_ylabel('Count', fontsize=14)
ax[0].set_title('Projects posted by Teachers as per Year', fontsize=16)



# Projects posted in 2018
new_index = ['0', 'Jan', 'Feb', 'March', 'April', 'May']
num_projects_posted_2018 = teachers.groupby(teachers[teachers['Teacher First Project Posted Date']\
                                            .dt.year == 2018]\
                                            ['Teacher First Project Posted Date']\
                                           .dt.month)\
                                           .count()['Teacher ID']

ax[1].plot(num_projects_posted_2018)
ax[1].set_xlabel('Months', fontsize=14)
ax[1].set_ylabel('Count', fontsize=14)
ax[1].set_xticks(np.arange(6))
labels = [new_index[i] for i, item in enumerate(ax[1].get_xticklabels())]

ax[1].set_xticklabels(labels)

ax[1].set_title('Projects posted by Teachers as per months for year 2018', fontsize=16)
plt.show()