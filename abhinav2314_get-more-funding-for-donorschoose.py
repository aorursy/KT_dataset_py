import numpy as np 
import pandas as pd 
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
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import cufflinks as cf
cf.go_offline()
from sklearn import preprocessing
import missingno as msno # to view missing values
import os
print(os.listdir("../input"))
teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv')
resources = pd.read_csv('../input/Resources.csv')
projects.head()
projects.info()
print(projects.isnull().sum())
msno.matrix(projects)
plt.show()
# how many total missing values do we have?
missing_values_count = projects.isnull().sum()
total_cells = np.product(projects.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Projects Data:')
print((total_missing/total_cells) * 100, "%")
donations.head()
donations.info()
print(donations.isnull().sum())

donors.head()
donors.info()
#donors.describe()
print('Missing Data Overview in Donors Table')
print(donors.isnull().sum())
msno.matrix(donors)
plt.title('Missing Data in Donors Table')
plt.show()
# how many total missing values do we have?
missing_values_count = donors.isnull().sum()
total_cells = np.product(donors.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Donors Table:')
print((total_missing/total_cells) * 100, "%")
teachers.head()
teachers.info()
print('Missing Data Overview in Teachers Table')
print(teachers.isnull().sum())
msno.matrix(teachers)
plt.title('Missing Data in Teachers Table')
plt.show()
schools.head()
schools.info()
print('Missing Data Overview in Schools Table')
print(schools.isnull().sum())
msno.matrix(schools)
plt.title('Missing Data in Schools Table')
plt.show()
resources.head()
resources.info()
print('Missing Data Overview in Resources Table')
print(resources.isnull().sum())
msno.matrix(resources)
plt.title('Missing Data in Resources Table')
plt.show()
project_subject_category = projects['Project Subject Category Tree'].value_counts().head(10)
project_subject_category.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project Subject Categories')
project_subject_sub_category = projects['Project Subject Subcategory Tree'].value_counts().head(10)
project_subject_sub_category.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project Subject Sub-Categories')
donor_cities = donors['Donor City'].value_counts().head(10)
donor_cities.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Donor Cities')
donor_states = donors['Donor State'].value_counts().head(10)
donor_states.iplot(kind='bar', xTitle = 'Donor State', yTitle = "Count", title = 'Donor States')
school_states = schools['School State'].value_counts().head(50)
school_states.iplot(kind='bar', xTitle = 'School State', yTitle = "Count", title = 'School States')
projects['Project Current Status'].describe()
projects['Project Current Status'].unique()
#fully_funded_projects.describe()
#projects[projects['Project Current Status'] == "Fully Funded"].head()
projects[projects['Project Current Status'] == "Fully Funded"].head(10)
projects['Project Title'].describe()
project_titles = projects['Project Title'].value_counts().head(50)
project_titles.iplot(kind='bar', xTitle = 'Project Titile', yTitle = "Count", title = 'Project Title')
#How to skip currency symbol and convert to numeric type
#df1['Avg_Annual'] = df1['Avg_Annual'].str.replace(',', '')
#df1['Avg_Annual'] = df1['Avg_Annual'].str.replace('$', '')
#df1['Avg_Annual'] = df1['Avg_Annual'].convert_objects(convert_numeric=True)

projects['Project Cost'] = projects['Project Cost'].str.replace(',', '')
projects['Project Cost'] = projects['Project Cost'].str.replace('$', '')
projects['Project Cost'] = projects['Project Cost'].convert_objects(convert_numeric=True)

print('Describe Project Cost: ')
print(projects['Project Cost'].describe())
print('View Some Project Cost data')
print(projects['Project Cost'].head())
print('Minimum Project Cost: ', projects['Project Cost'].min())
print('Maximum Project Cost: ', projects['Project Cost'].max())
print('Median Project Cost: ', projects['Project Cost'].median())
print('Total Project Cost: ', projects['Project Cost'].sum())
fully_funded_projects = projects[projects['Project Current Status'] == "Fully Funded"]
print('Total Funded Project Cost: ', fully_funded_projects['Project Cost'].sum())