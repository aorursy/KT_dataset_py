# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# get current working directory
os.getcwd()
# get approximate file sizes 
ten6 = 1000000
dict_filesize = {}

dict_filesize["Resources"] = round(os.path.getsize('../input/Resources.csv')/ten6, 2)
dict_filesize["Schools"] = round(os.path.getsize('../input/Schools.csv')/ten6, 2)
dict_filesize["Donors"] = round(os.path.getsize('../input/Donors.csv')/ten6, 2)
dict_filesize["Donations"] = round(os.path.getsize('../input/Donations.csv')/ten6, 2)
dict_filesize["Teachers"] = round(os.path.getsize('../input/Teachers.csv')/ten6, 2)
dict_filesize["Projects"] = round(os.path.getsize('../input/Projects.csv')/ten6, 2)

# display dict with filesize
dict_filesize

# Largest file is Projects.csv, which is around 2GB 
# load files using pandas

# schools = pd.read_csv('../input/Donors.csv')

# received the following error while running the above code
# /opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2698: DtypeWarning: Columns (4) have mixed types. 
# Specify dtype option on import or set low_memory=False. interactivity=interactivity, compiler=compiler, result=result)

schools = pd.read_csv('../input/Schools.csv', low_memory=False, skiprows=[59987])
teachers = pd.read_csv('../input/Teachers.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)


schools.head(5)
schools.tail(5)
print(schools.loc[[59987]])
# print(schools.loc[[59988]])

# one additional column is present in row no. 59987
# this can be fixed or we can skip and proceed for time being

teachers.head(5)
teachers.tail(5)
donors.head(5)
donors.tail(5)
# reading the remaining csv's

donations = pd.read_csv('../input/Donations.csv')

donations.head(5)
donations.tail(5)
# resources = pd.read_csv('../input/Resources.csv', skiprows=[1171,3431,5228,6492,7529,8885,11086,11530], warn_bad_lines=False, error_bad_lines=False)
resources = pd.read_csv('../input/Resources.csv', warn_bad_lines=False, error_bad_lines=False)

# [PENDING] need to handle these skipped rows 
resources.head(5)
resources.tail(5)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False)

# [PENDING] need to handle these skipped rows 
projects.head(5)
projects.tail(5)
# saving the metadata of the files
# count the number of rows, columns in schools
schools_rows = schools.shape[0]
schools_cols = schools.shape[1]
teachers_rows = teachers.shape[0]
teachers_cols = teachers.shape[1]
donors_rows = donors.shape[0]
donors_cols = donors.shape[1]
donations_rows = donations.shape[0]
donations_cols = donations.shape[1]
resources_rows = resources.shape[0]
resources_cols = resources.shape[1]
projects_rows = projects.shape[0]
projects_cols = projects.shape[1]

dict_schools = {'name': 'Schools', 'rows_count': schools_rows, 'cols_count': schools_cols, 'cols': list(schools), 'size': dict_filesize['Schools']}
dict_teachers = {'name': 'Teachers', 'rows_count': teachers_rows, 'cols_count': teachers_cols, 'cols': list(teachers), 'size': dict_filesize['Teachers']}
dict_donors = {'name': 'Donors', 'rows_count': donors_rows, 'cols_count': donors_cols, 'cols': list(donors), 'size': dict_filesize['Donors']}
dict_donations = {'name': 'Donations', 'rows_count': donations_rows, 'cols_count': donations_cols, 'cols': list(donations), 'size': dict_filesize['Donations']}
dict_resources = {'name': 'Resources', 'rows_count': resources_rows, 'cols_count': resources_cols, 'cols': list(resources), 'size': dict_filesize['Resources']}
dict_projects = {'name': 'Projects', 'rows_count': projects_rows, 'cols_count': projects_cols, 'cols': list(projects), 'size': dict_filesize['Projects']}


# dictionary of schools details
dict_metadata = {'Schools': dict_schools, 'Teachers': dict_teachers, 'Donors': dict_donors, 'Donations': dict_donations, 'Resources': dict_resources, 'Projects': dict_projects}

# display metadata
dict_metadata
# rows, column counts for each file
print("Type \t   Columns \t Rows")
for k,v in dict_metadata.items():
    print(k, "\t", dict_metadata[k]['cols_count'], "\t", dict_metadata[k]['rows_count'])
# Relation between Donor City/State and School/Project City/State needs to be found
# Donor ID, Donation ID and Project ID in Donations
# Donor City, Donor State from Donors based on Donor ID in Donations
# School ID from Schools based on Project ID in Projects
# School City, School State based on School ID from Schools

# merging donations and donors data frames
donations_donors = donations.merge(donors, on='Donor ID', how='inner')
donations_donors.head(5)
# merging projects and schools
projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.head(5)
total = projects_schools.merge(donations_donors, on='Project ID', how='inner')
total.head(5)
# adding a new column 'Same City' to find whether the donor city and school city are the same
# to find if there's a relation between donor city and school city
# if same city then value is set to 1
total.loc[total['School City'] == total['Donor City'], 'Same City'] = 1
# if not the same city then value is set to 0
total.loc[total['School City'] != total['Donor City'], 'Same City'] = 0
total.head(5)
total.groupby('Same City').size()
# approximately 25% of the donors are from the same city as the school
# adding a new column 'Same State' to find whether the donor state and school state are the same
# to find if there's a relation between donor state and school state
# if same state then value is set to 1
total.loc[total['School State'] == total['Donor State'], 'Same State'] = 1
# if not same state then value is set to 0
total.loc[total['School State'] != total['Donor State'], 'Same State'] = 0
total.tail(5)
total.groupby('Same State').size()
# chances of donors donating are more than double if donor and school belongs to the same state
# approximately 65% of the donations are from donors who belongs to the same state
# [WORKING] relation between project cost and donation amount
type(total['Project Cost'])
total.dtypes
# donation amount is of type float
# whereas project cost is an object
# as it's in the form $45
# we can create a new column for project cost
# without the dollar symbol
# total.drop('column name', axis=1, inplace=True)
total['Project Cost'].unique()
total['ProjCost'] = total['Project Cost'].apply(lambda x: float(str(x).replace('$','').replace(',','')))
total['ProjCost'].unique()
type(total['ProjCost'][5])
len(total['ProjCost'])
donations.dtypes
total['ProjCost'][5]
total['ProjCost'][445]
total.tail()
total['Donation Amount'].max()
total['ProjCost'].max()
import matplotlib.pyplot as plt
plt.scatter(total['ProjCost'], total['Donation Amount'])
plt.show() # Depending on whether you use IPython or interactive mode, etc.
# p1 - sum of donations by project id
p1 = total.groupby('Project ID')['Donation Amount'].sum()
p2 = total.groupby('Project ID', as_index=False)['Donation Amount'].sum()
p1.count()
# converting series to frame
p1f = p1.to_frame()
# not sure why Project ID and Donation Amount are on separate rows
p1f.head()
p1f.count()
p2
p3 = total.groupby(['Project ID','ProjCost'], as_index=False)['Donation Amount'].sum().sort_values(by='ProjCost',ascending=False)
p3.head()
p3.head().plot(x='Donation Amount', y='ProjCost', style='o')
plt.rcParams['agg.path.chunksize'] = 100000
p3.plot()
p3.head()
import seaborn as sns
sns.jointplot(x='ProjCost',y='Donation Amount',data=p3)
plt.title("Project Cost vs Donation Amount - by Project", loc='center')
p3.plot(x='Project ID', y='ProjCost' ,figsize=(12,8), grid=True, label="Project Cost", color="red") 
p3.plot(x='Project ID', y='Donation Amount' ,figsize=(12,8), grid=True, label="Donation Amount", color="blue")
p3[:500].set_index('Project ID').plot(figsize=(20,10), grid=True, alpha=0.45) 
list(p3)
p3.describe
p3['Donation Percentage'] = round((p3['Donation Amount']/p3['ProjCost'])*100,2)
p3
sns.jointplot(x='ProjCost',y='Donation Percentage',data=p3)
