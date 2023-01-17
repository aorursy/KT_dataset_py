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
dtypes_donations = {'Project ID' : 'object','Donation ID':'object','device': 'object','Donor ID' : 'object','Donation Included Optional Donation': 'object','Donation Amount': 'float64','Donor Cart Sequence' : 'int64'}
donations = pd.read_csv('../input/Donations.csv',dtype=dtypes_donations)

dtypes_donors = {'Donor ID' : 'object','Donor City': 'object','Donor State': 'object','Donor Is Teacher' : 'object','Donor Zip':'object'}
donors = pd.read_csv('../input/Donors.csv', low_memory=False,dtype=dtypes_donors)

dtypes_schools = {'School ID':'object','School Name':'object','School Metro Type':'object','School Percentage Free Lunch':'float64','School State':'object','School Zip':'int64','School City':'object','School County':'object','School District':'object'}
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False,dtype=dtypes_schools)

dtypes_teachers = {'Teacher ID':'object','Teacher Prefix':'object','Teacher First Project Posted Date':'object'}
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False,dtype=dtypes_teachers)
                   
dtypes_projects = {'Project ID' : 'object','School ID' : 'object','Teacher ID': 'object','Teacher Project Posted Sequence':'int64','Project Type': 'object','Project Title':'object','Project Essay':'object','Project Subject Category Tree':'object','Project Subject Subcategory Tree':'object','Project Grade Level Category':'object','Project Resource Category':'object','Project Cost':'object','Project Posted Date':'object','Project Current Status':'object','Project Fully Funded Date':'object'}
projects = pd.read_csv('../input/Projects.csv',parse_dates=['Project Posted Date','Project Fully Funded Date'], error_bad_lines=False, warn_bad_lines=False,dtype=dtypes_projects)

dtypes_resources = {'Project ID' : 'object','Resource Item Name' : 'object','Resource Quantity': 'float64','Resource Unit Price' : 'float64','Resource Vendor Name': 'object'}
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False,dtype=dtypes_resources)
donations.head()
donors.head()
teachers.head()
schools.head()
projects.head()
resources.head()
