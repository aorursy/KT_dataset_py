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

# Some more libraries
import time
import datetime

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style(style='darkgrid')

import traceback
# Read the file
sf = pd.read_csv('../input/Building_Permits.csv',low_memory=False)
sf.info()
# Conversion to datetime
import traceback
try :
    sf['Filed Date'] = pd.to_datetime(sf['Filed Date'],errors='coerce')
    sf['Issued Date'] = pd.to_datetime(sf['Issued Date'],errors='coerce')
    sf['Current Status Date'] = pd.to_datetime(sf['Current Status Date'],errors='coerce')
except :    
    traceback.print_exc()

# Keep a copy to reload
sfcpy = sf.copy()
# Sometimes when re-run is required, one can start from just here, to save time
sf = sfcpy.copy()
# Rename for brevity/readability
sf = sf.rename(columns =   {'Neighborhoods - Analysis Boundaries':'neighborhoods',
                            'Permit Type' : 'perm_typ',
                            'Permit Type Definition': 'perm_typ_def',
                            'Filed Date':'file_dt',
                            'Issued Date':'issue_dt',
                            'Permit Expiration Date' : 'perm_exp_dt',
                            'Current Status' : 'cur_st',
                            'Current Status Date' : 'cur_st_dt',
                            'Structural Notification':'strct_notif',
                            'Number of Existing Stories':'no_exist_stry',
                            'Number of Proposed Stories':'no_prop_stry',
                            'Fire Only Permit':'fire_only_permit',
                            'Estimated Cost':'est_cost',
                            'Revised Cost':'rev_cost',
                            'Existing Use':'exist_use',
                            'Proposed Use': 'prop_use',
                            'Plansets':'plansets',
                            'Existing Construction Type': 'exist_const_type',
                            'Existing Construction Type Description': 'exist_const_type_descr',
                            'Proposed Construction Type': 'prop_const_type',
                            'Proposed Construction Type Description': 'prop_const_type_descr',
                            'Site Permit':'site_permit',
                            'Supervisor District':'sup_dist',
                            'Location':'location'
                            })
sf.head()
sf['Permit Number'].nunique()
sf['Record ID'].nunique()
sfr = sf[['perm_typ','perm_typ_def','file_dt','issue_dt','cur_st','strct_notif','no_exist_stry','no_prop_stry',
          'fire_only_permit','est_cost','rev_cost','exist_use','prop_use','plansets','exist_const_type',
          'prop_const_type','site_permit','location']].copy()
sfr[['file_dt','issue_dt','perm_typ']].info()
# One way to introduce a new column
sfr = sfr.assign(wait_time = (sfr['issue_dt'] - sfr['file_dt']).dt.days)
# Another way to introduce a new column
# Extract month and year
sfr['month'] = sfr['file_dt'].dt.month
sfr['year'] = sfr['file_dt'].dt.year
_ = plt.figure(figsize=(12,6))
_ = plt.subplot(1,2,1)
_ = (sfr.groupby('year').wait_time.mean()).plot.barh()
_ = plt.title('Average wait time by year')
_ = plt.subplot(1,2,2)
_ = (sfr.groupby('year').wait_time.count()).plot.barh()
_ = plt.title('Permit count by year')
# create side-by-side boxplots for every permit type
_ = plt.figure(figsize=(10,6))
ax = sns.boxplot(y='perm_typ_def', x='wait_time', data = sfr, orient = 'h');
plt.title('permit_type vs. wait_time');
plt.tight_layout(pad=1)
plt.show()
