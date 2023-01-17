# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats

from sklearn import ensemble, tree, linear_model

import missingno as msno

import pandas_profiling
District_wise = pd.read_csv('../input/education-in-india/2015_16_Districtwise.csv')

State_wise_elementry = pd.read_csv('../input/education-in-india/2015_16_Statewise_Elementary.csv')

State_wise_secondary = pd.read_csv('../input/education-in-india/2015_16_Statewise_Secondary.csv')

District_wise_met = pd.read_csv('../input/education-in-india/2015_16_Districtwise_Metadata.csv')

State_wise_elementry_met = pd.read_csv('../input/education-in-india/2015_16_Statewise_Elementary_Metadata.csv')

State_wise_secondary_met = pd.read_csv('../input/education-in-india/2015_16_Statewise_Secondary_Metadata.csv')

District_wise.head()
District_wise_met.head()
District_wise_total = pd.DataFrame()
i=0

for name in District_wise_met['Description']:

    if 'Total' in name:

        District_wise_total[District_wise_met.iloc[i][1]] = District_wise[District_wise_met.iloc[i][0]]

    i=i+1
District_wise_total.info()
District_wise_total['Schools_By_Category: Total']/(District_wise_total['Schools_by_Category:_Government: Total']+District_wise_total['Schools_by_Category:_Private_: Total']+District_wise_total['Schools_by_Category:_Madarsas_&_Unrecognised: Total'])
District_wise.head()
District_wise_new = pd.DataFrame()
District_wise_new['STATNAME'] = District_wise['STATNAME']
District_wise_new['DISTNAME'] = District_wise['DISTNAME']
District_wise_new = pd.concat([District_wise_new, District_wise_total], axis = 1 )
Kerla_total = pd.DataFrame

UP_total = pd.DataFrame

District_wise_grouped = pd.DataFrame()
District_wise_grouped = District_wise_new.groupby(by = 'STATNAME')
State_wise_sum = District_wise_grouped.sum()
State_wise_sum.head()
State_wise_sum.index
State_wise_sum['People_per_School'] = State_wise_sum['Basic_data_from_Census_2011: Total_Population(in_1000\'s)']/State_wise_sum['Schools_By_Category: Total']
ax = sns.barplot(y=State_wise_sum.index, x='People_per_School', data = State_wise_sum  )

#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

#plt.tight_layout()



plt.figure(figsize=(16,4))