# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
dropout_ratio = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')
schools_with_electricity = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')
gross_enrolment_ratio = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')
schools_with_water_facility = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')
schools_with_boys_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
schools_with_girls_toilet = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
schools_with_comps = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')
gross_enrolment_ratio.head()
gross_enrolment_ratio.State_UT.value_counts().sort_index()
df = gross_enrolment_ratio[gross_enrolment_ratio.State_UT.isin(['MADHYA PRADESH', 'Madhya Pradesh', 'Pondicherry', 'Puducherry', 'Uttarakhand', 'Uttaranchal'])]
df
gross_enrolment_ratio.set_index('State_UT', inplace = True)
gross_enrolment_ratio.rename({'MADHYA PRADESH' : 'Madhya Pradesh', 'Pondicherry' : 'Puducherry', 'Uttaranchal' : 'Uttarakhand'}, inplace = True)
gross_enrolment_ratio.reset_index(inplace = True)
gross_enrolment_ratio.State_UT.value_counts().sort_index()
gb_year = gross_enrolment_ratio.groupby(['State_UT','Year'])
gb_year.first()
bihar = gross_enrolment_ratio[gross_enrolment_ratio.State_UT == 'Bihar']
all_india = gross_enrolment_ratio[gross_enrolment_ratio.State_UT == 'All India']
bihar_total = bihar[['Year', 'Primary_Total', 'Upper_Primary_Total', 'Secondary_Total', 'Higher_Secondary_Total']]
bihar_total
bihar_total['Higher_Secondary_Total'] = bihar_total.Higher_Secondary_Total.astype(float)
all_india['Higher_Secondary_Total'] = all_india.Higher_Secondary_Total.astype(float)
bihar_total.describe()
bihar
plt.figure(figsize = (15,8))

bihar.Primary_Girls.plot()
bihar.Primary_Boys.plot()
bihar.Primary_Total.plot()
plt.legend()
plt.figure(figsize = (15,8))

gross_enrolment_ratio.Primary_Total.hist()
good_states = set(gross_enrolment_ratio.State_UT[gross_enrolment_ratio.Primary_Total > 110])
modest_states = set(gross_enrolment_ratio.State_UT[gross_enrolment_ratio.Primary_Total <= 110])
not_good_states = set(gross_enrolment_ratio.State_UT[gross_enrolment_ratio.Primary_Total <= 90])
len(not_good_states)+len(modest_states)+len(good_states)
not_good_states
good_states
modest_states
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year',y = 'Primary_Total', data = bihar_total, label = 'Primary')
sns.lineplot(x = 'Year',y = 'Upper_Primary_Total', data = bihar_total, label = 'Upper Primary')
sns.lineplot(x = 'Year',y = 'Secondary_Total', data = bihar_total, label = 'Higher')
sns.lineplot(x = 'Year',y = 'Higher_Secondary_Total', data = bihar_total, label = 'Higher Secondary')
plt.legend()
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year',y = 'Primary_Total', data = all_india, label = 'Primary')
sns.lineplot(x = 'Year',y = 'Upper_Primary_Total', data = all_india, label = 'Upper Primary')
sns.lineplot(x = 'Year',y = 'Secondary_Total', data = all_india, label = 'Higher')
sns.lineplot(x = 'Year',y = 'Higher_Secondary_Total', data = all_india, label = 'Higher Secondary')
plt.legend()
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'Primary_Boys', data = bihar, label = 'Boys')
sns.lineplot(x = 'Year', y = 'Primary_Girls', data = bihar, label = 'Girls')
plt.legend()
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'Primary_Boys', data = all_india, label = 'Boys')
sns.lineplot(x = 'Year', y = 'Primary_Girls', data = all_india, label = 'Girls')
plt.legend()
print('Probability that a good state is selected = ', gross_enrolment_ratio.Primary_Total[gross_enrolment_ratio.Primary_Total>=110].sum() / gross_enrolment_ratio.Primary_Total.sum())
print('Probability that not a good state is selected = ', gross_enrolment_ratio.Primary_Total[gross_enrolment_ratio.Primary_Total<=90].sum() / gross_enrolment_ratio.Primary_Total.sum())
print('Probability that a modest state is selected = ', gross_enrolment_ratio.Primary_Total[gross_enrolment_ratio.Primary_Total<=110].sum() / gross_enrolment_ratio.Primary_Total.sum())









