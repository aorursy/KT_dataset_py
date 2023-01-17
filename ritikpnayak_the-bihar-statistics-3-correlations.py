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
from matplotlib import style
from statistics import mean, stdev
style.use('ggplot')
ger = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv', index_col = 'State_UT')
toilet_for_girls = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
toilet_for_boys = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
electricity = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-electricity-2013-2016.csv')
water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')
computer = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-comps-2013-2016.csv')
ger.rename({'MADHYA PRADESH' : 'Madhya Pradesh', 'Pondicherry' : 'Puducherry', 'Uttaranchal' : 'Uttarakhand'}, inplace = True)
ger.reset_index(inplace = True)

water.rename(columns = {'State/UT': 'State_UT'}, inplace = True)
ger['State_UT'].value_counts()
ger_2013 = ger.query(' State_UT != "All India" & Year == "2013-14" ')
tfg_2013 = toilet_for_girls.query(' State_UT != "All India" & year == "2013-14" ')
tfb_2013 = toilet_for_boys.query(' State_UT != "All India" & year == "2013-14" ')
electricity_2013 = electricity.query(' State_UT != "All India" & year == "2013-14" ')
water_2013 = water.query(' State_UT != "All India" & Year == "2013-14" ')
computer_2013 = computer.query(' State_UT != "All India" & year == "2013-14" ')

ger_2014 = ger.query(' State_UT != "All India" & Year == "2014-15" ')
tfg_2014 = toilet_for_girls.query(' State_UT != "All India" & year == "2014-15" ')
tfb_2014 = toilet_for_boys.query(' State_UT != "All India" & year == "2014-15" ')
electricity_2014 = electricity.query(' State_UT != "All India" & year == "2014-15" ')
water_2014 = water.query(' State_UT != "All India" & Year == "2013-14" ')
computer_2014 = computer.query(' State_UT != "All India" & year == "2013-14" ')

ger_2015 = ger.query(' State_UT != "All India" & Year == "2015-16" ')
tfg_2015 = toilet_for_girls.query(' State_UT != "All India" & year == "2015-16" ')
tfb_2015 = toilet_for_boys.query(' State_UT != "All India" & year == "2015-16" ')
electricity_2015 = electricity.query(' State_UT != "All India" & year == "2015-16" ')
water_2015 = water.query(' State_UT != "All India" & Year == "2013-14" ')
computer_2015 = computer.query(' State_UT != "All India" & year == "2013-14" ')
print(np.mean(ger_2013.Primary_Girls.values))
print(ger_2013.Primary_Girls.values)
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def covariance(x, y):
    n = len(x)
    return np.dot(de_mean(x), de_mean(y)) / (n - 1)
def correlation(x, y):
    std_x = stdev(x)
    std_y = stdev(y)
    if std_x > 0 and std_y >0:
        return covariance(x, y) / std_x / std_y
    else:
        return 0
ger_2013_val_girls = ger_2013.Primary_Girls.values
ger_2014_val_girls = ger_2014[ger_2014.State_UT != 'Telangana'].Primary_Girls.values
ger_2015_val_girls = ger_2015[ger_2015.State_UT != 'Telangana'].Primary_Girls.values

ger_2013_val_boys = ger_2013.Primary_Boys.values
ger_2014_val_boys = ger_2014[ger_2014.State_UT != 'Telangana'].Primary_Boys.values
ger_2015_val_boys = ger_2015[ger_2015.State_UT != 'Telangana'].Primary_Boys.values

tfb_2013_val = tfb_2013.Primary_Only.values
tfb_2014_val = tfb_2014[tfb_2014.State_UT != 'Telangana'].Primary_Only.values
tfb_2015_val = tfb_2015[tfb_2015.State_UT != 'Telangana'].Primary_Only.values

tfg_2013_val = tfg_2013.Primary_Only.values
tfg_2014_val = tfg_2014[tfg_2014.State_UT != 'Telangana'].Primary_Only.values
tfg_2015_val = tfg_2015[tfg_2015.State_UT != 'Telangana'].Primary_Only.values

electricity_2013_val = electricity_2013.Primary_Only.values
electricity_2014_val = electricity_2014[electricity_2014.State_UT != 'Telangana'].Primary_Only.values
electricity_2015_val = electricity_2015[electricity_2015.State_UT != 'Telangana'].Primary_Only.values

water_2013_val = water_2013.Primary_Only.values
water_2014_val = water_2014[water_2014.State_UT != 'Telangana'].Primary_Only.values
water_2015_val = water_2015[water_2015.State_UT != 'Telangana'].Primary_Only.values

computer_2013_val = computer_2013.Primary_Only.values
computer_2014_val = computer_2014[computer_2014.State_UT != 'Telangana'].Primary_Only.values
computer_2015_val = computer_2015[computer_2015.State_UT != 'Telangana'].Primary_Only.values
tfg_2014_val
print('Covariance between toilet for girls in 2013 and ger 2013: {}'.format(covariance(tfg_2013_val, ger_2013_val_girls)))
print('Correlation between toilet for girls in 2013 and ger 2013: {}'.format(correlation(tfg_2013_val, ger_2013_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2013_val_girls)
print('Covariance between toilet for girls in 2013 and ger 2014: {}'.format(covariance(tfg_2013_val, ger_2014_val_girls)))
print('Correlation between toilet for girls in 2013 and ger 2014: {}'.format(correlation(tfg_2013_val, ger_2014_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2014_val_girls)
print('Covariance between toilet for girls in 2013 and ger 2015: {}'.format(covariance(tfg_2013_val, ger_2015_val_girls)))
print('Correlation between toilet for girls in 2013 and ger 2015: {}'.format(correlation(tfg_2013_val, ger_2015_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2015_val_girls)
print('Covariance between toilet for girls in 2014 and ger 2014: {}'.format(covariance(tfg_2014_val, ger_2014_val_girls)))
print('Correlation between toilet for girls in 2014 and ger 2014: {}'.format(correlation(tfg_2014_val, ger_2014_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2014_val, ger_2014_val_girls)
print('Covariance between toilet for girls in 2014 and ger 2015: {}'.format(covariance(tfg_2014_val, ger_2015_val_girls)))
print('Correlation between toilet for girls in 2014 and ger 2015: {}'.format(correlation(tfg_2014_val, ger_2015_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2014_val, ger_2015_val_girls)
print('Covariance between toilet for girls in 2015 and ger 2015: {}'.format(covariance(tfg_2015_val, ger_2015_val_girls)))
print('Correlation between toilet for girls in 2015 and ger 2015: {}'.format(correlation(tfg_2015_val, ger_2015_val_girls)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2015_val, ger_2015_val_girls)
for i in ger_2014.State_UT.value_counts().index:
    if i not in ger_2013.State_UT.value_counts().index:
        print(i)
print('Covariance between toilet for boys in 2013 and ger 2013: {}'.format(covariance(tfb_2013_val, ger_2013_val_boys)))
print('Correlation between toilet for boys in 2013 and ger 2013: {}'.format(correlation(tfb_2013_val, ger_2013_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2013_val_girls)
print('Covariance between toilet for boys in 2013 and ger 2014: {}'.format(covariance(tfb_2013_val, ger_2014_val_boys)))
print('Correlation between toilet for boys in 2013 and ger 2014: {}'.format(correlation(tfb_2013_val, ger_2014_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2014_val_girls)
print('Covariance between toilet for boys in 2013 and ger 2015: {}'.format(covariance(tfb_2013_val, ger_2015_val_boys)))
print('Correlation between toilet for boys in 2013 and ger 2015: {}'.format(correlation(tfb_2013_val, ger_2015_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2013_val, ger_2015_val_girls)
print('Covariance between toilet for boys in 2014 and ger 2014: {}'.format(covariance(tfb_2014_val, ger_2014_val_boys)))
print('Correlation between toilet for boys in 2014 and ger 2014: {}'.format(correlation(tfb_2014_val, ger_2014_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2014_val, ger_2014_val_girls)
print('Covariance between toilet for boys in 2014 and ger 2015: {}'.format(covariance(tfb_2014_val, ger_2015_val_boys)))
print('Correlation between toilet for boys in 2014 and ger 2015: {}'.format(correlation(tfb_2014_val, ger_2015_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2014_val, ger_2015_val_girls)
print('Covariance between toilet for boys in 2015 and ger 2015: {}'.format(covariance(tfb_2015_val, ger_2015_val_boys)))
print('Correlation between toilet for boys in 2015 and ger 2015: {}'.format(correlation(tfb_2015_val, ger_2015_val_boys)))
plt.figure(figsize = (15, 8))

plt.scatter(tfg_2015_val, ger_2015_val_girls)






