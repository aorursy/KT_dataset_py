import numpy as np
import pandas as pd
pd.options.display.precision = 2

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('Set2')
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
drop = pd.read_csv('/kaggle/input/indian-school-education-statistics/dropout-ratio-2012-2015.csv')
water = pd.read_csv('/kaggle/input/indian-school-education-statistics/percentage-of-schools-with-water-facility-2013-2016.csv')
boy_t = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-boys-toilet-2013-2016.csv')
girl_t = pd.read_csv('/kaggle/input/indian-school-education-statistics/schools-with-girls-toilet-2013-2016.csv')
enroll = pd.read_csv('/kaggle/input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv')
drop['State_UT'] = drop['State_UT'].replace({'Madhya  Pradesh':'Madhya Pradesh','Arunachal  Pradesh':'Arunachal Pradesh',
                                            'Tamil  Nadu':'Tamil Nadu'})
drop = drop.replace({'NR':np.nan,'Uppe_r_Primary':np.nan})
for c in drop.columns[2:]:
    drop[c] = drop[c].astype(float)
drop_mean = pd.pivot_table(drop, index='State_UT', values=['Primary_Total','Upper Primary_Total',
                                                          'Secondary _Total','HrSecondary_Total'])
cm = sns.light_palette("blue", as_cmap=True)
drop_mean.style.background_gradient(cmap=cm)
water_all_sch = water[water['Year']=='2015-16'][['State/UT','All Schools']]
boy_t_all_sch = boy_t[boy_t['year']=='2015-16'][['State_UT','All Schools']]
girl_t_all_sch = girl_t[girl_t['year']=='2015-16'][['State_UT','All Schools']]

water_all_sch = water_allsch.rename(columns={'State/UT':'State_UT'})

t_sch = pd.merge(boy_t_all_sch, girl_t_all_sch, on='State_UT')
sch = pd.merge(t_sch, water_all_sch, on='State_UT')

sch = sch.rename(columns={'All Schools_x':'boys_toilet','All Schools_y':'girls_toilet',
                          'All Schools':'water_facility'})
cm = sns.light_palette("green", as_cmap=True)
sch.style.background_gradient(cmap=cm)
enroll_lastest = enroll[enroll['Year']=='2015-16']
enroll_lastest = enroll_lastest[['State_UT','Primary_Boys', 'Primary_Girls','Upper_Primary_Boys',
                         'Upper_Primary_Girls','Secondary_Boys', 'Secondary_Girls',
                         'Higher_Secondary_Boys', 'Higher_Secondary_Girls',]].reset_index(drop=True)

enroll_lastest['Higher_Secondary_Boys'] = enroll_lastest['Higher_Secondary_Boys'].astype(float)
enroll_lastest['Higher_Secondary_Girls'] = enroll_lastest['Higher_Secondary_Girls'].astype(float)
cm = sns.light_palette("red", as_cmap=True)
enroll_lastest.style.background_gradient(cmap=cm)