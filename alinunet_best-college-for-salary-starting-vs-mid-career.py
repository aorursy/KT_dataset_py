import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from re import sub

from decimal import Decimal

import warnings

warnings.filterwarnings('ignore')
regional_salaries = pd.read_csv("../input/salaries-by-region.csv")

type_salaries = pd.read_csv("../input/salaries-by-college-type.csv")

majors = pd.read_csv("../input/degrees-that-pay-back.csv")
type_salaries.head(n=5)
majors.head( n = 5)
regional_salaries = regional_salaries.drop(regional_salaries.columns[4:], axis=1)
for col in range(2,4):

    for i in range(len(regional_salaries)):

        regional_salaries.iloc[i][col] = sub(r'[^\d.]', '', regional_salaries.iloc[i][col]) 

regional_salaries_float= regional_salaries.apply(pd.to_numeric, errors='ignore')
regional_salaries_float.boxplot(column='Starting Median Salary',by='Region')

plt.suptitle('')
regional_salaries_float.boxplot(column='Mid-Career Median Salary',by='Region')

plt.suptitle('')
regional_salaries.sort_values(['School Name'],ascending = [True])

university_region_dict = {}

for jj in range(len(regional_salaries)):

    university_region_dict[regional_salaries.loc[jj,'School Name']] = regional_salaries.loc[jj,'Region']
for kk in range(len(type_salaries)):

    if type_salaries.loc[kk,'School Name'] in university_region_dict.keys():

        type_salaries.loc[kk,'Region'] = university_region_dict[type_salaries.loc[kk,'School Name']]

    else:

        type_salaries.loc[kk,'Region'] = 'Southern'

type_salaries
for col in range(2,4):

    for i in range(len(type_salaries)):

        type_salaries.iloc[i,col] = sub(r'[^\d.]', '', type_salaries.iloc[i,col]) 

type_salaries_float= type_salaries.apply(pd.to_numeric, errors='ignore')
type_salaries_float['Salary Increase'] = type_salaries_float['Mid-Career Median Salary'] - type_salaries_float['Starting Median Salary']
type_salaries_float.boxplot(column='Starting Median Salary',by='School Type')

plt.suptitle('')
type_salaries_float.boxplot(column='Mid-Career Median Salary',by='School Type')

plt.suptitle('')
type_salaries_float.boxplot(column='Salary Increase',by='School Type')

plt.suptitle('')
type_salaries_float.boxplot(column='Salary Increase',by='Region')

plt.suptitle('')
byType = type_salaries_float.groupby("School Type")

byType.groups.keys()
for name, group in byType:

    print(name)

    print(group["Mid-Career Median Salary"].describe())
for col in [1,2,4,5,6,7]:

    for i in range(len(majors)):

        majors.iloc[i,col] = sub(r'[^\d.]', '', majors.iloc[i,col]) 

majors_float= majors.apply(pd.to_numeric, errors='ignore')

majors_float['maj'] = majors_float['Undergraduate Major']
from bokeh.plotting import figure,output_file, show,output_notebook

from bokeh.models import HoverTool,sources

from collections import OrderedDict

output_notebook()



x=majors_float['Starting Median Salary']

y=majors_float['Mid-Career Median Salary']

increase = 100 * (y-x)/x

label=majors_float['maj']

#from bokeh.plotting import *

source = sources.ColumnDataSource(

    data=dict(

        x=x,

        y=y,

        increase = increase,

        label=label

    )

)

TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"

p = figure( x_axis_label = "Starting Salary",

            y_axis_label = "Mid-career Salary",title="Starting vs mid-career salary (Major)",plot_width = 800,plot_height = 800, tools=TOOLS)

p.circle('x', 'y', color="#2222aa", size=10, source=source)



hover =p.select(dict(type=HoverTool))

hover.tooltips = OrderedDict([

    ("increase salary[%]","@increase"),

    ("label", "@label"),

])



show(p)
colormap = {'California': 'red', 'Western': 'green', 'Northeastern': 'blue','Midwestern':'orange','Southern':'purple'}

colors = [colormap[x] for x in type_salaries_float['Region']]
x=type_salaries_float['Starting Median Salary']

y=type_salaries_float['Mid-Career Median Salary']

increase = 100 * (y-x)/x

label=type_salaries_float['School Name']

region = type_salaries_float['Region']

#from bokeh.plotting import *

source = sources.ColumnDataSource(

    data=dict(

        x=x,

        y=y,

        increase = increase,

        label=label,

        region = region

    )

)

TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"

p1 = figure(x_axis_label = "Starting Salary",

            y_axis_label = "Mid-career Salary",title="Starting vs mid-career salary (Schools)",plot_width = 800,plot_height = 800, tools=TOOLS)

p1.circle('x', 'y', color=colors, size=10, source=source)



hover =p1.select(dict(type=HoverTool))

hover.tooltips = OrderedDict([

    ("increase salary[%]","@increase"),

    ("label", "@label"),

    ("Region","@region")

])



show(p1)