# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import networkx as nx
import matplotlib.pyplot as plt
from plotly.graph_objs import *
import pyproj as proj
import numpy as np
import pandas as pd
import plotly as py
import sys
import pip
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
from IPython.display import display
import plotly.figure_factory as ff
# Any results you write to the current directory are saved as output.
sch_explor = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
shsat_reg_testers = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
safe_report = pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv")
math_results = pd.read_csv("../input/new-york-state-math-test-results/2006-2011-nys-math-test-results-by-grade-school-level-all-students.csv")

#format School Income Estimate column to remove special characters and turn the data to numeric
sch_explor['School Income Estimate'] = sch_explor['School Income Estimate'].str.strip('$')
sch_explor['School Income Estimate'] = sch_explor['School Income Estimate'].str.replace(",", "")
sch_explor[['School Income Estimate']] = sch_explor[['School Income Estimate']].apply(pd.to_numeric)

#render the chart
limits = [(1,0.8),(0.8,0.6),(0.6,0.4),(0.4,0.2),(0.2,0.0)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","#009688"]
py.offline.init_notebook_mode(connected=True)
mapbox_access_token = 'pk.eyJ1Ijoibml6YXNpd2FsZSIsImEiOiJjaml3dnppMGQyNXNxM3dudjEzN3BoNTRyIn0.ayvpbxyR5P8anOkX9RLvZw'
schs = []
scale = 4000

for i in range(len(limits)):
    lim = limits[i]
    sch_explor_sub = sch_explor.loc[((sch_explor['Grade 8 Math 4s - Economically Disadvantaged']>0)&(sch_explor['Economic Need Index']<lim[0]) &(sch_explor['Economic Need Index']>lim[1]))]
    #sch_explor_sub['School Income Estimate Value'] =sch_explor_sub['School Income Estimate'].map('${:,.2f}'.format)
    sch = dict(
        type = 'scattermapbox',
        locationmode = 'USA-states',
        lon = sch_explor_sub['Longitude'],
        lat = sch_explor_sub['Latitude'],
        text = sch_explor_sub['School Name'],
        locations='New York',
        mode='markers',
        marker = dict(
            size = 5,
            color = colors[i],
            line = dict(width=0.5, color='#fafafa'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1])
    )
    schs.append(sch)
data = [
    go.Scattermapbox(
        lat=['45.5017'],
        lon=['-73.5673'],
        mode='markers',
        marker=dict(
            size=14
        ),
        text=['Montreal'],
    )
]

layout = go.Layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.785091,
            lon=-73.968285
        ),
        pitch=3,
        zoom=8.3
    ),
)

fig = dict(data=schs, layout=layout)

py.offline.iplot( fig, validate=False, filename='School Location Mapbox' )
#formate columns by dropping any NaN values from the safe report
safe_report = safe_report.fillna(0)
unique_safe_report = safe_report[['DBN','Latitude','Longitude','Major N','Oth N','Prop N','NoCrim N','Vio N']].copy()
#unique_safe_report = unique_safe_report.drop_duplicates('DBN') 
sch_explor_mod = sch_explor.copy()
sch_explor_mod = sch_explor_mod.rename(columns={'Location Code': 'DBN'})
sch_explor_dbn = pd.unique(sch_explor_mod[['DBN']].values.ravel('K'))  

sch_explor_crime = unique_safe_report[((unique_safe_report['DBN'].isin(sch_explor_dbn)))]
#some all safety reports and the merge with school
sch_explor_crime = sch_explor_crime.groupby('DBN')[['Major N','Oth N','Prop N','NoCrim N','Vio N']].sum()
sch_explor_crime = sch_explor_crime.reset_index()
merged_sch_explor = sch_explor_mod.merge(sch_explor_crime, how = 'inner', on = ['DBN'])

display(merged_sch_explor)
#table = ff.create_table(merged_sch_explor)
#py.offline.iplot( table, validate=False, filename='jupyter-table1' )

#change some columns to numeric
merged_sch_explor[['Major N','Oth N','NoCrim N','Prop N','Vio N']] = merged_sch_explor[['Major N','Oth N','NoCrim N','Prop N','Vio N']].apply(pd.to_numeric)

merged_sch_explor['safety index']= (1.50 * merged_sch_explor.loc[:, ['Major N']].sum(axis=1))+(0.65*merged_sch_explor.loc[:, ['Oth N']].sum(axis=1))+(0.70*merged_sch_explor.loc[:, ['NoCrim N']].sum(axis=1))+(0.70*merged_sch_explor.loc[:, ['Prop N']].sum(axis=1))+(2.0*merged_sch_explor.loc[:, ['Vio N']].sum(axis=1))
display(merged_sch_explor)
safety_index_q3 = merged_sch_explor[['safety index']].quantile(.75)
sch_explor_issues = merged_sch_explor.loc[(((merged_sch_explor['safety index'] > safety_index_q3[0]) )
                                          &(merged_sch_explor['Economic Need Index'] > 0.5) )]

sch_explor_issues_non = merged_sch_explor.loc[((merged_sch_explor['safety index'] == 0)
                                               &(merged_sch_explor['Economic Need Index'] < 0.5))]
sch_explor_dbn = pd.unique(sch_explor_issues[['DBN']].values.ravel('K'))
sch_explor_dbn_all = pd.unique(sch_explor_issues_non[['DBN']].values.ravel('K'))

#display(sch_explor_issues)

math_results_safety = math_results[((math_results['DBN'].isin(sch_explor_dbn)))]
math_results_top = math_results[((math_results['DBN'].isin(sch_explor_dbn_all)))]
#merged_sat.dropna(how='any')

def showForGrade(grade):    
    years = pd.unique(math_results_safety[['Year']].values.ravel('K'))
    years = pd.unique(math_results_safety[['Year']].values.ravel('K'))
    math_results_safety_7s = math_results_safety.loc[((math_results_safety['Grade'] == grade))]

    math_results_safety_7s_top = math_results_top.loc[((math_results_top['Grade'] == grade))]

    math_results_safety_7s[['Mean Scale Score']] = math_results_safety_7s[['Mean Scale Score']].apply(pd.to_numeric)
    math_results_safety_7s_top[['Mean Scale Score']] = math_results_safety_7s_top[['Mean Scale Score']].apply(pd.to_numeric)

    q2 = math_results_safety_7s_top[['Mean Scale Score']].quantile(.75)
    display(q2)
    display(math_results_safety_7s[['Mean Scale Score']].quantile(.75))


    math_results_safety_7s_up = math_results_safety_7s.loc[((math_results_safety_7s['Mean Scale Score']>q2[0]))]

    schs_names = pd.unique(math_results_safety_7s_up[['DBN']].values.ravel('K'))
    #display(math_results_safety_7s)
    sch_explor_issues_top = sch_explor_issues.loc[(((sch_explor_issues['DBN'].isin(schs_names))))]

    sch_explor_issues_top['safety index']= sch_explor_issues_top.loc[:, ['Major N','Oth N','NoCrim N','Vio N']].sum(axis=1)
    #display(sch_explor_issues_top)

    traces = []

    for i in range(len(schs_names)):
        scho_name = schs_names[i]
        sch_results_safety = math_results_safety_7s_up.loc[((math_results_safety_7s_up['DBN'] == scho_name))]
        results = pd.unique(sch_results_safety[['Mean Scale Score']].values.ravel('K'))
        dates = pd.unique(sch_results_safety[['Year']].values.ravel('K'))
    
        trace = go.Scatter(x=dates,y=results,name = scho_name)
    
        traces.append(trace)
    layout = dict(title = 'Math Test Mean Score Grade '+grade,
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Mean Score'),
              )
    fig = dict(data=traces,layout=layout)

    py.offline.init_notebook_mode(connected=True)
    py.offline.iplot( fig, validate=False, filename='map' )
    return schs_names
    
performing_7th_grade = showForGrade('7')
performing_8th_grade =showForGrade('8')
performing_7th_grade_explor = merged_sch_explor.loc[((merged_sch_explor['DBN'].isin(performing_7th_grade)))]
performing_8th_grade_explor = merged_sch_explor.loc[((merged_sch_explor['DBN'].isin(performing_8th_grade)))]

display(sch_explor_issues_non[['School Income Estimate']].quantile(.5))
display(performing_8th_grade_explor[['School Income Estimate']].quantile(.5))

#display(performing_7th_grade_explor)

display(performing_8th_grade_explor)
