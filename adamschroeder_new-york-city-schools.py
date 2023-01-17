import pandas as pd
import numpy as np
import seaborn as sns

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print (__version__)
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import plotly.offline as py
# NYC Open Data
df2 = pd.read_csv("../input/2019_DOE_High_School_Directory.csv")

# https://infohub.nyced.org/reports-and-policies/school-quality/school-quality-reports-and-resources (Citywide Data)
df3 = pd.read_excel("../input/201718_hs_sqr_results.xlsx", sheet_name='Closing the Achievement Gap')
df4 = pd.read_excel("../input/201718_hs_sqr_results.xlsx", sheet_name='Summary')
df5 = pd.read_excel("../input/201718_hs_sqr_results.xlsx", sheet_name='Student Achievement')

# https://tools.nycenet.edu/dashboard/#dbn=01M539&report_type=HS&view=City
df6 = pd.read_excel("../input/impact_performance.xlsx", sheet_name='2018') 
df4 = df4.merge(df3[['Metric Value - College Persistence','DBN']], on="DBN")

df4 = df4.merge(df5[['Percentage of Students with 90%+ Attendance','Metric Value - 4-Year Graduation Rate',
                     'DBN','Metric Value - College and Career Preparatory Course Index']], on="DBN")
df2.rename(columns={'dbn':'DBN'}, inplace=True)
df4 = df4.merge(df2[['DBN','Latitude','Longitude','school_type','boro','interest1','link','pct_stu_safe']], on='DBN')

df4 = df4.merge(df6[['DBN','Impact','Performance']], on='DBN')
df4 = df4.loc[:,['DBN','School Name','Enrollment','Supportive Environment - Percent Positive',
            'Percent Students with Disabilities','Economic Need Index','Percent Asian','Percent Black','Percent White',
            'Percent Hispanic','Latitude','Longitude','school_type','boro','interest1','link','Impact','Performance',
            'Percentage of Students with 90%+ Attendance','Metric Value - 4-Year Graduation Rate',
            'Metric Value - College and Career Preparatory Course Index','Metric Value - College Persistence', 
            'pct_stu_safe']]
# my own calculation of diversity of students' race
df4['from_25_percent_hispanic'] = np.abs(0.25-df4['Percent Hispanic'])
df4['from_25_percent_white'] = np.abs(0.25-df4['Percent White'])
df4['from_25_percent_black'] = np.abs(0.25-df4['Percent Black'])
df4['from_25_percent_asian'] = np.abs(0.25-df4['Percent Asian'])

# 1 minus because we want the higher values to represent more diversity
df4['student_diversity'] = (1-((df4['from_25_percent_hispanic'] + df4['from_25_percent_white'] +\
                              df4['from_25_percent_black'] + df4['from_25_percent_asian'])/2)).round(2) 
# Normalize the values so they are between 0 and 1.
df4['Supportive Environment - Percent Positive']=(df4['Supportive Environment - Percent Positive'] - 
                                                  df4['Supportive Environment - Percent Positive'].min()) / (df4['Supportive Environment - Percent Positive'].max()- df4['Supportive Environment - Percent Positive'].min())
# Create text for hover in map below
df4['text']=df4['School Name'].astype(str)+'<br>'+\
'Students: '+df4['Enrollment'].astype(str)+'<br>'+\
'Performance Rate: '+df4['Performance'].round(2).astype(str)+'<br>'+\
'4-Year Graduation Rate: '+(df4['Metric Value - 4-Year Graduation Rate']*100).round(2).astype(str)+'%<br>'+\
'College Persistence: '+(df4['Metric Value - College Persistence']*100).round(2).astype(str)+'%<br>'+\
'Diverstiy Rate: '+df4['student_diversity'].round(2).astype(str)+'<br>'+\
'Safety: '+(df4['pct_stu_safe']*100).round(2).astype(str)+'%<br>'+\
'Supportive Environment: '+df4['Supportive Environment - Percent Positive'].round(2).astype(str)
# Combine repetitive interest areas and replace NANs
df4['interest1'].replace({"Computer Science, Math & Technology":"Computer Science & Technology", np.nan:'No Data'}, 
                         inplace=True)
interest_areas = list(df4.interest1.unique())
# Replace Nan Performance with 0.3 size, otherwise scattermapbox in Figurewidget doens't work
df4.Performance.replace({np.nan:0.3}, inplace=True)
"""Homework"""
# safety info from charter schools
# Ask DOE how to get Interest areas for missing 61 charter schools (No Data)
import webbrowser
import plotly.graph_objs as go

mapbox_access_token = 'pk.eyJ1IjoiYWRhbXNjaHJvZWRlciIsImEiOiJjanEwaTZucDAwbjQ5NDlxbTZrZXVtMWduIn0.jIN1SfSxeiHs7205kVvWYQ'

fig = go.FigureWidget(layout = dict(
    autosize=True,
    hovermode='closest',
    hoverdistance=2,
    title='NYC High Schools<br>(bubble sized according to school performance relative to city average)',
    showlegend = True,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=25,
        style='light',
        center=dict(
            lat=40.705005,
            lon=-73.93879
        ),
        pitch=20,
        zoom=11.5,
    ),
))

def do_click(trace, points, state):
    if points.point_inds:
        ind = points.point_inds[0]
        url = df4.link.iloc[ind]
        webbrowser.open_new_tab(url)

school=[]
for i in interest_areas:
    df_sub=df4[df4.interest1==i]
    school = fig.add_scattermapbox(
        lon = df_sub['Longitude'],
        lat = df_sub['Latitude'],
        mode='markers',
        marker = dict(
            size = (df_sub['Performance']*10)**1.8
                     ),
        hovertext=df_sub['text'],
        hoverlabel=dict(namelength=-1),
        hoverinfo='name+text',
        name = df_sub['interest1'].values[0], #legend interest name

    )
    school.on_click(do_click)
    
fig

# fig = dict( data=schools, layout=layout)
# py.iplot(fig, filename = 'nyc_schools')






