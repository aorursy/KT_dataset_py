# For data

import pandas as pd

from pandas import Series,DataFrame

import numpy as np



# For visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')

from __future__ import division

import plotly.plotly as py

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



#Take care of warnings

import warnings

warnings.filterwarnings("ignore")
#Read in dataset and remove unnecessary data

df=pd.read_csv('../input/HospInfo.csv')

init_notebook_mode(connected=True)

df_ratings=df.drop(['Address','ZIP Code','Phone Number','Emergency Services',

'Patient experience national comparison footnote','Efficient use of medical imaging national comparison footnote',

'Hospital overall rating footnote','Mortality national comparison footnote','Mortality national comparison footnote',

'Safety of care national comparison footnote','Readmission national comparison footnote',

'Patient experience national comparison footnote','Effectiveness of care national comparison footnote',

'Timeliness of care national comparison footnote','Timeliness of care national comparison footnote',

'Efficient use of medical imaging national comparison footnote','Meets criteria for meaningful use of EHRs',

'Provider ID'],axis=1)
#Converting verbal rankings into integers so we can measure.

df1=df_ratings.replace('Below the National average', 1)

df2=df1.replace('Same as the National average', 2)

df3=df2.replace('Above the National average', 3)

    

#Converting data types to numeric

df3['Mortality national comparison'] = df3['Mortality national comparison'].convert_objects(convert_numeric=True)

df3['Safety of care national comparison'] = df3['Safety of care national comparison'].convert_objects(convert_numeric=True)

df3['Readmission national comparison'] = df3['Readmission national comparison'].convert_objects(convert_numeric=True)

df3['Patient experience national comparison'] = df3['Patient experience national comparison'].convert_objects(convert_numeric=True)

df3['Mortality national comparison'] = df3['Mortality national comparison'].convert_objects(convert_numeric=True)

df3['Hospital overall rating'] = df3['Hospital overall rating'].convert_objects(convert_numeric=True)
#Prepping our dataframe for the map

df4=df3.groupby('State')['Hospital overall rating'].mean()

df4=pd.DataFrame(df4)

df4=df4.reset_index()



#Map syntax

df['text'] = df4['State']



data = [dict(

    type='choropleth',

    autocolorscale = False,

    locations = df4['State'],

    z = df4['Hospital overall rating'].astype(float),

    locationmode = 'USA-states',

    text = df['text'],

    hoverinfo = 'location+z',

    marker = dict(

        line = dict (

            color = 'rgb(255,255,255)',

            width = 2

        )

    ),

    colorbar = dict(

        title = "Rating"

    )

)]

layout = dict(

    title = 'Hospital Ratings by State',

    geo = dict(

        scope='usa',

        projection=dict( type='albers usa' ),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'

    )

)    

fig = dict(data=data, layout=layout)



import plotly.graph_objs as go



iplot(fig, validate=False, filename='d3-Ratings-map')
#Now let's explore what categories of hospital ownership recieve the best ratings.

sns.barplot(x='Hospital overall rating', y='Hospital Ownership', ci=0, data=df3, palette="deep")

plt.title('Hospital Ownership & Average Rating',x=.3, y=1.05)

plt.xlabel('Rating')

plt.ylabel('Ownership Category')

plt.subplots_adjust(right=.75)
#Creating pivot table for next viz.

df5=df3.pivot_table(index=['State'], values = ['Hospital overall rating','Safety of care national comparison',

                    'Patient experience national comparison'], aggfunc='mean')

df5=pd.DataFrame(df5)

df5=df5.reset_index()

df5=df5.sort_values("Hospital overall rating", ascending = False).dropna()
#Ratings Correlation

g = sns.PairGrid(df5,

                 x_vars=['Hospital overall rating','Patient experience national comparison','Safety of care national comparison']

                 , y_vars=["State"],

                 size=10, aspect=.3)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=7, orient="h", palette="hls")



# Use the same x axis limits on all columns and add better labels

g.set(xlim=(0, 5), xlabel="", ylabel="")



# Use semantically meaningful titles for the columns

titles = ['Overall Rating','Patient Rating','Safety Rating']



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(True)

    ax.yaxis.grid(True)

   



sns.despine(left=True, bottom=True)
df6=df3['State'].value_counts()

df6=pd.DataFrame(df6)

df6=df6.reset_index()

df6=df6.sort('State',ascending=False)

df6=df6.dropna()



sns.factorplot(x="State", y='index', data=df6, kind='bar',

                   palette="Set2", size=10, aspect=.75)

plt.ylabel('State')

plt.xlabel('Count')

plt.title('Total Hospitals Per State',y=1.05, size=15)