# Important Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime # for time manipulation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Interactive Visualization

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/school-shootings-us-1990present/pah_wikp_combo.csv")

df1=pd.read_csv('/kaggle/input/school-shootings-us-1990present/cps_01_formatted.csv')
df.drop_duplicates(subset ="Date", inplace = True) 

df
df1.columns
df.head()
# Top 10 states for school shootings



# Assign the value_counts() results to a new data frame

state_shootings=df.State.value_counts().head(10).reset_index()

state_shootings.columns=['State','Total Shootings']



# Visualize the results

fig = px.bar(state_shootings, x="Total Shootings", y="State", orientation='h',color='State')

fig.show()
# Which Cities have the most school shootings



# Assign the value_counts() results to a new data frame

city_shootings=df.City.value_counts().head(10).reset_index()

city_shootings.columns=['City','Total Shootings']



# Visualize the results

fig = px.bar(city_shootings, x="Total Shootings", y="City", orientation='h',color='City')

fig.show()
# Which school shootings are more brutal: Elementary School, Middle School, High School or College?



# Find those wounded in elemetary school shootings

wounded_elementary_school=df[

    (df['Wounded']>=1)&(df['School']=='ES')

]

totalwes=len(wounded_elementary_school)



# Find those killed in elemetary school shootings

fatality_elementary_school=df[

    (df['Fatalities']>=1)&(df['School']=='ES')

]

totalfes=len(fatality_elementary_school)



# Find those wounded in middle school shootings

wounded_middle_school=df[

    (df['Wounded']>=1)&(df['School']=='MS')

]

totalwms=len(wounded_middle_school)



# Find those killed in middle school shootings

fatality_middle_school=df[

    (df['Fatalities']>=1)&(df['School']=='MS')

]

totalfms=len(fatality_middle_school)





# Find those wounded in high school shootings

wounded_high_school=df[

    (df['Wounded']>=1)&(df['School']=='HS')

]

totalwhs=len(wounded_high_school)



# Find those killed in high school shootings

fatality_high_school=df[

    (df['Fatalities']>=1)&(df['School']=='HS')

]

totalfhs=len(fatality_high_school)



# Find those wounded in college shootings

wounded_college=df[

    (df['Wounded']>=1)&(df['School']=='C')

]

totalwc=len(wounded_college)



# Find those killed in college shootings

fatality_college=df[

    (df['Fatalities']>=1)&(df['School']=='C')

]

totalfc=len(fatality_college)



# Visualize the results

schools=['Elementary School','Middle School','High School','College']



fig = go.Figure()

fig.add_trace(go.Bar(x=schools, y=[totalwes,totalwms,totalwhs,totalwc],

                base=0,

                marker_color='lightslategrey',

                name='Wounded'))

fig.add_trace(go.Bar(x=schools, y=[totalfes,totalfms,totalfhs,totalfc],

                base=0,

                marker_color='crimson',

                name='Fatalities'

                ))

fig.update_layout(title_text='Shootings per School Type')



fig.show()
# Are there more cases of school shootings now? 



# Create a new colum that contains the year value from the Date column

df['year'] = pd.DatetimeIndex(df['Date']).year



# Assign the value_counts() results to a new data frame

annual_shootings=df.year.value_counts().reset_index()

annual_shootings.columns=['Year','Total Shootings']



# Visualize this 

fig = px.scatter(annual_shootings, x="Year", y="Total Shootings", size="Total Shootings", color="Total Shootings",

           hover_name="Total Shootings", log_x=True, size_max=60)

fig.show()
# Determine the most severe of shootings, anything more than 3 fatalities is severe

df['Severity']=df['Fatalities'].apply(lambda x: 'Severe' if x>3 else 'Bad' )

severe_shootings=df[df['Severity']=='Severe']
# Which States have the most severe shootings

state_severe=severe_shootings.State.value_counts().reset_index()

state_severe.columns=['State','Total Shootings']

fig=px.bar(state_severe,x='Total Shootings',y='State',orientation='h',color='State')

fig.show()
states=severe_shootings.groupby('State')['Fatalities'].apply(sum).reset_index().sort_values('Fatalities',ascending=False)

states
# Which states have the most severe shootings based on fatalities, visualized

fig = px.scatter(states, x="State", y="Fatalities",

                 size="Fatalities", color="State",

                 log_x=True, size_max=60)

fig.show()