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
import altair as alt

df = pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")
df.columns = ["Entity", "Code","Year","Annual CO2"]

df.info()
print("Shape of Data :")

df.shape
print("Number of Unique Value :")

df.nunique()
print("Number of Null Value:")

df.isnull().sum()
null_code = df.loc[df["Code"].isnull()]

no_emmi = df.loc[df["Annual CO2"] == 0]

df.drop(no_emmi.index,axis=0,inplace=True)

df.shape
significant_list = ['Africa', 'Americas (other)', 'Asia and Pacific (other)', 'EU-28','Europe (other)', 'Middle East', 'United States', 'China',

                        'International transport']

g20_list = ['Argentina', 'Australia', 'Brazil', 'Canada', 'Saudi Arabia','China', 'France', 'Germany',

               'India', 'United States','Indonesia', 'Italy', 'Japan', 'Mexico', 'Russia','South Africa',

               'South Korea', 'Turkey', 'United Kingdom', 'Spain']

significant = df.query(f"Entity == {significant_list}").copy()

g20 = df.query(f"Entity == {g20_list}").copy()



g20["Annual CO2"] = g20["Annual CO2"] / 1e9

significant["Annual CO2"] = significant["Annual CO2"] / 1e9
alt.Chart(significant).mark_area(opacity=0.5,line=True).encode(

    x=alt.X("Year:O",title="Year",),

    y=alt.Y("Annual CO2",title="CO2 Emission (Billions/Tonnes)"),

    color = alt.Color("Entity",title="Region"),

    tooltip = ["Entity","Annual CO2","Year"]

).properties(width=600,height=400,title="CO2 Emission trends for Significant Places")
binder = alt.binding_range(min=1850,max=2017,step=1)

slider = alt.selection_single(bind=binder,fields=["Year"],name="Select",init={"Year":2017})

over = alt.selection_single(on="mouseover")

alt.Chart(g20).mark_bar(opacity=0.9).encode(

    x = alt.X("Entity",title="G20 Country"),

    y = alt.Y("Annual CO2",title="CO2 Emission (millions/tonnes)"),

    color= alt.condition(over,"Entity",alt.value("lightgrey"),title="Region"),

    tooltip=["Entity","Annual CO2"]

).properties(

    title="Annual CO2 Emission by G20 Countries",

    width=600,

    height=400,   

    selection=slider

).transform_filter(slider).add_selection(over)
alt.Chart(g20).mark_area(opacity=0.5, line=True).encode(

        x = alt.X('Year:O'),

        y = alt.Y('Annual CO2', title='Annual Emissions (in billion tonnes)'),

        color = alt.Color("Entity",title="Region"),

        tooltip = ["Year",

                   'Entity', 'Annual CO2']

).properties(

        width = 600,

        height = 400,

        title='CO2 emissions trends by G20 Country',

        

)
me_list = ['Algeria', 'Bahrain', 'Djibouti', 'Egypt', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Libya', 'Malta', 'Morocco', 'Oman', 'Qatar', 'Saudi Arabia', 'Syria', 'Tunisia', 'United Arab Emirates', 'Palestine','Yemen']

middle_east = df.query(f"Entity == {me_list}").copy()

middle_east["Annual CO2"] = middle_east["Annual CO2"] / 1e9
year_bind = alt.binding_range(min=1920,max=2017,step=1)

slider = alt.selection_single(bind=year_bind,fields=["Year"],name="Select",init={"Year":2017})



alt.Chart(middle_east).mark_bar().encode(

    x = alt.X("Entity:N",title="Country"),

    y = alt.Y("Annual CO2",title="CO2 Emission (Billions/Tonnes)"),

    color = alt.Color("Entity",title="Country"),

    tooltip=["Annual CO2","Year"]

).properties(

     width = 600 , height = 400 , title = "Annual CO2 Emission for Middle East Countries" , selection=slider

).transform_filter(slider)
alt.Chart(middle_east).mark_area(opacity=0.5,line=True).encode(

    x= alt.X("Year:O"),

    y = alt.Y("Annual CO2",title="Annual CO2 Emission (Billions/Tonnes)"),

    color = alt.Color("Entity",title="Region"),

    tooltip = ["Year","Entity","Annual CO2"]

).properties(

    title="CO2 Emission Trend for Middle East Country",

    width = 600,

    height = 400,

)