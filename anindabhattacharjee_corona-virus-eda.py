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
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df
len(df)
print("list of affected countries",df['Country'].unique())
df['Last Update']= df['Last Update'].astype('str')

df.info()
# splitting Last Update to date 

df['date'] = df['Date'].str.split(" ").str[0]

df
#df['date'] = pd.to_datetime(df['date'])

#df

#len(df[df.date<'2020-01-02'])

#df[df.date>'1/31/2020']

#df[df.date<'2020-01-02'].tail(100)
# cleaning the date section

for i in range(len(df[df.date<'2020-01-02'])):

    date = df['date'][i].split("/")

    df['date'][i] = date[2]+"-"+date[1]+"-"+date[0]



df
# assign date in yyyy-mm-dd format

for i in range(len(df.date)):

    date = df['date'][i].split("-")

    df['date'][i] = date[0]+"-"+date[2]+"-"+date[1]

    

df


df['date'][df.date=='2023-01-20']="2020-01-23"

df['date'] = df['date'].astype('datetime64')
china_df= df[df.Country=='China'].append(df[df.Country=='Mainland China'])

china_df['date'][china_df.date=='2023-01-20']="2020-01-23"

#china_df['date'] = china_df['date'].str.replace('-','/')

china_df['date'] = china_df['date'].astype('datetime64')

china_df['date'].unique()
df1 = china_df.groupby(['date'],as_index =False).sum()

df1
#plotting infection count vs day

import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(x=df1['date'],

                y=df1['Confirmed'],

                name='Confirmed Cases',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=df1['date'],

                y=df1['Deaths'],

                name='Deaths',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=df1['date'],

                y=df1['Recovered'],

                name='Recovered Cases',

                marker_color='Green'

                ))



fig.update_layout(

    title='Corona Virus Convirmed vs Deaths vs Recovered in china',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='count',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
# finding day wise infection rate/death rate/ recovery rate

df1['inf_rate']=""

df1['inf_rate'][0]=0

df1['death_rate']=""

df1['death_rate'][0]=0

df1['recovery_rate']=""

df1['recovery_rate'][0]=0

for i in range(1,len(df1)):

    df1['inf_rate'][i]= (df1['Confirmed'][i]-df1['Confirmed'][i-1])/24

    df1['death_rate'][i]= (df1['Deaths'][i]-df1['Deaths'][i-1])/24

    df1['recovery_rate'][i]= (df1['Recovered'][i]-df1['Recovered'][i-1])/24
df1
import plotly.graph_objects as go

from plotly.subplots import make_subplots





fig = make_subplots(

    rows=3, cols=1,

    shared_xaxes=True,

    vertical_spacing=0.03,

    specs=[[{"type": "scatter"}],

           [{"type": "scatter"}],

           [{"type": "scatter"}]]

)



fig.add_trace(

    go.Scatter(

        x=df1["date"],

        y=df1["inf_rate"],

        mode="lines",

        name="Infection Rate "

    ),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(

        x=df1["date"],

        y=df1["death_rate"],

        mode="lines",

        name="Death Rate "

    ),

    row=2, col=1

)

fig.add_trace(

    go.Scatter(

        x=df1["date"],

        y=df1["recovery_rate"],

        mode="lines",

        name="Recovery Rate "

    ),

    row=3, col=1

)







fig.update_layout(

    height=800,

    showlegend=True,

    title_text="Rate of infection/Death/Recovery in China",

)



fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(

    y=china_df['Province/State'],

    x=china_df['Confirmed'],

    name='Confirmed',

    orientation='h'

))

fig.add_trace(go.Bar(

    y=china_df['Province/State'],

    x=china_df['Deaths'],

    name='Deaths',

    orientation='h'

))

fig.add_trace(go.Bar(

    y=china_df['Province/State'],

    x=china_df['Recovered'],

    name='Recovered',

    orientation='h'

    

))



fig.update_layout(barmode='stack')

fig.show()
df1
country = df['Country'].unique()

country
df['Country_code']=""

df['Country_code'][df.Country=='China']="CHN"

df['Country_code'][df.Country=='US']="USA"

df['Country_code'][df.Country=='Japan']="JPN"

df['Country_code'][df.Country=='Thailand']="THA"

df['Country_code'][df.Country=='South Korea']="PRK"

df['Country_code'][df.Country=='Mainland China']="CHN"

df['Country_code'][df.Country=='Hong Kong']="HKG"

df['Country_code'][df.Country=='Macau']="MAC"

df['Country_code'][df.Country=='Taiwan']="TWN"

df['Country_code'][df.Country=='Singapore']="SGP"

df['Country_code'][df.Country=='Philippines']="PHL"

df['Country_code'][df.Country=='Malaysia']="MYS"

df['Country_code'][df.Country=='Vietnam']="VNM"

df['Country_code'][df.Country=='Australia']="AUS"

df['Country_code'][df.Country=='Mexico']="MEX"

df['Country_code'][df.Country=='Brazil']="BRA"

df['Country_code'][df.Country=='France']="FRA"

df['Country_code'][df.Country=='Nepal']="NPL"

df['Country_code'][df.Country=='Canada']="CAN"

df['Country_code'][df.Country=='Cambodia']="KHM"

df['Country_code'][df.Country=='Sri Lanka']="LKA"

df['Country_code'][df.Country=='Ivory Coast']="CIV"

df['Country_code'][df.Country=='Germany']="DEU"

df['Country_code'][df.Country=='Finland']="FIN"

df['Country_code'][df.Country=='United Arab Emirates']="ARE"

df['Country_code'][df.Country=='India']="IND"

df['Country_code'][df.Country=='Italy']="ITA"

df['Country_code'][df.Country=='Sweden']="SWE"

df['Country_code'][df.Country=='Russia']="RUS"

df['Country_code'][df.Country=='Spain']="ESP"

df['Country_code'][df.Country=='UK']="GBR"

tot_case = df.groupby(['Country_code','Country'],as_index =False).sum()



#tot_case['Rank'] = tot_case['Confirmed'].rank(method ='max')

tot_case['Rank']=""

tot_case['Rank'][tot_case.Confirmed>=1000 ]='6'

tot_case['Rank'][tot_case.Confirmed>=100&(tot_case.Confirmed<1000) ]='4'

tot_case['Rank'][(tot_case.Confirmed>=30)&(tot_case.Confirmed<100)]='3'

tot_case['Rank'][(tot_case.Confirmed>7)&(tot_case.Confirmed<30)]='2'

tot_case['Rank'][tot_case.Confirmed<=7]='1'

tot_case['Rank'] = pd.to_numeric(tot_case['Rank'])

tot_case
import plotly.express as px

fig = px.scatter_geo(tot_case, locations="Country_code",color="Country",

                     hover_name='Confirmed',size='Rank',projection="natural earth")

fig.update_layout(

        title_text = '2020 confirmed cases of Coronavirus-World Wide',

        showlegend = True,

        geo = dict(

            landcolor = 'rgb(217, 217, 217)',

            showcoastlines = True

        )

        )

fig.update_geos(

    visible=False, resolution=50,

    showcountries=True, countrycolor="RebeccaPurple"

)

fig.show()