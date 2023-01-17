import requests

import pandas as pd



# Incep cu DailyCaseReport



r = requests.get('https://covid19.geo-spatial.org/api/dashboard/getDailyCaseReport')



def extract_values(obj, key): # e o functie care ma asigura ca pot citi jsonuri complexe

    

    arr = []



    def extract(obj, arr, key):

        

        if isinstance(obj, dict):

            for k, v in obj.items():

                if isinstance(v, (dict, list)):

                    extract(v, arr, key)

                elif k == key:

                    arr.append(v)

        elif isinstance(obj, list):

            for item in obj:

                extract(item, arr, key)

        return arr



    rezultate = extract(obj, arr, key)

    return rezultate



day_case = extract_values(r.json(), 'day_case')

day_no = extract_values(r.json(), 'day_no')

new_case_no = extract_values(r.json(), 'new_case_no')

new_dead_no = extract_values(r.json(), 'new_dead_no')

new_healed_no = extract_values(r.json(), 'new_healed_no')

total_case = extract_values(r.json(), 'total_case')

total_dead = extract_values(r.json(), 'total_dead')

total_healed = extract_values(r.json(), 'total_healed')



CazuriZilnice = pd.DataFrame()

CazuriZilnice['Data'] = day_case

CazuriZilnice['Ziua'] = day_no

CazuriZilnice['CazuriNoi'] = new_case_no

CazuriZilnice['DeceseNoi'] = new_dead_no

CazuriZilnice['VindecariNoi'] = new_healed_no

CazuriZilnice['TotalCazuri'] = total_case

CazuriZilnice['TotalDecese'] = total_dead

CazuriZilnice['TotalVindecati'] = total_healed



CazuriZilnice.Data = pd.to_datetime(CazuriZilnice.Data)



print(CazuriZilnice.dtypes)



CazuriZilnice.head()
# Deaths



r = requests.get('https://covid19.geo-spatial.org/api/dashboard/getDeadCasesByCounty')



DeceseJudete = pd.DataFrame()

DeceseJudete['TotalDecese'] = extract_values(r.json(), 'total_county')

DeceseJudete['Judet'] = extract_values(r.json(), 'county')

DeceseJudete['CodJudet'] = extract_values(r.json(), 'county_code')

DeceseJudete.head()
# Healed by counties



r = requests.get('https://covid19.geo-spatial.org/api/dashboard/getHealthCasesByCounty')



VindecatiJudete = pd.DataFrame()

VindecatiJudete['Judet'] = extract_values(r.json(), 'county')

VindecatiJudete['Vindecari'] = extract_values(r.json(), 'total_county')

VindecatiJudete.head()
# Total cases evolution



import plotly.express as px

import numpy as np



x = CazuriZilnice.Data

y = CazuriZilnice.TotalCazuri



fig = px.bar(x=x, y=y)



fig.update_layout(title_text='EVOLUTIE TOTAL CAZURI', title_x=0.5)

fig.show()
# New daily cases



x = CazuriZilnice.Data

y = CazuriZilnice.CazuriNoi



fig = px.line(x=x, y=y)



fig.update_layout(title_text='EVOLUTIE CAZURI NOI', title_x=0.5)

fig.show()
# Age groups



r = requests.get('https://covid19.geo-spatial.org/api/dashboard/getCasesByAgeGroup')

Varsta = pd.DataFrame()

#Varsta['data'] = extract_values(r.json(), 'timestamp')

Varsta['0-9'] = extract_values(r.json(), '0-9')

Varsta['10-19'] = extract_values(r.json(), '10-19')

Varsta['20-29'] = extract_values(r.json(), '20-29')

Varsta['30-39'] = extract_values(r.json(), '30-39')

Varsta['40-49'] = extract_values(r.json(), '40-49')

Varsta['50-59'] = extract_values(r.json(), '50-59')

Varsta['60-69'] = extract_values(r.json(), '60-69')

Varsta['70-79'] = extract_values(r.json(), '70-79')

Varsta['80 +'] = extract_values(r.json(), '>80')

Varsta = Varsta.T

Varsta.reset_index()



fig = px.pie(Varsta.index, values=[int(item) for item in Varsta.values], names=Varsta.index,

            title='Cazuri pe grupe de varsta')

fig.show()
# Deaths by county



x = DeceseJudete.Judet

y = DeceseJudete.TotalDecese



fig = px.bar(x=x, y=y)



fig.update_layout(title_text='Numar de decese pe judet', title_x=0.5)

fig.show()
# Healed by county



x = VindecatiJudete.Judet # elimin necunoscutii

y = VindecatiJudete.Vindecari # elimin necunoscutii



fig = px.bar(x=x, y=y)



fig.update_layout(title_text='Numar de vindecari pe judet', title_x=0.5)

fig.show()
# Healed by county, unknown removed



x = VindecatiJudete.Judet[1:] # elimin necunoscutii

y = VindecatiJudete.Vindecari[1:] # elimin necunoscutii



fig = px.bar(x=x, y=y)



fig.update_layout(title_text='Numar de vindecari pe judet', title_x=0.5)

fig.show()
# Deaths and Healed by county



situatieDV = pd.merge(VindecatiJudete, DeceseJudete, on='Judet')

situatieDV.Judet = situatieDV.Judet.apply(str)

situatieDV.head(), situatieDV.dtypes
# Deaths vs Healed by county



import plotly.graph_objects as go



x = situatieDV.Judet



fig = go.Figure(data=[

    go.Bar(name='Vindecari', x=x, y=situatieDV.Vindecari),

    go.Bar(name='Decese', x=x, y=situatieDV.TotalDecese)

])



fig.update_layout(barmode='group')

fig.update_layout(title_text='Decese vs Vindecari', title_x=0.5)

fig.show()
# get geographic coordinates of each county



import json

path = '../input/regiuni-ro/regions.json'



with open(path) as jsonfile:

    data = json.load(jsonfile)



name = []

lat = []

lng = []



situatieDV.Judet[13] = 'SATU-MARE'



for item in situatieDV.Judet:

    x = data[item][0] 

    name.append(item)

    lat.append(x['lat'])

    lng.append(x['long'])

coord = pd.DataFrame()

coord['Judet'] = name 

# names of counties capitals are not spelled right, so I get approx location of any small city in that county

# don't need to be too precise

coord['lat'] = lat

coord['lng'] = lng

geo_sit = pd.merge(situatieDV, coord, on='Judet')

geo_sit.lat = geo_sit.lat.astype('float')

geo_sit.lng = geo_sit.lng.astype('float')

print(geo_sit.dtypes)

geo_sit
# map with population (nothing about covid)



import plotly.express as px

fig = px.scatter_mapbox(geo_sit, lat=geo_sit.lat, lon=geo_sit.lng, hover_name="Judet", 

                        hover_data=["TotalDecese", "Vindecari" ],

                        color_discrete_sequence=["fuchsia"], zoom=6, height=600, width=850, size=geo_sit.TotalDecese )

fig.update_layout(mapbox_style="carto-positron")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(

    rows=5, cols=2,

    specs=[[{}, {"rowspan": 2}],

           [{}, None],

           [{"rowspan": 2, "colspan": 2}, None],

           [None, None],

           [{},{"type": "domain"}]],

    print_grid=False)



fig.add_trace(go.Bar(x=CazuriZilnice.Data, y=CazuriZilnice.TotalCazuri, name="Total Cazuri"),

              row=1, col=1)

fig.add_trace(go.Bar(x=CazuriZilnice.Data, y=CazuriZilnice.CazuriNoi, name="Cazuri zilnice noi"), row=1, col=2)



fig.add_trace(go.Bar(x=Varsta.index, y=[int(item) for item in Varsta.values], name="Cazuri pe grupe de varsta"), row=2, col=1)



fig.add_trace(go.Bar(x = VindecatiJudete.Judet[1:], y = VindecatiJudete.Vindecari[1:], name="Vindecari pe judete"), row=3, col=1)



fig.add_trace(go.Bar(x = DeceseJudete.Judet[1:], y = DeceseJudete.TotalDecese[1:], name="Decese pe judet"), row=5, col=1)



fig.update_layout(height=1200, width=900, title_text="Covid sintetic")

fig.show()