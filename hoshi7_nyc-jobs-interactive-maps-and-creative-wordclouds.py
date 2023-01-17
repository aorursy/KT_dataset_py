import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.offline as py

import plotly.tools as tls

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go

import palettable

init_notebook_mode(connected=True)  

plt.style.use('ggplot')

from geopy.geocoders import Nominatim

import plotly_express as px

import folium

from folium.plugins import FastMarkerCluster

from folium.plugins import HeatMap

import geopandas as gpd

import plotly.figure_factory as ff

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from gensim.models import word2vec

import nltk

from collections import Counter

from wordcloud import WordCloud

from PIL import Image

import urllib.request

import random

from palettable.colorbrewer.sequential import Greens_9, Greys_9, Oranges_9, PuRd_9

df = pd.read_csv("../input/nyc-jobs.csv")
df.head()

df.info()
def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig

df['Full-Time/Part-Time indicator'].fillna("Unknown", inplace=True)

value_counts = df['Full-Time/Part-Time indicator'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts, ['#1B9E77', '#D95F02', '#7570B3'], "Job Types"))
df['Salary Frequency'].fillna("Unknown", inplace=True)

value_counts = df['Salary Frequency'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts, ['#7F3C8D', '#11A579', '#3969AC'], "Salary Options"))











value_counts = df['Posting Type'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts, ['#3969AC', '#E73F74'], "Posting Type"))













high_sal_range = (df.groupby('Civil Service Title')['Salary Range To'].mean().nlargest(10)).reset_index()



fig = px.bar(high_sal_range, y="Civil Service Title", x="Salary Range To", orientation='h', title = "Highest High Salary Range",color=  "Salary Range To", color_continuous_scale= px.colors.qualitative.G10).update_yaxes(categoryorder="total ascending")

fig.show()

high_sal_range = (df.groupby('Civil Service Title')['Salary Range From'].mean().nlargest(10)).reset_index()



fig = px.bar(high_sal_range, y="Civil Service Title", x="Salary Range From", orientation='h', title = "Highest (Low) Salary Ranges",color=  "Salary Range From", color_continuous_scale= px.colors.qualitative.T10).update_yaxes(categoryorder="total ascending")



fig.show()

max_positions = (df.groupby('Civil Service Title')['# Of Positions'].mean().nlargest(10)).reset_index()



fig = px.bar(max_positions, y="Civil Service Title", x="# Of Positions", orientation='h', title = "Max Number of Positions",color=  "# Of Positions", color_continuous_scale= px.colors.qualitative.Prism).update_yaxes(categoryorder="total ascending")



fig.show()

top_work_unitdf = df['Division/Work Unit'].value_counts().rename_axis('Work Unit').reset_index(name='counts')[:10]



fig = px.bar(top_work_unitdf, y="Work Unit", x='counts', orientation='h', title = "Popular Work Units",color=  "counts", color_continuous_scale=px.colors.qualitative.D3).update_yaxes(categoryorder="total ascending")



fig.show()

hourly = df[df['Salary Frequency']=='Hourly'][['Salary Range To']]

fig = ff.create_distplot([hourly['Salary Range To']], ['Salary Range To'], bin_size = 10)

fig.show()
job_categorydf = df['Job Category'].value_counts(sort=True, ascending=False)[:10].rename_axis('Job Category').reset_index(name='Counts')

job_categorydf = job_categorydf.sort_values('Counts')
trace = go.Scatter(y = job_categorydf['Job Category'],x = job_categorydf['Counts'],mode='markers',

                   marker=dict(size= job_categorydf['Counts'].values/2,

                               color = job_categorydf['Counts'].values,

                               colorscale='Viridis',

                               showscale=True,

                               colorbar = dict(title = 'Opening Counts')),

                   text = job_categorydf['Counts'].values)



data = [(trace)]



layout= go.Layout(autosize= False, width = 1000, height = 750,

                  title= 'Top 10 Job Openings Count',

                  hovermode= 'closest',

                  xaxis=dict(showgrid=False,zeroline=False,

                             showline=False),

                  yaxis=dict(title= 'Job Openings Count',ticklen= 2,

                             gridwidth= 5,showgrid=False,

                             zeroline=True,showline=False),

                  showlegend= False)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
dict_loc = {'253 Broadway New York Ny': (40.7134096, -74.0075231),

 '42 Broadway, N.Y.': (40.705974, -74.01261459999999),

 '1 Murray Hulbert Ave, Staten I': (40.63786959999999, -74.0736872),

 '1 Centre St., N.Y.': (40.7134096, -74.0075231),

 '15 Metrotech': (40.705974, -74.01261459999999),

 '31-00 47 Ave, 3 FL, LIC NY': (40.63786959999999, -74.0736872),

 '350 Jay St, Brooklyn  Ny': (40.7128306, -74.00371489999999),

 'Lead Hazard - Office of Dir': (40.694028, -73.9839499),

 'Design-Architecture': (40.74209, -73.93551579999999),

 '105 St & 5 Ave': (40.6935265, -73.9876482),

 '20 Nyc Highway 30A Downsville,': (40.7134096, -74.0075231),

 'CP Sustainability Programs': (40.705974, -74.01261459999999),

 'Marlboro': (40.63786959999999, -74.0736872),

 '420 East 26Th St Ny Ny': (40.7128306, -74.00371489999999),

 'R E DEV FOR New Construction': (40.694028, -73.9839499),

 '198 E 161st Street': (40.74209, -73.93551579999999),

 'Law Dept-Civil Litigation': (40.6935265, -73.9876482),

 '600 W 168Th St., N.Y.': (40.7134096, -74.0075231),

 'Resident Engagement': (40.705974, -74.01261459999999),

 '1270 Victory Blvd, Staten Isl': (40.63786959999999, -74.0736872),

 'Operations-EVP': (40.7128306, -74.00371489999999),

 '335 Adams Street, Brooklyn Ny': (40.694028, -73.9839499),

 '100 Gold Street': (40.74209, -73.93551579999999),

 'OCSE Central Court Svcs': (40.6935265, -73.9876482),

 '88-20 Pitkin Ave., Ozone Park': (40.7134096, -74.0075231),

 'Prevention & Intervention-DIR': (40.705974, -74.01261459999999),

 'Brooklyn Storage Management': (40.63786959999999, -74.0736872),

 'LHD-Budget, Personnel & Stats': (40.7128306, -74.00371489999999),

 '1 Metro Tech, Brooklyn Ny': (40.694028, -73.9839499),

 '130 Stuyvesant Place, S.I.': (40.74209, -73.93551579999999),

 'LHD-OFFICE OF THE EVP': (40.6935265, -73.9876482),

 '42-09 28th Street': (40.7128, -74.0060),

 'Design-Engineering': (43.0625186, -88.40262659999999),

 '1826 Arthur Ave., Bronx': (40.7936207, -73.9516092),

 'Office of the Vice-President': (42.0564209, -75.0115939),

 '470 Vanderbilt Ave': (40.7128, -74.0060),

 '22 Reade St, Ny': (40.3380949, -74.26872910000002),

 'DMP-Contract & Analysis Unit': (40.7383829, -73.9761524),

 '1932 Arthur Ave, Bronx': (39.77327469999999, -86.1398544),

 'East 91St St & East River, N Y': (40.82585600000001, -73.921089),

 '75 Park Place New York Ny': (40.7128, -74.0060),

 'Red Hook East': (40.8410669, -73.940253),

 'Real Estate Development-SVP': (40.7128, -74.0060),

 '40 Rector Street New York Ny': (40.6150542, -74.10562469999999),

 'Randalls Island 5-Boro Shops': (40.7128, -74.0060),

 '160 West 100Th Street Ny': (40.6934634, -73.9882453),

 '10 Richmond Terrace, S.I. N.Y.': (40.7103379, -74.0033029),

 'Hazen St-Trans. Div., E. Elm,': (40.7128, -74.0060),

 '215 W. 125Th St., N.Y.': (40.6737517, -73.8480622),

 'Hughes Apartments': (40.7128, -74.0060),

 '18-39 42Nd St, Long Island Cit': (40.6809543, -73.9650276),

 '1 Fordham Plaza, Bronx': (40.7128, -74.0060),

 '33 Beaver St, New York Ny': (40.6931259, -73.9866365),

 '12 W 14Th St., N.Y.': (40.6428301, -74.0770605),

 '105 East 106 Street, New York,': (40.7128, -74.0060),

 '164-21 110 Ave. Jamaica Queens': (40.7493572, -73.9390619),

 'Office Of The Inspector Genera': (37.41836139999999, -95.68028989999999),

 'Tech Serv-Elevator Div (Hrly)': (40.8441212, -73.89414049999999),

 'Louis Armstrong': (38.8975669, -77.0383),

 '80 Maiden Lane': (40.6830766, -73.9680195),

 'Codes/Standards': (40.7142973, -74.0043792),

 'Brooklyn Floating Staff': (40.7128, -74.0060),

 '316 East 88 Street, New York N': (40.8457866, -73.89285439999999),

 '44 Beaver St., N.Y.': (40.7814746, -73.95035539999999),

 '1775 Grand Concourse Bronx N.Y': (40.7144626, -74.0109797),

 '96-05 Horace Harding Expway': (40.6755798, -74.0048104),

 'Environmental Health & Safety': (32.8100274, -96.806286),

 '24 West 61 Street': (40.708606, -74.0147714),

 'Budget-Office Of Director': (40.7992515, -73.9230296),

 '5 Dubois Ave., Staten Island': (40.7959294, -73.9681724),

 'Ravenswood': (40.6424311, -74.0761249),

 'Lower E. Side Consolidation': (40.7128, -74.0060),

 '120 Broadway, New York, NY': (40.8096896, -73.94888639999999),

 '1201 Metropolitan Ave, Bklyn': (34.0257905, -118.4010249),

 '51-02 59th Street': (40.778042, -73.8966049),

 'Health Initiatives': (40.8605938, -73.8902626),

 '5503 Route 9W marlboro': (40.7050972, -74.01207099999999),

 'Capital Projects-EVP': (40.73613340000001, -73.9944189),

 '58-50 57 Road, Maspeth, N.Y.': (40.792935, -73.94716799999999),

 'Office Of The General Manager': (40.6940612, -73.787419),

 'CP Cap Plan-Financial Planning': (38.8996816, -77.03252739999999),

 '88-26 Pitkin Avenue': (29.9622873, -90.1808181),

 '215 Bay St, Staten Island Ny': (40.7545794, -73.861519),

 '465 Columbus Ave. Valhalla, Ny': (40.7071228, -74.00777149999999),

 'EVP-Compliance': (38.6098124, -121.5080607),

 '17 Bristol Street Brooklyn Ny': (40.6781784, -73.9441579),

 'Manhattan Planning Unit': (40.7783133, -73.9495293),

 '280 Broadway, 7th Floor, N.Y.': (40.704911, -74.0108723),

 '356 Flushing Ave, Brooklyn': (40.8464748, -73.91035459999999),

 '100 Church St., N.Y.': (40.7348796, -73.86348939999999),

 'Morrisania Air Rights': (35.7856856, -78.68273669999999),

 '30-30 Thomson Ave L I City Qns': (40.7697706, -73.982895),

 '420 East 38Th St.': (39.7680942, -86.1626787),

 '280 Broadway, 6th Floor, N.Y.': (40.63215700000001, -74.127624),

 '492 First Avenue, New York, Ny': (41.9688072, -87.6791713),

 'Construction, Safety & Quality': (40.7128, -74.0060),

 'Office of the Director': (40.7084773, -74.01059839999999),

 '345 East 59th Street': (40.7144428, -73.9287523),

 'Randalls Island, N.Y.': (40.7361482, -73.9057223),

 '151 East 151st St, Bronx, NY': (29.290794, -94.82650579999999),

 'Law-EVP': (41.5768723, -73.9916574),

 'Office Of Public Information': (40.7128, -74.0060),

 '52-35 58Th St., Woodside, Ny': (40.7223117, -73.9099849),

 '4 World Trade Center': (42.1949775, -71.1998496),

 '137 Centre St., N.Y.': (35.1004385, -106.5710205),

 '11 Park Place, New York, Ny': (40.673379, -73.84703999999999),

 'Queens-SI Floating Staff': (40.6369029, -74.07581499999999),

 'Law Dept - Corporate Affairs': (41.1071626, -73.7815093),

 '2405 Amsterdam Ave., N.Y.': (40.7128, -74.0060),

 '34-02 Queens Boulevard Long Is': (40.6708076, -73.9128186),

 'Pelham Parkway Houses': (40.7082038, -74.01051350000002),

 '71 Smith Avenue, Kingston, Ny': (40.714299, -74.0058154),

 '165 Cadman Plaza East': (40.6979686, -73.96055009999999),

 '30-48 Linden Place': (40.7132993, -74.0101098),

 '55 East 115Th Street, N.Y.': (40.8245054, -73.9170288),

 '31 Chambers St., N.Y.': (40.7441315, -73.9360615),

 'Capital Projects-VP': (39.8252298, -86.150325),

 'Office Of The Chair': (40.714299, -74.0058154),

 '255 Greenwich Street': (40.7400973, -73.9755923),

 '17 Battery Place': (40.7128, -74.0060),

 '60 Bay St. S.I. Ny': (38.9327, -77.20556859999999),

 'R E DEV FOR Preservation': (40.7600927, -73.96309409999999),

 'East New York, Brooklyn Ny': (40.7932271, -73.92128579999999),

 '16 Court Street': (40.8192567, -73.92388079999999),

 '28-11 Queens Plaza No., L.I.C.': (40.7128, -74.0060),

 '132 W 125Th St., N.Y.': (33.7493562, -84.39120439999999),

 '329 Greenpoint Ave., Brooklyn': (40.734384, -73.9094432),

 'Flushing Meadow Pk Olmsted Ctr': (40.71028889999999, -74.01229010000002),

 'Boro Hall Richmond, Staten Isl': (40.71720080000001, -74.0008917),

 '1274 Bedford Ave., Brooklyn': (40.7130634, -74.0082771),

 '4 Metrotech, Brooklyn Ny Ny': (40.7282239, -73.7948516),

 '1 Court Square, Queens': (34.0567061, -118.246127),

 '130-30 28th Ave': (40.847041, -73.9313437),

 'Manhattan Floating Staff': (40.7441363, -73.930965),

 '1010 East 178th Street': (40.8619444, -73.86),

 '158 E 115Th St., N.Y.': (41.9299757, -73.9992371),

 '9 Bond Street': (40.6990819, -73.9893388),

 '330 Jay Street': (40.7696172, -73.83315329999999),

 'Visual Assessment/Remediation': (40.7992309, -73.94475349999999),

 '150 William Street, New York N': (40.7136304, -74.00444949999999),

 '455 First Ave., N.Y.': (40.7128, -74.0060),

 'Field Operations Office of DIR': (36.1761772, -95.90546669999999),

 '248 Duffield St, Brooklyn': (40.7143284, -74.01091439999999),

 '120-55 Queens Blvd, Queens Ny': (40.7051592, -74.01598539999999),

 '9 Metrotech Center, Brooklyn N': (40.6404885, -74.0760463),

 '150-14 Jamaica Ave': (40.8183027, -73.95577449999999),

 'Tech Svcs-CSS': (40.6590529, -73.8759245),

 '280 Broadway, 5th Floor, N.Y.': (40.6936215, -73.99085989999999),

 '10 Walker Rd, Valhalla NY10595': (40.7502888, -73.9380487),

 '375 Pearl Street': (40.8082615, -73.9472409),

 '2551 Bainbridge Ave., Bronx': (40.7316928, -73.9462305),

 '210 Joralemon St., Brooklyn': (40.7515114, -73.8491477),

 'Surfside Gardens': (40.6424095, -74.0760356),

 '250 Church St., N.Y.': (40.67993939999999, -73.9535243),

 '1075 Ralph Avenue Bklyn, N.Y.': (40.6922183, -73.9839956),

 '24-55 Bklyn Qns Expy Woodside': (40.7472946, -73.94433049999999),

 '248 E 161St Street, Bronx': (40.7718656, -73.8391331),

 '65-10 Douglaston Pkwy., Queens': (40.7830603, -73.9712488),

 'REES - Zone Coordination': (40.8410336, -73.88008409999999),

 '520 1St Ave., N.Y.': (40.7975917, -73.9419063),

 '78-88 Park Drive East QueensNY': (40.7257993, -73.9786235),

 '110 William St. N Y': (40.6947467, -73.98760010000001),

 '855 Remsen., Brooklyn': (40.7128, -74.0060),

 'Dept of Mixed Finance': (40.7096402, -74.0057361),

 '182 Joline Ave, Staten Isl': (40.7394052, -73.9776358),

 'NYC - All Boroughs': (32.4844361, -93.7724282),

 '295 Flatbush Ext Brooklyn': (40.69088410000001, -73.9846159),

 'Law-Housing Litigation': (40.7137004, -73.8281571),

 '59-17 Junction Blvd Corona Ny': (40.6943147, -73.9842832),

 '1 Bx Rvr Pkwy & Garage': (40.7041696, -73.8056742),

 '2 Metro Tech': (40.7128, -74.0060),

 '66 John Street, New York, Ny': (40.714299, -74.0058154),

 '1218 Prospect Place Bklyn, N.Y': (41.0790113, -73.8141354),

 '109 E 16Th St., N.Y.': (40.7109599, -74.0014202),

 '66-26 Metropolitan Ave., Queen': (40.86277080000001, -73.8931143),

 '421 East 26th Street NY NY': (40.6923814, -73.99083619999999),

 '1 Police Plaza, N.Y.': (40.57436300000001, -73.99794709999999),

 '75-20 Astoria Blvd': (40.7179316, -74.0059174),

 '149-40 134 Street, Queens Ny': (40.6492631, -73.9200707),

 '24 Ontario Ave., Staten Island': (40.75633699999999, -73.9066129),

 '100 Central Park Ave N Yonkers': (40.8255371, -73.9187548),

 '890 Garrison Avenue': (40.7524397, -73.7413491),

 '90-27 Sutphin Blvd, Queens Ny': (40.68219, -73.96821),

 '161 William St  New York N Y': (40.7412595, -73.9746112),

 '11 Metrotech Center': (40.718124, -73.825285),

 'Real Estate Development-VP': (40.7087931, -74.0068363),

 '59 Maiden Lane': (40.646947, -73.912397),

 '7870 State Rd 42 Grahamsville,': (40.7128, -74.0060),

 '900 Sheridan Ave., Bronx': (40.5080805, -74.2357454),

 '235 E 20Th St., N.Y.': (40.72764300000001, -73.937443),

 '5 Manhattan West': (40.691787, -73.9820294),

 '55 Water St Ny Ny': (38.906653, -77.043036),

 'Jefferson Houses': (40.7348102, -73.8644434),

 'Arsenal 830 Fifth Ave, New Yor': (41.0659999, -73.7735373),

 'Office for Exec Proj Manager': (40.693364, -73.9857147),

 '1 Centre Street Ny, Ny': (40.7088657, -74.0078458),

 '1601 Ave. S, Brooklyn': (40.6733051, -73.9358673),

 'Analysis & Reporting': (40.735627, -73.988013),

 '16 Little Hollow Road': (40.7105133, -73.89170589999999),

 'Wards Island, N.Y.': (40.7383514, -73.97570259999999),

 'Heating Mgt-Operations': (40.7121101, -74.0018864),

 '22 Cortlandt Street': (40.7645473, -73.89321269999999),

 '55 West 125 St, New York, Ny': (40.665278, -73.80603099999999),

 '2389 Route 28A, Shokan, Ny': (40.614849, -74.105334),

 '125 Worth Street, Nyc': (40.910008, -73.8768959),

 '2 Lafayette St., N.Y.': (40.8171143, -73.89077979999999),

 'Hazen St-Sod-Supp.Svcs., E.Elm': (40.7024295, -73.8079068),

 'Edenwald': (40.710219, -74.00619),

 '3701 Jerome Ave, Bx NY 10467': (40.6951244, -73.9847252),

 'OFC OF PUBLIC/PRIVATE PARTNERS': (33.5103069, -112.0271809),

 '1 Bay St., S.I.,Ny': (40.7087357, -74.0081164),

 'CP Cap Plan-Technical Planning': (41.8478701, -74.547935),

 'Mold ASMT & Remediation-DIR': (40.8267469, -73.92069459999999),

 'Compliance & Training': (40.736597, -73.9831951),

 '83 Maiden Lane, New York Ny': (40.7530688, -73.9994561),

 'EVP-NextGen Ops': (40.7033226, -74.0088962),

 'Rikers Island': (40.7958142, -73.93974349999999),

  '275-285 Bergen St, Brooklyn Ny' :(40.683570,-73.982130),

'48-34 35Th St., Queens':(40.774860,-73.908900),

'Tech Svcs-Central Office Staff': (40.7033226, -74.0088962),

'430 East 30 Street, New York N':(44.848660,-74.291740),

'Morris Houses':(42.548680, -75.247720),

'345 Adams St., Brooklyn':(40.692720,-73.988450),

'PIM-Office of Director':(41.194650,-74.185140),

'Monroe-Clason Point':(43.233110,-77.927690),

'111 Livingston St., Bklyn., N.':(40.410160,-73.989800),

'Cypress Hills': (40.619270,-73.956960),

'Melrose': (42.845260,-73.618480),

'101-07 Farragut Road, Brooklyn': (40.644300,-73.907570),

'Roosevelt':(40.766600,-73.945220),

'City Hall':(40.708390,-73.834100),

'1200 Waters Place, Bronx Ny':(40.852945,-73.836788),

'151-20 Jamaica Avenue':(40.702222,-73.803333),

'18-22 Jackson St. New York N Y':(40.716055,-73.950047),

'263 Tompkins Ave., Brooklyn':(40.688992,-73.944925),

'400 8Th Ave., N.Y.':(40.749330, -73.995070),

'516 Bergen St., Brooklyn':(40.680216,-73.973402),

'88-11 165 Street Jamaica':(40.708042, -73.796363),

'Brooklyn Navy Yard, Brooklyn':(40.698351,-73.972434),

'Prospect Pk 95 Ppw &5Th St':(40.658025,-73.974456),

'Default':(40.710219, -74.00619),



           

           }
df['coordinates'] = df['Work Location'].apply(lambda x: dict_loc[x] )

df['lat'] = df['coordinates'].apply(lambda x: float(x[0]) if x != 0 else 0)

df['lon'] = df['coordinates'].apply(lambda x: float(x[1]) if x != 0 else 0)



#Creating two columns for latitude and longitude( so value can be easily converted to float)
# This is used to set the marker type (The red icons on the map). It changes it from the default Home-Blue based one, which would have lost this map it's oomph.

callback = ('function (row) {'

                    'icon = L.AwesomeMarkers.icon(({icon: "map-marker", markerColor: "red"}));'

                'marker = L.marker(new L.LatLng(row[0], row[1]));'

                'marker.setIcon(icon);'

                'return marker};')
coordinates = list(zip(list(df['lat']), list(df['lon'])))

map1 = folium.Map(location=[40.7128, -74.0060], zoom_start=12.03, icon = 'cloud')

FastMarkerCluster(data=coordinates, callback = callback).add_to(map1)

map1
mapbox_access_token = "pk.eyJ1Ijoic3JhbGxpIiwiYSI6ImNqeWx4NzMzMTBkN2ozZXBoaTZrd2RhaWQifQ.237WfJXIXNdEQfA6veC3MQ"
coordinates = pd.DataFrame(df[['Work Location','coordinates']])
def generateBaseMap(default_location=[40.7128, -74.0060], default_zoom_start=12):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map
basemap = generateBaseMap()
def produce_openings(name):

    data_openings=pd.DataFrame(df[df["Business Title"]==name]['Work Location'].value_counts().reset_index())

    data_openings.columns=['Work Location','Counts']

    

    data_openings=data_openings.merge(coordinates,on="Work Location",how="left").dropna()

    data_openings['lat'] , data_openings['lon'] =zip(*data_openings['coordinates'].values)

    data_openings.drop_duplicates(keep = 'first', inplace = True)

    data_openings = data_openings.reset_index()

    return data_openings[['Work Location','Counts','lat','lon']]



def produce_trace(data_openings,name):

        data_openings['text']=data_openings['Work Location']+'<br>'+data_openings['Counts'].astype(str)

        trace =  go.Scattermapbox(

           

                lat=data_openings['lat'],

                lon=data_openings['lon'],

                mode='markers',

                marker=go.scattermapbox.Marker(

                    size=data_openings['Counts']*7

                ),

                text=data_openings['text'],name=name

            )

        

        return trace



def produce_data(col, name):

    data=pd.DataFrame(df[df[col]==name]['Work Location'].value_counts().reset_index())

    data.columns=['Work Location','Counts']

    data=data.merge(coordinates,on="Work Location",how="left").dropna()

    data['lat'] , data['lon'] =zip(*data['coordinates'].values)

    data.drop_duplicates(keep = 'first', inplace = True)

    data = data.reset_index()

    return data[['Work Location','Counts','lat','lon']]
Engineering_heatmap=produce_data('Job Category','Engineering, Architecture, & Planning')

HeatMap(Engineering_heatmap[['lat','lon','Counts']].values.tolist(),zoom=20,radius=15).add_to(basemap)

basemap
engineer=df[df['Job Category']=='Engineering, Architecture, & Planning'][['Business Title','coordinates']]

engineerdf = engineer['Business Title'].value_counts(sort=True, ascending=False)[:10].rename_axis('Business Title').reset_index(name='Counts')

engineerdf



data=[] 

for row in engineerdf['Business Title']:

    data_openings=produce_openings(row) 

    trace_0=produce_trace(data_openings,row)

    data.append(trace_0)







layout = go.Layout(title="Top 10 Engineer Placement Openings",

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,style="streets",

        center=dict(

            lat=40.7128,

            lon=-74.0060

        ),

        pitch=0,

        zoom=10

    ),

)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Montreal Mapbox')
Tech_heatmap=produce_data('Job Category','Technology, Data & Innovation')

HeatMap(Tech_heatmap[['lat','lon','Counts']].values.tolist(),zoom=20,radius=15).add_to(basemap)

basemap
technical=df[df['Job Category']=='Technology, Data & Innovation'][['Business Title','coordinates']]

techincaldf = technical['Business Title'].value_counts(sort=True, ascending=False)[:10].rename_axis('Business Title').reset_index(name='Counts')

techincaldf



data=[] 

for row in techincaldf['Business Title']:

    data_openings=produce_openings(row) 

    trace_0=produce_trace(data_openings,row)

    data.append(trace_0)







layout = go.Layout(title="Top 10 Techincal Job Openings",

    autosize=True,

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,style="streets",

        center=dict(

            lat=40.7128,

            lon=-74.0060

        ),

        pitch=0,

        zoom=10

    ),

)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='Montreal Mapbox')
Qual_mask = np.array(Image.open(urllib.request.urlopen('https://i.imgur.com/XnFmbtf.png')))

skill_mask = np.array(Image.open(urllib.request.urlopen('https://i.imgur.com/V3R3KZS.png')))

residency_mask = np.array(Image.open(urllib.request.urlopen('https://i.imgur.com/eodMnCZ.png')))

job_mask = np.array(Image.open(urllib.request.urlopen('https://i.imgur.com/JVuN0kA.png')))

df['Min_req']=df['Minimum Qual Requirements'].apply(lambda x : x.split(',') if type(x)==str else [''])

df['Job_desc'] = df['Job Description'].apply(lambda x : x.split(',') if type(x)==str else [''])

df['res_req']=df['Residency Requirement'].apply(lambda x : x.split(',') if type(x)==str else [''])

df['Pref_skill'] = df['Preferred Skills'].apply(lambda x : x.split(',') if type(x)==str else [''])
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple(Greys_9.colors[random.randint(2,8)])





def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple(Greens_9.colors[random.randint(2,8)])



def orange_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple(Oranges_9.colors[random.randint(2,8)])



def PuRd_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple(PuRd_9.colors[random.randint(2,8)])



def produce_wordcloud(dataframe, title, mask, color):

    

  

    plt.figure(figsize=(10, 10))

    corpus=dataframe.values.tolist()

    corpus=','.join(x  for list_words in corpus for x in list_words)

    wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False, height=1500,

                 mask = mask).generate(corpus)

    wordcloud.recolor(color_func=color)

    plt.axis("off")

    plt.title(title)    

    return plt.imshow(wordcloud)

    

produce_wordcloud(df['Job_desc'], "Job Description", job_mask, orange_color_func)
produce_wordcloud(df['Min_req'], "Minimum Qualification Required", Qual_mask, grey_color_func)
produce_wordcloud(df['res_req'], "Residency Requirement", residency_mask, PuRd_color_func)
produce_wordcloud(df['Pref_skill'], "Preferred Skills", skill_mask, green_color_func)