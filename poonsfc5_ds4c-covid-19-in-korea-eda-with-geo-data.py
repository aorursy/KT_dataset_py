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
df_time           = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')

df_peroid         = df_time[df_time["date"] >= '2020-02-25'].copy()

df_peroid["rate"] = df_peroid["confirmed"]/df_peroid["test"] * 100 

import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))

barplot = sns.barplot(x=df_peroid['date'], y=df_peroid['rate'], palette="rocket")

plt.xticks(rotation=90)



plt.show()
# Create a data frame to show the new case 

df_non_acc = df_time.copy()

r, d = df_time.shape

for i in range(1, r):              # skip the first row

    for j in range(2,d) :          # skip the first two columns

        df_non_acc.iloc[i,j] = df_non_acc.iloc[i,j] - df_time.iloc[(i-1),j]



#Save the dataframe into a file

file_name='/kaggle/working/Time_daily.csv'

df_non_acc.to_csv(file_name, sep=',', encoding='utf-8')



df_peroid      = df_non_acc[df_non_acc["date"] >= '2020-02-25'].copy()

df_peroid["rate"] = df_peroid["confirmed"]/df_peroid["test"] * 100 



plt.figure(figsize=(10,5))

barplot = sns.barplot(x=df_peroid['date'], y=df_peroid['rate'], palette="rocket")

plt.xticks(rotation=90)



plt.show()
df_age        = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')

df_daily_age  = df_age.copy()



r, d = df_daily_age.shape

for i in range(9, r):        # skip the first nine row

    for j in range(3,d) :   # skip the first three columns

        df_daily_age.iloc[i,j] = df_daily_age.iloc[i,j] - df_age.iloc[(i-9),j]

    #print(df_daily_age.iloc[i,:])



# Create TimeAge daily stats file

file_name2='/kaggle/working/TimeAge_daily.csv'

df_daily_age.to_csv(file_name2, sep=',', encoding='utf-8')



# Plot Accuminated Total Confirmed Cases by Age

sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="age", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_age)

plt.title('Accuminated Total Confirmed Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot Daily Confirmed Cases by Age

sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="age", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_age[df_daily_age["date"] > '2020-03-02'])

plt.title('Daily Confirmed Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot Accuminated Total Deceased Cases by Age

sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="age", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_age)

plt.title('Accuminated Total Deceased Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot Daily Deceased Cases by Age

sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="age", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_age[df_daily_age["date"] > '2020-03-02'])

plt.title('Daily Deceased Cases by Age', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)
df_gender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')

df_daily_gender  = df_gender.copy()



r, d = df_daily_gender.shape

for i in range(2, r):              # skip the first row

    for j in range(3,d) :          # skip the first two columns

        df_daily_gender.iloc[i,j]   = df_daily_gender.iloc[i,j] - df_gender.iloc[(i-2),j]



# save the daily output to file

file_name1='/kaggle/working/TimeGender_daily.csv'

df_daily_gender.to_csv(file_name1, sep=',', encoding='utf-8')



# Plot accuminated total confirmed cases by sex

sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="sex", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=df_gender)

plt.title('Accuminated Total Confirmed Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot daily confirmed cases by sex

sns.set(style="darkgrid")

sns.relplot(x="date", y="confirmed", hue="sex", size="confirmed",

            sizes=(40, 400), alpha=.5, palette="muted", height=6, data=df_daily_gender[df_daily_gender["date"] > '2020-03-02'])

plt.title('Daily Confirmed Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot accuminated total of deceased cases by Sex

sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="sex", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_gender[df_gender["date"] > '2020-03-02'])

plt.title('Accuminated Total Deceased Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)



# Plot Daily deceased cases by Sex

sns.set(style="darkgrid")

sns.relplot(x="date", y="deceased", hue="sex", size="deceased",

            sizes=(40, 400), alpha=.5, palette="muted",

            height=6, data=df_daily_gender[df_daily_gender["date"] > '2020-03-02'])

plt.title('Daily Deceased Cases by Sex', loc = 'left', fontsize = 12)

plt.xticks(rotation=90)
def wordcloud_column(dataframe):

    from wordcloud import WordCloud, STOPWORDS 

 

    comment_words = ' '

    stopwords = set(STOPWORDS) 

  

    # iterate through the csv file 

    for k in range(len(dataframe)):

        # typecaste each val to string 

        val = str(dataframe.iloc[k,0]) 

        # split the value 

        tokens = val.split()

    

        # Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 

          

        for words in tokens: 

            comment_words = comment_words + words + ' '

  

    # lower max_font_size

    wordcloud = WordCloud(width=400, height=200,background_color ='white', max_font_size=60).generate(comment_words)

    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()



# Find the word cloud for infection case in Patientinfo file

df_patient = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')

df_reason  = df_patient[['infection_case']]

df_reason  = df_reason[(df_reason['infection_case'].notna())]

wordcloud_column(df_reason)



# Find the word cloud for infection case in Case file

df_case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

df_case = df_case[['infection_case']]

df_case = df_case[(df_case['infection_case'].notna())]

wordcloud_column(df_case)
# Create a legend

legend_html = '''

        <div style="position: fixed; bottom: 300px; left: 50px; width: 160px; height: 110px; 

                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"

                    >&nbsp; <b>Legend</b> <br>

                    &nbsp; Confirmed < 100 &nbsp&nbsp&nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff9900"></i><br>

                    &nbsp; Confirmed < 1000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#cc33ff"></i><br>

                    &nbsp; Confirmed < 3000 &nbsp; 

                        <i class="fa fa-circle" style="font-size:14px;color:#ff0000"></i><br>

                    &nbsp; Confirmed >= 3000

                        <i class="fa fa-circle" style="font-size:14px;color:#660000"></i>

        </div>

        ''' 



def color(total):

    # Color range

    col_100  = "#ff9900"

    col_1000 = "#cc33ff"

    col_3000 = "#ff0000"

    over     = "#660000"

    if (total < 100):   

            rad = total/10

            color = col_100

    elif (total < 1000): 

            rad = min(total/10, 20)

            color = col_1000

    elif (total < 3000): 

            rad = min(total/10, 30)

            color = col_3000

    else: 

            rad = 35

            color = over

    return rad, color



# read the region coordinates from region.csv



import folium

from   folium import plugins



df_province = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')

df_region   = pd.read_csv('/kaggle/input/region-new/Region_New.csv')

df_current = df_province[df_province['date']==df_province['date'].max()]

df_row = df_current.join(df_region.set_index('city')[['latitude','longitude']], on='province')



# map0 = folium.Map(location=[35.7982008,125.6296572], control_scale=True, tiles='Stamen Toner', zoom_start=7)

map0 = folium.Map(location=[35.7982008,125.6296572], control_scale=True, zoom_start=7)

folium.TileLayer('openstreetmap').add_to(map0)

folium.TileLayer('CartoDB positron',name='Positron').add_to(map0)

folium.TileLayer('CartoDB dark_matter',name='Dark Matter').add_to(map0)

folium.TileLayer('Stamen Terrain',name='Terrain').add_to(map0)

folium.TileLayer('Stamen Toner',name='Toner').add_to(map0)

# Enable the layer control 

folium.LayerControl().add_to(map0)

# Enable Expand fullscreen feature

plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(map0) 

map0.get_root().html.add_child(folium.Element(legend_html))



for index, row in df_row.iterrows():

    date      = row['date']

    confirmed = row["confirmed"]

    deceased  = row["deceased"]

    released  = row['released']

    province  = row["province"]

    lat   = row["latitude"]

    long  = row["longitude"]

    

    # generate the popup message that is shown on click.

    popup_text = "<b>Date:</b> {}<br><b>Province: </b>{}<br><b>Confirmed:</b> {}<br><b>Deceased: </b>{}"

    popup_text = popup_text.format(date, province, confirmed, deceased)          

    

    # select colors and radius

    rad, col = color(confirmed)

    folium.CircleMarker(location=(lat,long), radius = rad, color=col, popup=popup_text, 

                        opacity= 4.0, fill=True).add_to(map0)



map0.save('SKConfirmed_Mar20.html')

display(map0)
df_data = df_province[df_province['date']>='2020-02-18']

df_row = df_data.join(df_region.set_index('city')[['latitude','longitude']], on='province')



from folium.plugins import TimestampedGeoJson



map1 = folium.Map(location=[35.7982008,125.6296572], zoom_start=7, control_scale=True, tiles='openstreetmap')

folium.TileLayer('openstreetmap').add_to(map1)

folium.TileLayer('CartoDB positron',name='Positron').add_to(map1)

folium.TileLayer('CartoDB dark_matter',name='Dark Matter').add_to(map1)

folium.TileLayer('Stamen Terrain',name='Terrain').add_to(map1)

folium.TileLayer('Stamen Toner',name='Toner').add_to(map1)

# Enable the layer control 

folium.LayerControl().add_to(map1)

# Enable Expand fullscreen feature

plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(map1) 

map1.get_root().html.add_child(folium.Element(legend_html))



features = []

for index, row in df_row.iterrows():

    date      = row['date']

    confirmed = row["confirmed"]

    deceased  = row["deceased"]

    released  = row['released']

    province  = row["province"]

    lat       = row["latitude"]

    long      = row["longitude"]

    rad, col = color(confirmed)

    popup_text = "<b>Date</b>:{}<br><b>Region:</b> {}<br><b>Confirmed: </b>{}"

    popup_text = popup_text.format(date, province, confirmed)

    feature = {

            'type': 'Feature',

            'geometry': {

                'type':'Point', 'coordinates':[long, lat]},

                'properties': {

                    'time': date.__str__(),

                    'style': {'color' : col},

                    'popup': popup_text,

                    'icon': 'circle',

                'iconstyle':{

                    'fillColor': col,

                    'fillOpacity': 0.8,

                    'fill': 'true',

                    #'stroke': 'true',

                    'radius': rad}

                    }

                }

    features.append(feature)

TimestampedGeoJson(

        {'type': 'FeatureCollection',

        'features': features}

        , period='P1D'

        , add_last_point=True

        , auto_play=True

        , loop=True

        , max_speed=1

        , loop_button=True

        , date_options='YYYY-MM-DD'

        , time_slider_drag_update=True).add_to(map1)



filename='SKGrowth_Mar20Time.html'

map1.save(filename)

display(map1)
def zoom_prov(province_in, zoomstart=12):

    province = df_route[df_route["province"] == province_in]

    # Initialize the area

    

    init_points  = (np.average(province.iloc[:,5]),np.average(province.iloc[:,6]))

    label_points = (np.average(province.iloc[:,5])-0.23,np.average(province.iloc[:,6])-0.23)



    temp_map     = folium.Map(location=init_points, zoom_start=zoomstart, control_scale=True,tiles='CartoDB Positron')

    plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(temp_map)



    # Create a City Name

    name1 = '''

        <div style="position: fixed; bottom: 50px; left: 50px; width: 130px; height: 65px; 

                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"

                    >&nbsp; <br>  <b>Province: '''  

    name2 =  province_in + ' </b> </div> '

    name  = name1 + name2

    temp_map.get_root().html.add_child(folium.Element(name))

    

    for index, row in province.iterrows():

        # Calculate the radius

        id    = row["patient_id"]

        date  = row["date"]

        city  = row["city"]

        case  = row["infection_case"]

        loc   = (row["latitude"], row["longitude"])

     

        # generate the popup message that is shown on click.

        popup_text = "<b>Patient Id:</b> {}<br><b>Date:</b> {}<br><b>City:</b> {}<br><b>Loc:</b> {}<br><b>Case:</b> {}"

        popup_text = popup_text.format(id,date,city,loc,case)

        icon       = folium.Icon(color='red', icon='info-sign')

        popup = folium.Popup(popup_text, max_width=300, min_width=80)

        folium.Marker(loc, popup=popup, icon=icon).add_to(temp_map)

    display(temp_map)

    

df_route   = pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')

df_route   = df_route.join(df_patient.set_index('patient_id')[['infection_case']], on='patient_id')

zoom_prov('Seoul')

zoom_prov("Gangwon-do", zoomstart=12)

zoom_prov("Daegu")

zoom_prov("Gyeonggi-do",zoomstart=9)
def trace_route(id, zoomstart):

    

    route        = df_route[df_route["patient_id"] == id]

    

    init_points  = (np.average(route.iloc[:,5]),np.average(route.iloc[:,6]))

    label_points = (np.average(route.iloc[:,5])-0.25,np.average(route.iloc[:,6])-0.25)

    temp_map = folium.Map(location=init_points, zoom_start=zoomstart, control_scale=True,tiles='CartoDB Positron')

    plugins.Fullscreen( position='topleft', title='Expand', title_cancel='Exit', force_separate_button=True ).add_to(temp_map)

    

    prev_date   = '0000-01-01'

    prev_loc    = ''

    for index, row in route.iterrows():

        # Calculate the radius

        id    = row["patient_id"]

        date  = row["date"]

        case = row["infection_case"]

        city  = row["city"]

        loc   = (row["latitude"], row["longitude"])

    

        text = 'Case:' + str(id) 

        folium.map.Marker(label_points, icon=DivIcon(icon_size=(200,45), icon_anchor=(0,0),

            html='<div style="top:; background-color: white; font-size: 18pt">%s</div>' % text)).add_to(temp_map)

    

        # generate the popup message that is shown on click.

        popup_text = "ID:{}<br><b>Date</b>:{}<br><b>Case</b>:{}<br><b>City</b>:{}<br><b>Loc</b>:{}"

        popup_text = popup_text.format(id,date,case,city,loc)

        icon       = folium.Icon(color='red', icon='info-sign')

        popup      = folium.Popup(popup_text, max_width=300, min_width=80)

        folium.Marker(loc, popup=popup, icon=icon).add_to(temp_map)



        if (prev_date == date):

            folium.PolyLine(locations=(prev_loc, loc), line_opacity = 0.2, color='blue').add_to(temp_map)

    

        prev_loc  = loc

        prev_date = date

    

    display(temp_map)



# Find the path of ID 1000000002

from folium.features import DivIcon



trace_route(1000000002,10)

trace_route(2000000003,8)
