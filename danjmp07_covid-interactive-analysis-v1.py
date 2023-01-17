# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



#Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

import datetime

from ipywidgets import widgets, Layout

from IPython.display import update_display, HTML

import plotly.express as px

import plotly.graph_objects as go



df = pd.read_csv('/kaggle/input/datacsv/data.csv', sep = ',')

df['Date'] = df['Date'].astype(str)

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

#data['Date']= pd.to_datetime(data['Date']) 

#df.head()



dic_colors = {'Afghanistan': '#FD3216',

 'Albania': '#00FE35',

 'Algeria': '#6A76FC',

 'Andorra': '#FED4C4',

 'Angola': '#FE00CE',

 'Antigua and Barbuda': '#0DF9FF',

 'Argentina': '#F6F926',

 'Armenia': '#FF9616',

 'Australia': '#479B55',

 'Austria': '#EEA6FB',

 'Azerbaijan': '#DC587D',

 'Bahamas': '#D626FF',

 'Bahrain': '#6E899C',

 'Bangladesh': '#00B5F7',

 'Barbados': '#B68E00',

 'Belarus': '#C9FBE5',

 'Belgium': '#FF0092',

 'Belize': '#22FFA7',

 'Benin': '#E3EE9E',

 'Bhutan': '#86CE00',

 'Bolivia': '#BC7196',

 'Bosnia and Herzegovina': '#7E7DCD',

 'Brazil': '#008000',

 'Brunei': '#E48F72',

 'Bulgaria': '#FD3216',

 'Burkina Faso': '#00FE35',

 'Cabo Verde': '#6A76FC',

 'Cambodia': '#FED4C4',

 'Cameroon': '#FE00CE',

 'Canada': '#0DF9FF',

 'Central African Republic': '#F6F926',

 'Chad': '#FF9616',

 'Chile': '#479B55',

 'China': '#EEA6FB',

 'Colombia': '#DC587D',

 'Congo (Brazzaville)': '#D626FF',

 'Congo (Kinshasa)': '#6E899C',

 'Costa Rica': '#00B5F7',

 "Cote d'Ivoire": '#B68E00',

 'Croatia': '#C9FBE5',

 'Cuba': '#FF0092',

 'Cyprus': '#22FFA7',

 'Czechia': '#E3EE9E',

 'Denmark': '#86CE00',

 'Diamond Princess': '#BC7196',

 'Djibouti': '#7E7DCD',

 'Dominica': '#FC6955',

 'Dominican Republic': '#E48F72',

 'Ecuador': '#FD3216',

 'Egypt': '#00FE35',

 'El Salvador': '#6A76FC',

 'Equatorial Guinea': '#FED4C4',

 'Eritrea': '#FE00CE',

 'Estonia': '#0DF9FF',

 'Eswatini': '#F6F926',

 'Ethiopia': '#FF9616',

 'Fiji': '#479B55',

 'Finland': '#EEA6FB',

 'France': '#DC587D',

 'Gabon': '#D626FF',

 'Gambia': '#6E899C',

 'Georgia': '#00B5F7',

 'Germany': '#B68E00',

 'Ghana': '#C9FBE5',

 'Greece': '#FF0092',

 'Grenada': '#22FFA7',

 'Guatemala': '#E3EE9E',

 'Guinea': '#86CE00',

 'Guinea-Bissau': '#BC7196',

 'Guyana': '#7E7DCD',

 'Haiti': '#FC6955',

 'Holy See': '#E48F72',

 'Honduras': '#FD3216',

 'Hungary': '#00FE35',

 'Iceland': '#6A76FC',

 'India': '#FED4C4',

 'Indonesia': '#FE00CE',

 'Iran': '#0DF9FF',

 'Iraq': '#F6F926',

 'Ireland': '#FF9616',

 'Israel': '#479B55',

 'Italy': '#0000FF',

 'Jamaica': '#DC587D',

 'Japan': '#D626FF',

 'Jordan': '#6E899C',

 'Kazakhstan': '#00B5F7',

 'Kenya': '#B68E00',

 'Korea, South': '#C9FBE5',

 'Kuwait': '#FF0092',

 'Kyrgyzstan': '#22FFA7',

 'Laos': '#E3EE9E',

 'Latvia': '#86CE00',

 'Lebanon': '#BC7196',

 'Liberia': '#7E7DCD',

 'Libya': '#FC6955',

 'Liechtenstein': '#E48F72',

 'Lithuania': '#FD3216',

 'Luxembourg': '#00FE35',

 'Madagascar': '#6A76FC',

 'Malaysia': '#FED4C4',

 'Maldives': '#FE00CE',

 'Mali': '#0DF9FF',

 'Malta': '#F6F926',

 'Mauritania': '#FF9616',

 'Mauritius': '#479B55',

 'Mexico': '#EEA6FB',

 'Moldova': '#DC587D',

 'Monaco': '#D626FF',

 'Mongolia': '#6E899C',

 'Montenegro': '#00B5F7',

 'Morocco': '#B68E00',

 'Mozambique': '#C9FBE5',

 'Namibia': '#FF0092',

 'Nepal': '#22FFA7',

 'Netherlands': '#E3EE9E',

 'New Zealand': '#86CE00',

 'Nicaragua': '#BC7196',

 'Niger': '#7E7DCD',

 'Nigeria': '#FC6955',

 'North Macedonia': '#E48F72',

 'Norway': '#FD3216',

 'Oman': '#00FE35',

 'Pakistan': '#6A76FC',

 'Panama': '#FED4C4',

 'Papua New Guinea': '#FE00CE',

 'Paraguay': '#0DF9FF',

 'Peru': '#F6F926',

 'Philippines': '#FF9616',

 'Poland': '#479B55',

 'Portugal': '#EEA6FB',

 'Qatar': '#DC587D',

 'Romania': '#D626FF',

 'Russia': '#6E899C',

 'Rwanda': '#00B5F7',

 'Saint Kitts and Nevis': '#B68E00',

 'Saint Lucia': '#C9FBE5',

 'Saint Vincent and the Grenadines': '#FF0092',

 'San Marino': '#22FFA7',

 'Saudi Arabia': '#E3EE9E',

 'Senegal': '#86CE00',

 'Serbia': '#BC7196',

 'Seychelles': '#7E7DCD',

 'Singapore': '#FC6955',

 'Slovakia': '#E48F72',

 'Slovenia': '#FD3216',

 'Somalia': '#00FE35',

 'South Africa': '#6A76FC',

 'Spain': '#FFFF00',

 'Sri Lanka': '#FE00CE',

 'Sudan': '#0DF9FF',

 'Suriname': '#F6F926',

 'Sweden': '#FF9616',

 'Switzerland': '#479B55',

 'Syria': '#EEA6FB',

 'Taiwan*': '#DC587D',

 'Tanzania': '#D626FF',

 'Thailand': '#6E899C',

 'Timor-Leste': '#00B5F7',

 'Togo': '#B68E00',

 'Trinidad and Tobago': '#C9FBE5',

 'Tunisia': '#FF0092',

 'Turkey': '#22FFA7',

 'US': '#FF0000',

 'Uganda': '#86CE00',

 'Ukraine': '#BC7196',

 'United Arab Emirates': '#7E7DCD',

 'United Kingdom': '#FC6955',

 'Uruguay': '#E48F72',

 'Uzbekistan': '#FD3216',

 'Venezuela': '#00FE35',

 'Vietnam': '#6A76FC',

 'Zambia': '#FED4C4',

 'Zimbabwe': '#FE00CE'}



style = {'description_width': 'initial'}

#First Widget - Choose the number of variables





info = widgets.Dropdown(

    options=['Cases', 'Deaths', 'Cases per Day', 'Deaths per Day', 'Cases (Each 100000)', 'Mortality Rate (%)'],

    value='Cases',

    description='Choose the information:',

    disabled=False,

    continous_update = True,

    style = style,

    layout=Layout(width='35%', height='25px', align_self = 'center')

)



log_scale = widgets.Checkbox(

    value=False,

    description='Log Scale',

    disabled=False,

    indent=False

)





cl = sorted(list(set(df['Country'])))



countries_list = []

for i in range(10):

    globals()['Country_%s' % (i+1)] = widgets.Combobox(

    # value='John',

    placeholder='Choose Country',

    options=cl,

    description='Country ' + str(i+1) + ':',

    ensure_option=True,

    disabled=False

    )

    countries_list.append(globals()['Country_%s' % (i+1)])





container = widgets.VBox(children=countries_list)





period = widgets.ToggleButtons(

    options=['All Period', 'First Week', 'First Two Weeks', 'First Month'],

    description='Period of time since first...',

    disabled=False,

    button_style='',

    style = style# 'success', 'info', 'warning', 'danger' or ''

    #tooltips=['Description of slow', 'Description of regular', 'Description of fast'],

#     icons=['check'] * 3

)



f = go.FigureWidget()

f.layout.width = 900

f.layout.height = 600

f.layout.title = 'Cases - COVID19'

f.layout.paper_bgcolor = 'black'

f.layout.plot_bgcolor = 'black'

f.layout.xaxis.gridcolor = 'gray'

f.layout.yaxis.gridcolor = 'gray'

f.layout.yaxis.color = 'white'

f.layout.xaxis.color = 'white'

f.layout.title.font['color'] = 'white'

f.layout.legend.font['color'] = '#ffffff' 

f.update_layout(showlegend=True)



values = ['Brazil']

for i in range(len(values)):

    f.add_scatter(x = df[df['Country'].isin([values[i]])]['Date'].astype(str), y=df[df['Country'].isin([values[i]])]['Cases']

                  , marker={'color': dic_colors.get(values[i]), 'size': 15, 'opacity':0.7}, 

                  mode = 'markers', name = values[i], 

                 text=df[df['Country'].isin([values[i]])]['Country'],

        hovertemplate=

        "<b>%{text}</b><br><br>" +

        "Number of Cases: %{y}<br>" +

        "Date: %{x}<br>" +

        "<extra></extra>")

    

    

    f.layout.yaxis.type = 'linear'

    f.layout.yaxis.title = 'Cases'

    f.layout.xaxis.title = 'Date'

    



    





def response(change):

    

    with f.batch_update():

        f.data = []

        

        lt = []

        for i in range(10):

            temp = container.children[i].value

            if len(temp)>0:

                lt.append(temp)

        

        for i in range(len(lt)):

            if ((period.value in ['First Week', 'First Two Weeks', 'First Month']) & (info.value in ['Cases', 'Cases per Day', 'Cases (Each 100000)'])):

                if(period.value=='First Week'):

                    data = df[(df['StartCases']<8) & (df['StartCases']>0)]

                    x = np.array(data[data['Country']==lt[i]]['StartCases'])

                    y = np.array(data[data['Country']==lt[i]][info.value])

                    marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                    customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                    text = data[data['Country'].isin([lt[i]])]['Country']

                    f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.5},

                                  mode = 'markers', name = lt[i],

                                  text = text,

                                  customdata = customdata.T,

                                    hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date: <br>%{customdata[0]}<br>" +

                                    info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                  #marker_size=marker

                                 )

                    f.layout.xaxis.type = 'linear'

                else:

                    if(period.value=='First Two Weeks'):

                        data = df[(df['StartCases']<16) & (df['StartCases']>0)]

                        x = np.array(data[data['Country']==lt[i]]['StartCases'])

                        y = np.array(data[data['Country']==lt[i]][info.value])

                        marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                        customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                        text = data[data['Country'].isin([lt[i]])]['Country']

                        f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                                      mode = 'markers', name = lt[i],

                                    text = text,

                                    customdata = customdata.T,

                                   hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date: <br>%{customdata[0]}<br>" +

                                    info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                  #marker_size=marker

                                 )

                        f.layout.xaxis.type = 'linear'

                    else:

                        if(period.value=='First Month'):

                            data = df[(df['StartCases']<32) & (df['StartCases']>0)]

                            x = np.array(data[data['Country']==lt[i]]['StartCases'])

                            y = np.array(data[data['Country']==lt[i]][info.value])

                            marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                            customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                            text = data[data['Country'].isin([lt[i]])]['Country']

                            f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                                          mode = 'markers', name = lt[i],

                                        text = text,

                                        customdata = customdata.T,

                                        hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date: <br>%{customdata[0]}<br>" +

                                    info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                  #marker_size=marker

                                 )

                            f.layout.xaxis.type = 'linear'

            else:

                if ((period.value in ['First Week', 'First Two Weeks', 'First Month']) & (info.value in ['Deaths', 'Deaths per Day', 'Mortality Rate (%)'])):

                    if(period.value=='First Week'):

                        data = df[(df['StartDeaths']<8) & (df['StartDeaths']>0)]

                        x = np.array(data[data['Country']==lt[i]]['StartDeaths'])

                        y = np.array(data[data['Country']==lt[i]][info.value])

                        marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                        customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                        text = data[data['Country'].isin([lt[i]])]['Country']

                        f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                                      mode = 'markers', name = lt[i],

                                    text = text,

                                    customdata = customdata.T,

                                    hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date:<br> %{customdata[0]}<br>" +

                                    info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                  #marker_size=marker

                                 )

                        f.layout.xaxis.type = 'linear'

                    else:

                        if(period.value=='First Two Weeks'):

                            data = df[(df['StartDeaths']<16) & (df['StartDeaths']>0)]

                            x = np.array(data[data['Country']==lt[i]]['StartDeaths'])

                            y = np.array(data[data['Country']==lt[i]][info.value])

                            marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                            customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                            text = data[data['Country'].isin([lt[i]])]['Country']

                            f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                                          mode = 'markers', name = lt[i],

                                        text = text,

                                        customdata = customdata.T,

                                        hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date: <br>%{customdata[0]}<br>" +

                                     info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                 #marker_size=marker

                                 )

                            f.layout.xaxis.type = 'linear'

                        else:

                            if(period.value=='First Month'):

                                data = df[(df['StartDeaths']<32) & (df['StartDeaths']>0)]

                                x = np.array(data[data['Country']==lt[i]]['StartDeaths'])

                                y = np.array(data[data['Country']==lt[i]][info.value])

                                marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                                customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                                         tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                                text = data[data['Country'].isin([lt[i]])]['Country']

                                f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                                              mode = 'markers', name = lt[i],

                                            text = text,

                                            customdata = customdata.T,

                                            hovertemplate=

                                    "<b>%{text}</b><br><br>" +

                                    "Day: <br>%{x}<br>" +

                                    "Date: <br>%{customdata[0]}<br>" +

                                     info.value + ": <br> %{y}<br>" +

                                    #"Life Expectancy: <br>%{customdata[1]}<br>" +

                                    "<extra></extra>",

                                  #marker_size=marker

                                 )

                                f.layout.xaxis.type = 'linear'

            

            if(period.value=='All Period'):

                data = df

                x = np.array(data[data['Country']==lt[i]]['Date'].astype(str))

                y = np.array(data[data['Country']==lt[i]][info.value])

                marker = (((np.array((data.loc[data['Country']==lt[i], 'LifeExp_2017'])) - np.array(50))/(np.array(85) - np.array(50)))*10)+5

                customdata = np.array([tuple(np.array(data[data['Country']==lt[i]]['Date'].astype(str))), 

                                 tuple(np.array(data[data['Country']==lt[i]]['LifeExp_2017']))])

                text = data[data['Country'].isin([lt[i]])]['Country']

                

                hovertemplate = "<b>%{text}</b><br><br>" + info.value + ": <br> %{y}<br>" + "Date: <br>%{x}<br>" +  "<extra></extra>" #"Life Expectancy: <br>%{customdata[1]}<br>" +

                

                f.add_scatter(x = x, y = y,marker={'color': dic_colors.get(lt[i]), 'size': 15, 'opacity':0.7},

                            mode = 'markers', name = lt[i],

                            text = text,

                            customdata = customdata.T,

                            hovertemplate = hovertemplate)

                            #marker_size = marker)



                f.layout.xaxis.type = 'date'

            

    if log_scale.value == True:

        f.layout.yaxis.type = 'log'

    else:

        f.layout.yaxis.type = 'linear'

            

    f.layout.title = info.value + ' - COVID19'

    f.layout.title.font['color'] = '#ffffff'

    f.layout.legend.font['color'] = '#ffffff' 

    f.layout.yaxis.title = info.value

        

        

        

        

[var.observe(response, names = 'value') for var in countries_list]

info.observe(response, names = 'value')

log_scale.observe(response, names = 'value')

period.observe(response, names = 'value')

#time_range.observe(response, names = 'value')





display(widgets.VBox([container, info, log_scale, period]), display_id = True);

display(widgets.VBox([f]), display_id = True);



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

style = {'description_width': 'initial'}

#First Widget - Choose the number of variables





var = widgets.Dropdown(

    options=['Cases', 'Deaths',

             'Cases (Each 100000)', 

             'Mortality Rate (%)'],

    

    value='Cases',

    description='COVID Variable',

    disabled=False,

    continous_update = True,

    style = style,

    layout=Layout(width='35%', height='25px', align_self = 'center')

)



var1 = widgets.Dropdown(

    options=['Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

             'HDI_2018', 'LifeExp_2017', 

             'UrbPop_2018 (%)'],

    

    value='Arrivals_2018 (Millions)',

    description='Variable:',

    disabled=False,

    continous_update = True,

    style = style,

    layout=Layout(width='35%', height='25px', align_self = 'center')

)







log_scale_2 = widgets.Checkbox(

    value=False,

    description='Log Scale',

    disabled=False,

    indent=False

)





period2 = widgets.ToggleButtons(

    options=['All Period', 'First Week', 'First Two Weeks', 'First Month'],

    description='Period of time since first...',

    disabled=False,

    button_style='',

    style = style

)



fig = go.FigureWidget()

fig.layout.width = 900

fig.layout.height = 600

fig.layout.title = 'COVID19'

fig.layout.paper_bgcolor = 'black'

fig.layout.plot_bgcolor = 'black'

fig.layout.xaxis.gridcolor = 'gray'

fig.layout.yaxis.gridcolor = 'gray'

fig.layout.yaxis.color = 'white'

fig.layout.xaxis.color = 'white'

fig.layout.title.font['color'] = 'white'

fig.layout.legend.font['color'] = '#ffffff' 







dt = df.groupby(['Country'], sort=False, as_index=False)[['Cases', 'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                 'HDI_2018', 'LifeExp_2017', 

                                                                 'UrbPop_2018 (%)']].max().dropna()



fig.add_scatter(x = dt['Arrivals_2018 (Millions)'], y= dt['Cases']

              , marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

             name = 'Arrivals_2018 (Millions)',

             text = dt['Country'],

             hovertemplate=

            "<b>%{text}</b><br><br>" +

            "Cases: <br>%{y}<br>" +

            "Arrivals_2018 (Millions):<br> %{x}<br>" +

            "<extra></extra>")





fig.layout.yaxis.type = 'linear'

fig.layout.yaxis.title = 'Cases'

fig.layout.xaxis.title = 'Arrivals_2018 (Millions)'

    



def response2(change):

    

    with fig.batch_update():

        fig.data = []

        if ((period2.value in ['First Week', 'First Two Weeks', 'First Month']) & (var.value in ['Cases', 'Cases (Each 100000)'])):

            if(period2.value=='First Week'):

                



                data = df[df['StartCases']==7]



                df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Cases', 'Cases (Each 100000)',

                                                                                 'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                 'HDI_2018', 'LifeExp_2017', 

                                                                 'UrbPop_2018 (%)']].max().dropna()

                

                x = df_temp[var1.value]

                y = df_temp[var.value]

                text = df_temp['Country']

                hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"

                

                fig.add_scatter(x = x, y= y,

                                marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                name = var1.value,

                                text = text,

                         hovertemplate = hovertemplate

                        )

                

                fig.layout.xaxis.type = 'linear'

                fig.layout.yaxis.title = var.value

                fig.layout.xaxis.title = var1.value

                

            else:

                if(period2.value=='First Two Weeks'):

                    data = df[df['StartCases']==15]

                    df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Cases', 'Cases (Each 100000)',

                                                                                     'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                     'HDI_2018', 'LifeExp_2017', 

                                                                     'UrbPop_2018 (%)']].max().dropna()



                    x = df_temp[var1.value]

                    y = df_temp[var.value]

                    text = df_temp['Country']

                    hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



                    fig.add_scatter(x = x, y= y,

                                    marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                    name = var1.value,

                                    text = text,

                             hovertemplate = hovertemplate

                            )



                    fig.layout.xaxis.type = 'linear'

                    fig.layout.yaxis.title = var.value

                    fig.layout.xaxis.title = var1.value

                    

                else:

                    if(period2.value=='First Month'):

                        data = df[df['StartCases']==30]

                        df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Cases', 'Cases (Each 100000)', 

                                                                                         'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                         'HDI_2018', 'LifeExp_2017', 

                                                                         'UrbPop_2018 (%)']].max().dropna()



                        x = df_temp[var1.value]

                        y = df_temp[var.value]

                        text = df_temp['Country']

                        hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



                        fig.add_scatter(x = x, y= y,

                                        marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                        name = var1.value,

                                        text = text,

                                 hovertemplate = hovertemplate

                                )



                        fig.layout.xaxis.type = 'linear'

                        fig.layout.yaxis.title = var.value

                        fig.layout.xaxis.title = var1.value

                        

        else:

            if ((period2.value in ['First Week', 'First Two Weeks', 'First Month']) & (var.value in ['Deaths', 'Mortality Rate (%)'])):

                if(period2.value=='First Week'):

                    data = df[df['StartDeaths']==7]



                    df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Deaths', 'Mortality Rate (%)',

                                                                                     'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                     'HDI_2018', 'LifeExp_2017', 

                                                                     'UrbPop_2018 (%)']].max().dropna()



                    x = df_temp[var1.value]

                    y = df_temp[var.value]

                    text = df_temp['Country']

                    hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



                    fig.add_scatter(x = x, y= y,

                                    marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                    name = var1.value,

                                    text = text,

                             hovertemplate = hovertemplate

                            )



                    fig.layout.xaxis.type = 'linear'

                    fig.layout.yaxis.title = var.value

                    fig.layout.xaxis.title = var1.value



                else:

                    if(period2.value=='First Two Weeks'):

                        data = df[df['StartDeaths']==15]

                        df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Deaths', 'Mortality Rate (%)',

                                                                                         'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                         'HDI_2018', 'LifeExp_2017', 

                                                                         'UrbPop_2018 (%)']].max().dropna()



                        x = df_temp[var1.value]

                        y = df_temp[var.value]

                        text = df_temp['Country']

                        hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



                        fig.add_scatter(x = x, y= y,

                                        marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                        name = var1.value,

                                        text = text,

                                 hovertemplate = hovertemplate

                                )



                        fig.layout.xaxis.type = 'linear'

                        fig.layout.yaxis.title = var.value

                        fig.layout.xaxis.title = var1.value



                    else:

                        if(period2.value=='First Month'):

                            data = df[df['StartDeaths']==30]

                            df_temp = data.groupby(['Country'], sort=False, as_index=False)[['Deaths', 'Mortality Rate (%)', 

                                                                                             'Arrivals_2018 (Millions)', 'GDP_2018 (Billions)', 

                                                                             'HDI_2018', 'LifeExp_2017', 

                                                                             'UrbPop_2018 (%)']].max().dropna()



                            x = df_temp[var1.value]

                            y = df_temp[var.value]

                            text = df_temp['Country']

                            hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



                            fig.add_scatter(x = x, y= y,

                                            marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                                            name = var1.value,

                                            text = text,

                                     hovertemplate = hovertemplate

                                    )



                            fig.layout.xaxis.type = 'linear'

                            fig.layout.yaxis.title = var.value

                            fig.layout.xaxis.title = var1.value

            



        if(period2.value=='All Period'):

            df_temp = df.groupby(['Country'], sort=False, as_index=False)[['Cases', 'Cases (Each 100000)',

                                                                           'Arrivals_2018 (Millions)', 'Deaths', 

                                                                           'Mortality Rate (%)',

                                                                           'GDP_2018 (Billions)', 

                                                                     'HDI_2018', 'LifeExp_2017', 

                                                                     'UrbPop_2018 (%)']].last()

                        

            x = df_temp[var1.value]

            y = df_temp[var.value]

            text = df_temp['Country']

            hovertemplate = "<b>%{text}</b><br><br>" + var.value + ": <br>%{y}<br>" + var1.value + ":<br> %{x}<br>" + "<extra></extra>"



            fig.add_scatter(x = x, y= y,

                            marker={'color': 'red', 'size': 15, 'opacity':0.7}, mode='markers',

                            name = var1.value,

                            text = text,

                     hovertemplate = hovertemplate

                    )



            fig.layout.xaxis.type = 'linear'

            fig.layout.yaxis.title = var.value

            fig.layout.xaxis.title = var1.value



    if log_scale_2.value == True:

        fig.layout.yaxis.type = 'log'

    else:

        fig.layout.yaxis.type = 'linear'

            

#     f.layout.title = info.value + ' - COVID19'

#     f.layout.title.font['color'] = '#ffffff'

#     f.layout.legend.font['color'] = '#ffffff' 

#     f.layout.yaxis.title = info.value

        

        

              

#[var.observe(response, names = 'value') for var in countries_list]

var.observe(response2, names = 'value')

var1.observe(response2, names = 'value')

log_scale_2.observe(response2, names = 'value')

period2.observe(response2, names = 'value')

#time_range.observe(response, names = 'value')

display(widgets.VBox([var, var1, log_scale_2, period2]), display_id = True);

display(widgets.VBox([fig]), display_id = True);



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')