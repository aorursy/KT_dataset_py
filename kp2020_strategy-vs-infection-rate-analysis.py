# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filesList = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        filesList.append(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# read relevant datasets



data_path1 = '/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv'

data_path2 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'

data_path3 = '/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv'



def data_details(data, path):

    

    # read file

    print(path)

    print('-', 60)

    

    # quick check data size

    nrow = data.shape[0]

    ncol = data.shape[1]

    print("No. of Rows: {0} \nNo. of Columns: {1}\n".format(nrow, ncol))



    # display dataset

    display(data.head())



    # columns names

    data_col = data.columns.tolist()

    print("CSV - Columes Name: \n{0}\n".format(data_col))



    # data types and missing values

    print("Data Info")

    data.info()

    print('\n')



    # missing values cleaning: 

    null_col = data.columns[data.isnull().any()].tolist()

    print("Columns with Null: \n{0}\n".format(null_col))



    # how many missing values in the columns with null

    if len(null_col) > 0:

        print("No. of Nulls")

    for nc in null_col:

        print(f"{nc:<70}{':'}{len(data[nc][data[nc].isnull()].index.tolist()):>10}")

    print('\n')



data1 = pd.read_csv(data_path1)

data2 = pd.read_csv(data_path2)

data3 = pd.read_csv(data_path3)



# an overview of the dataset

data_details(data1, data_path1)

data_details(data2, data_path2)

data_details(data3, data_path3)
# an overview of applied strategies

import plotly.express as px

fig = px.histogram(data1, y='category', orientation="h", color='category', 

                   color_discrete_sequence=px.colors.sequential.Agsunset)

fig.update_yaxes(categoryorder='total ascending')

fig.update_layout(title_text='Count of Category (Strategy)', 

                   title_x=0.5)

fig.show() 
import plotly.express as px



# rearrange data for treemap presentation 

data_treemap = pd.DataFrame({'count':data1.groupby(['category','measure'])['measure'].size().sort_values(ascending=False)}).reset_index()



fig = px.treemap(data_treemap, path=['category', 'measure'], values='count', color='category', color_discrete_sequence=px.colors.sequential.Plasma)

fig.data[0].textinfo = 'label+text+value'

fig.update_layout(title='Applied Techniques with Count', title_x=0.5)

fig.show()
import seaborn as sns

cm=sns.light_palette("red", as_cmap=True)



# rearrange data for dataframe presentation 

data_df = pd.DataFrame({'count':data1.groupby(['category','measure'])['measure'].size().sort_values(ascending=False)}).reset_index()



data_df.style.background_gradient(cmap='viridis').set_caption('The Popularity of Measure Techniques')
# China, Iran, Italy, Spain, US, and data1 = 'United States of America', 'Korea Republic of'

countryList = ['China', 'Iran', 'Italy', 'Spain', 'France', 'United States of America']

def country_techniques(Country):

    data_Country = data1[data1.country==Country].reset_index()

    return pd.DataFrame({'count':data_Country.groupby(['category','measure']).size().sort_values(ascending=True)}).reset_index()





for nation in countryList:

    data_pie = country_techniques(nation)

    fig = px.pie(data_pie, values='count', names='measure', color_discrete_sequence=px.colors.sequential.OrRd)

    fig.update_layout(title='{0} Govs Actions'.format(nation), title_x=0.5)

    fig.show()
# 'Korea Republic of', 'Germany', 'New Zealand', 'Finland', 'Iceland', 'Viet Nam', 'Singapore', 'France'  data1 = 'United States of America', 'Korea Republic of', 'Viet Nam'



countryList = ['Korea Republic of', 'Germany', 'New Zealand', 'Finland', 'Iceland', 'Viet Nam', 'Singapore']

def country_techniques(Country):

    data_Country = data1[data1.country==Country].reset_index()

    return pd.DataFrame({'count':data_Country.groupby(['category','measure']).size().sort_values(ascending=True)}).reset_index()



for nation in countryList:

    data_pie = country_techniques(nation)

    fig = px.pie(data_pie, values='count', names='measure', color_discrete_sequence=px.colors.sequential.Plasma)

    fig.update_layout(title='{0} Govs Actions'.format(nation), title_x=0.5)

    fig.show()
countryList = ['Australia', 'United Kingdom', 'India', 'Indonesia', 'Japan', 'Canada', 'Viet Nam', 'Turkey', 'Brazil']



def country_techniques(Country):

    data_Country = data1[data1.country==Country].reset_index()

    return pd.DataFrame({'count':data_Country.groupby(['category','measure']).size().sort_values(ascending=True)}).reset_index()



for nation in countryList:

    data_pie = country_techniques(nation)

    fig = px.pie(data_pie, values='count', names='measure', color_discrete_sequence=px.colors.sequential.YlGnBu)

    fig.update_layout(title='{0} Govs Actions'.format(nation), title_x=0.5)

    fig.show()
import plotly.express as px

fig = px.histogram(data1, x='date_implemented', color='measure', title='Governments Actions by Time', 

                   color_discrete_sequence=px.colors.sequential.Aggrnyl)

fig.update_yaxes(categoryorder='total ascending')

fig.update_layout(title_x=0.5)

fig.show()
# When The earliest strategy is applied and does it bring any impact?

#data1:

#countryList = ['China', 'Iran', 'Italy', 'Spain', 'France', 'United States of America']

#countryList = ['Korea Republic of', 'Germany', 'New Zealand', 'Finland', 'Iceland', 'Viet Nam', 'Singapore']

#countryList = ['Australia', 'United Kingdom', 'India', 'Indonesia', 'Japan', 'Canada', 'Viet Nam', 'Turkey', 'Brazil']



def country_start_date(Country, groupby_list):

    # focus on any method when is the first lockdown among the countries

    data_Country = data1[data1.country==Country].reset_index()



    # remove NaN on date_implemented column

    data_Country = data_Country[~data_Country.date_implemented.isnull()]



    # implemented date

    df_country = pd.DataFrame({'count':data_Country.groupby(groupby_list).size()}).reset_index()



    # get the earliest day of measure implementation

    earliest_ = data_Country.date_implemented.min()



    # get the latest day of measure implementation

    latest_ = data_Country.date_implemented.max()

    

    # get the technique type

    if pd.notna(earliest_):

        earliest_measure = data_Country[data_Country.date_implemented==earliest_].measure.tolist()[0]

    else:

        earliest_measure = "No Date / No Measure Type"

    return [df_country, earliest_, earliest_measure]



import plotly.graph_objects as go

from plotly.subplots import make_subplots



df_usa = country_start_date('United States of America', ['date_implemented'])

df_esp = country_start_date('Spain', ['date_implemented'])

df_ita = country_start_date('Italy', ['date_implemented'])

df_fra = country_start_date('France', ['date_implemented'])

df_deu = country_start_date('Germany', ['date_implemented'])

df_gbr = country_start_date('United Kingdom', ['date_implemented'])

df_cna = country_start_date('China', ['date_implemented'])

df_tur = country_start_date('Turkey', ['date_implemented'])

df_irn = country_start_date('Iran', ['date_implemented'])

df_bra = country_start_date('Brazil', ['date_implemented'])

df_can = country_start_date('Canada', ['date_implemented'])

df_ind = country_start_date('India', ['date_implemented'])

df_kor = country_start_date('Korea Republic of', ['date_implemented'])

df_jap = country_start_date('Japan', ['date_implemented'])

df_pak = country_start_date('Pakistan', ['date_implemented'])

df_dnk = country_start_date('Denmark', ['date_implemented'])

df_nor = country_start_date('Norway', ['date_implemented'])

df_aus = country_start_date('Australia', ['date_implemented'])

df_sin = country_start_date('Singapore', ['date_implemented'])

df_idn = country_start_date('Indonesia', ['date_implemented'])

df_fin = country_start_date('Finland', ['date_implemented'])

df_nzl = country_start_date('New Zealand', ['date_implemented'])

df_vnm = country_start_date('Viet Nam', ['date_implemented'])

df_isl = country_start_date('Iceland', ['date_implemented'])





#data2

country_time_rapid = ['China', 'Iran', 'Italy', 'Spain', 'France', 'United States']

country_time_slower = ['South Korea', 'Germany', 'New Zealand', 'Finland', 'Iceland', 'Viet Nam', 'Singapore']

country_time_monitor = ['Australia', 'United Kingdom', 'India', 'Indonesia', 'Japan', 'Canada', 'Vietnam', 'Turkey', 'Brazil']



from plotly.subplots import make_subplots

import plotly.graph_objects as go

def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



fig = make_subplots(rows=3, cols=2,

                    subplot_titles=('{0}'.format(country_time_rapid[0]),

                                    '{0}'.format(country_time_rapid[1]),

                                    '{0}'.format(country_time_rapid[2]),

                                    '{0}'.format(country_time_rapid[3]),

                                    '{0}'.format(country_time_rapid[4]),

                                    '{0}'.format(country_time_rapid[5])),

                    horizontal_spacing=0.25,

                    x_title='Date',

                    vertical_spacing=0.085,

                    y_title='Count of Confirm Cases')



ROW = 1

COL = 1

color_rapid = ['rgb(253,212,158)','rgb(252,141,89)',

               'rgb(239,101,72)','rgb(215,48,31)',

               'rgb(179,0,0)','rgb(127,0,0)']

for cnt, nation in enumerate(country_time_rapid):

    dfcountry = confirmed_case_country(nation)

    fig.add_trace(

        go.Bar(x=dfcountry.date, y=dfcountry.total_confirmed_cases_of_covid_19_cases,

               marker=dict(color=color_rapid[cnt]), name=country_time_rapid[cnt]), row=ROW, col=COL

    )

    

    if COL >=2:

        COL = 1

        ROW += 1

    elif COL < 2:

        COL = COL + 1

    else:

        ROW += 1



y1=55000; y2=20000; y3=55000; y4=40000; y5=20000; y6=55000;



fig.update_layout(

    shapes=[

        dict(type="line", xref="x1", yref="y1",

             x0=df_cna[1], y0=0, x1=df_cna[1], 

             y1=y1, line_width=2),

        dict(type="line", xref="x2", yref="y2",

             x0=df_irn[1], y0=0, x1=df_irn[1], 

             y1=y2, line_width=2),

        dict(type="line", xref="x3", yref="y3",

             x0=df_ita[1], y0=0, x1=df_ita[1], 

             y1=55000, line_width=2),

        dict(type="line", xref="x4", yref="y4",

             x0=df_esp[1], y0=0, x1=df_esp[1], 

             y1=55000, line_width=2),

        dict(type="line", xref="x5", yref="y5",

             x0=df_fra[1], y0=0, x1=df_fra[1], 

             y1=55000, line_width=2),

        dict(type="line", xref="x6", yref="y6",

             x0=df_usa[1], y0=0, x1=df_usa[1], 

             y1=55000, line_width=2)],

    title='Confirmed Case vs First Strategy<br>(Countries with a rapid spread)',

    title_x=0.5

)



b = 'first announcement'

a=df_cna[1]

c=df_cna[2]

fig.add_annotation(

    dict(

        xref="x1", yref="y1",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_cna[1],y=61000,

        font=dict(size=10))

)

a=df_irn[1]

c=df_irn[2]

fig.add_annotation(

    dict(

        xref="x2", yref="y2",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_irn[1],y=23000,

        font=dict(size=10))

)

a=df_ita[1]

c=df_ita[2]

fig.add_annotation(

    dict(

        xref="x3", yref="y3",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_ita[1],y=60000,

        font=dict(size=10))

)

a=df_esp[1]

c=df_esp[2]

fig.add_annotation(

    dict(

        xref="x4", yref="y4",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_esp[1],y=60000,

        font=dict(size=10))

)

a=df_fra[1]

c=df_fra[2]

fig.add_annotation(

    dict(

        xref="x5", yref="y5",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_fra[1],y=60000,

        font=dict(size=10))

)



a=df_usa[1]

c=df_usa[2]

b='1st announcement'

fig.add_annotation(

    dict(

        xref="x6", yref="y6",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_usa[1],y=60000,

        font=dict(size=10))

)

   

fig.update_layout(height=1300)    

fig.show()
# When The earliest strategy is applied and does it bring any impact?

#data2

country_time_slower = ['South Korea', 'Germany', 'New Zealand', 'Finland', 'Iceland', 'Vietnam']

country_time_monitor = ['Australia', 'United Kingdom', 'India', 'Indonesia', 'Japan', 'Canada', 'Vietnam', 'Turkey', 'Brazil']





def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



fig = make_subplots(rows=3, cols=2,

                    subplot_titles=('{0}'.format(country_time_slower[0]),

                                    '{0}'.format(country_time_slower[1]),

                                    '{0}'.format(country_time_slower[2]),

                                    '{0}'.format(country_time_slower[3]),

                                    '{0}'.format(country_time_slower[4]),

                                    '{0}'.format(country_time_slower[5])),

                    horizontal_spacing=0.25,

                    x_title='Date',

                    vertical_spacing=0.085,

                    y_title='Count of Confirm Cases')



ROW = 1

COL = 1

color_rapid = ['#0d0887','#7201a8',

               '#9c179e','#bd3786',

               '#ed7953','#fdca26']

for cnt, nation in enumerate(country_time_slower):

    dfcountry = confirmed_case_country(nation)

    fig.add_trace(

        go.Bar(x=dfcountry.date, y=dfcountry.total_confirmed_cases_of_covid_19_cases,

               marker=dict(color=color_rapid[cnt]), name=country_time_slower[cnt]), row=ROW, col=COL

    )

    

    if COL >=2:

        COL = 1

        ROW += 1

    elif COL < 2:

        COL = COL + 1

    else:

        ROW += 1



y1=7800; y2=30000; y3=200; y4=700; y5=650; y6=80;



fig.update_layout(

    shapes=[

        dict(type="line", xref="x1", yref="y1",

             x0=df_kor[1], y0=0, x1=df_kor[1], 

             y1=y1, line_width=3),

        dict(type="line", xref="x2", yref="y2",

             x0=df_deu[1], y0=0, x1=df_deu[1], 

             y1=y2, line_width=3),

        dict(type="line", xref="x3", yref="y3",

             x0=df_nzl[1], y0=0, x1=df_nzl[1], 

             y1=y3, line_width=3),

        dict(type="line", xref="x4", yref="y4",

             x0=df_fin[1], y0=0, x1=df_fin[1], 

             y1=y4, line_width=3),

        dict(type="line", xref="x5", yref="y5",

             x0=df_isl[1], y0=0, x1=df_isl[1], 

             y1=y5, line_width=3),

        dict(type="line", xref="x6", yref="y6",

             x0=df_vnm[1], y0=0, x1=df_vnm[1], 

             y1=y6, line_width=3)],

    title='Confirmed Case vs First Strategy<br>(Countries with a slower spread)',

    title_x=0.5

)



a=df_kor[1]

c=df_kor[2]

fig.add_annotation(

    dict(

        xref="x1", yref="y1",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_kor[1],y=8300,

        font=dict(size=10))

)



a=df_deu[1]

c=df_deu[2]

fig.add_annotation(

    dict(

        xref="x2", yref="y2",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_deu[1],y=32000,

        font=dict(size=10))

)

a=df_nzl[1]

c=df_nzl[2]

fig.add_annotation(

    dict(

        xref="x3", yref="y3",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_nzl[1],y=230,

        font=dict(size=10))

)

a=df_fin[1]

c=df_fin[2]

fig.add_annotation(

    dict(

        xref="x4", yref="y4",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_fin[1],y=750,

        font=dict(size=10))

)

a=df_isl[1]

c=df_isl[2]

fig.add_annotation(

    dict(

        xref="x5", yref="y5",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_isl[1],y=750,

        font=dict(size=10))

)

a=df_vnm[1]

c=df_vnm[2]

fig.add_annotation(

    dict(

        xref="x6", yref="y6",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_vnm[1],y=100,

        font=dict(size=10))

)



fig.update_layout(height=1300)



fig.show()

#data2

country_time_monitor = ['Australia', 'United Kingdom', 'India', 'Indonesia', 'Japan', 'Canada', 'Turkey', 'Brazil']



def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



fig = make_subplots(rows=4, cols=2,

                    subplot_titles=('{0}'.format(country_time_monitor[0]),

                                    '{0}'.format(country_time_monitor[1]),

                                    '{0}'.format(country_time_monitor[2]),

                                    '{0}'.format(country_time_monitor[3]),

                                    '{0}'.format(country_time_monitor[4]),

                                    '{0}'.format(country_time_monitor[5]),

                                    '{0}'.format(country_time_monitor[6]),

                                    '{0}'.format(country_time_monitor[7])),

                    horizontal_spacing=0.25,

                    x_title='Date',

                    vertical_spacing=0.085,

                    y_title='Count of Confirm Cases')



ROW = 1

COL = 1

color_rapid = ['rgb(237,248,177)','rgb(199,233,180)',

               'rgb(127,205,187)','rgb(62,182,196)',

               'rgb(29,145,192)','rgb(34,94,168)',

               'rgb(37,52,148)','rgb(8,29,88)']

for cnt, nation in enumerate(country_time_monitor):

    dfcountry = confirmed_case_country(nation)

    fig.add_trace(

        go.Bar(x=dfcountry.date, y=dfcountry.total_confirmed_cases_of_covid_19_cases,

               marker=dict(color=color_rapid[cnt]), name=country_time_monitor[cnt]), row=ROW, col=COL

    )

    

    if COL >=2:

        COL = 1

        ROW += 1

    elif COL < 2:

        COL = COL + 1

    else:

        ROW += 1



y1=2000; y2=9300; y3=650; y4=750; y5=1000; y6=2500; y7=2200; y8=2200

ydealta=200; line_ = 2;

fig.update_layout(

    shapes=[

        dict(type="line", xref="x1", yref="y1",

             x0=df_aus[1], y0=0, x1=df_aus[1], 

             y1=y1-ydealta, line_width=line_),

        dict(type="line", xref="x2", yref="y2",

             x0=df_gbr[1], y0=0, x1=df_gbr[1], 

             y1=y2-500, line_width=line_),

        dict(type="line", xref="x3", yref="y3",

             x0=df_ind[1], y0=0, x1=df_ind[1], 

             y1=y3-70, line_width=line_),

        dict(type="line", xref="x4", yref="y4",

             x0=df_idn[1], y0=0, x1=df_idn[1], 

             y1=y4-70, line_width=line_),

        dict(type="line", xref="x5", yref="y5",

             x0=df_jap[1], y0=0, x1=df_jap[1], 

             y1=y5-ydealta, line_width=line_),

        dict(type="line", xref="x6", yref="y6",

             x0=df_can[1], y0=0, x1=df_can[1], 

             y1=y6-250, line_width=line_),

        dict(type="line", xref="x7", yref="y7",

             x0=df_tur[1], y0=0, x1=df_tur[1], 

             y1=y7-ydealta, line_width=line_),

        dict(type="line", xref="x8", yref="y8",

             x0=df_bra[1], y0=0, x1=df_bra[1], 

             y1=y8-ydealta, line_width=line_)],

    title='Confirmed Case vs First Strategy<br>(Countries required Monitoring and Attentions)', 

    title_x=0.5 

)



a=df_aus[1]

c=df_usa[2]

fig.add_annotation(

    dict(

        xref="x1", yref="y1",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_aus[1],y=y1,

        font=dict(size=10))

)

a=df_gbr[1]

c=df_gbr[2]

fig.add_annotation(

    dict(

        xref="x2", yref="y2",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_gbr[1],y=y2,

        font=dict(size=10))

)

a=df_ind[1]

c=df_ind[2]

fig.add_annotation(

    dict(

        xref="x3", yref="y3",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_ind[1],y=y3,

        font=dict(size=10))

)

a=df_idn[1]

c=df_idn[2]

fig.add_annotation(

    dict(

        xref="x4", yref="y4",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_idn[1],y=y4,

        font=dict(size=10))

)

a=df_jap[1]

c=df_jap[2]

fig.add_annotation(

    dict(

        xref="x5", yref="y5",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_jap[1],y=y5,

        font=dict(size=10))

)

a=df_can[1]

c=df_can[2]

fig.add_annotation(

    dict(

        xref="x6", yref="y6",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_can[1],y=y6,

        font=dict(size=10))

)

a=df_tur[1]

c=df_tur[2]

fig.add_annotation(

    dict(

        xref="x7", yref="y7",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_tur[1],y=y7,

        font=dict(size=10))

)

a=df_bra[1]

c=df_bra[2]

fig.add_annotation(

    dict(

        xref="x8", yref="y8",

        showarrow=False,

        text=f'{c}<br>{a}<br>{b}',

        x=df_bra[1],y=y8,

        font=dict(size=10))

)



fig.update_layout(height=1600)

fig.show()
CLASS = ['Local transmission', 'Imported cases only', 'Under investigation','Local Transmission']

CL = CLASS[0]

data_class = data3[data3.transmission_classification==CL].reset_index()

df1 = pd.DataFrame({'count':data_class.groupby(['reporting_country_territory']).size().sort_values(ascending=True)}).reset_index()

fig = px.bar(df1, y='count',x='reporting_country_territory', color='reporting_country_territory',

             color_discrete_sequence=px.colors.sequential.Plasma, title='{0}'.format(CL))

fig.update_layout(height=1000)

fig.show()
CLASS = ['Local transmission', 'Imported cases only', 'Under investigation','Local Transmission']

CL = CLASS[1]

data_class = data3[data3.transmission_classification==CL].reset_index()

df1 = pd.DataFrame({'count':data_class.groupby(['reporting_country_territory']).size().sort_values(ascending=True)}).reset_index()

fig = px.bar(df1, y='count',x='reporting_country_territory', color='reporting_country_territory',

             color_discrete_sequence=px.colors.sequential.Plasma, title='{0}'.format(CL))

fig.update_layout(height=700)

fig.show()
# Germany Analysis based on time series:

# how frequent they applied the covid-19 strategy



Country = 'Germany'

data_Country = data1[data1.country==Country].reset_index()

data_Country = data_Country[~data_Country.date_implemented.isnull()].reset_index()



def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



def deaths_case_country(country_name):

    dc = data3[data3.reporting_country_territory==country_name]

    return dc[~dc.total_deaths.isnull()]



df_confirmed_case_Country = confirmed_case_country(Country)



df_deaths_case_Country = deaths_case_country(Country)



df_deaths_case_Country = df_deaths_case_Country[~df_deaths_case_Country.confirmed_cases.isnull()]

df_deaths_case_Country.confirmed_cases = df_deaths_case_Country.confirmed_cases.astype(np.int64)

df_deaths_case_Country = df_deaths_case_Country[df_deaths_case_Country.confirmed_cases<7000000]

 

fig = go.Figure()



fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.confirmed_cases,

          name='confirmed_cases')

)

fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.total_deaths,

          name='total_deaths')

)





date_implemented_list = data_Country.date_implemented.tolist()



ply_shapes = {}

for t in date_implemented_list:

    ply_shapes['shape_' + str(t)]=go.layout.Shape(type="line",

                                                  x0=t, y0=0, x1=t,

                                                  y1=750, line_width=2

                                                 )



lst_shapes=list(ply_shapes.values())

comments_list = data_Country.comments.tolist()



ply_annotation = {}

n= 0

ydelta1 = 770

ydelta2 = 820

keepn = []

for com in date_implemented_list:

    if n % 2 == 0:

        ydelta_ = ydelta1

    else:

        ydelta_ = ydelta2

    ply_annotation['annotation_' + str(com)]=go.layout.Annotation(showarrow=False,

                                                                   text=f'{n}',

                                                                   x=com,y=ydelta_,

                                                                   font=dict(size=10))

    keepn.append(date_implemented_list.index(com))

    n+=1

                                                                  

annot=list(ply_annotation.values())

fig.update_layout(shapes=lst_shapes, annotations=annot, barmode='overlay', 

                  title_text='{0} applied Strategy<br>(kindly refer to the following dataframe for further descriptions on each applied strategy)'.format(Country),

                  title_x=0.5,

                  titlefont_size=14)



fig.show()



date_implemented_list

measure_descriptions = pd.DataFrame({'reference':keepn,'date_implemented_list':date_implemented_list,'comments_list':comments_list, 'measure':data_Country.measure.tolist()})

def f(dat, c='rgb(160,214,91)'):

    return [f'background-color: {c}' for i in dat]

measure_descriptions = measure_descriptions.sort_values(by='date_implemented_list')



display(measure_descriptions.style.apply(f, axis=0, subset=['reference']))



measure_descriptions.date_implemented_list = pd.to_datetime(measure_descriptions.date_implemented_list)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions.date_implemented_list.diff()

measure_descriptions['days_diff_btw_strategy'] = pd.to_timedelta(measure_descriptions['days_diff_btw_strategy'])

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].fillna(0)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].dt.days.astype('int16')





length = measure_descriptions.shape[0]-1

labelList = [str(i+1)+'-'+str(i+2) for i in measure_descriptions.index]

daysList = measure_descriptions.days_diff_btw_strategy.tolist()



label1 = labelList[0].split('-')[0]

label2 = labelList[0].split('-')[1]

fig = go.Figure()

fig.add_trace(go.Bar(

                     y=labelList,

                     x=daysList,

                     orientation='h',

                     marker_color=list(range(0,45))))

                     # y=labelList, hovertemplate = "%{y}: %{customdata}" name='days_diff_btw_strategy',,customdata=measure_descriptions.days_diff_btw_strategy,



    

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(yaxis=dict(type='category'), height=700, title_text='Days Differences between Strategies<br>(e.g. {0} = {2} strategy is taken after {3} days from {1} strategy)'.format(labelList[0], label1, label2, daysList[0]), title_x=0.5)

fig.update_yaxes(categoryorder='total ascending')





fig.show()

# how frequent they applied the covid-19 strategy



Country = 'China'

data_Country = data1[data1.country==Country].reset_index()

data_Country = data_Country[~data_Country.date_implemented.isnull()].reset_index()

#for i in range (data_Country.shape[0]):

#    print(data_Country.loc[i,'comments'])    



def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



def deaths_case_country(country_name):

    dc = data3[data3.reporting_country_territory==country_name]

    return dc[~dc.total_deaths.isnull()]



df_confirmed_case_Country = confirmed_case_country(Country)



df_deaths_case_Country = deaths_case_country(Country)



df_deaths_case_Country = df_deaths_case_Country[~df_deaths_case_Country.confirmed_cases.isnull()]



df_deaths_case_Country.confirmed_cases = df_deaths_case_Country[pd.to_numeric(df_deaths_case_Country.confirmed_cases, errors='coerce').notnull()]



 

fig = go.Figure()



#'#bd4844', '#c25653', ligtest '#e1abaa'

color_ = ['#672725', '#a03c39', '#c76561','#cc7370',

          '#d1817e','#d68f8d','#db9d9b','#ebc8c7']



fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.total_deaths,

          name='total_deaths')

)





date_implemented_list = data_Country.date_implemented.tolist()



ply_shapes = {}

for t in date_implemented_list:

    ply_shapes['shape_' + str(t)]=go.layout.Shape(type="line",

                                                  x0=t, y0=0, x1=t,

                                                  y1=1000, line_width=2)



lst_shapes=list(ply_shapes.values())

comments_list = data_Country.comments.tolist()



ply_annotation = {}

n= 0

ydelta1 = 1200

ydelta2 = 1500

keepn = []

for com in date_implemented_list:

    if n % 2 == 0:

        ydelta_ = ydelta1

    else:

        ydelta_ = ydelta2

    ply_annotation['annotation_' + str(com)]=go.layout.Annotation(showarrow=False,

                                                                   text=f'{n}',

                                                                   x=com,y=ydelta_,

                                                                   font=dict(size=10))

    keepn.append(date_implemented_list.index(com))

    n+=1

                                                                  

annot=list(ply_annotation.values())

fig.update_layout(shapes=lst_shapes, annotations=annot, barmode='overlay', 

                  title_text='{0} applied Strategy<br>(kindly refer to the following dataframe for further descriptions on each applied strategy)'.format(Country),

                  title_x=0.5,

                  titlefont_size=14)



fig.show()



date_implemented_list

measure_descriptions = pd.DataFrame({'reference':keepn,'date_implemented_list':date_implemented_list,'comments_list':comments_list, 'measure':data_Country.measure.tolist()})

def f(dat, c='rgb(160,214,91)'):

    return [f'background-color: {c}' for i in dat]

measure_descriptions = measure_descriptions.sort_values(by='date_implemented_list')



display(measure_descriptions.style.apply(f, axis=0, subset=['reference']))



measure_descriptions.date_implemented_list = pd.to_datetime(measure_descriptions.date_implemented_list)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions.date_implemented_list.diff()

measure_descriptions['days_diff_btw_strategy'] = pd.to_timedelta(measure_descriptions['days_diff_btw_strategy'])

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].fillna(0)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].dt.days.astype('int16')





length = measure_descriptions.shape[0]-1

labelList = [str(i+1)+'-'+str(i+2) for i in measure_descriptions.index]

daysList = measure_descriptions.days_diff_btw_strategy.tolist()



label1 = labelList[0].split('-')[0]

label2 = labelList[0].split('-')[1]

fig = go.Figure()

fig.add_trace(go.Bar(

                     y=labelList,

                     x=daysList,

                     orientation='h',

                     marker_color=list(range(0,45))))

                     # y=labelList, hovertemplate = "%{y}: %{customdata}" name='days_diff_btw_strategy',,customdata=measure_descriptions.days_diff_btw_strategy,



    

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(yaxis=dict(type='category'), height=700, title_text='Days Differences between Strategies<br>(e.g. {0} = {2} strategy is taken after {3} days from {1} strategy)'.format(labelList[0], label1, label2, daysList[0]), title_x=0.5)

fig.update_yaxes(categoryorder='total ascending')



fig.show()
# Germany Analysis based on time series:

# how frequent they applied the covid-19 strategy



Country = 'Korea Republic of'#'South Korea'

data_Country = data1[data1.country==Country].reset_index()

data_Country = data_Country[~data_Country.date_implemented.isnull()].reset_index()





def confirmed_case_country(country_name): # 'South Korea'

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



def deaths_case_country(country_name): # 'Republic of Korea'

    dc = data3[data3.reporting_country_territory==country_name]

    return dc[~dc.total_deaths.isnull()]



if 'Korea' in Country:

    df_confirmed_case_Country = confirmed_case_country('South Korea')

else:

    df_confirmed_case_Country = confirmed_case_country(Country)



if 'Korea' in Country:

    df_deaths_case_Country = deaths_case_country('Republic of Korea')

else:

    df_deaths_case_Country = deaths_case_country(Country)



df_deaths_case_Country = df_deaths_case_Country[~df_deaths_case_Country.confirmed_cases.isnull()]

df_deaths_case_Country.confirmed_cases = df_deaths_case_Country.confirmed_cases.astype(np.int64)

df_deaths_case_Country = df_deaths_case_Country[df_deaths_case_Country.confirmed_cases<900000]

 

fig = go.Figure()



#'#bd4844', '#c25653', ligtest '#e1abaa'

color_ = ['#672725', '#a03c39', '#c76561','#cc7370',

          '#d1817e','#d68f8d','#db9d9b','#ebc8c7']



fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.confirmed_cases,

          name='confirmed_cases')

)

fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.total_deaths,

          name='total_deaths')

)





date_implemented_list = data_Country.date_implemented.tolist()



ply_shapes = {}

for t in date_implemented_list:

    ply_shapes['shape_' + str(t)]=go.layout.Shape(type="line",

                                                  x0=t, y0=0, x1=t,

                                                  y1=150, line_width=2

                                                 )



lst_shapes=list(ply_shapes.values())

comments_list = data_Country.comments.tolist()



ply_annotation = {}

n= 0

ydelta1 = 155

ydelta2 = 160

keepn = []

for com in date_implemented_list:

    if n % 2 == 0:

        ydelta_ = ydelta1

    else:

        ydelta_ = ydelta2

    ply_annotation['annotation_' + str(com)]=go.layout.Annotation(showarrow=False,

                                                                   text=f'{n}',

                                                                   x=com,y=ydelta_,

                                                                   font=dict(size=10))

    keepn.append(date_implemented_list.index(com))

    n+=1

                                                                  

annot=list(ply_annotation.values())

fig.update_layout(shapes=lst_shapes, annotations=annot, barmode='overlay', 

                  title_text='{0} applied Strategy<br>(kindly refer to the following dataframe for further descriptions on each applied strategy)'.format(Country),

                  title_x=0.5,

                 titlefont_size=14)



fig.show()



date_implemented_list

measure_descriptions = pd.DataFrame({'reference':keepn,'date_implemented_list':date_implemented_list,'comments_list':comments_list, 'measure':data_Country.measure.tolist()})

def f(dat, c='rgb(160,214,91)'):

    return [f'background-color: {c}' for i in dat]

measure_descriptions = measure_descriptions.sort_values(by='date_implemented_list')



display(measure_descriptions.style.apply(f, axis=0, subset=['reference']))



measure_descriptions.date_implemented_list = pd.to_datetime(measure_descriptions.date_implemented_list)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions.date_implemented_list.diff()

measure_descriptions['days_diff_btw_strategy'] = pd.to_timedelta(measure_descriptions['days_diff_btw_strategy'])

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].fillna(0)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].dt.days.astype('int16')





length = measure_descriptions.shape[0]-1

labelList = [str(i+1)+'-'+str(i+2) for i in measure_descriptions.index]

daysList = measure_descriptions.days_diff_btw_strategy.tolist()



label1 = labelList[0].split('-')[0]

label2 = labelList[0].split('-')[1]

fig = go.Figure()

fig.add_trace(go.Bar(

                     y=labelList,

                     x=daysList,

                     orientation='h',

                     marker_color=list(range(0,45))))

                     # y=labelList, hovertemplate = "%{y}: %{customdata}" name='days_diff_btw_strategy',,customdata=measure_descriptions.days_diff_btw_strategy,



    

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(yaxis=dict(type='category'), height=700, title_text='Days Differences between Strategies<br>(e.g. {0} = {2} strategy is taken after {3} days from {1} strategy)'.format(labelList[0], label1, label2, daysList[0]), title_x=0.5)

fig.update_yaxes(categoryorder='total ascending')





fig.show()

# how frequent they applied the covid-19 strategy



Country = 'Italy'

data_Country = data1[data1.country==Country].reset_index()

data_Country = data_Country[~data_Country.date_implemented.isnull()].reset_index()





def confirmed_case_country(country_name): # 'South Korea'

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



def deaths_case_country(country_name): # 'Republic of Korea'

    dc = data3[data3.reporting_country_territory==country_name]

    return dc[~dc.total_deaths.isnull()]



if 'Korea' in Country:

    df_confirmed_case_Country = confirmed_case_country('South Korea')

else:

    df_confirmed_case_Country = confirmed_case_country(Country)



if 'Korea' in Country:

    df_deaths_case_Country = deaths_case_country('Republic of Korea')

else:

    df_deaths_case_Country = deaths_case_country(Country)



df_deaths_case_Country = df_deaths_case_Country[~df_deaths_case_Country.confirmed_cases.isnull()]

df_deaths_case_Country.confirmed_cases = df_deaths_case_Country.confirmed_cases.astype(np.int64)

df_deaths_case_Country = df_deaths_case_Country[df_deaths_case_Country.confirmed_cases<900000]

 

fig = go.Figure()



#'#bd4844', '#c25653', ligtest '#e1abaa'

color_ = ['#672725', '#a03c39', '#c76561','#cc7370',

          '#d1817e','#d68f8d','#db9d9b','#ebc8c7']



fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.confirmed_cases,

          name='confirmed_cases')

)

fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.total_deaths,

          name='total_deaths')

)





date_implemented_list = data_Country.date_implemented.tolist()



ply_shapes = {}

for t in date_implemented_list:

    ply_shapes['shape_' + str(t)]=go.layout.Shape(type="line",

                                                  x0=t, y0=0, x1=t,

                                                  y1=9000, line_width=2

                                                 )



lst_shapes=list(ply_shapes.values())

comments_list = data_Country.comments.tolist()



ply_annotation = {}

n= 0

ydelta1 = 9050

ydelta2 = 9100

keepn = []

for com in date_implemented_list:

    if n % 2 == 0:

        ydelta_ = ydelta1

    else:

        ydelta_ = ydelta2

    ply_annotation['annotation_' + str(com)]=go.layout.Annotation(showarrow=False,

                                                                   text=f'{n}',

                                                                   x=com,y=ydelta_,

                                                                   font=dict(size=10))

    keepn.append(date_implemented_list.index(com))

    n+=1

                                                                  

annot=list(ply_annotation.values())

fig.update_layout(shapes=lst_shapes, annotations=annot, barmode='overlay', 

                  title_text='{0} applied Strategy<br>(kindly refer to the following dataframe for further descriptions on each applied strategy)'.format(Country),

                  title_x=0.5,

                 titlefont_size=14)



fig.show()



date_implemented_list

measure_descriptions = pd.DataFrame({'reference':keepn,'date_implemented_list':date_implemented_list,'comments_list':comments_list, 'measure':data_Country.measure.tolist()})

def f(dat, c='rgb(160,214,91)'):

    return [f'background-color: {c}' for i in dat]

measure_descriptions = measure_descriptions.sort_values(by='date_implemented_list')



display(measure_descriptions.style.apply(f, axis=0, subset=['reference']))



measure_descriptions.date_implemented_list = pd.to_datetime(measure_descriptions.date_implemented_list)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions.date_implemented_list.diff()

measure_descriptions['days_diff_btw_strategy'] = pd.to_timedelta(measure_descriptions['days_diff_btw_strategy'])

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].fillna(0)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].dt.days.astype('int16')





length = measure_descriptions.shape[0]-1

labelList = [str(i+1)+'-'+str(i+2) for i in measure_descriptions.index]

daysList = measure_descriptions.days_diff_btw_strategy.tolist()



label1 = labelList[0].split('-')[0]

label2 = labelList[0].split('-')[1]

fig = go.Figure()

fig.add_trace(go.Bar(

                     y=labelList,

                     x=daysList,

                     orientation='h',

                     marker_color=list(range(0,45))))

                     # y=labelList, hovertemplate = "%{y}: %{customdata}" name='days_diff_btw_strategy',,customdata=measure_descriptions.days_diff_btw_strategy,



    

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(yaxis=dict(type='category'), height=700, title_text='Days Differences between Strategies<br>(e.g. {0} = {2} strategy is taken after {3} days from {1} strategy)'.format(labelList[0], label1, label2, daysList[0]), title_x=0.5)

fig.update_yaxes(categoryorder='total ascending')





fig.show()

# how frequent they applied the covid-19 strategy



Country = 'Spain'#'South Korea'

data_Country = data1[data1.country==Country].reset_index()

data_Country = data_Country[~data_Country.date_implemented.isnull()].reset_index()





def confirmed_case_country(country_name): # 'South Korea'

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



def deaths_case_country(country_name): # 'Republic of Korea'

    dc = data3[data3.reporting_country_territory==country_name]

    return dc[~dc.total_deaths.isnull()]



if 'Korea' in Country:

    df_confirmed_case_Country = confirmed_case_country('South Korea')

else:

    df_confirmed_case_Country = confirmed_case_country(Country)



if 'Korea' in Country:

    df_deaths_case_Country = deaths_case_country('Republic of Korea')

else:

    df_deaths_case_Country = deaths_case_country(Country)



df_deaths_case_Country = df_deaths_case_Country[~df_deaths_case_Country.confirmed_cases.isnull()]

df_deaths_case_Country.confirmed_cases = df_deaths_case_Country.confirmed_cases.astype(np.int64)

df_deaths_case_Country = df_deaths_case_Country[df_deaths_case_Country.confirmed_cases<900000]

 

fig = go.Figure()



#'#bd4844', '#c25653', ligtest '#e1abaa'

color_ = ['#672725', '#a03c39', '#c76561','#cc7370',

          '#d1817e','#d68f8d','#db9d9b','#ebc8c7']



fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.confirmed_cases,

          name='confirmed_cases')

)

fig.add_trace(

    go.Bar(x=df_deaths_case_Country.reported_date,

           y=df_deaths_case_Country.total_deaths,

          name='total_deaths')

)





date_implemented_list = data_Country.date_implemented.tolist()



ply_shapes = {}

for t in date_implemented_list:

    ply_shapes['shape_' + str(t)]=go.layout.Shape(type="line",

                                                  x0=t, y0=0, x1=t,

                                                  y1=9000, line_width=2

                                                 )



lst_shapes=list(ply_shapes.values())

comments_list = data_Country.comments.tolist()



ply_annotation = {}

n= 0

ydelta1 = 9500

ydelta2 = 9200

keepn = []

for com in date_implemented_list:

    if n % 2 == 0:

        ydelta_ = ydelta1

    else:

        ydelta_ = ydelta2

    ply_annotation['annotation_' + str(com)]=go.layout.Annotation(showarrow=False,

                                                                   text=f'{n}',

                                                                   x=com,y=ydelta_,

                                                                   font=dict(size=10))

    keepn.append(date_implemented_list.index(com))

    n+=1

                                                                  

annot=list(ply_annotation.values())

fig.update_layout(shapes=lst_shapes, annotations=annot, barmode='overlay', 

                  title_text='{0} applied Strategy<br>(kindly refer to the following dataframe for further descriptions on each applied strategy)'.format(Country),

                  title_x=0.5,

                 titlefont_size=14)



fig.show()



date_implemented_list

measure_descriptions = pd.DataFrame({'reference':keepn,'date_implemented_list':date_implemented_list,'comments_list':comments_list, 'measure':data_Country.measure.tolist()})

def f(dat, c='rgb(160,214,91)'):

    return [f'background-color: {c}' for i in dat]

measure_descriptions = measure_descriptions.sort_values(by='date_implemented_list').reset_index(drop=True)

display(measure_descriptions.style.apply(f, axis=0, subset=['reference']))



measure_descriptions.date_implemented_list = pd.to_datetime(measure_descriptions.date_implemented_list)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions.date_implemented_list.diff()

measure_descriptions['days_diff_btw_strategy'] = pd.to_timedelta(measure_descriptions['days_diff_btw_strategy'])

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].fillna(0)

measure_descriptions['days_diff_btw_strategy'] = measure_descriptions['days_diff_btw_strategy'].dt.days.astype('int16')





length = measure_descriptions.shape[0]-1

labelList = [str(i+1)+'-'+str(i+2) for i in measure_descriptions.index]

daysList = measure_descriptions.days_diff_btw_strategy.tolist()



label1 = labelList[0].split('-')[0]

label2 = labelList[0].split('-')[1]

fig = go.Figure()

fig.add_trace(go.Bar(

                     y=labelList,

                     x=daysList,

                     orientation='h',

                     marker_color=list(range(0,45))))

                     # y=labelList, hovertemplate = "%{y}: %{customdata}" name='days_diff_btw_strategy',,customdata=measure_descriptions.days_diff_btw_strategy,



    

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(yaxis=dict(type='category'), height=700, title_text='Days Differences between Strategies<br>(e.g. {0} = {2} strategy is taken after {3} days from {1} strategy)'.format(labelList[0], label1, label2, daysList[0]), title_x=0.5)

fig.update_yaxes(categoryorder='total ascending')





fig.show()

df1 = data[~data.total_confirmed_cases_of_covid_19_cases.isnull()]

df1 = df1[~df1.total_covid_19_tests.isnull()]

df1.total_covid_19_tests.apply(np.int64)

import plotly.express as px



fig = px.scatter(df1, x="date", y="total_covid_19_tests",

                 size="total_covid_19_tests", color="entity",

                 color_discrete_sequence=px.colors.sequential.Plasma,

                 hover_name="entity", size_max=60)

fig.show()
# testing focus

foc_cat = 'Public health measures'

foc_mea = 'Mass population testing'



def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



countries = []

days_btw_confirmed_strategy = []

df_cat = data1[data1.category==foc_cat].reset_index()

df_cat_mea = df_cat[df_cat.measure==foc_mea].reset_index()

countryList = df_cat_mea.country.unique().tolist()

print(countryList)

#display(df_cat)

def measure_extract(NATION):

    df_1 = df_cat[df_cat.country==NATION]

    df_2 = df_1[df_1.measure==foc_mea].reset_index()

    return df_2



# first case vs Full Lockdown

for nation in countryList:

    df_ita_ = measure_extract(nation)

    if 'United States' in nation:

        df_ita_confirmed_ = confirmed_case_country('United States')

    else:

        df_ita_confirmed_ = confirmed_case_country(nation)

    

    temp = df_ita_confirmed_.sort_values(by='total_confirmed_cases_of_covid_19_cases', ascending=True).reset_index()

    for idx_ in temp.total_confirmed_cases_of_covid_19_cases:

        if idx_ !=0.0:

            found_time = temp.total_confirmed_cases_of_covid_19_cases[temp.total_confirmed_cases_of_covid_19_cases == idx_].index.tolist()

            idx_ = min(found_time) # get the first confirmed case

            break

    try:

        df_ita_confirmed_ = df_ita_confirmed_.reset_index()

        # negative means announcement announced after the first confirmed case

        days_btw_confirmed_strategy.append([nation, (pd.to_datetime(df_ita_confirmed_.loc[idx_,'date']) - pd.to_datetime(df_ita_.date_implemented.min())).days,

                                           df_ita_confirmed_.loc[idx_,'date'], df_ita_.date_implemented.min()])

    except Exception as e:

        #print(e)

        pass



df_confirmed_case_vs_fulllockdown = pd.DataFrame(days_btw_confirmed_strategy, columns=['country','days_diff','first_confirmed_date', 'mass_population_testing_date'])

df_confirmed_case_vs_fulllockdown = df_confirmed_case_vs_fulllockdown.sort_values(by='days_diff')

fig = go.Figure()



fig.add_trace(go.Bar(x=df_confirmed_case_vs_fulllockdown.days_diff.values,

                     y=df_confirmed_case_vs_fulllockdown.country,

                     orientation='h',

                     name='days_diff',

                     marker_color=list(range(35,45)),

                     customdata=df_confirmed_case_vs_fulllockdown.days_diff,

                     hovertemplate = "%{y}: %{customdata}"))   

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(barmode='relative', 

                  yaxis_autorange='reversed',

                  bargap=0.01,

                  legend_orientation ='h',

                  legend_x=-0.05, legend_y=1.1,

                  title_text = 'First Case vs {0}'.format(foc_mea),

                  title_x = 0.5

                 )



fig.show()
# testing focus

foc_cat = 'Lockdown'

foc_mea = 'Full lockdown'



def confirmed_case_country(country_name):

    dc = data2[data2.entity==country_name]

    return dc[~dc.total_confirmed_cases_of_covid_19_cases.isnull()]



countries = []

days_btw_confirmed_strategy = []

df_cat = data1[data1.category==foc_cat].reset_index()

df_cat_mea = df_cat[df_cat.measure==foc_mea].reset_index()

countryList = df_cat_mea.country.unique().tolist()

print(countryList)

#display(df_cat)

def measure_extract(NATION):

    df_1 = df_cat[df_cat.country==NATION]

    df_2 = df_1[df_1.measure==foc_mea].reset_index()

    return df_2



# first case vs Full Lockdown

for nation in countryList:

    df_ita_ = measure_extract(nation)

    if 'United States' in nation:

        df_ita_confirmed_ = confirmed_case_country('United States')

    else:

        df_ita_confirmed_ = confirmed_case_country(nation)

    

    temp = df_ita_confirmed_.sort_values(by='total_confirmed_cases_of_covid_19_cases', ascending=True).reset_index()

    for idx_ in temp.total_confirmed_cases_of_covid_19_cases:

        if idx_ !=0.0:

            found_time = temp.total_confirmed_cases_of_covid_19_cases[temp.total_confirmed_cases_of_covid_19_cases == idx_].index.tolist()

            idx_ = min(found_time) # get the first confirmed case

            break

    try:

        df_ita_confirmed_ = df_ita_confirmed_.reset_index()

        # negative means announcement announced after the first confirmed case

        days_btw_confirmed_strategy.append([nation, (pd.to_datetime(df_ita_confirmed_.loc[idx_,'date']) - pd.to_datetime(df_ita_.date_implemented.min())).days,

                                           df_ita_confirmed_.loc[idx_,'date'], df_ita_.date_implemented.min()])

    except Exception as e:

        #print(e)

        pass



df_confirmed_case_vs_fulllockdown = pd.DataFrame(days_btw_confirmed_strategy, columns=['country','days_diff','first_confirmed_date', 'mass_population_testing_date'])

df_confirmed_case_vs_fulllockdown = df_confirmed_case_vs_fulllockdown.sort_values(by='days_diff')

fig = go.Figure()



fig.add_trace(go.Bar(x=df_confirmed_case_vs_fulllockdown.days_diff.values,

                     y=df_confirmed_case_vs_fulllockdown.country,

                     orientation='h',

                     name='days_diff',

                     marker_color=list(range(30,45)),

                     customdata=df_confirmed_case_vs_fulllockdown.days_diff,

                     hovertemplate = "%{y}: %{customdata}"))   

fig.update_xaxes(title='The Numbers of Days Differences')

fig.update_layout(barmode='relative', 

                  yaxis_autorange='reversed',

                  bargap=0.01,

                  legend_orientation ='h',

                  legend_x=-0.05, legend_y=1.1,

                  title_text = 'First Case vs {0}'.format(foc_mea),

                  title_x = 0.5

                 )



fig.show()
data.reporting_country_territory.unique() # reporting_country_territory

import plotly.express as px

fig = px.bar(data, x='reported_date', y='new_confirmed_cases',

                  color_discrete_sequence=['#3810dc'])

fig.show()

# I will end with the following chart to remind myself again about the key objective of solving which problems and how these countries could help reasoning out which strategy is better to be applied.           

data_path_2 = '/kaggle/input/uncover/UNCOVER/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv'

data_2 = pd.read_csv(data_path_2)

df_test_vs_confirmed = data_2[~data_2.total_confirmed_cases_of_covid_19_cases.isnull()]

import plotly.graph_objects as go

fig = go.Figure()

countryList = data_2.entity.unique().tolist()

countryList.remove('World')

for nation in countryList:

    df_ = df_test_vs_confirmed[df_test_vs_confirmed.entity==nation]

    fig.add_trace(go.Scatter(x=df_.date, 

                             y=df_.total_confirmed_cases_of_covid_19_cases,

                             name=nation,

                             mode='lines+markers'

                            ))



fig.update_xaxes(showspikes=True, title='Date')

fig.update_yaxes(showspikes=True, title='Total of Confirmed Cases')

fig.update_layout(

    title_text='Total confirmed cases of Nations',

    title_x=0.5

)

fig.show()