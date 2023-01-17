#Numpy/Pandas

import numpy as np

import pandas as pd



# plotly

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

import plotly.offline as off



#For HTML Rendering

from IPython.core.display import display, HTML



#matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



#folium for Map

import folium

from folium import plugins





# word cloud

from wordcloud import WordCloud



%matplotlib inline
#Setting Matplotlib Params

plt.rcParams["font.size"] = 12

plt.rcParams["font.weight"] = 'bold'

plt.rcParams["figure.figsize"] = (15,10)



#Setting Seaborn Style

sns.set_style("whitegrid")
df = pd.read_csv('../input/Attacks on Political Leaders in Pakistan.csv', encoding='latin1')
#Fixing misspelled column name

lat_col = df.columns.values

lat_col[11] = 'Longitude'

df.columns = lat_col

df.columns
df.head()
df.info()
#Dropping Irrelevant columns

df.drop('S#', axis=1, inplace=True)
#Replace NULL values in Location Category with UNKNOWN

df['Location Category'].fillna('UNKNOWN', inplace=True)



#Giving same values but with different text, the same text

df.loc[df['Location Category'] == 'Details Missing', 'Location Category'] = 'UNKNOWN'

df.loc[df['Province'] == 'Fata', 'Province'] = 'FATA'

df.loc[df['City'] == 'ATTOCK', 'City'] = 'Attock'
#Convert Categorical variables to Category type

columns = ['Target Status', 'Day', 'Day Type', 'Time', 'City', 'Location Category',

          'Province', 'Target Category', 'Space (Open/Closed)', 'Party']

df[columns] = df[columns].astype('category')
#Get Month and Year of Attack

df['month'] = pd.DatetimeIndex(df['Date']).month

df['year'] = pd.DatetimeIndex(df['Date']).year



#correct wrong interpretation of data

df['year']=df['year'].replace(2051, 1951)

df['year']=df['year'].replace(2058, 1958)
df.describe()
print(df['Target Category'].value_counts())

print(df['Target Status'].value_counts())

print(df['Space (Open/Closed)'].value_counts())
df['marker_popup'] = ''

for index, row in df.iterrows():

    df.loc[index, 'marker_popup'] = df.loc[index,'City'].strip() + '(' + str(df.loc[index,'Date']) + '  |  <b>Killed</b>: ' + str(df.loc[index,'Killed']) + '  |  Injured: ' + str(df.loc[index,'Injured'])  + ')'
df['Target Attack'] = (df['Target Category'] == 'Target').astype(int)

df['Suicide Attack'] = (df['Target Category'] != 'Target').astype(int)

df['Open Space'] = (df['Space (Open/Closed)'] == 'Open').astype(int)

df['Closed Space'] = (df['Space (Open/Closed)'] != 'Open').astype(int)

df['Politician Killed'] = (df['Target Status'] == 'Killed').astype(int)

df['Politician Escaped'] = (df['Target Status'] != 'Killed').astype(int)
month_lookup = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',

            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}



df['month'] = df['month'].apply(lambda x: month_lookup[x])
pk_map = folium.Map(location=[30.3753, 69.3451],

                   zoom_start=5)

# mark each station as a point

for index, row in df.iterrows():

    folium.Marker([df.loc[index,'Latitude'], df.loc[index,'Longitude']],

                  icon=folium.Icon(color= 'red' if df.loc[index, 'Target Status'] == 'Killed' else 'green'),

                  popup=df.loc[index,'marker_popup']).add_to(pk_map)

pk_map
sns.countplot(y='Day', data = df)
sns.countplot(y='Time', data = df)
sns.countplot(y='City', data = df)
sns.countplot(x='Province', data = df)
sns.countplot(y='Location Category', data = df)
sns.countplot(y='Party', data = df)
def draw_barchart(dataframe, x_col, y_cols, chart_title='', x_title='', y_title='', agg_func = 'sum', tick_angle=0):

    

    if dataframe is None:

        raise ValueError('dataframe is not Provided')

    if not isinstance(dataframe, pd.DataFrame):

        raise ValueError('dataframe should be of type Pandas Dataframe')

    if type(x_col) is not str:

        raise ValueError('x_col should be of string type')

    if not isinstance(y_cols,(list,)):

        raise ValueError('x_col should be passed as a list')

    

    Province = dataframe[x_col]

    data = []



    for i in range(len(y_cols)):

        data.append(

            dict(

            type = 'bar',

            x = Province,

            y = dataframe[y_cols[i]],

            name = y_cols[i],

            transforms = [

                dict(

                    type = 'aggregate',

                    groups = Province,

                    aggregations = [dict(

                        target = 'y', func = agg_func, enabled = True)]

                )

            ]

            )

        )





    if tick_angle > 0:

        layout = dict(

            title = '<b>' + chart_title + '</b>',

            xaxis = dict(title = x_col if len(x_title) == 0 else x_title, tickangle=tick_angle),

            yaxis = dict(title = y_title),

            barmode = 'relative'

        )

    else:

        layout = dict(

            title = '<b>' + chart_title + '</b>',

            xaxis = dict(title = x_col if len(x_title) == 0 else x_title),

            yaxis = dict(title = y_title),

            barmode = 'relative'

    )



    off.iplot({

        'data': data,

        'layout': layout

    }, validate = False)
def draw_bubblechart(dataframe, x_col, y_cols):

    

    if dataframe is None:

        raise ValueError('dataframe is not Provided')

    if not isinstance(dataframe, pd.DataFrame):

        raise ValueError('dataframe should be of type Pandas Dataframe')

    if type(x_col) is not str:

        raise ValueError('x_col should be of string datatype')

    if not isinstance(y_cols,(list,)):

        raise ValueError('x_col should be passed as a list')

    

    ycol_size = len(y_cols)

    data = []

    updatemenu_list = []

    

    for i in range(ycol_size):

        visible = [True if j == i else False for j in range(ycol_size)]

        data.append(

            dict(

                type = 'scatter',

                mode = 'markers',

                x = dataframe[x_col],

                y = dataframe[y_cols[i]],

                text = dataframe[y_cols[i]],

                hoverinfo = 'text',

                name = y_cols[i],

                opacity = 0.8,

                marker = dict(

                    size = dataframe[y_cols[i]],

                    sizemode = 'area'

                ),

                transforms = [

                    dict(

                        type = 'aggregate',

                        groups = dataframe[x_col],

                        aggregations = [dict(

                            target = 'y', func = 'sum', enabled = True)]

                    )

                ]

            ))

        updatemenu_list.append(

            dict(label = y_cols[i],

                method = 'update',

                args = [{

                        'visible': visible

                    },

                    {

                        'title': y_cols[i] + ' Per Year',

                        'yaxis.title': y_cols[i]

                    }

                ])

        )



    layout = dict(

        title = '<b>Casualty Rate Per Year</b>',

        xaxis = dict(

            title = x_col,

            showgrid = False

        ),

        yaxis = dict(

            type = 'exp'

        ),

        updatemenus = list([

            dict(

                active = -1,

                buttons = updatemenu_list,

                direction = 'down',

                pad = {'r': 10, 't': 10},

                showactive = True,

                x = 0.05,

                xanchor = 'left',

                y = 1.1,

                yanchor = 'top'

            )

        ])

    )



    off.iplot({

        'data': data,

        'layout': layout

    }, validate = False)
draw_barchart(df, 'Province', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate')
draw_barchart(df, 'Province', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Politician Casualty Rate')
draw_barchart(df, 'Province', ['Target Attack', 'Suicide Attack'], 'Suicide Attack vs Target Attack By Province','Province', 'Attack Type')
draw_barchart(df, 'Province', ['Open Space', 'Closed Space'], 'Open/Closed Space Attacks By Province','Province', 'Open/Closed Space')
#Group By Province and Day

df_province_day = df.groupby(['Province', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_province_day.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_province_day['ProvinceByDay'] = df_province_day['Day'].str.cat(df_province_day['Province'],sep='-' )

df_province_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_province_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_province_day, 'ProvinceByDay', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)
draw_barchart(df_province_day, 'ProvinceByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)
#Group By Province and Time

df_province_time = df.groupby(['Province', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_province_time.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_province_time['ProvinceByTime'] = df_province_time['Time'].str.cat(df_province_time['Province'],sep='-' )

df_province_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_province_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_province_time, 'ProvinceByTime', ['Killed', 'Injured'], 'Total Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)
draw_barchart(df_province_time, 'ProvinceByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Province','Province', 'Casualty Rate', tick_angle=45)
draw_barchart(df, 'City', ['Killed', 'Injured'], 'Total Casualty By City ','City', 'Total Casualty', tick_angle=90)
draw_barchart(df, 'City',  ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By City ','City', 'Politician Casualty', tick_angle=90)
draw_barchart(df, 'City', ['Target Attack', 'Suicide Attack'], 'Suicide Attack vs Target Attack By City','City', 'Attack Type')
draw_barchart(df, 'City', ['Open Space', 'Closed Space'], 'Open/Closed Space Attacks By City','City', 'Open/Closed Space')
#Group By City and Time

df_city_time = df.groupby(['City', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_city_time.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_city_time['CityByTime'] = df_city_time['Time'].str.cat(df_city_time['City'],sep='-' )

df_city_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_city_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_city_time, 'CityByTime', ['Killed', 'Injured'], 'Total Casualty Rate By City and Time','City', 'Casualty Rate', tick_angle=45)
draw_barchart(df_city_time, 'CityByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By City and Time','City', 'Casualty Rate', tick_angle=45)
#Group By City and Day

df_city_day = df.groupby(['City', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_city_day.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_city_day['CityByDay'] = df_city_day['Day'].str.cat(df_city_day['City'],sep='-' )

df_city_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_city_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_city_day, 'CityByDay', ['Killed', 'Injured'], 'Total Casualty Rate By City and Day','City', 'Casualty Rate', tick_angle=45)
draw_barchart(df_city_day, 'CityByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By City and Day','City', 'Casualty Rate', tick_angle=45)
draw_barchart(df, 'Day', ['Target Attack', 'Suicide Attack'], 'Attack Type By Days','Day', 'Attack Type')
draw_barchart(df, 'Time', ['Target Attack', 'Suicide Attack'], 'Attack Type By Time of Day','Time of Day', 'Attack Type')
draw_barchart(df, 'Location Category', ['Target Attack', 'Suicide Attack'], 'Attack Type By Time of Day','Time of Day', 'Attack Type')
draw_barchart(df, 'Day', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Day','Day', 'Politician Casualty')
draw_barchart(df, 'Location Category', ['Killed', 'Injured'], 'Total Casualty By Location ','Location', 'Total Casualty')
draw_barchart(df, 'Location Category', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Location ','Location', 'Politician Casualty')
#Group By Location and Time

df_loc_time = df.groupby(['Location Category', 'Time'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_loc_time.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_loc_time['LocationByTime'] = df_loc_time['Time'].str.cat(df_loc_time['Location Category'],sep='-' )

df_loc_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_loc_time[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_loc_time, 'LocationByTime', ['Killed', 'Injured'], 'Total Casualty Rate By Location Category and Time','Location Category', 'Casualty Rate', tick_angle=45)
draw_barchart(df_loc_time, 'LocationByTime', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Location Category and Time','Location Category', 'Casualty Rate', tick_angle=45)
#Group By Location and Time

df_loc_day = df.groupby(['Location Category', 'Day'])['Politician Killed', 'Politician Escaped', 'Killed', 'Injured'].sum()

#Reset Index

df_loc_day.reset_index(level=[0, 1], inplace=True)

#Join and Clean Columns

df_loc_day['LocationByDay'] = df_loc_day['Day'].str.cat(df_loc_day['Location Category'],sep='-' )

df_loc_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']] = df_loc_day[['Politician Killed', 'Politician Escaped', 'Killed', 'Injured']].fillna(0)
draw_barchart(df_loc_day, 'LocationByDay', ['Killed', 'Injured'], 'Total Casualty Rate By Location Category and Day','Location Category', 'Casualty Rate', tick_angle=45)
draw_barchart(df_loc_day, 'LocationByDay', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty Rate By Location Category and Day','Location Category', 'Casualty Rate', tick_angle=45)
draw_barchart(df, 'Target Category', ['Killed', 'Injured'], 'Total Casualty By Attack Type','Attack Type', 'Total Casualty')
draw_barchart(df, 'Target Category', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Attack Type','Attack Type', 'Politician Casualty')
draw_barchart(df, 'Party', ['Killed', 'Injured'], 'Total Casualty By Party ','Party', 'Total Casualty')
draw_barchart(df, 'Party', ['Politician Killed', 'Politician Escaped'], 'Politician Casualty By Party ','Party', 'Politician Casualty')
draw_barchart(df, 'Party', ['Target Attack', 'Suicide Attack'], 'Attack Type By Party ','Party', 'Attack Type')
#Group By Year

df_year = df.groupby('year')['Injured', 'Killed'].sum()

df_year.reset_index(level=0, inplace=True)
draw_bubblechart(df_year, 'year', ['Killed', 'Injured'])
wordcloud = WordCloud(background_color='white').generate(" ".join(df['Location']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()