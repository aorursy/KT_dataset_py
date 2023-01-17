# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)



import os

debug = False 

if debug : 

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



#sns.set(style="darkgrid")
infected = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_casos_long.csv')

uci_beds = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_camas_uci_2017.csv')

recovered = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_altas_long.csv')

death = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_fallecidos_long.csv')

hospitalized = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_hospitalizados_long.csv')

masks = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_mascarillas.csv')

uci = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_uci_long.csv')

national = pd.read_csv('/kaggle/input/covid19-in-spain/nacional_covid19.csv')

age_range = pd.read_csv('/kaggle/input/covid19-in-spain/nacional_covid19_rango_edad.csv')



debug = False

if debug : 

 print(infected)

 print(recovered)

 print(death)

 print(hospitalized)

 print(uci)


def get_location_list():

    """Return the list of the 19 spain regions (also called comunidades autonomas). 

    We remove the totals (last element [:-1])"""

    return pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_casos_long.csv').CCAA.unique()[:-1]



def load_location_data(location):

    """Return the dataframe for a given region"""



    total_df = pd.DataFrame()



    infected = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_casos_long.csv')

    infected = infected[(infected['CCAA'] == location)][['fecha','total']].sort_values(by='fecha')

    infected.rename(columns={'fecha':'date','Total':'total' }, inplace=True)

    infected.set_index('date', inplace=True)

    total_df['infected'] = infected['total']



    recovered = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_altas_long.csv')

    recovered.rename(columns={'fecha':'date','Total':'total' }, inplace=True)

    recovered = recovered[(recovered['CCAA'] == location)][['date','total']].sort_values(by='date')

    recovered.set_index('date', inplace=True)

    total_df['recovered'] = recovered['total']



    death = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_fallecidos_long.csv')

    death.rename(columns={'fecha':'date','Total':'total' }, inplace=True)

    death = death[(death['CCAA'] == location)][['date','total']].sort_values(by='date')

    death.set_index('date', inplace=True)

    total_df['death'] = death['total']



    hospitalized = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_hospitalizados_long.csv')

    hospitalized.rename(columns={'fecha':'date','Total':'total' }, inplace=True)

    hospitalized = hospitalized[(hospitalized['CCAA'] == location)][['date','total']].sort_values(by='date')

    hospitalized.set_index('date', inplace=True)

    total_df['hospitalized'] = hospitalized['total']



    uci = pd.read_csv('/kaggle/input/covid19-in-spain/ccaa_covid19_uci_long.csv')

    uci.rename(columns={'fecha':'date','Total':'total' }, inplace=True)

    uci = uci[(uci['CCAA'] == location)][['date','total']].sort_values(by='date')

    uci.set_index('date', inplace=True)

    total_df['intensive care unit'] = uci['total']

    

    total_df['location'] = location



    return total_df



df = load_location_data('Madrid')

df.tail()
debug = False

if debug : print(get_location_list()) 
# Functions for data wrangling



import numpy as np

    

def enrich_data(df):

    """Add daily increases , porcentage, derived and other columns"""

    

    if 'date' in df.columns :

        df.set_index('date', inplace=True) 

	

    # Headers 

    #infected	recovered	death	hospitalized	uci	location



    # Death : daily increase, daily porcentage and daily derived 

    df['death daily increase'] = df['death'] - df['death'].shift(1)

    df['death daily increase percentage'] = df['death daily increase']  / df['death'] 

    df['death daily increase derived'] = df['death daily increase'] - df['death daily increase'].shift(1)





    # Infected : daily increase, daily porcentage and daily derived 

    df['infected daily increase'] = df['infected'] - df['infected'].shift(1)

    df['infected daily increase percentage'] = df['infected daily increase']  / df['infected'] 

    df['infected daily increase derived'] = df['infected daily increase'] - df['infected daily increase'].shift(1)



    # Recovered : daily increase, daily porcentage and daily derived 

    df['recovered daily increase'] = df['recovered'] - df['recovered'].shift(1)

    df['recovered daily increase percentage'] = df['recovered daily increase']  / df['recovered'] 

    df['recovered daily increase derived'] = df['recovered daily increase'] - df['recovered daily increase'].shift(1)



    # hospitalized : daily increase, daily porcentage and daily derived 

    df['hospitalized daily increase'] = df['hospitalized'] - df['hospitalized'].shift(1)



                                                                                                     



    # Other columns    

    df['recovered / infected rate'] = df['recovered'] / df['infected'] 

    df['infected non recovered yet'] = df['infected'] - df['recovered']   - df['death']

    df['death rate'] = df['death'] / df['infected'] 

    

    # convert to integer

    CONVERT_INT_COLUMNS = ['infected daily increase',

        'infected daily increase derived',

       'death daily increase',

       'death daily increase derived', 'recovered daily increase',

       'recovered daily increase derived', 

       'infected non recovered yet', 'intensive care unit',

       'hospitalized', 'hospitalized daily increase']

    for column in CONVERT_INT_COLUMNS :

        df[column] = df[column].fillna(0)

        df[column] = df[column].astype(np.int64)

  

    # order columns

    columnsTitles = ['location', 

                     'infected','infected daily increase'         , 'infected daily increase derived'      , 'infected daily increase percentage', 

                     'death','death daily increase'         , 'death daily increase derived'      , 'death daily increase percentage', 

                     'recovered','recovered daily increase'         , 'recovered daily increase derived'      , 'recovered daily increase percentage', 

                     'death rate', 

                     'recovered / infected rate',    'infected non recovered yet',

                     'intensive care unit',  

                     'hospitalized', 'hospitalized daily increase']

    df = df.reindex(columns=columnsTitles)

    df = df.sort_values(by=['date'], ascending=False)

    df = df.rename(columns = {'CCAA':'Lugar'})



    return df



def get_location(location):

    """Load data for a given location"""

    df = load_location_data(location)

    df = enrich_data(df)

    return df



def get_dimension(dimension, debug = False):

    """ Return a given dimension for all location. 

    We only count those days with > 100 infected, so every location start at the same time"""

    dimension_df = pd.DataFrame()

    LOCATION_LIST=get_location_list()

    for location in LOCATION_LIST:

        if debug: print (location)

        df = pd.DataFrame()

        df = get_location(location)

        df = df.sort_values(by='date')

        df = df.reset_index() # Resets the index, makes factor a column    

        df = df[df["infected"] >= 100] 

        if debug: print (df[dimension])



        dimension_df[location] = df[dimension]

    return dimension_df 
df = get_location("Madrid")

df.head()
get_dimension("infected")

# Functions for : Single dimension report

from IPython.display import display, HTML

import pandas as pd

from matplotlib import pyplot as plt    





def compare_charts_median(Dimension,df): 

    short_df = df.tail(1)

    short_df = short_df.T

    short_df.columns = [Dimension]

    short_df



    mean_y = short_df.median(axis=1)[0]

    x = short_df.index

    y = short_df[Dimension]



    plt.figure(figsize = (10, 5))

    plt.scatter(x, y, c= "red", alpha = 0.5)

    plt.title(Dimension + " by region")

    color = 'blue'

    plt.xticks(rotation=90)

    plt.axhline(mean_y, c = color, alpha = 0.5, lw = 1)

    plt.annotate('Median ' + Dimension+  ' is {}'.format(round(mean_y, 2)),

            xy=(12, mean_y),

            xycoords='data',

            xytext=(50, 50), 

            textcoords='offset points',

            arrowprops=dict(arrowstyle="->", color = "k", alpha = 0.5),

            color = color)

    return



def compare_charts_time(Dimension,df):

    fig = plt.figure(figsize=(8, 6), dpi=80)

    for ca in df.columns:

        plt.plot(df[ca])

    plt.legend(df.columns)

    fig.suptitle('Comparing : '+Dimension+', starting at 100 cases', fontsize=15)

    plt.show()

    return 



def report_single_dimension_comparative(dimension):

    """ Report, show a dataframe, and two charts, 1) comparing all location, 2) median value """

    # Ger Data

    display(HTML(f"<h2>Comparative of : {dimension}<h2>"))

    df = get_dimension(dimension)

    # Compare chart

    display(HTML(f"<h3>Evolution of : {dimension}<h3>"))

    compare_charts_time(dimension,df)



    compare_charts_median(dimension,df)

    display(HTML(f"<h3>Raw data : {dimension}<h3>"))

    with pd.option_context("display.max_rows", 1000):

        display(HTML(df.to_html()))

    # Compare median chart

    display(HTML(f"<h3>Median chart of : {dimension}<h3>"))

    return 
report_single_dimension_comparative('infected')

report_single_dimension_comparative('infected daily increase derived')

report_single_dimension_comparative('infected')



report_single_dimension_comparative('infected daily increase')

report_single_dimension_comparative('death')

report_single_dimension_comparative('death daily increase')

report_single_dimension_comparative('death rate')

report_single_dimension_comparative('recovered')

report_single_dimension_comparative('recovered / infected rate')
# # Functions for : two  dimensions report

import pandas as pd





def get_dimensions_all_locations(attributes, debug = False):

    array = []

    LOCATION_LIST=get_location_list()

    for location in LOCATION_LIST:

        if debug: print (location)

        comunidad = get_location(location).head(1)

        comunidad = comunidad.reset_index() # Resets the index, makes factor a column

        temp_dict = {}

        temp_dict['Lugar'] = location

        for attr in attributes:

            temp_dict[attr] = comunidad[attr].iloc[0]

        array.append(temp_dict)



    return pd.DataFrame.from_records(array)



def print_two_cordinates_CCAA(df):

    fig,ax = plt.subplots()

    fig.set_figheight(8) 

    fig.set_figwidth(8)

    ax.axhline(y=0, color='blue')

    ax.axvline(x=0, color='blue')

    for k,d in df.groupby('Lugar'):

        ax.scatter(d[df.columns[1]], d[df.columns[2]], label=k)



    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)



    ax.set_xlabel(df.columns[1])

    ax.set_ylabel(df.columns[2])

    ax.set_title(df.columns[1]+ ' VS. ' + df.columns[2])

    return plt





def report_two_dimension_comparative(attributes):

    """ Report, for two attributes 

    df with the two dimensions, and a chart comparing it """



    df = get_dimensions_all_locations(attributes)

    display(HTML(f"<h2>Analysis of : {attributes[0]} Vs {attributes[1]}</h2>"))

                

    # Compare 2d chart

    plt = print_two_cordinates_CCAA(df)

    plt.show()

 

              

    # show raw data

    display(HTML(f"<h3>Raw data : {attributes[0]} Vs {attributes[1]}<h3>"))

    df = df.append(df.sum(numeric_only=True), ignore_index=True)

    df.iat[-1, 0] = "Total"



    display(HTML(df.sort_values(by=[df.columns[1],df.columns[2]]).to_html()))

 

    return 
report_two_dimension_comparative(['infected', 'death'] )   
report_two_dimension_comparative(['infected', 'death'] )   
report_two_dimension_comparative(['infected daily increase', 'death daily increase'] ) 
report_two_dimension_comparative(['infected daily increase derived', 'death daily increase derived'] )   
report_two_dimension_comparative(['infected', 'recovered'] ) 
def report_single_location_single_dimension(location,dimension):

    """ Report, for two attributes 

    df with the two dimensions, and a chart comparing it """



    MOVING_AVERAGE_WINDOW = 4

    df_location = get_location(location)

    df_location = df_location.sort_values(by=['date'], ascending=True)



    df = pd.DataFrame()

    df[dimension] = df_location[dimension]

    df_location['Moving Average ' + dimension] = df_location[dimension].rolling(window=4).mean()

    df['Moving Average ' + dimension] = df_location['Moving Average ' + dimension]

    

    display(HTML(f"<h2>Analysis of : {dimension} for COVID-19 in {location}</h2>"))





    fig = plt.figure(figsize=(8, 6), dpi=80)

    plt.plot(df, marker='o') 

    plt.xticks(rotation=90)



    plt.legend(df.columns)

    fig.suptitle( dimension + ' in ' + location, fontsize=20)

    display(HTML(f"<h3>Raw data : {dimension} in {location}<h3>"))



    display(HTML(pd.DataFrame(df).to_html()))

    display(HTML(f"<h3>Analysis of : {dimension} in {location}</h3>"))

    display(HTML(f" with moving average window = {MOVING_AVERAGE_WINDOW}"))

    return 





 

report_single_location_single_dimension('Madrid','death daily increase')
report_single_location_single_dimension('Cataluña','death daily increase')
report_single_location_single_dimension('Madrid','infected daily increase')
report_single_location_single_dimension('Cataluña','infected daily increase')