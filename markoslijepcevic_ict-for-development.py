import pandas as pd

import numpy as np

import seaborn as sns

import operator

import matplotlib.pyplot as plt



# Set up 

%matplotlib inline 



pd.options.display.max_columns = None

pd.set_option('display.max_colwidth', -1)

sns.set(style="whitegrid") # sns.set(style="darkgrid")    

palette = sns.color_palette("YlGnBu", 20)

plt.figure(figsize=(20, 10))
# Read the data file

mcr=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')



platform_col = ['Q11', 'Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7'

                ,'Q13_Part_8','Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']



platform_data = mcr[platform_col]

platform_data.columns = platform_data.iloc[0]

platform_data = platform_data.drop(platform_data.index[0])



list_of_platforms = ['Udacity', 'Coursera', 'edX', 'DataCamp', 'DataQuest', 'Kaggle Courses (i.e. Kaggle Learn)',

                    'Fast.ai', 'Udemy', 'LinkedIn Learning', 'University Courses (resulting in a university degree)',

                    'None', 'Other']



col_to_rename = list(platform_data.columns)



for i in range(len(list_of_platforms)):

    str_val = str(list_of_platforms[i])

    for j in range(1,len(col_to_rename)):

        col_name = str(col_to_rename[j])

        if col_name.endswith(str_val):

            platform_data.rename(index=str, columns={col_name:str_val}, inplace=True)

        else:

            continue

# We will doo a copy of this dataframe becaus the same will be used later

platform_data_grouped = platform_data.copy()



c = ['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?',

    'Kaggle Courses (i.e. Kaggle Learn)'] 

platform_data_grouped = platform_data_grouped[c]



platform_data_grouped = platform_data_grouped.groupby(

    ['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?'

    ,'Kaggle Courses (i.e. Kaggle Learn)']

    ).size().unstack()



platform_data_grouped = platform_data_grouped['Kaggle Courses (i.e. Kaggle Learn)']

platform_data_grouped = platform_data_grouped.dropna().sort_values()
ax = platform_data_grouped.plot(kind='bar', width = 0.75, color=(0.2, 0.4, 0.6, 1), rot=10, )

ax.set_xlabel("Amount of money spent on machine learning and/or cloud computing products at work")

ax.set_ylabel("Number of participants")

ax.legend(['Kaggle Courses (i.e. Kaggle Learn)'])

mcr_group = platform_data.copy()

# Again, we will simplify the column name

mcr_group = mcr_group.rename(index=str, columns={"Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?" : "Amount of money spent"})

# Filter only the data with end users

mcr_group = mcr_group[(mcr_group['Amount of money spent'] == '$0 (USD)') | (mcr_group['Amount of money spent'] == '> $100,000 ($USD)')]



dc=mcr_group.groupby(['DataCamp','Amount of money spent']).size().unstack()

dq=mcr_group.groupby(['DataQuest','Amount of money spent']).size().unstack()

kgl=mcr_group.groupby(['Kaggle Courses (i.e. Kaggle Learn)','Amount of money spent']).size().unstack()



col = ['$0 (USD)', '> $100,000 ($USD)']

dc, dq, kgl = dc[col], dq[col], kgl[col]



platform_grouped = dc

platform_grouped = platform_grouped.append(dq)

platform_grouped = platform_grouped.append(kgl)
ax = platform_grouped.plot(kind='bar', width = 0.75, title = 'Competition Platforms Comparison', cmap='YlGnBu', rot=15)

ax.set_xlabel("Platform name")

ax.set_ylabel("Number of participants")

sns.set(font_scale=1,rc={'figure.figsize':(20,10)})
import geopandas as gpd

import json

from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool

from bokeh.palettes import brewer
def geodata(shapefile):

    """

    Prepares the geospatial data file for ploting the world map



    Parameters

    ----------

    shapefile : path to the shape file



    Returns

    -------

    gdf : geopandas.geodataframe.GeoDataFrame

    """

    #Read the shapefile using Geopandas

    gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

    #Rename columns.

    gdf.columns = ['country', 'country_code', 'geometry']

    gdf.head()

    # Remove Antarctica because it is unnecessary

    gdf = gdf.drop(gdf.index[159])



    return gdf
def prepare_data(file_path, target_col, year_var):

    """

    Prepares the dataset for being ploted on a world map.

    This function is customized for the World Data excel file with the structure they provide.

    It has a special condition for ploting GDP per capita, PPP (current international $) data because 

    it needs to be an integer.



    Parameters

    ----------

    file_path : path to the data file

    target_col : Name of the target variable which you want to plot.

    year_var : Define the year for being prepared

    

    Returns

    -------

    data : pandas.core.frame.DataFrame

    """

    df = pd.read_excel(file_path)

    # Clean useless data

    yr_var = r'{} [YR{}]'.format(year_var, year_var)

    col = [yr_var]

    df[col] = df[col].replace({'..':0})

    # Take only needed columns

    gdp_col = ['Country Name', 'Country Code', 'Series', yr_var]

    data = df[gdp_col]

    # Rename columns

    data.rename(index=str, columns={"Country Name": "Country_Name", "Country Code": "Country_Code", yr_var: year_var}, inplace=True)  

    data = data[data['Series'] == target_col] # for the desired variable.

    

    if(target_col == "GDP per capita, PPP (current international $)"):

        data[year_var] = data[year_var].astype(int)



    else:

        pass

    

    return data
def plot(file_path, shapefile, target_col, year_var):

    """

    This function calls geodata(shapefile) and prepare_data(file_path, target_col)



    Parameters

    ----------

    file_path : path to the data file

    shapefile : path to the shape file

    target_col : Name of the target variable which you want to plot.

    year_var : Define the year for ploting

    

    Returns

    -------

    A plot with the world map data for the target variable from the World Bank data set

    """

    data = prepare_data(file_path, target_col, year_var)

    gdf = geodata(shapefile)

    

    #Merge dataframes gdf and data.

    merged = gdf.merge(data, left_on = 'country_code', right_on = 'Country_Code', how='left')

   

    #Read data to json.

    json_data = json.loads(merged.to_json())

    #Convert to String like object.

    json_data = json.dumps(json_data)

    #Input GeoJSON source that contains features for plotting.

    geosource = GeoJSONDataSource(geojson = json_data)

    #Define a sequential multi-hue color palette.

    palette = brewer['YlGnBu'][8]

    #Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

    min_val = data[year_var].min()

    max_val = data[year_var].max()

    color_mapper = LinearColorMapper(palette = palette[::-1], low = min_val, high = max_val)

    #Add the hover tool feature

    hover_tool = HoverTool(tooltips = [ ('Country','@country'),(target_col, r'@{}'.format(year_var))])

    #Create color bar. 

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,

        border_line_color=None,location = (0,0), orientation = 'horizontal')

    #Create a figure object.

    fig = figure(title = target_col, plot_height = 600 , plot_width = 950, toolbar_location = None, tools = [hover_tool])

    fig.xgrid.grid_line_color = None

    fig.ygrid.grid_line_color = None

    #Add patch renderer to figure. 

    fig.patches('xs','ys', source = geosource,fill_color = {'field' :year_var, 'transform' : color_mapper},

              line_color = 'black', line_width = 0.25, fill_alpha = 1)

    #Specify figure layout.

    fig.add_layout(color_bar, 'below')

    #Display figure inline in Jupyter Notebook.

    output_notebook()

    #Display figure.

    show(fig)
plot("../input/world-data/WorldData.xlsx", "../input/shapedata/country_shapes.shp", 'GDP per capita, PPP (current international $)' , '2017')
# We are going to use here a new data set, from the official UN website

un = pd.read_csv('../input/undataict/UNdata_ICT.csv')

un_ict = un.copy()

# By analizing the data we came accross some useless data, and found the need to rename some columns for better estetics  

un_ict.rename(index=str, columns={"Country or Area": "Country Name"}, inplace=True)

un_ict = un_ict[un_ict['Country Name'] != 'Middle East & North Africa (excluding high income)']

un_ict = un_ict[un_ict['Country Name'] != 'Middle East & North Africa (IDA & IBRD)']

un_ict = un_ict.sort_values(by=['Value'])

ax = sns.barplot(x='Value', y='Country Name', data=un_ict, palette=np.array(palette[:])).set_title("Investment in ICT")

#plt.xlabel("Value/$")

#plt.ylabel("Country")

plt.ticklabel_format(style='plain', axis='x')

sns.set(font_scale=1,rc={'figure.figsize':(20,10)})

plt.xticks(size=16)

plt.yticks(size=16)

plt.xlabel('Value/$', fontsize=16)

plt.ylabel('Country', fontsize=16)
# Get copies of original data to aply changes

mcr_cpy = mcr.copy()

un_mcr = un.copy()



un_mcr.rename(index=str, columns={"Country or Area": "Country_Name"}, inplace=True)

un_mcr_data = mcr.loc[mcr_cpy['Q3'].isin(un_mcr.Country_Name.unique())]

un_mcr_data=un_mcr_data.groupby(['Q3','Q11']).size().unstack()

col = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999', '$10,000-$99,999',

       '> $100,000 ($USD)']

un_mcr_data = un_mcr_data[col]

ax = un_mcr_data.plot(kind='bar', width = 0.75, title = 'Money spent on ML and CCP', cmap='YlGnBu', rot=0,)

ax.set_xlabel("Country")

ax.set_ylabel("Number of participants")

sns.set(font_scale=1,rc={'figure.figsize':(20,10)})

plt.show()
plot("../input/timss-data/TIMSS.xlsx", "../input/shapedata/country_shapes.shp", 'TIMSS: Mean performance on the mathematics scale for eighth grade students, total', '2015')
plot("../input/timss-data/TIMSS.xlsx", "../input/shapedata/country_shapes.shp",'TIMSS: Mean performance on the science scale for eighth grade students, total', '2015')
plot("../input/world-data/WorldData.xlsx", "../input/shapedata/country_shapes.shp",'Internet users (per 100 people)' , '2017')