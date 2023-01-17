# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
donors = pd.read_csv('../input/Donors.csv', low_memory = False)

donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])

donations_donor = donations.merge(donors, on='Donor ID',how='inner' )



projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])

resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)

schools = pd.read_csv('../input/Schools.csv', error_bad_lines = False)

teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines = False)



# creating a new dataframe "donations_donor" by merging "donations" & "donors"



all_data = [donors, donations, donations_donors, projects, resources, schools, teachers]
donors.head()
donations.head()
projects.head()
schools.head()
teachers.head()
resources.head()
def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    ## the two following line may seem complicated but its actually very simple. 

    total = df.isnull().sum().sort_values(ascending = False)

    total = total[total > 0]

    percent = total/len(df)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_percentage(donors)
missing_percentage(donations)
missing_percentage(projects)
missing_percentage(schools)
missing_percentage(teachers)
missing_percentage(resources)
donors.info(memory_usage='deep')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

## -------------------
for df in [donors, donations, projects, resources, schools, teachers]:

    original = df.copy()

    df = reduce_mem_usage(df)



    for col in list(df):

        if df[col].dtype!='O':

            if (df[col]-original[col]).sum()!=0:

                df[col] = original[col]

                print('Bad transformation', col)
donors.info(memory_usage='deep')
# We're going to be calculating memory usage a lot,

# so we'll create a function to save us some time!

def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)



# We’ll write a loop to iterate over each object column, 

# check if the number of unique values is more than 50%, 

# and if so, convert it to the category atype.

def reduce_by_category_type(df):

    converted_obj = pd.DataFrame()

    for col in df.columns:

        num_unique_values = len(df[col].unique())

        num_total_values = len(df[col])

        if num_unique_values / num_total_values < 0.5 and df[col].dtype == 'object':

            converted_obj.loc[:,col] = df[col].astype('category')

        else:

            converted_obj.loc[:,col] = df[col]

    return converted_obj
donors = reduce_by_category_type(donors)
donors.info(memory_usage='deep')
donations = reduce_by_category_type(donations)

donations_donor = reduce_by_category_type(donations_donor)

projects = reduce_by_category_type(projects)

resources = reduce_by_category_type(resources)

schools = reduce_by_category_type(schools)
import plotly.graph_objects as go



temp = donations_donor.groupby(['Donor State'])['Donation Amount'].sum().sort_values(ascending = False).reset_index()

states_dict = {

    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 

    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 

    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 

    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 

    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 

    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 

    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 

    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',

    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 

    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 

    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 

    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 

    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 

    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 

    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}



temp['code'] = temp['Donor State'].apply(lambda x: states_dict[x])



fig = go.Figure(data=go.Choropleth(

    locations=temp['code'], # Spatial coordinates

    z = temp['Donation Amount'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Greens',

    colorbar_title = "Millions(USD)",

))



fig.update_layout(

#     title_text = 'States with Most Donations',

    geo_scope='usa', # limite map scope to USA

    geo_showlakes = True,

    geo_lakecolor = 'rgb(0, 200, 255)',

#     template="plotly_dark",

)



fig.show()
from plotly.subplots import make_subplots



temp = donations_donor.groupby(['Donor State'])['Donation Amount'].sum().sort_values(ascending = False)



fig = make_subplots( 

    rows=1,

    cols=2,

    # shared_yaxes=True,

    #vertical_spacing=11,

#     specs=[[{"colspan": 2}, None], 

#            [{}, {}]], ## distribution of chart spacing

    subplot_titles = ('Most',"Least"))



x_values = temp.head(10).sort_values(ascending= True).values

fig.add_trace(

    go.Bar(x=x_values,

           y=temp.head(10).sort_values(ascending= True).index,

           orientation = 'h',

           marker=dict(color = (x_values/x_values.sum()),colorscale = 'Greens')



          ),

    row = 1, 

    col = 1,

)



x_values = temp.tail(10).sort_values(ascending= False).values

fig.add_trace(

    go.Bar(x=x_values,

           y=temp.tail(10).sort_values(ascending = False).index,

           orientation = 'h',

           marker=dict(color = (x_values/x_values.sum()),colorscale = 'Greens')

           ),

    row = 1, 

    col = 2,

)

fig['layout']['xaxis1'].update(title = 'Amount($)')

fig['layout']['yaxis1'].update(title = 'States', showgrid = True)



fig['layout']['xaxis2'].update(title = 'Amount($)')

fig['layout']['yaxis2'].update(title = 'States', showgrid = True)







# fig['layout']['margin'].update()





fig.update_layout(height = 600, 

                  showlegend = False, 

                  title_text = 'Most and Least Donated States', 

#                   template="plotly_dark",

                 );

#fig.layout.update(title = 'testing')

fig.show()



temp = donors[donors['Donor State'] != 'other']

temp = temp['Donor State'].value_counts()



fig = make_subplots( 

    rows=1,

    cols=2,

    # shared_yaxes=True,

    #vertical_spacing=11,

#     specs=[[{"colspan": 2}, None], 

#            [{}, {}]], ## distribution of chart spacing

    subplot_titles = ('Most',"Least"))



x_values = temp.head(10).sort_values(ascending= True).values

fig.add_trace(

    go.Bar(x=x_values,

           y=temp.head(10).sort_values(ascending= True).index,

           orientation = 'h',

           marker=dict(color = (x_values/x_values.sum()),colorscale = 'Greens')



          ),

    row = 1, 

    col = 1,

)



x_values = temp.tail(10).sort_values(ascending= False).values

fig.add_trace(

    go.Bar(x=x_values,

           y=temp.tail(10).sort_values(ascending = False).index,

           orientation = 'h',

           marker=dict(color = (x_values/x_values.sum()),colorscale = 'Greens')

           ),

    row = 1, 

    col = 2,

)

fig['layout']['xaxis1'].update(title = 'Amount($)')

fig['layout']['yaxis1'].update(title = 'States', showgrid = True)



fig['layout']['xaxis2'].update(title = 'Amount($)')

fig['layout']['yaxis2'].update(title = 'States', showgrid = True)







# fig['layout']['margin'].update()





fig.update_layout(height = 600, 

                  showlegend = False, 

                  title_text = 'Most and Least donor States', 

#                   template="plotly_dark",

                 );

#fig.layout.update(title = 'testing')

fig.show()





temp = donations_donor.groupby(['Donor State'])['Donation Amount'].sum().sort_values(ascending = False).reset_index()

temp['code'] = temp['Donor State'].apply(lambda x: states_dict[x])



# Initialize figure with subplots

fig = make_subplots(

    rows=2, cols=2,

    column_widths=[0.6, 0.4],

    row_heights=[0.5, 0.5],

    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar"}],

           [            None                    , {"type": "bar"}]])



# Add Chotopleth

fig.add_trace(

    go.Choropleth(locations=temp['code'], # Spatial coordinates

                  z = temp['Donation Amount'].astype(float), # Data to be color-coded

                  locationmode = 'USA-states', # set of locations match entries in `locations`

                  colorscale = 'greens',

                  showscale = False,

                  colorbar_title = "Millions USD"),

    row=1, col=1

)





# Add locations bar chart

fig.add_trace(

    go.Bar(x=donations_donor['Donor State'].value_counts().head(10).sort_values(ascending = False).index,

           y=donations_donor['Donor State'].value_counts().head(10).sort_values(ascending = False).values,

           marker=dict(color=[i for i in range(10,0,-1)], colorscale = 'Greens'), 

           showlegend=False),

    row=1, col=2

)



# Add locations bar chart

fig.add_trace(

    go.Bar(x=donations_donor['Donor State'].value_counts().tail(10).sort_values(ascending = True).index,

           y=donations_donor['Donor State'].value_counts().tail(10).sort_values(ascending = True).values,

           marker=dict(color=[i for i in range(10)], colorscale = 'Greens'), 

           showlegend=False),

    row=2, col=2

)



# # Update geo subplot properties

# fig.update_geos(

#     projection_type="orthographic",

#     landcolor="white",

#     oceancolor="MidnightBlue",

#     showocean=True,

#     lakecolor="LightBlue"

# )



# Rotate x-axis labels

fig.update_xaxes(tickangle=45)





# Set theme, margin, and annotation in layout

fig.update_layout(

#     title_text = 'Most and Least Donations and Donors', 

#     template="plotly_dark",

    geo_scope='usa', # limite map scope to USA

    geo_showlakes = True,

    geo_lakecolor = 'rgb(0, 200, 255)',

    margin=dict(r=100, t=25, b=40, l=60),

    annotations=[

        go.layout.Annotation(

            text="Source: Rumi",

            showarrow=False,

            xref="paper",

            yref="paper",

            x=0,

            y=0)

    ]

)



fig.show()
# ## getting Mean donations

# mean_donation_amount = donations_donor.groupby(['Donor State'])['Donation Amount'].mean().reset_index()

# mean_donation_amount.columns = ['Donor State', 'Mean Donation Amount']



# ## Getting total Donors

# total_donors = pd.DataFrame(donors['Donor State'].value_counts()).reset_index()

# total_donors.columns = ['Donor State', 'total_donors']



# ## getting mean donation and total donors in one dataframe called "states"

# states = pd.merge(mean_donation_amount, total_donors, on = 'Donor State', how = 'inner')

# ## getting total donations

# total_donations = donations_donor.groupby(['Donor State'])['Donation Amount'].sum().reset_index()

# ## merging total donations with states dataframe

# states = states.merge(total_donations, on = 'Donor State', how = 'inner')









# ## creating total school column. 

# temp_school = pd.DataFrame(schools['School State'].value_counts()).reset_index()

# temp_school = temp_school.rename(columns={'index':'state',

#                     'School State':'total_school'},

#            )



# # merging total school column with states df

# states = states.merge(temp_school, how = 'inner', left_on='Donor State', right_on='state')

# states.drop('state', axis=1, inplace=True)



# # merging total 

# states = states.merge(temp, on="Donor State", how='inner')

# # states.drop(['state',"total_school"] ,axis=1, inplace=True)



# ## creating a text column for visualization chart

# # making them as string type

# for col in states.columns:

#     states[col] = states[col].astype(str)

# # writing the text column

# states['text'] = states['Donor State'] + '<br>' + 'Mean Donations: ' + states['Mean Donation Amount'] + '<br>' + 'Total Donors: ' + states['total_donors'] + '<br>' + 'Total Donations: ' + states['Donation Amount']

# # this is for the size part of the chart

# states['Mean Donation Amount'] = states['Mean Donation Amount'].astype(float)

# #states['text'] = states.apply(lambda x: (states['Donor State']+ '<br>' + "Mean Donations:" + states['Mean Donation Amount'] + '<br>' + "Total Donors:" + states['total_donors'] + "<br" + "Total Donations:" + states['Donation Amount']))

# #states.text = states.text.astype(str)



# ## Doing some rounding up

# states['Donation Amount'] = states['Donation Amount'].apply(lambda x: float(x)).apply(lambda x : "%.2f"%x)

# states['average_projects'] = states['average_projects'].apply(lambda x: float(x)).apply(lambda x : "%.2f"%x).apply(lambda x: float(x))

# states['total_donors'] = states['total_donors'].apply(lambda x: float(x))

# states['total_schools'] = states['total_schools'].apply(lambda x: float(x)).apply(lambda x : "%.2f"%x).apply(lambda x: float(x))

# states['total_projects'] = states['total_projects'].astype(float)

# states = states.astype({"Mean Donation Amount": float, "Donation Amount": float})



# ## Get the states for

# states['code'] = states['Donor State'].apply(lambda x: states_dict[x])
# # difine our data for plotting

# data = [ dict(

#         type='choropleth',

#         colorscale = "Greens",

#         autocolorscale = False,

#         locations = states['code'], # location (states)

#         z = states['Donation Amount'],

#         locationmode = 'USA-states', # let's define the location mode to USA_states

#         text = states['text'],

#         marker = dict(

#             line = dict (

#                 color = 'rgb(255,255,255)',

#                 width = 2

#             ) ),

#         colorbar = dict(

#             title = "Donation<br>Amount($)")

#         ) ]



# layout = dict(

#         title = 'States with Most Donations',

#         geo = dict(

#             scope='usa',

#             projection=dict( type='albers usa' ),

#             showlakes = True,

#             lakecolor = 'rgb(0, 200, 255)'),

#              )









    

# fig = go.Figure(data=data, layout=layout)



# fig.update_layout(template = 'plotly_dark')

# fig.show()
data = go.Bar(

    x = donations_donor.groupby(['Donor State'])['Donation Amount'].max().sort_values(ascending = False).head(20).index,

    y = donations_donor.groupby(['Donor State'])['Donation Amount'].max().sort_values(ascending = False).head(20).values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'States'

fig.layout.yaxis.title = 'Donation Amount'

# fig.update_layout(template = 'plotly_dark')

fig.show()
## Getting lats and lons of the city

city_code = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')

city_code.drop(columns = ['pop'], inplace = True)

city_code.name = city_code.name.apply(lambda x: x.strip())

df = donations_donor.groupby(['Donor City'])['Donation Amount'].max().sort_values(ascending = False).head(50).reset_index()

df = df.merge(city_code, how ='inner', left_on = 'Donor City', right_on='name')

df.drop_duplicates(inplace = True)

df.drop(columns = ['name'], inplace = True)



df['text'] = df['Donor City'] + '<br>Donation Amount: ' + '$'+(df['Donation Amount']).astype(str)

# limits = [(0,2),(3,10),(11,20),(21,50),(50,3000)]

# colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

fig = go.Figure()



fig.add_trace(go.Scattergeo(

    locationmode = 'USA-states',

    

    lon = df['lon'],

    lat = df['lat'],

    text = df['text'],

    marker = dict(

        size = df['Donation Amount']/20,

#         color = (df['Donation Amount']/df['Donation Amount'].sum())*100,

        line_color='rgb(40,40,40)',

        line_width=0.5,

        sizemode = 'area'

    ),))

#     name = '{0} - {1}'.format(lim[0],lim[1])))



fig.update_layout(

    

        title_text = 'Some of the highest single donated cities',

#         showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

            showlakes = True,

            lakecolor = 'rgb(0, 200, 255)'),

        )

fig.show()
# # Brookln is techincally part of New York City therefore replacing it with New York. 

# donations_donor['Donor City'].replace('Brooklyn', 'New York', inplace = True)



# data = go.Bar(

#     x = donations_donor.groupby(['Donor City'])['Donation Amount'].max().sort_values(ascending = False).head(20).index,

#     y = donations_donor.groupby(['Donor City'])['Donation Amount'].max().sort_values(ascending = False).head(20).values

# )

# fig = go.Figure(data = data)

# fig.layout.xaxis.title = 'City'

# fig.layout.yaxis.title = 'Donation Amount'

# fig.update_layout(template = 'plotly_dark')

# fig.show()
## Setting up the dataframe

df = donations_donor.groupby(['Donor City'])['Donation Amount'].sum().sort_values(ascending = False).reset_index()

df = df.merge(city_code, how = 'inner', left_on = 'Donor City', right_on ='name').drop(columns=['name'])



## Setting up viz



df['text'] = df['Donor City'] + '<br>Donation Amount ' + (round(df['Donation Amount'])).astype(str)

limits = [(0,10),(11,20),(21, 30),(31,40),(41,50)]

colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

cities = []

scale = 5000



fig = go.Figure()

for i in range(len(limits)):

    lim = limits[i]

    df_sub = df[lim[0]:lim[1]]

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lon'],

        lat = df_sub['lat'],

        text = df_sub['text'],

        marker = dict(

            size = df_sub['Donation Amount']/scale,

            color = colors[i],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(lim[0],lim[1])))



fig.update_layout(title_text = 'Most Donated Cities<br>(Click legend to toggle traces)',

#                   template = 'plotly_dark',

                  showlegend = True,

                  geo = dict(

                      scope = 'usa',

                      landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()
# import plotly.express as px

# temp_df = donations_donor.groupby(['Donor City'])['Donation Amount'].sum().sort_values(ascending = False).reset_index()

# temp_df = temp_df.merge(city_code, how = 'inner', left_on = 'Donor City', right_on ='name').drop(columns=['name'])

# fig = px.scatter_mapbox(temp_df, lat="lat", lon="lon", hover_name="Donor City", hover_data=["Donor City", "Donation Amount"],

#                         color_discrete_sequence=["fuchsia"], zoom=3, height=400)

# fig.update_layout(mapbox_style="open-street-map", ## other mabbox_styles are 'dark', 'white-bg'

#                   mapbox_layers=[

#         {

#             "below": 'traces',

#             "sourcetype": "raster",

#             "source": [

#                 "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

#             ]

#         }

#       ])

# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# fig.show()
temp_df = donations_donor['Donor City'].value_counts()

most = go.Bar(

    y=temp_df.head(10).sort_values(ascending= True).index,

    x=temp_df.head(10).sort_values(ascending= True).values,

    orientation = 'h',

)



least = go.Bar(

    y=temp_df.tail(10).sort_values(ascending = False).index,

    x=temp_df.tail(10).sort_values(ascending = False).values,

    orientation = 'h',

)



fig = make_subplots(rows=1, # row #'s

                          cols=2, # column #'s

                          #specs=[[{'colspan': 2}, None], 

                               # [{}, {}]], ## distribution of chart spacing

                          #shared_yaxes=True, 

                          subplot_titles = ['Most Donor Cities',

                                            "Least Donor Cities", 

                                            #'Countries with least loans', 

                                           ]);

#fig.append_trace(data, 1,1);##fig.append_trace(data1,raw #,col #);

fig.append_trace(most,1,1);

fig.append_trace(least,1,2);



fig['layout']['xaxis1'].update(title = 'Amount($)')

fig['layout']['yaxis1'].update(title = 'Cities', showgrid = True)



fig['layout']['xaxis2'].update(title = 'Amount($)')

fig['layout']['yaxis2'].update(title = 'Cities', showgrid = True)







#fig['layout']['xaxis3'].update(title = 'Count',

                               #type = 'log'

                            #  )



fig['layout'].update(height = 600, 

                     showlegend = False, 

#                      title = 'Most and Least Cities in terms of Donations', 

#                      template = 'plotly_dark'

                    );

#fig.layout.update(title = 'testing')

fig.show()





temp_1 = schools['School State'].value_counts().sort_values(ascending = True).head(20)

temp_school = go.Bar(

    y=temp_1.index,

    x=temp_1.values,

    orientation = 'h',

)





temp = projects.merge(schools[['School ID','School State']], how = 'inner', on='School ID')['School State'].value_counts().sort_values(ascending = True).head(20)

temp_project = go.Bar(

    y=temp.index,

    x=temp.values,

    orientation = 'h',

)





fig = make_subplots(rows=1, # row #'s

                          cols=2, # column #'s

                          #specs=[[{'colspan': 2}, None], 

                               # [{}, {}]], ## distribution of chart spacing

#                           shared_yaxes=True,

                          vertical_spacing=0.009,

                          subplot_titles = ['Schools',

                                            "Projects", 

                                            #'Countries with least loans', 

                                           ]);

#fig.append_trace(data, 1,1);##fig.append_trace(data1,raw #,col #);

fig.append_trace(temp_school,1,1);

fig.append_trace(temp_project,1,2);

fig['layout']['yaxis1'].update(title = 'States', 

                               showgrid = True

                              )

fig['layout']['yaxis2'].update(title = 'States', showgrid = True)



fig['layout']['xaxis1'].update(title = '# of Schools')

fig.layout.xaxis2.update(title = '# of Projects')





fig['layout'].update(height = 700, 

                     showlegend = False, 

                     title = 'Schools VS Projects', 

#                      template = 'plotly_dark'

                    );

#fig.layout.update(title = 'testing')

fig.show()
## getting total_schools per state

temp = pd.DataFrame(schools['School State'].value_counts()).reset_index()

temp.rename(columns = {'index':'state','School State': 'total_schools'}, inplace = True)



## getting total projects per state

temp2 = projects.merge(schools[['School ID','School State']], how = 'inner', on='School ID')

temp2 = pd.DataFrame(temp2['School State'].value_counts()).reset_index()

temp2.rename(columns = {'index':'state','School State': 'total_projects'}, 

             inplace = True)



## merging 

temp = temp.merge(temp2, on = 'state', how = 'inner')



## getting average projects per school per state. 

temp['average_projects'] = temp['total_projects']/temp['total_schools']





trace1 = go.Bar(

    x=temp.sort_values(by = 'average_projects', ascending = False).head(20).state.tolist(),

    y=temp.sort_values(by = 'average_projects', ascending = False).head(20).average_projects.tolist(),

    name='Projects'

)





data = [trace1]





fig = go.Figure(data=data)

fig.layout.update(title = 'Average Projects Per School in Each States', 

#                   template = 'plotly_dark'

                 )

fig.layout.xaxis.title = 'States'

fig.layout.yaxis.title = 'Project Count'

fig.show()
labels = donors['Donor Is Teacher'].value_counts().index

values = donors['Donor Is Teacher'].value_counts().values



colors = ['gold', 'lightgreen']

# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])



fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

# fig.update_layout(template = 'plotly_dark')

fig.show()
labels = teachers['Teacher Prefix'].value_counts().sort_values(ascending = True).index

values = teachers['Teacher Prefix'].value_counts().sort_values(ascending = True).values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='black', width=2)))



# fig.update_layout(template = 'plotly_dark')

fig.show()
labels = donations['Donation Included Optional Donation'].value_counts().index

values = donations['Donation Included Optional Donation'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='black', width=3)))



# fig.update_layout(template = 'plotly_dark')

fig.show()
labels = projects['Project Type'].value_counts().index

values = projects['Project Type'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='black', width=2)))



# fig.update_layout(template = 'plotly_dark')

fig.show()


temp =  pd.DataFrame(projects['Project Subject Category Tree'].dropna().str.split(',').tolist())

temp = temp.applymap(lambda x: x.strip() if x else x).stack().value_counts()



data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Subject Category Type'

fig.layout.yaxis.title = 'Project count'

# fig.update_layout(template = 'plotly_dark')

fig.show()

%timeit
temp =  pd.DataFrame(projects['Project Subject Subcategory Tree'].dropna().str.split(',').tolist())

temp = temp.applymap(lambda x: x.strip() if x else x).stack().value_counts()



data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Subject Subcategory Type'

fig.layout.yaxis.title = 'Project Count'

# fig.update_layout(template = 'plotly_dark')

fig.show()


temp = projects['Project Resource Category'].value_counts().head(15)



data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Project Resource Category'

fig.layout.yaxis.title = 'Project Count'

# fig.update_layout(template = 'plotly_dark')

fig.show()
temp = projects['Project Grade Level Category'].value_counts()

data = go.Bar(

    x = temp.index,

    y = temp.values

)

fig = go.Figure(data = data)

fig.layout.xaxis.title = 'Grade Level'

fig.layout.yaxis.title = 'Project Count'

# fig.update_layout(template = 'plotly_dark')

fig.show()
labels = projects['Project Current Status'].value_counts().index

values = projects['Project Current Status'].value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='black', width=2)))



# fig.update_layout(template = 'plotly_dark')

fig.show()
# expired_projects = projects[projects['Project Current Status'] == 'Expired']

# expired_projects.shape



# expired_projects.head()



# expired_projects['Project Type'].value_counts()



# pd.DataFrame({'labels':temp.index,"values":temp.values}).iplot(kind = 'pie', labels = "labels", values = "values")



# expired_projects['Project Cost'].describe()



# expired_projects['Project Cost'].iplot(kind = 'box', )



# expired_projects[expired_projects['Project Cost'] >= 1300].sort_values(by = ['Project Cost'], ascending = False).head()



# round(donations['Donation Amount'].describe(),2)



# projects.head()



# projects.head()



# expired_projects = projects[projects['Project Current Status'] == 'Expired']

# print ('There are ' + str(expired_projects.shape[0]) + ' projects expired before getting funded.')



# That is a huge number of projects that are just not getting funded. I am interested in finding out why these projects were not funded. 



# expired_projects.head()



# round(expired_projects['Project Cost'].describe(),2)



# round(projects['Project Cost'].describe(),2).iplot(kind = 'box')
for name in dir():

    if not name.startswith('_'):

        del globals()[name]