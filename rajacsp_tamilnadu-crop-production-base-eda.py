# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#ignore warnings

import warnings

warnings.filterwarnings('ignore')
FILEPATH = '/kaggle/input/tamilnadu-cropproduction/Tamilnadu agriculture yield data.csv'
df = pd.read_csv(FILEPATH)
df.sample()
df.describe()
df.info()
list(df.columns)
df.isnull().any().any()
df.isnull().any()
import missingno as mino
mino.matrix(df)
mino.dendrogram(df)
mino.bar(df)
df.sample(3)
df.reset_index(inplace=True)
df.sample(3)
df['State_Name'].unique()
# We can remove the state as there is no use of it.



df = df.drop(['State_Name'], axis = 1)
df.sample(3)
df['District_Name'] = df['District_Name'].apply(lambda x: x.title())
df.dropna(how='any', inplace=True)
# Google Translator



!pip install googletrans
# clean up season



from googletrans import Translator



translator = Translator()
def convert_me(msg):

    translation = translator.translate(msg, dest='en')

    return(translation.text)
df['Season'].unique()
convert_me('Kharif')
convert_me('Rabi')
season_map = {

    'Kharif' : 'Autumn',

    'Rabi' : 'Spring',

    'Whole Year' : 'Whole Year'

}



def convert_season(season):

    

    return season_map[season]
df['Season'] = df['Season'].apply(convert_season)
df.sample(5)
df.sample(3)
df['Production'].unique()
import plotly.express as px

import plotly.graph_objects as go
district_df = df[['District_Name', 'Production']]
district_df.sample(2)
district_df = pd.DataFrame(district_df['Production'].value_counts().head(10)).reset_index()
district_df.sample(3)
state_fig = go.Figure(data=[go.Pie(labels=district_df['index'],

                             values=district_df['Production'],

                             hole=.7,

                             title = 'Count by District',

                             marker_colors = px.colors.sequential.Blues_r,

                            )

                     ])

state_fig.update_layout(title = '% by District')

state_fig.show()
df.sample(2)
district_production_group = df.groupby("District_Name")["Production"].sum().sort_index(ascending=True)
dist_prod_df = pd.DataFrame({'District_Name': district_production_group.index,

                        'Production': district_production_group.values})



# dist_prod_df
import plotly.express as px



fig = px.bar(dist_prod_df, x="Production", y="District_Name", orientation='h', color = 'Production')

fig.show()
# Unique crop



df['Crop'].unique()
# Which district produce more Brinjal?



def show_production_by_crop_and_district(df, crop):

    

    df = df[df['Crop'] == crop]

    

    current_group = df.groupby("District_Name")["Production"].sum().sort_index(ascending=True)

    

    current_df = pd.DataFrame({'District_Name': current_group.index,

                        'Production': current_group.values})

    

    fig = px.bar(current_df, x="Production", y="District_Name", orientation='h', color = 'Production')

    fig.show()
show_production_by_crop_and_district(df, 'Grapes')
show_production_by_crop_and_district(df, 'Sunflower')
show_production_by_crop_and_district(df, 'Coconut')
show_production_by_crop_and_district(df, 'Tapioca')
df.sample(4)
def show_production_by_crop_and_dist(df, district):

    

    df = df[df['District_Name'] == district]

    

#     return df

    

    current_group = df.groupby("Crop_Year")["Production"].sum().sort_index(ascending=True)

    

    current_df = pd.DataFrame({'Crop_Year': current_group.index,

                        'Production': current_group.values})

    

    fig = px.bar(current_df, x="Production", y="Crop_Year", orientation='h', color = 'Production')

    fig.show()
show_production_by_crop_and_dist(df, 'Madurai')
import matplotlib.pyplot as plt



def show_donut_plot(col, max_cols = 10):

    

    rating_data = df.groupby(col)[['index']].count().head(max_cols)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['index']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot by ' +str(col), loc='center')

    

    plt.show()
show_donut_plot('District_Name')
show_donut_plot('Crop_Year', 8)
import squarify



def show_treemap(col, max_labels = 10):

    

    df_type_series = df.groupby(col)['index'].count().sort_values(ascending = False).head(20)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:max_labels],  # show labels for only first 10 items

                  alpha=.2 )

    

    plt.title('TreeMap: Count by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap('Crop_Year')
show_treemap('District_Name')
df.sample(2)
fig = px.sunburst(df, path=['District_Name', 'Crop_Year'], values='Production',

                  color='Production', hover_data=['Production'])

fig.show()
fig = px.sunburst(df, path=['Crop_Year', 'Season'], values='Production',

                  color='Production', hover_data=['Production'])

fig.show()
df['Crop_Year'].max()
last_4_years_df = df[df['Crop_Year'] > 2009]
fig = px.sunburst(last_4_years_df, path=['Crop_Year', 'Season'], values='Area',

                  color='Area', hover_data=['Area'])

fig.show()
fig = px.sunburst(last_4_years_df, path=['Crop_Year', 'Crop'], values='Area',

                  color='Area', hover_data=['Area'])

fig.show()
df.sample(2)
theni_df = df[df['District_Name'] == 'Theni']

theni_df = theni_df[theni_df['Crop_Year'] > 2009]
fig = px.sunburst(theni_df, path=['Crop_Year', 'Crop'], values='Area',

                  color='Area', hover_data=['Area'])

fig.show()
def show_crop_sunburtst_by_district(district = 'Theni'):

    

    current_df = df[df['District_Name'] == district]

    current_df = current_df[current_df['Crop_Year'] > 2009]

    

    fig = px.sunburst(current_df, path=['Crop_Year', 'Crop'], values='Area',

                  color='Area', hover_data=['Area'])

    fig.show()
show_crop_sunburtst_by_district('Madurai')
show_crop_sunburtst_by_district('Kanniyakumari')