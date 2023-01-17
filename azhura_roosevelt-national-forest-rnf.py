#Load Functions & Load DataSet

import os # File 

import pandas_profiling as pp # Report

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd # geodata

from geopandas.tools import geocode # geodata

import folium # geodata

from folium import Marker # geodata

import numpy as np # linear algebra

import matplotlib.pyplot as plt # Plot

%matplotlib inline  

import seaborn as sns # Plot

sns.set(style="darkgrid") # Style seaborn

sns.set(color_codes=True) # Color seaborn 



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Function for exploring data

def df_eda(df):

    print("SUMMARY")

    # Select categorical columns with relatively low cardinality (convenient but arbitrary)

    categorical_col = [cname for cname in df.columns if df[cname].nunique() > 0 and 

                       df[cname].dtype == "object"]

    # Select numerical columns

    numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only

    my_cols = categorical_col + numerical_cols

    data = df[my_cols].copy()

    print("Categorical columns :",categorical_col)

    print("")

    print("Numerical columns :",numerical_cols)

    print("")

    print("Structure")

    print(data.info())

    print("")

    # Number of missing values in each column of training data

    print("Missing Values")

    missing_val_count_by_column = (data.isnull().sum())

    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    print("duplicated :",data.duplicated(subset=None, keep='first').sum())# Check

    print("")

    print("Cardinality")

    print(data.nunique())

    print("")

    print("HEAD")

    print("")

    print(data.head().transpose())

    print("")

    # Shape of training data (num_rows, num_columns)

    print("data Shape: ",data.shape)

    print("")

    print("Statistics")

    print("")

    print(data.describe())

    

# Function for plot

def visual_eda(df):

    sns.set(style="darkgrid")

    # Select categorical columns with relatively low cardinality (convenient but arbitrary)

    categorical_col = [cname for cname in df.columns if df[cname].nunique() > 0 and 

                       df[cname].dtype == "object"]

    # Select numerical columns

    numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only

    my_cols = categorical_col + numerical_cols

    data = df[my_cols].copy()

    # Histogram

    f, ax = plt.subplots(figsize=(30, 30))

    data[numerical_cols].hist(ax=ax,align='mid',orientation='horizontal',bins=50)

    plt.show()

    # Boxplot

    f, ax = plt.subplots(figsize=(30, 30))

    sns.boxplot(data=data[numerical_cols],palette="tab20")

    plt.xticks(rotation=90)

    plt.title('BoxPlot Train',fontsize=20)

    plt.ylabel('Values',fontsize=20)

    plt.show()

    # Heatmap correlation

    f,ax = plt.subplots(figsize=(24,24))

    sns.heatmap(data[numerical_cols].corr(), linewidths=.8, fmt='.1f', ax=ax,label=True,cmap="RdBu",alpha=.9,annot=True)

    plt.xticks(rotation=90)

    plt.title('Correlation ',fontsize=20)

    plt.show()

    

    return data[numerical_cols].corr()

        

# Function for displaying the map

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')



# Function for Create map with location marks 

def Create_map(op):

    if (op == 1):

        text = 'openstreetmap'

        save = 'map_1.html'

        zoom = 3

    elif(op == 2):

        text = 'Stamen Terrain'

        save = 'map_2.html'

        zoom = 6

    else :

        text = 'stamentoner'

        save = 'map_3.html'

        zoom = 4

        

    m = folium.Map(location=[39.656,-100.437], tiles=text, zoom_start=zoom)

        

    folium.Marker(

    location=[40.7972, -105.661],

    popup='Roosevelt National Forest'

    ).add_to(m)



    folium.Marker(

        location=[40.7124, -105.9453],

        popup='Rawah Wilderness Area',

        icon=folium.Icon(color='green', icon='info-sign')

    ).add_to(m)



    folium.Marker(

        location=[40.519, -105.85],

        popup='Neota Wilderness Area',

        icon=folium.Icon(color='green', icon='info-sign')

    ).add_to(m)



    folium.Marker(

        location=[40.5717, -105.7571],

        popup='Comanche Peak Wilderness Area',

        icon=folium.Icon(color='green', icon='info-sign')

    ).add_to(m)



    folium.Marker(

        location=[40.6568, -105.4751],

        popup='Cache La Poudre Wilderness Area',

        icon=folium.Icon(color='green', icon='info-sign')

    ).add_to(m)

    folium.LayerControl().add_to(m)

    embed_map(m, save)

    return m        





#read data

Path_train = "../input/learn-together/train.csv"

Path_test = "../input/learn-together/test.csv"

Path_Shape = "../input/wildernessareas/S_USA.Wilderness.shp"

#path_sub = "../input/comp-01/best_submission.csv"

df_train = pd.read_csv(Path_train,index_col='Id')# ->> "train.csv"

df_test = pd.read_csv(Path_test,index_col='Id')# ->>"test.csv"

#submission =  pd.read_csv(path_sub,index_col='Id')

full_geodata = gpd.read_file(Path_Shape)

data_train = df_train.copy()

data_test = df_test.copy()

data_geo = full_geodata.copy()

print("Train dataset shape: "+ str(df_train.shape))

print("Test dataset shape:  "+ str(df_test.shape))

print("Geo dataset shape:  "+ str(data_geo.shape))
df_eda(data_train) # Call of function 
df_eda(data_test) # Call of function 
visual_eda(df_train)# Call Function
new_data = df_train.iloc[:,10:14].join(df_train.iloc[:,54])

new_data.head()
new_data.describe().transpose()
wilderness_areas = df_train.iloc[:,10:14].sum(axis=0)



plt.figure(figsize=(7,5))

sns.barplot(x=wilderness_areas.index,y=wilderness_areas.values, palette="Set3")

plt.xticks(rotation=90)

plt.title('Wilderness Areas',color = 'g',fontsize=20)

plt.ylabel('Cover_type',color = 'g',fontsize=20)

plt.show()
sns.set(style="darkgrid")

sns.pairplot(new_data,hue="Cover_Type", palette="Set1",diag_kind="kde",

             height=2.5,markers=["+", "p", "x","d", "s", "D","o"])

plt.show()
plt.matshow(new_data.corr())
new_data.corr()
Create_map(1)
Create_map(2)
Create_map(3)
data_geo.head().transpose()
new_data_geo = data_geo[data_geo.WILDERNE_1.isin([

    'Rawah Wilderness', 'Neota Wilderness', 'Comanche Peak Wilderness', 'Cache La Poudre Wilderness'])]

new_data_geo
Rawah = new_data_geo.loc[new_data_geo.WILDERNE_1.isin(['Rawah Wilderness', 'Wilderness'])].copy()

Neota = new_data_geo.loc[new_data_geo.WILDERNE_1.isin(['Neota Wilderness', 'Wilderness'])].copy()

Comanche = new_data_geo.loc[new_data_geo.WILDERNE_1.isin(['Comanche Peak Wilderness', 'Wilderness'])].copy()

Cache = new_data_geo.loc[new_data_geo.WILDERNE_1.isin(['Cache La Poudre Wilderness', 'Wilderness'])].copy()
new_data_geo.plot(column='WID', cmap='tab20c',scheme='quantiles',figsize=(15, 15),

           legend=True,alpha=.95)

plt.title('Wilderness Areas',size=20)

plt.show()
Rawah.plot(column='GIS_ACRES', cmap='RdBu',figsize=(10, 10),legend=True,alpha=0.7)

plt.title('Rawah Wilderness',size=20)

plt.xlabel('Geometry',size=16)

plt.show()
Neota.plot(column='GIS_ACRES', cmap='RdBu',figsize=(10, 10),legend=True,alpha=0.7)

plt.title('Neota Wilderness',size=20)

plt.xlabel('Geometry',size=16)

plt.show()
Comanche.plot(column='GIS_ACRES', cmap='RdBu',figsize=(10, 10),legend=True,alpha=0.7)

plt.title('Comanche Peak Wilderness',size=20)

plt.xlabel('Geometry',size=16)

plt.show()
Cache.plot(column='GIS_ACRES', cmap='RdBu',figsize=(10, 10),legend=True,alpha=0.7)

plt.title('Cache la Poudre Wilderness',size=20)

plt.xlabel('Geometry',size=16)

plt.show()
profile = data_train.profile_report (title = 'Train Report') 

profile.to_file(output_file="Train_Report.html")

profile