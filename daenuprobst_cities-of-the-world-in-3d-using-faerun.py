!pip install faerun



import math

import numpy as np

import pandas as pd

from faerun import Faerun

from scipy.stats import rankdata

def latlong_to_3d(latitude, longitude, alt = 0, r = 6371):

    X = []

    Y = []

    Z = []

    

    for lat, long in zip(latitude, longitude):

        lat, long = np.deg2rad(lat), np.deg2rad(long)

        

        x = r * math.cos(lat) * math.cos(long)

        y = r * math.cos(lat) * math.sin(long)

        z = r * math.sin(lat)

        

        a = np.deg2rad(-90)        

        

        y_r = y * math.cos(a) - z * math.sin(a)

        z_r = y * math.sin(a) + z * math.cos(a)

        

        X.append(x)

        Y.append(y_r)

        Z.append(z_r)



    return X, Y, Z

data = pd.read_csv("/kaggle/input/world-cities-database/worldcitiespop.csv")

data = data[data["Population"] > 0]



x, y, z = latlong_to_3d(data["Latitude"], data["Longitude"])



f = Faerun(coords=False, legend_title="Legend")

f.add_scatter(

    "world", 

    {

        "x": x, "y": y, "z": z, 

        "c": rankdata(data["Population"].values), 

        "labels": data["City"].str.replace("'", "`")

    },

    shader="circle",

    fog_intensity=3.5,

    point_scale=0.5,

    max_point_size=10,

    colormap="viridis",

    has_legend=True,

    max_legend_label=f'{int(max(data["Population"].values)):,}',

    min_legend_label=f'{int(min(data["Population"].values)):,}',

    legend_title="Population Size",

)



f.plot(notebook_height=750)