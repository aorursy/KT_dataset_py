import pandas as pd

crimes = pd.read_csv("../input/crime.csv")
crimes[['Latitude', 'Longitude', 'NEIGHBOURHOOD']].head()
coords = crimes.query('NEIGHBOURHOOD == "Strathcona"')[['Latitude', 'Longitude']].values
pd.DataFrame(coords).dropna().plot.scatter(x=0, y=1)

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
from shapely.geometry import Polygon



Polygon(

    list(

        pd.DataFrame(coords).apply(lambda srs: (srs[0], srs[1]), axis='columns').values

    )

).convex_hull