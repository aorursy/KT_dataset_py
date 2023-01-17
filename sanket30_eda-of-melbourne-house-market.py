
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import IFrame
df=pd.read_csv("../input/Melbourne_housing_FULL.csv")

df.head()
df.isna().sum()
df=df.dropna()
IFrame("https://public.tableau.com/views/RegionwisePropertyCount/RegionwisePropertyCount?:embed=y&:showVizHome=no", width=600, height=500)

IFrame("https://public.tableau.com/views/CostlySuburb/CostlySuburb?:embed=y&:showVizHome=no", width=1100, height=600)

IFrame("https://public.tableau.com/views/MostPopularTypeamongallSuburb/MostPopularTypeamongallSuburb?:embed=y&:showVizHome=no", width=1200, height=800)

IFrame("https://public.tableau.com/views/Moreroommoreprice/Moreroommoreprice?:embed=y&:showVizHome=no", width=700, height=400)

IFrame("https://public.tableau.com/views/LargeLandsizemoreprice/LargeLandsizemoreprice?:embed=y&:showVizHome=no", width=1200, height=500)

