import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('../input/oec.csv')

df = df[['PlanetIdentifier','PlanetaryMassJpt','RadiusJpt','DiscoveryMethod','DiscoveryYear','DistFromSunParsec','HostStarMassSlrMass','HostStarRadiusSlrRad']]

# Remove unnecessary rows

df = df[df['PlanetIdentifier']!='Mercury']

df = df[df['PlanetIdentifier']!='Venus']

df = df[df['PlanetIdentifier']!='Earth']

df = df[df['PlanetIdentifier']!='Mars']

df = df[df['PlanetIdentifier']!='Jupiter']

df = df[df['PlanetIdentifier']!='Saturn']

df = df[df['PlanetIdentifier']!='Uranus']

df = df[df['PlanetIdentifier']!='Neptune']

df = df[df['PlanetIdentifier']!='Pluto']



dfXPyear = df.dropna()



dftest = pd.crosstab(dfXPyear.DiscoveryYear, dfXPyear.DiscoveryMethod)



dftest



#dftest.plot(kind='bar', stacked=True)