import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/oec.csv')

df.head()
df.describe()
df.columns.values
disc = df.DiscoveryYear.dropna()

plt.hist(disc, bins=200)

plt.show()
disc_pre2000 = disc[disc < 2000]

plt.hist(disc_pre2000, bins=200)

plt.show()
disc_post2000 = disc[disc > 2000]

plt.hist(disc_post2000)

plt.show()
df_2000 = df[df.DiscoveryYear > 2000]

plt.scatter(df_2000.DiscoveryYear, df_2000.RadiusJpt)
plt.scatter(df_2000.DiscoveryYear, df_2000.DistFromSunParsec)
plt.scatter(df_2000.DiscoveryYear, df_2000.PlanetaryMassJpt)
plt.scatter(df_2000.DiscoveryYear, df_2000.HostStarRadiusSlrRad)
plt.scatter(df_2000.DiscoveryYear, df_2000.SurfaceTempK)
plt.scatter(df_2000.DiscoveryYear, df_2000.HostStarTempK)