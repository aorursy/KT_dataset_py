import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv("/kaggle/input/ca-oshpd-adveventhospitalizationspsi-county2005-2015q3.csv", delimiter=',')

df1.dataframeName = "Adverse Hospital Events"

nRow, nCol = df1.shape

print(f"There are {nRow} rows and {nCol} columns")
for i in range(len(df1.ObsRate)):

    df1.Count[i] = int(df1.Count[i].replace(',', ''))

    df1.Population[i] = int(df1.Population[i].replace(',', ''))

    df1.ObsRate[i] = 100000 * df1.Count[i] / df1.Population[i]



df1 = df1[df1.County != "STATEWIDE"]
geos = {"Alameda": (37.640366, -121.880578),

       "Alpine": (38.599504, -119.802560),

        "Amador": (38.453800, -120.660010),

        "Butte": (39.687971, -121.600976),

        "Calaveras": (38.243448, -120.506967),

        "Colusa": (39.191769, -122.207145),

        "Contra Costa": (37.885338, -121.918506),

        "Del Norte": (41.782004, -123.916940),

        "El Dorado": (38.743386, -120.476513),

        "Fresno": (36.736274, -119.789114),

        "Glenn": (39.593418, -122.369708),

        "Humboldt": (40.613720, -123.874477),

        "Imperial": (33.011018, -115.229721),

        "Inyo": (36.459372, -117.263410),

        "Kern": (35.312492, -118.624158),

        "Kings": (36.018154, -119.807741),

        "Lake": (39.045063, -122.723313),

        "Lassen": (40.647286, -120.485798),

        "Los Angeles": (34.260708, -118.159933),

        "Madera": (37.181234, -119.790605),

        "Marin": (38.056252, -122.680623),

        "Mariposa": (37.558958, -119.991418),

        "Mendocino": (39.464229, -123.394887),

        "Merced": (37.201817, -120.669482),

        "Modoc": (41.595553, -120.678051),

        "Mono": (37.870508, -118.844046),

        "Monterey": (36.109402, -121.152795),

        "Napa": (38.489498, -122.312825),

        "Nevada": (39.336061, -120.849519),

        "Orange": (33.688426, -117.744370),

        "Placer": (39.084290, -120.731134),

        "Plumas": (40.004003, -120.771274),

        "Riverside": (33.702231, -115.888371),

        "Sacramento": (38.472165, -121.305147),

        "San Benito": (36.599524, -121.070537),

        "San Bernardino": (34.862832, -116.156556),

        "San Diego": (32.991834, -116.714210),

        "San Francisco": (37.756454, -122.442409),

        "San Joaquin": (37.946840, -121.255427),

        "San Luis Obispo": (35.367193, -120.361390),

        "San Mateo": (37.403316, -122.300386),

        "Santa Barbara": (34.726410, -120.005606),

        "Santa Clara": (37.226030, -121.680308),

        "Santa Cruz": (37.033096, -121.948246),

        "Shasta": (40.751708, -122.028946),

        "Sierra": (39.590462, -120.409292),

        "Siskiyou": (41.615659, -122.404116),

        "Solano": (38.247722, -121.907008),

        "Sonoma": (38.478863, -122.824891),

        "Stanislaus": (37.568779, -120.973042),

        "Sutter": (39.026107, -121.681325),

        "Tehama": (40.122409, -122.160088),

        "Trinity": (40.707422, -123.048124),

        "Tulare": (36.229581, -118.744465),

        "Tuolumne": (38.057249, -119.913258),

        "Ventura": (34.396072, -119.060124),

        "Yolo": (38.691382, -121.856415),

        "Yuba": (39.254952, -121.373879)

       }
df1.describe(include="all")
# Use HTML5 to request ^^

Coor = (33.643244, -117.595689)



# Use dropdown

PSI = 22
dists = []

dist = float("inf")

for count in df1.County.unique():

    d = np.linalg.norm(np.array(Coor)-np.array(geos[count]))

    dists.append(d)

    if d < dist:

        dist = d

        County = count
print(f"County: {County}\nPSI: {PSI}")
def rate(df, PSI, County):

    return float(df.loc[(df.Year == max(df.Year)) & (df.PSI == PSI) & (df.County == County)].ObsRate)
data = df1.loc[(df1.Year == df1.Year.max()) & (df1.PSI == PSI)]



data["Distance"] = dists
best = data.loc[(data.ObsRate == data.ObsRate.min()) & (data.Population == data[(data.ObsRate == data.ObsRate.min())].Population.max())].County.iloc[0]



good = data.loc[(data.ObsRate < data.ObsRate.quantile(0.25))]

good = good.loc[(good.Population > good.Population.quantile(0.50))]

good = list(good.loc[(good.Distance < good.Distance.quantile(0.25))].County)
print(f"Go to {best} County")

if len(good) > 0:

    print("\n\tAdditional:")

    for county in good:

          print("\t", county)