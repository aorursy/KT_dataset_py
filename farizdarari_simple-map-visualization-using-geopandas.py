import geopandas as gpd



filename = '../input/indonesiaprovincejmlpenduduk/indonesia-province-jml-penduduk.json' # population data for irian jaya timur and irian jaya tengah is taken from that of papua province divided by two

df = gpd.read_file(filename)

print(type(df))

df # take a look at the whole data
df.plot() # plot the whole Indonesia, x-axis is for longitude, y-axis is for latitude
df.set_index("Propinsi", inplace=True) # so you can get a specific row by its province

df.head() # return first 5 rows, look how Propinsi is now set as index (leftmost column)
df.loc['JAWA TIMUR'] # show only the specific data of JAWA TIMUR
series = gpd.GeoSeries(df.loc['JAWA TIMUR']['geometry']) # get the geometry of JAWA TIMUR and convert it into GeoSeries

type(series)
ax = series.plot() # make a plot of JAWA TIMUR

ax
series = gpd.GeoSeries(df.loc['DKI JAKARTA']['geometry']) # get the geometry of DKI Jakarta and convert it into GeoSeries

type(series)

ax = series.plot() # make a plot of DKI Jakarta

ax
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

df = df[df['Jumlah Penduduk'].notnull()] # pick rows where Jumlah Penduduk is not null, just in case

df.plot(column='Jumlah Penduduk', ax=ax, legend=True)
# in case you are not that fond of the above color scheme, let's redo it with another color scheme

# also this time with a bigger size

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

df.plot(column='Jumlah Penduduk', ax=ax, legend=True, cmap='OrRd')