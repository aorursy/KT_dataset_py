import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Read the file into a variable flight_data

gltbs_data = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv', index_col="Country")

gltbs_data
dd = pd.DataFrame(gltbs_data.loc[(gltbs_data.index == "China") & (gltbs_data.dt == "2013-08-01")])

de = pd.DataFrame(dd.set_index('State'))

df = pd.DataFrame(de.loc[:,'AverageTemperature'])

df
# Set the width and height of the figure

plt.figure(figsize=(30,6))



# Add title

plt.title("Suhu Rata-Rata China bulan Oktober 2013")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.barplot(x=df.index, y=df['AverageTemperature'])



# Add label for vertical axis

plt.ylabel("Suhu Rata-Rata")

plt.xlabel("Negara Bagian")
db = pd.DataFrame(gltbs_data.loc[(gltbs_data.index == "China") & (gltbs_data.dt >= "2013-01-01")])

dc = pd.DataFrame(db.set_index('dt'))

da = pd.DataFrame(dc.loc[:,'AverageTemperature'])

da
# Set the width and height of the figure

plt.figure(figsize=(30,6))



# Add title

plt.title("Perubahan Suhu per Waktu")



# Bar chart showing average arrival delay for Spirit Airlines flights by month

sns.lineplot(x=da.index, y=da['AverageTemperature'])



# Add label for vertical axis

plt.ylabel("Suhu Rata-Rata")

plt.xlabel("Waktu")