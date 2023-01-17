import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/weather_madrid_LEMD_1997_2015.csv") # 1997 - 2015 ilmaandmed Madridis
# Vajalike andmete leidmine

kuupäev = pd.to_datetime(df["CET"])

temp_niiskus = pd.DataFrame({"Temperatuur": df["Mean TemperatureC"],"Niiskus": df[" Mean Humidity"]})

suvekuud = temp_niiskus[kuupäev.dt.month > 5][kuupäev.dt.month < 9]

suvekuud = suvekuud.dropna()
# Suvekuude sagedaseimad temperatuurid

suvekuud.Temperatuur.plot.hist(bins=15, grid=False, rwidth=0.95);
# Seos temperatuuri ja niiskuse vahel suvekuudel

suvekuud.plot.scatter("Temperatuur", "Niiskus", alpha = 0.1);
# Suvede keskmised temperatuurid ja keskmine niiskus

suvekuud.groupby(kuupäev.dt.year)["Niiskus", "Temperatuur"] .mean().round(1).sort_values("Niiskus", ascending=False)