

import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')
data.info()
data.head()
# data cleaning 



# create Origin or Crossing From variable

data["CrossingFrom"] = ["Mexico" if i == "US-Mexico Border" else "Canada" for i in data.Border]



# format date column

data["DateAsDateObj"] = pd.to_datetime(data.Date)

data = data.set_index("DateAsDateObj")
#Plot all border crossings by Month



dataForPlot = data.resample("M").mean()



dataForPlot.loc[:,["Value"]].plot()
# Which measurements have the hightest totals?



dataForPlot = data.loc[:,["Measure","Value"]]

dataForPlot.groupby("Measure").Value.sum().plot(kind="bar")
# Have the categories changed over time?



dataForPlot = data.loc[:,["Measure","Value"]]

# dataForPlot = dataForPlot.reset_index()



dataForPlot = dataForPlot.groupby("Measure").resample("M").mean()

dataForPlot.reset_index().pivot(index="DateAsDateObj",columns="Measure", values="Value").plot(subplots=True, figsize=(8,14))


