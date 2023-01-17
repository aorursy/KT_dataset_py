import pandas as pd

#Import data and save it as MPData
MPIData = pd.read_csv("../input/kiva-augmented-data/MPIData_augmented.csv")
#(rows,columns)
MPIData.shape
#Object = String, float* or int* = numeric
MPIData.info()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import numpy as np


trace0 = go.Box(
    y=MPIData["precipitation"].values.tolist()
)
data = [trace0]
py.iplot(data)

corr = MPIData[["precipitation",'PercPoverty', 'DepvrIntensity', 
                'popDensity',  'TimeToCity',"MPI_Region",
       'AvgNightLight', 'LandClassification', 'Elevation', 'Temperature',
       'Evaporation', 'Modis_LAI', 'Modis_EVI', 'Conflicts_total']].corr()

trace = go.Heatmap(z=corr.values.tolist(),
                   x=corr.columns,
                   y=corr.columns)
data=[trace]
py.iplot(data, filename='labelled-heatmap')

trace0 = go.Histogram(
    x=MPIData["precipitation"].values.tolist(),name = "Precipitation"
)
trace1 = go.Histogram(
    x=MPIData["MPI_Region"].values.tolist(),name = "MPI"
)
trace2 = go.Histogram(
    x=np.log(MPIData["MPI_Region"]+0.0001).values.tolist(),name = "MPI (Log)"
)
data = [trace0]
py.iplot(data)
data = [trace1]
py.iplot(data)
data = [trace2]
py.iplot(data)

# Create a trace
trace = go.Scatter(
    x = MPIData["precipitation"].values.tolist(),
    y = MPIData["MPI_Region"].values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')
import statsmodels.api as sm

MPIData_no_na = MPIData[["MPI_Region","AvgNightLight","Evaporation"]].dropna()
y = MPIData_no_na["MPI_Region"]
X = MPIData_no_na[["AvgNightLight","Evaporation"]]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
import numpy as np

y = np.log(MPIData_no_na["MPI_Region"]+.0001)
X = MPIData_no_na[["AvgNightLight","Evaporation"]]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
# Create a trace
res = predictions - y

trace = go.Scatter(
    y = res.values.tolist(),
    x = predictions.values.tolist(),
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')