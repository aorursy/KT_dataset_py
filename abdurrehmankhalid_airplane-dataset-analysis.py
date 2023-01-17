import pandas as pandasInstance
import numpy as numpyInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotlibInstance
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
%matplotlib inline
init_notebook_mode(connected=True)
cf.go_offline()
airplanesRecord = pandasInstance.read_csv('../input/DelayedFlights.csv')
airplanesRecord.head()
airplanesRecord.info()
groupedByYear = airplanesRecord.groupby(by='Year')
groupedByYearCount = groupedByYear.count()
groupedByYearCount
groupedByMonth = airplanesRecord.groupby(by='Month').count()
groupedByMonth
matplotlibInstance.figure(figsize=(12,10))
groupedByMonth['FlightNum'].iplot(title='Flights to Month Comparison',color='red')
tailNumberMonth = airplanesRecord.groupby(by=['TailNum','Month']).count()['FlightNum'].unstack()
matplotlibInstance.figure(figsize=(25,20))
matplotlibInstance.tight_layout()
seabornInstance.heatmap(tailNumberMonth)
seabornInstance.jointplot(x='Month',y='WeatherDelay',data=groupedByMonth.reset_index(),kind='kde',color='red')
matplotlibInstance.figure(figsize=(25,20))
matplotlibInstance.tight_layout()
seabornInstance.countplot(x='TailNum',data=airplanesRecord)
airplanesRecord['TailNum'].value_counts().head(10)
groupedByDay = airplanesRecord.groupby(by=['DayofMonth']).count()
matplotlibInstance.figure(figsize=(12,10))
groupedByDay['FlightNum'].iplot(color='red')
airplanesRecord['Dest'].value_counts().head(1)
