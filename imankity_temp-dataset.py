from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from matplotlib import pyplot as plt
import pandas as pd
sc = SparkContext()
spark = SparkSession(sc)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

tempDF=pd.read_excel("../input/temp.xls")
sparkDF = spark.createDataFrame(tempDF)
sparkDF.count()
def dfToList(dfnm,colnm):
    return dfnm.rdd.map(lambda row : row[colnm]).collect()
year = dfToList(sparkDF,'YEAR')
annual = dfToList(sparkDF,'ANNUAL')
trace = go.Scatter(
    x = year,
    y = annual,
    mode = 'markers',
    name = 'temp'
)
data = [trace]

iplot(data, filename='line-mode')
topAnnualDF=sparkDF.groupBy('YEAR').agg({'ANNUAL' : 'max'}).orderBy('max(ANNUAL)',ascending=False).limit(10)
topAnnualDF.show()
monthsDF=sparkDF.select('*').filter(sparkDF['YEAR'] == 2016)
month=list(monthsDF.toPandas())
month=month[1:13]
temperature=list(monthsDF.rdd.flatMap(lambda x: x).collect())
temperature=temperature[1:13]
temperature
trace = go.Scatter(
    x = month,
    y = temperature
)

data = [trace]

print("Temperature over months in 2016")
iplot(data, filename='basic-line')
monthsDF=sparkDF.select('JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC').filter(sparkDF['YEAR'] >= 2000)
month=list(monthsDF.toPandas())
month=month[0:13]
temperature=list(monthsDF.rdd.flatMap(lambda x: x).collect())
l=[]
for i in range(0,len(temperature),12):
    chunk = temperature[i : i + 12]
    l.append(chunk)
    
trace=[]
yearDF=sparkDF.select('YEAR').filter(sparkDF['YEAR'] >= 2000)
year=dfToList(yearDF,'YEAR')

for i in range(0,len(year)):
    trace0 = go.Scatter(
    x = month,
    y = l[i][:11],
    name=year[i]
    )
    trace.append(trace0)

print("Temperature over months from year 2000-2017 ")
iplot(trace, filename='line-mode')

