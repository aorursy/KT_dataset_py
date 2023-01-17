import pandas as pd 
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, BooleanType

sc = SparkContext()
spark = SparkSession(sc)


fireSchema = StructType([StructField('CallNumber', IntegerType(), True),
                     StructField('UnitID', StringType(), True),
                     StructField('IncidentNumber', IntegerType(), True),
                     StructField('CallType', StringType(), True),
                     StructField('CallDate', StringType(), True),
                     StructField('WatchDate', StringType(), True),
                     StructField('ReceivedDtTm', StringType(), True),
                     StructField('EntryDtTm', StringType(), True),
                     StructField('DispatchDtTm', StringType(), True),
                     StructField('ResponseDtTm', StringType(), True),
                     StructField('OnSceneDtTm', StringType(), True),
                     StructField('TransportDtTm', StringType(), True),
                     StructField('HospitalDtTm', StringType(), True),
                     StructField('CallFinalDisposition', StringType(), True),
                     StructField('AvailableDtTm', StringType(), True),
                     StructField('Address', StringType(), True),
                     StructField('City', StringType(), True),
                     StructField('ZipcodeofIncident', IntegerType(), True),
                     StructField('Battalion', StringType(), True),
                     StructField('StationArea', StringType(), True),
                     StructField('Box', StringType(), True),
                     StructField('OriginalPriority', StringType(), True),
                     StructField('Priority', StringType(), True),
                     StructField('FinalPriority', IntegerType(), True),
                     StructField('ALSUnit', BooleanType(), True),
                     StructField('CallTypeGroup', StringType(), True),
                     StructField('NumberofAlarms', IntegerType(), True),
                     StructField('UnitType', StringType(), True),
                     StructField('Unitsequenceincalldispatch', IntegerType(), True),
                     StructField('FirePreventionDistrict', StringType(), True),
                     StructField('SupervisorDistrict', StringType(), True),
                     StructField('NeighborhoodDistrict', StringType(), True),
                     StructField('Location', StringType(), True),
                     StructField('RowID', StringType(), True)])


fireServiceCalldDF=spark.read.csv("../input/Fire_Department_Calls_for_Service.csv",header=True,schema=fireSchema)

incidentsDF=spark.read.csv("../input/Fire_Incidents.csv",header=True,inferSchema=True)

fireDataDF=fireServiceCalldDF.join(incidentsDF,'IncidentNumber')


#drop_list = ['IncidentNumber','Address','CallNumber','City','SupervisorDistrict','Location','Battalion','StationArea','Box','NumberofAlarms','NeighborhoodDistrict']

drop_list = [c for c in fireServiceCalldDF.columns if c in incidentsDF.columns]
for col in drop_list:
    fireDataDF = fireDataDF.drop(incidentsDF[col])
    
fireDataDF

callDF = fireDataDF.select('IncidentNumber','CallType').groupBy('CallType').agg({'IncidentNumber' : 'count'}).orderBy('count(IncidentNumber)',ascending=False).limit(10).toPandas()

call=callDF['CallType'].tolist()
incident_count=callDF['count(IncidentNumber)'].tolist()
pie_visual = go.Pie(labels=call, values=incident_count, marker=dict(colors=['#25e475', '#ee1c96',]))

layout = go.Layout(title='Types of Call', width=800, height=500)
fig = go.Figure(data=[pie_visual], layout=layout)
iplot(fig)
alarmsDF = fireDataDF.select('NumberofAlarms','City').groupBy('City').agg({'NumberofAlarms' : 'count'}).orderBy('count(NumberofAlarms)',ascending=False).limit(10).toPandas()

alarm=alarmsDF['City'].tolist()
alarm_count=alarmsDF['count(NumberofAlarms)'].tolist()
alarm
trace = go.Bar(
    x=alarm,
    y=alarm_count,
    name='Alarms',
    marker=dict(color='#f7bb31'),
    opacity=0.8
)
data = [trace]
layout = go.Layout(
    barmode='group',
    legend=dict(dict(x=-.1, y=1.2)),
    margin=dict(b=120),
    title = 'Alarms Raised in Different Locations',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')





