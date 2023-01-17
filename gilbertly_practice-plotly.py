import plotly as py
from plotly.graph_objs import Scatter, Layout, Figure
from plotly.offline import init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
layout = Layout(
    title="Simple graph",
    xaxis=dict(title="xaxis"),
    yaxis=dict(title="yaxis")
)
trace_1 = Scatter(
    x=[1,2,3,4,5],
    y=[5,4,3,2,1],
    name="Line 1"
)
trace_2 = Scatter(
    x=[10,9,8,7,6],
    y=[6,7,8,9,10],
    name="Line 2"
)
data = [trace_1, trace_2]
figure = Figure(data=data, layout=layout)
iplot(figure)
from pyspark.sql import SparkSession

sc = SparkSession\
    .builder\
    .master("local[*]")\
    .appName('plotly_pyspark')\
    .getOrCreate()

data = [
    (1, "Gilbert", "24", "125"), 
    (2, "Someone", "26", "150"),
    (3, "Gathara", "28", "165"),
    (4, "Kariuki", "30", "172"),
    (5, "Python", "42", "180"),
]
headers = ("id", "Name", "Age", "Height")
df = sc.createDataFrame(data, headers)
df.show()
layout = Layout(
    title="Age vs. Height graph",
    xaxis=dict(title="Age"),
    yaxis=dict(title="Height")
)
trace_1 = Scatter(
    x=[x.Age for x in df.collect()],
    y=[x.Height for x in df.collect()],
    name="Age vs. Height"
)
data = [trace_1]
figure = Figure(data=data, layout=layout)
iplot(figure)





