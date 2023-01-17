# Import our Pandas, calling our instance 'pd' for a handy shortcut
import pandas as pd

# Import our Plotly library 
import plotly.express as px
# Create a Data Frame by using our Pandas instance to read in the CSV data
df = pd.read_csv("../input/coviddatacsv/Covid.csv")
# Create the figure with appropriate data and range
fig = px.bar(df,
	     x="CountyName",
	     y="PopulationProportionCovidCases",
	     animation_frame="TimeStamp",
	     range_y=[0,1500])

# Create the chart
fig.show()