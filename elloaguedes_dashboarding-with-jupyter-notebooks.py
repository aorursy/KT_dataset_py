import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
# to use the csvvalidator package, you'll need to 
# install it. Turn on the internet (in the right-hand
# panel; you'll need to have phone validated your account)

import sys
!{sys.executable} -m pip install csvvalidator
df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
df.head(10)
df = df[["School_Year","Reason","Number_Of_Students_On_The_Bus","Occurred_On"]]
df.head(7)
df.dropna(inplace=True)
df.head(12)
df['Occurred_On'] = pd.to_datetime(df['Occurred_On'])
df.head(15)
len(df)
### Validating

# import everything from the csvvalidator package
from csvvalidator import *

# Specify which fields (columns) your .csv needs to have
# You should include all fields you use in your dashboard
field_names = ("School_Year","Reason","Number_Of_Students_On_The_Bus","Occurred_On")

# create a validator object using these fields
validator = CSVValidator(field_names)

# write some checks to make sure specific fields 
# are the way we expect them to be
validator.add_value_check("School_Year", # the name of the field
                          str, 
                          'EX1', # code for exception
                          'School_Year invalid'# message to report if error thrown
                         )
validator.add_value_check("Reason", 
                          # check for a date with the sepcified format
                          str, 
                          'EX2',
                          'Reason'
                         )
validator.add_value_check('Number_Of_Students_On_The_Bus',
                          # makes sure the number of units sold is an integer
                          int,
                          'EX3',
                          'Number_Of_Students_On_The_Bus invalid'
                         )
validator.add_value_check("Occurred_On", 
                          str,
                          'EX4', 
                          'Occurred_On" invalid')

results = validator.validate(df)
lines_remove = []
for di in results:
    lines_remove.append(di['row'])
    
df.drop(df.index[lines_remove],inplace=True)
    
len(df)
newdf = pd.DataFrame(df["School_Year"].value_counts())
newdf.rename(index=str, columns={"School_Year": "Bus Breakdowns"},inplace=True)
newdf.index.name = "Year"
newdf.sort_index(ascending=True,inplace=True)
newdf
from matplotlib.pyplot import figure

newdf.plot.bar(align='center', alpha=0.8,color='blue')
plt.title("Counting the number of bus breakdowns per school year")
plt.show()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# copying df to insert new index and make manipulation easier
newdf2 = newdf.copy()
newdf2.reset_index(level=0, inplace=True)

data = [
    go.Bar(
        x=newdf2['Year'], # assign x as the dataframe column 'x'
        y=newdf2['Bus Breakdowns']
    )
]

# specify the layout of our figure
layout = dict(title = "Number of Bus Breakdowns per School Year",
              xaxis= dict(title= 'Year',ticklen= 5,zeroline= False))

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)

df2 = df[["School_Year","Reason"]]


for ano in df2["School_Year"].unique():
    f, axes = plt.subplots(figsize=(8,8))
    dados = df2.loc[df["School_Year"] == ano]
    dados = pd.DataFrame(dados["Reason"].value_counts())
    
    total = sum(dados["Reason"])
    novo = [x/total for x in dados["Reason"]]
    dados["Fraction"] = novo

    axes.pie(dados["Fraction"],labels=dados.index, autopct='%.2f')
    plt.title("Year "+ str(ano))
    plt.show()
    plt.close('all')
    
    
    

## organizing data
df3 = df[["School_Year","Reason"]]
df3.reset_index(level=0, inplace=True)




year = "2015-2016"
df3.loc[df3["School_Year"]== year]

dados = pd.DataFrame(df3["Reason"].value_counts())

total = sum(dados["Reason"])
novo = [x/total for x in dados["Reason"]]
dados["Fraction"] = novo
dados.reset_index(level=0, inplace=True)
dados["Reason"] = dados["index"]
dados.drop(["index"],axis=1,inplace=True)
dados.head()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()



data = [
    go.Pie(
        labels=dados["Reason"],
        values=dados["Fraction"]
    )
]

# specify the layout of our figure
layout = dict(title = "Reasons of Bus Breakdowns")

# create and show our figure
fig = dict(data = data, layout = layout)
iplot(fig)




interest = df.loc[df['School_Year'] == '2018-2019']
interest.set_index(pd.to_datetime(interest["Occurred_On"]),inplace=True)
interest.drop(["School_Year","Reason","Occurred_On"],axis = 1, inplace= True)
interest.head(10)
interest = interest['Number_Of_Students_On_The_Bus'].resample('D').sum()
interest.head(10)
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(interest)
plt.gcf().autofmt_xdate()
plt.title("Students affected per day in 2018-2019")
plt.show()
## organizing data
newdf = interest.to_frame()
newdf.reset_index(level=0, inplace=True)
newdf.head(4)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

trace1 = go.Scatter(
    x = newdf["Occurred_On"],
    y = newdf["Number_Of_Students_On_The_Bus"],
    mode = 'lines+markers',
    name = 'lines+markers'
)

# specify the layout of our figure
layout = dict(title = "Students Affected by Day")

# create and show our figure
fig = dict(data = [trace1], layout = layout)
iplot(fig)