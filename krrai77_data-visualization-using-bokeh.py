#Load the pandas_bokeh library



!pip install pandas_bokeh
import numpy as np

import pandas as pd

import pandas_bokeh

import seaborn as sns
#load the dataset



iris=sns.load_dataset("iris")

iris.head()
#set the bokeh output to the notebook



pandas_bokeh.output_notebook()
#Bar plot



iris.plot_bokeh(

    kind='bar',

    x='species',

    y=['sepal_length', 'sepal_width','petal_length','petal_width'],

    xlabel='Species',

    ylabel='Length and Width',

    title='Flowers',

)
#Line plot



iris.plot_bokeh(

    kind='line',

    x='species',

    y=['sepal_length', 'sepal_width','petal_length','petal_width'],

    xlabel='Species',

    ylabel='Length and Width',

    title='Flowers',

)
iris.plot_bokeh(

    kind='scatter',

    x='species',

    y=['sepal_length', 'sepal_width'],

    xlabel='Species',

    ylabel='Length and Width',

    title='Flowers',

)
iris.plot_bokeh(

    kind='scatter',

    x='species',

    y=['petal_length', 'petal_width'],

    xlabel='Species',

    ylabel='Length and Width',

    title='Flowers',

)
#Scatter plot



iris.plot_bokeh.scatter(

    x='species', 

    y=['sepal_length', 'petal_length'],

    figsize=(1000, 700),

    zooming=False,

    panning=False

)
#Save the bokeh output as a html file and share it with others



pandas_bokeh.output_file('./outout/kaggle/working/chart.html')
pandas_bokeh.output_notebook()
#Histogram



iris.plot_bokeh(kind="hist",title ="Iris feature distribution",

                   figsize =(1000,800),

                   xlabel = "Features",

                   ylabel="Measure"

                )
sns.get_dataset_names()
car=sns.load_dataset("mpg")

car
#Top 10 cars based on horsepower

carhp=car.nlargest(10,'horsepower')

carhp
#Horizontal barchart

carhp.plot_bokeh(

    kind="barh",

    figsize =(1000,800),

    x="name",

    ylabel="Car Models", 

    title="Top 10 Car Features", 

    alpha=0.6,

    legend = "bottom_right",

    show_figure=True)



#Stacked horizontal bar

carhp.plot_bokeh.barh(

    figsize =(1000,800),

    x="name",

    stacked=True,

    ylabel="Car Models", 

    title="Top 10 Car Features", 

    alpha=0.6,

    legend = "bottom_right",

    show_figure=True)



#Bottom 10 cars based on horsepower

carhpbot=car.nsmallest(10,'horsepower')

carhpbot
#Vertical barchart

carhpbot.plot_bokeh(

    kind="bar",

    figsize =(1000,800),

    x="name",

    xlabel="Car Models", 

    title="Bottom 10 Car Features", 

    alpha=0.6,

    legend = "top_right",

    show_figure=True)



#Stacked vertical bar

carhpbot.plot_bokeh.bar(

    figsize =(1000,800),

    x="name",

    stacked=True,

    xlabel="Car Models", 

    title="Bottom 10 Car Features", 

    alpha=0.6,

    legend = "top_right",

    show_figure=True)
car.plot_bokeh.scatter(

    x='horsepower', 

    y=['weight'],

    figsize=(1000, 700),

    zooming=False,

    panning=False

)
carhpbot.plot_bokeh(

    kind='line',

    x='horsepower',

    y=['mpg','acceleration'],

    figsize=(1000, 700),

    xlabel='Horsepower',

    ylabel='Mpg and Acceleration',

    title='Average Cars',

)
carhp.plot_bokeh(

    kind='line',

    x='horsepower',

    y=['mpg','acceleration'],

    figsize=(1000, 700),

    xlabel='Horsepower',

    ylabel='Mpg and Acceleration',

    title='Top Performing Cars',

)
#Pie plot

carhp.plot_bokeh.pie(

    x="name",

    title="Cars",

    )
#Area plot



carhp.plot_bokeh.area(

    x="name",

    stacked=True,

    figsize=(1300, 700),

    title="Compare Car Models",

    xlabel="Top 10 Car models",

    )
mapplot = pd.read_csv(r"https://raw.githubusercontent.com/PatrikHlobil/Pandas-Bokeh/master/docs/Testdata/populated%20places/populated_places.csv")

mapplot.head()
#Plotting Map



mapplot["size"] = mapplot["pop_max"] / 1000000

mapplot.plot_bokeh.map(

    x="longitude",

    y="latitude",

    hovertool_string="""<h2> @{name} </h2> 

    

                        <h3> Population: @{pop_max} </h3>""",

    tile_provider="STAMEN_TERRAIN_RETINA",

    size="size", 

    figsize=(1200, 600),

    title="Cities with more than 1000K population")