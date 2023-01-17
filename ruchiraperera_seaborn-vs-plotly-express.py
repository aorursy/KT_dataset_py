import pandas as pd



import seaborn as sns ## These Three lines are necessary for Seaborn to work   

import matplotlib.pyplot as plt 

sns.set(color_codes=True)



%matplotlib inline 





import plotly_express as px ##Plotly Express need only one line to load the libraries
auto = pd.read_csv('../input/auto-data-set-with-automotive-information/Auto.csv')## Loading the data set 
auto.head()## Let's take look on the data set
sns.boxplot(auto['number_of_doors'], auto['horsepower']);
fig = px.box(auto, x="number_of_doors", y="horsepower")

fig.show()
sns.set(font="Verdana") ## Global font type , this will apply for all seaborn plots

plt.figure(figsize=(9,6)) ## Changing the Figure Size 

box=sns.boxplot(auto['number_of_doors'], auto['horsepower']) # Define the plot with variables 

box.axes.set_title("Seaborn Box Plot",fontsize=20) # Set the Tittle for the defined plot

box.set_xlabel("No of Doors",fontsize=16) # Set the x-axis Label and fornt size

box.set_ylabel("Horsepower",fontsize=16); # Set the y-axis Label and fornt size
# Defining the plot with Size and Title text 

fig = px.box(auto, x="number_of_doors", y="horsepower",width=700, height=500,title='Plotly Express Box Plot',labels={"horsepower": "Horsepower",  "number_of_doors": "No of Doors"}) 

fig.update_layout(font_family="Courier New", # Changing Styling of the plot 

    font_color="black",

    font_size=16,              

    title_font_family="Times New Roman",

    title_font_color="green",

    title_font_size=26,              

    title={'y':0.9,'x':0.5}) # Change the Title Alignment

fig.show()
fig, axs = plt.subplots(ncols = 2, figsize = (30, 7))

sns.distplot(auto['highway_mpg'],ax=axs[0],kde=False);

sns.distplot(auto['highway_mpg'],ax=axs[1]);
fig = px.histogram(auto, x="highway_mpg",histnorm='probability density',width=600, height=400)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots





fig = make_subplots(rows=1, cols=2,subplot_titles=("Histograp with count", "histogram with probability density"))



trace0 = go.Histogram(x=auto['highway_mpg'])

trace1 = go.Histogram(x=auto['highway_mpg'],histnorm='probability density')



fig.append_trace(trace0, 1, 1,)

fig.append_trace(trace1, 1, 2)



fig.show()
sns.jointplot(auto['engine_size'], auto['horsepower']);
fig = px.scatter(auto, x="engine_size", y="horsepower",marginal_y="histogram",marginal_x="histogram",width=600, height=600)

fig.show()
fig = px.scatter(auto, x="engine_size", y="horsepower",color="drive_wheels",size='price',hover_data=['number_of_doors'],width=900, height=500)

fig.show()
sns.jointplot(auto['engine_size'], auto['horsepower'], kind="kde");
fig = go.Figure(go.Histogram2dContour(

        x = auto['engine_size'],

        y = auto['horsepower'],

        colorscale = 'Blues'

))

fig.update_layout(width=500, height=500)

fig.show()
sns.pairplot(auto[['normalized_losses', 'engine_size', 'horsepower']]);
fig = px.scatter_matrix(auto, dimensions=auto[['normalized_losses', 'engine_size', 'horsepower']],width=700, height=700)

fig.show()
fig = px.scatter_matrix(auto, dimensions=auto[['normalized_losses', 'engine_size', 'horsepower']],color="fuel_type",width=700, height=700)

fig.show()
sns.stripplot(auto['fuel_type'], auto['horsepower'], jitter=True);
fig = px.strip(auto, x="fuel_type", y="horsepower",width=600, height=400)

fig.show()
sns.boxplot(auto['number_of_doors'], auto['horsepower']);
fig = px.box(auto, x="number_of_doors", y="horsepower",width=600, height=400)

fig.show()
plt.figure(figsize=(10,6))

sns.boxplot(auto['number_of_doors'], auto['horsepower'], hue=auto['fuel_type']);
fig = px.box(auto, x="number_of_doors", y="horsepower", color="fuel_type",width=800, height=500)

fig.show()
plt.figure(figsize=(10,6))

sns.violinplot(y=auto["horsepower"], x=auto["number_of_doors"], hue=auto["fuel_type"]);
fig = px.violin(auto, y="horsepower", x="number_of_doors", color="fuel_type", box=True, points="all",width=800, height=500)

fig.show()
sns.barplot(auto['body_style'], auto['horsepower'], hue=auto['fuel_type']);
sns.countplot(auto['body_style'],hue=auto['fuel_type']);
fig = px.bar(auto, x="body_style", y="horsepower", color='fuel_type',barmode='group',width=600, height=400)

fig.show()
plt.figure(figsize=(8,8))

sns.heatmap(auto.corr())

plt.show()
fig = px.imshow(auto.corr(),x=list(auto.corr().columns),y=list(auto.corr().columns),width=600, height=600)

fig.show()
fig = px.scatter_3d(auto, x='normalized_losses', y='engine_size', z='horsepower', color='fuel_type')

fig.show()