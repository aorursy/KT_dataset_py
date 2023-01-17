import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
#read dataset for all analysis and visualization

data=pd.read_csv('../input/googleplaystore.csv')
#show first five rows in data set

data.head()
data.sample(5)
#show last five rows in dataset

data.tail()
#it is a process that shows the property value in the data set and shows the numbers in the register values.

data.info()
#It is a function that shows the analysis of numerical values.

data.describe()
#It shows the data types in the data set

data.dtypes
data.Size=data.Size.replace('Varies with device',np.nan)

data.Size=data.Size.str.replace('M','000')

data.Size=data.Size.str.replace('k','')

data.Size=data.Size.replace('1,000+',1000)



data.Installs=data.Installs.str.replace(",","")

data.Installs=data.Installs.apply(lambda x:x.strip('+'))

data.Installs=data.Installs.replace('Free',np.nan)

data.Price=data.Price.str.replace('$','')

data=data.drop(data.index[10472])

data[['Size','Installs','Reviews','Price']]=data[['Size','Installs','Reviews','Price']].astype(float)
data.info()
#It is a function that shows the analysis of proximity values between data.

data.corr()
#show data columns

for col in data.columns:

    print(col)
#I will apply to rename data columns because of analysis of problem

data=data.rename(columns={'Content Rating':'ContentRating',

                    'Last Updated':'LastUpdated',

                    'Current Ver':'CurrentVer',

                    'Android Ver':'AndroidVer'})
#show columns

for i,col in enumerate(data.columns):

    print(i+1,". column is",col)
#random data

data.sample(4)
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.3f',ax=ax)

plt.show()
#the total number of vacancies between our values

data.isnull().sum()

#Reviews,Size,Type,Current Ver,Android Ver some nulls
im=Imputer(missing_values='NaN', strategy='mean')

data[['Rating','Size']]=im.fit_transform(data[['Rating','Size']])
data.isnull().sum()
data[(data['Type']!='Free')&(data['Type']!='Paid')]
data[data['CurrentVer'].isnull()]
data[data['AndroidVer'].isnull()]
#show drop index

data=data.drop([9148,15,4453,4490,1553,10342,7730,7407,7333,6803,6322],axis=0)
#show data rows and columns

data.shape
data.isnull().sum()

#As you can see, if there is an empty value in any way.
data.Category.value_counts()

#There are many different categories. These categories are composed of many numbers.
data.Type.unique()
plot = figure(x_axis_label = "x",y_axis_label = "y",tools = "pan,box_zoom")

plot.circle(x=[5,4,3,2,1],y=[1,2,3,4,5],size = 5,color = "black",alpha = 0.7)

output_file("my_first_bokeh_plot.html")

show(plot)
# There are other types of glyphs

plot = figure()

plot.diamond(x=[5,4,3,2,1],y=[1,2,3,4,5],size = 10,color = "black",alpha = 0.7)

plot.cross(x=[1,2,3,4,5],y=[1,2,3,4,5],size = 10,color = "red",alpha = 0.7)

show(plot)
plot = figure()

plot.line(x=[1,2,3,4,5,6,7],y = [1,2,3,4,5,5,5],line_width = 2)

plot.circle(x=[1,2,3,4,5,6,7],y = [1,2,3,4,5,5,5],fill_color = "white",size = 10)

show(plot)
plot = figure()

plot.patches(xs = [[1,1,2,2],[2,2,3,3]],ys = [[1,2,1,2],[1,2,1,2]],fill_color = ["purple","red"],line_color = ["black","black"])

show(plot)
source = ColumnDataSource(data)

plot = figure()

plot.circle(x="Size",y="Rating",source = source)

show(plot)
plot = figure(tools="box_select,lasso_select")

plot.circle(y= "Rating",x = "Reviews",source=source,color = "black",

            selection_color = "orange",

            nonselection_fill_alpha = 0.2,

           nonselection_fill_color = "blue")

show(plot)
# Hover appearance

hover = HoverTool(tooltips = [("Genre of game","@Genre"),("Publisher of game","@Publisher")], mode="hline")

plot = figure(tools=[hover,"crosshair"])

plot.circle(x= "Rating",y = "Size",source=source,color ="black",hover_color ="red")

show(plot)
# Color mapping

factors = list(data.Category.unique()) # what we want to color map. I choose genre of games

colors = ["red","green","blue","black","orange","brown","grey","purple","yellow","white","pink","peru"]

mapper = CategoricalColorMapper(factors = factors,palette = colors)

plot=figure()

plot.circle(x= "Rating",y = "Size",source=source,color = {"field":"Category","transform":mapper})

show(plot)

# plot looks like confusing but I think you got the idea of mapping 
# Row and column

p1 = figure()

p1.circle(x = "Reviews",y= "Rating",source = source,color="red")

p2 = figure()

p2.circle(x = "Reviews",y= "Reviews",source = source,color="black")

p3 = figure()

p3.circle(x = "Reviews",y= "Size",source = source,color="blue")

p4 = figure()

p4.circle(x = "Reviews",y= "Installs",source = source,color="orange")

layout1 = row(p1,p2)

layout2 = row(p3,p4)

layout3= column(layout1,layout2)

show(layout3)
layout = row(column(p1,p2),p3)

show(layout)
#Tabbed layout

#I use p1 and p2 that are created at above

tab1 = Panel(child = p1,title = "Reviews")

tab2 = Panel(child = p2,title = "Ratings")

tabs = Tabs(tabs=[tab1,tab2])

show(tabs)
# linking axis

# We will use p1 and p2 that are created at above

p2.x_range = p1.x_range

p2.y_range = p1.y_range

layout4=column(p1,p2)

show(layout4)
def make_plot(title, hist, edges, x, pdf, cdf):

    p = figure(title=title, tools='', background_fill_color="#fafafa")

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],

           fill_color="navy", line_color="white", alpha=0.5)

    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")

    p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")



    p.y_range.start = 0

    p.legend.location = "center_right"

    p.legend.background_fill_color = "#fefefe"

    p.xaxis.axis_label = 'x'

    p.yaxis.axis_label = 'Pr(x)'

    p.grid.grid_line_color="white"

    return p



# Normal Distribution



mu, sigma = 0, 0.5



measured = np.random.normal(mu, sigma, 1000)

hist, edges = np.histogram(measured, density=True, bins=50)



x = np.linspace(-2, 2, 1000)

pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2



p1 = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)



# Log-Normal Distribution



mu, sigma = 0, 0.5



measured = np.random.lognormal(mu, sigma, 1000)

hist, edges = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.0001, 8.0, 1000)

pdf = 1/(x* sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))

cdf = (1+scipy.special.erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))/2



p2 = make_plot("Log Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf)



# Gamma Distribution



k, theta = 7.5, 1.0



measured = np.random.gamma(k, theta, 1000)

hist, edges = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.0001, 20.0, 1000)

pdf = x**(k-1) * np.exp(-x/theta) / (theta**k * scipy.special.gamma(k))

cdf = scipy.special.gammainc(k, x/theta)



p3 = make_plot("Gamma Distribution (k=7.5, θ=1)", hist, edges, x, pdf, cdf)



# Weibull Distribution



lam, k = 1, 1.25

measured = lam*(-np.log(np.random.uniform(0, 1, 1000)))**(1/k)

hist, edges = np.histogram(measured, density=True, bins=50)



x = np.linspace(0.0001, 8, 1000)

pdf = (k/lam)*(x/lam)**(k-1) * np.exp(-(x/lam)**k)

cdf = 1 - np.exp(-(x/lam)**k)



p4 = make_plot("Weibull Distribution (λ=1, k=1.25)", hist, edges, x, pdf, cdf)



output_file('histogram.html', title="histogram.py example")



show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))
N = 500

x = np.linspace(0, 10, N)

y = np.linspace(0, 10, N)

xx, yy = np.meshgrid(x, y)

d = np.sin(xx)*np.cos(yy)



p = figure(x_range=(0, 10), y_range=(0, 10),

           tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])



# must give a vector of image data for image parameter

p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11")



output_file("image.html", title="image.py example")



show(p)  # open a browser
from bokeh.plotting import figure, show, output_file

from bokeh.sampledata.iris import flowers



colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

colors = [colormap[x] for x in flowers['species']]



p = figure(title = "Iris Morphology")

p.xaxis.axis_label = 'Petal Length'

p.yaxis.axis_label = 'Petal Width'



p.circle(flowers["petal_length"], flowers["petal_width"],

         color=colors, fill_alpha=0.2, size=10)



output_file("iris.html", title="iris.py example")



show(p)