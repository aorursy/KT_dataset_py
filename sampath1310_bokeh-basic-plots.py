from bokeh.io import output_notebook, show #,output_file 
output_notebook()
import numpy as np
from bokeh.plotting import figure
#figure will create an empty plot where you can visualize data
#lets take a look at simple sin plot
#create values for plotting data
x = np.arange(0,10,0.5)
#sin values of x
y=np.sin(x)
#create empty figure having axis lables 
#having axis label with title of plot and height and width of 500
p = figure(x_axis_label='number',y_axis_label='sin(number)',title="Sin Plot", plot_width=500, plot_height=500)
#line is a glyph lets talk a little later for now you can see that it creates line plot
p.line(x,y)
#show will show rendered plot
show(p)
import pandas as pd 
from sklearn.datasets import load_iris # iris dataset
iris = load_iris()
data = iris.data
column_names = iris.feature_names
#Creating dataframe out of data and features
#convert target name from numeric to names
df = pd.DataFrame(data,columns=column_names)
df = df.assign(target=iris.target)
df.target = df.target.apply(lambda x: iris.target_names[x])
df.head()
from bokeh.models import ColumnDataSource
#columndatasource takes dataframe in parameter
source = ColumnDataSource(data=df)
#plot data with glyph
p = figure(x_axis_label='sepal length (cm)', y_axis_label='petal length (cm)')
p.circle(x='sepal length (cm)', y='petal length (cm)', source=source)
show(p)
from bokeh.models import CategoricalColorMapper

#CategoricalColorMapper is used to map color to factors all we need to do is initialize CategoricalColorMapper
#with factor that you want to and mention palette i.e color
color_mapper = CategoricalColorMapper(factors=np.unique(df.target),
                                      palette=['red', 'green', 'blue'])
#add dictionary to circle with field and transformer 
p.circle(x='sepal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),
            legend='target')

show(p)
from bokeh.palettes  import Spectral3
from bokeh.layouts import gridplot
#make a source for dataframe this was already initialized already in above part 
source = ColumnDataSource(data=df)

#create CategoricalColorMapper
color_mapper = CategoricalColorMapper(factors=np.unique(df.target),
                                      palette=Spectral3)
#tools 
TOOLS = "box_select,lasso_select,help"

# create a new plot and add a renderer
left = figure(tools=TOOLS, plot_width=400, plot_height=400, title=None)
left.circle(x='sepal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),legend='target')
left.legend.location='top_left'
# create another new plot and add a renderer
right = figure(tools=TOOLS, plot_width=400, plot_height=400, title=None)
right.circle(x='petal length (cm)', y='petal length (cm)', source=source,
         color=dict(field='target',transform=color_mapper),legend='target')
right.legend.location='top_left'

#Here we use gridplot for side by side plotting 
p = gridplot([[left, right]])

show(p)
new_df = df.sample(50)
#get count of target data 
group=new_df.groupby('target').count()
#change x axis  to factors 
p = figure(x_range=group.index.tolist(), plot_height=250, toolbar_location=None)
#apply vbar glyph for bar plot
p.vbar(x=group.index.tolist(), top=group['sepal length (cm)'].tolist(), width=0.9,
       line_color='white')
show(p)