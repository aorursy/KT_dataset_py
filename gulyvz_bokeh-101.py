#importing Bokeh

from bokeh.plotting import figure

from bokeh.io import output_file, show, output_notebook

output_notebook()
#prepare some data

x = [1,2,3,4,5]

y = [6,7,8,9,10]

#create a figure object

f = figure(plot_width=400, plot_height=400)

#create line plot

f.line(x,y,line_width=2)

#write the plot in the figure object

show(f)
f = figure(plot_width=400, plot_height=400)



# add a steps renderer

f.step([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2, mode="center")



show(f)
f = figure(plot_width=400, plot_height=400)



f.multi_line([[1, 3, 2], [3, 4, 6, 6]], [[2, 1, 4], [4, 7, 8, 5]],

             color=["purple", "green"], alpha=[0.8, 0.3], line_width=4)



show(f)
f = figure(plot_width=400, plot_height=400)

f.vbar(x=[1, 2, 3], width=0.5, bottom=0,

       top=[1.2, 2.5, 3.7], color="purple")



show(f)
f = figure(plot_width=400, plot_height=400)

f.hbar(y=[1, 2, 3], height=0.5, left=0,

       right=[1.2, 2.5, 3.7], color="green")



show(f)
from bokeh.sampledata.iris import flowers
# Print the first 5 rows of the data

flowers.head()
# Print the last 5 rows of the data

flowers.tail()
colormap={'setosa':'red','versicolor':'green','virginica':'blue'}

flowers['color'] = [colormap[x] for x in flowers['species']]

flowers['size'] = flowers['sepal_width'] * 4
#after adding color and size columns

flowers.head()
from bokeh.models import ColumnDataSource



setosa = ColumnDataSource(flowers[flowers["species"]=="setosa"])

versicolor = ColumnDataSource(flowers[flowers["species"]=="versicolor"])

virginica = ColumnDataSource(flowers[flowers["species"]=="virginica"])
#Create the figure object

f = figure(plot_width=1000, plot_height=400)



#adding glyphs

f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2, 

color="color", line_dash=[5,3], legend_label='Setosa', source=setosa)



f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2, 

color="color", line_dash=[5,3], legend_label='Versicolor', source=versicolor)



f.circle(x="petal_length", y="petal_width", size='size', fill_alpha=0.2,

color="color", line_dash=[5,3], legend_label='Virginica', source=virginica)



show(f)
#Style the legend

f.legend.location = (500,500)

f.legend.location = 'top_left'

f.legend.background_fill_alpha = 0

f.legend.border_line_color = None

f.legend.margin = 10

f.legend.padding = 18

f.legend.label_text_color = 'black'

f.legend.label_text_font = 'times'



show(f)