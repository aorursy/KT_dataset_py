import pandas as pd

import os



from bokeh.models import (LinearInterpolator,

                          CategoricalColorMapper,

                          ColumnDataSource,

                          HoverTool,

                          NumeralTickFormatter)



from bokeh.palettes import Spectral5

from bokeh.io import output_notebook, show, push_notebook

from bokeh.plotting import figure



from ipywidgets import interact

from IPython.display import IFrame



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data = pd.read_csv(os.path.join(dirname, filename), sep='\t', index_col = 'year')



output_notebook()
print(data.dtypes)

data.head()


source = ColumnDataSource(data.loc[data.index.min()])



PLOT_OPTS = dict(

    height=400, width = 800, x_axis_type='log',

    x_range = (100,100000), y_range = (0,100),

    background_fill_color = 'black'

)



# Making Hover

hover = HoverTool(tooltips = '@country', show_arrow = False)





fig = figure(tools = [hover],

             toolbar_location = 'above',

             **PLOT_OPTS)



def update(year):

    """ Build New Data Based On The Year, method that update the source object """

    try:

        new_data =data.loc[year]

        source.data = new_data

    except KeyError:

        new_data = dict()

    fig.title.text = str(year)

    push_notebook()





# Interactive Widget

interact(update, year =(data.index.min(), data.index.max(),1))         



# Mapping biggest population to size 50

# Linearly interpolating all of the one in between

size_mapper = LinearInterpolator(

    x = [data['pop'].min(), data['pop'].max()],

    y = [10,60]

)



color_mapper = CategoricalColorMapper(

    factors = data.continent.unique().tolist(),

    palette = Spectral5,

)







fig.circle('gdpPercap', 'lifeExp',

           size = {'field':'pop', 'transform':size_mapper},

           color = {'field': 'continent', 'transform': color_mapper},

           alpha = 0.6,

           legend_field = 'continent',

           source = source,

           hover_color='white',

           line_color="white"

           )



# Move Legends off Canvas

fig.legend.border_line_color = None

fig.legend.location = (0,200)

fig.right.append(fig.legend[0])



fig.axis[0].formatter = NumeralTickFormatter(format = "$0")

show(fig, notebook_handle = True)