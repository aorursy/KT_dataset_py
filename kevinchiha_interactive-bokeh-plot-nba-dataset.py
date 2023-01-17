import numpy as np

import pandas as pd

from bokeh.plotting import figure, ColumnDataSource

from bokeh.models import CategoricalColorMapper, HoverTool, Slider, Select

from bokeh.layouts import row, column, widgetbox

from bokeh.io import curdoc



# Read in the Datasets

players = pd.read_csv('../input/Players.csv')

seasons = pd.read_csv('../input/Seasons_Stats.csv')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
seasons.columns
seasons.drop(seasons.columns[0], axis=1, inplace=True)
seasons.head()

seasons.info()
seasons_1990_on = seasons[seasons.Year >= 1990]

seasons_1990_on = seasons_1990_on[(seasons_1990_on['3P'].notnull()) &

                                  (seasons_1990_on['3P'] > 0) &

                                  (seasons_1990_on['3PA'].notnull()) &

                                  (seasons_1990_on['3PA'] > 0) &

                                  (seasons_1990_on['2PA'].notnull()) &

                                  (seasons_1990_on['2PA'] > 0) &

                                  (seasons_1990_on['2P'].notnull()) &

                                  (seasons_1990_on['2P'] > 0)]

# Reindex

new_index = np.arange(0, len(seasons_1990_on)).tolist()

seasons_1990_on.index = new_index

seasons_1990_on.head()
len(seasons_1990_on['Tm'].unique().tolist())
palette = ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black',

           'blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse',

           'chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue','darkcyan',

           'darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen',

           'darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue',

           'darkslategray','darkturquoise','darkviolet','red']

color_mapper = CategoricalColorMapper(factors=seasons_1990_on['Tm'].unique().tolist(),

                                      palette=palette)



p1 = figure(x_axis_label='3 Points Attempted', y_axis_label='3 Points Made', tools='box_select')

p2 = figure(x_axis_label='2 Points Attempted', y_axis_label='2 Points Made', tools='box_select')
slider = Slider(title='Year', start=1990, end=2017, step=1, value=2006)

menu = Select(options=seasons_1990_on['Tm'].unique().tolist(), value='GSW', title='Team')
source = ColumnDataSource(data={'x_3p': seasons_1990_on['3PA'], 'y_3p': seasons_1990_on['3P'],

                                'Tm': seasons_1990_on['Tm'], 'x_2p': seasons_1990_on['2PA'],

                                'y_2p': seasons_1990_on['2P'], 'Year': seasons_1990_on['Year'],

                                'Player': seasons_1990_on['Player']})
def callback(attr, old, new):

	new_x_3p = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                           (seasons_1990_on['Tm'] == menu.value)]['3PA']



	new_y_3p = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                           (seasons_1990_on['Tm'] == menu.value)]['3P']



	new_tm = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                         (seasons_1990_on['Tm'] == menu.value)]['Tm']



	new_x_2p = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                           (seasons_1990_on['Tm'] == menu.value)]['2PA']



	new_y_2p = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                           (seasons_1990_on['Tm'] == menu.value)]['2P']



	new_year = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                           (seasons_1990_on['Tm'] == menu.value)]['Year']



	new_player = seasons_1990_on[(seasons_1990_on['Year'] == slider.value) &

	                             (seasons_1990_on['Tm'] == menu.value)]['Player']



	source.data = {'x_3p': new_x_3p, 'y_3p': new_y_3p, 'Tm': new_tm, 'x_2p': new_x_2p,

	               'y_2p': new_y_2p, 'Year': new_year, 'Player': new_player}





slider.on_change('value', callback)

menu.on_change('value', callback)
p1.circle('x_3p', 'y_3p', source=source, alpha=0.8, nonselection_alpha=0.1,

          color=dict(field='Tm', transform=color_mapper), legend='Tm')



p2.circle('x_2p', 'y_2p', source=source, alpha=0.8, nonselection_alpha=0.1,

          color=dict(field='Tm', transform=color_mapper), legend='Tm')



p1.legend.location = 'bottom_right'

p2.legend.location = 'bottom_right'

hover1 = HoverTool(tooltips=[('Player', '@Player')])

p1.add_tools(hover1)

hover2 = HoverTool(tooltips=[('Player', '@Player')])

p2.add_tools(hover2)



column1 = column(widgetbox(menu), widgetbox(slider))

layout = row(column1, p1, p2)



curdoc().add_root(layout)