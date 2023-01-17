import numpy as np

import pandas as pd

import geopandas as gpd



from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS

from bokeh.layouts import row, column, layout

from bokeh.transform import cumsum, linear_cmap

from bokeh.palettes import Blues8



output_notebook()
full = pd.read_csv('../input/full.csv')

full.head()
df1 = full[['Lifeboat', 'Sex']].copy()

df1.head()
df1 = df1[(df1.Lifeboat.notna()) & (df1.Lifeboat != '?')]



df1.loc[df1.Lifeboat == '14?', 'Lifeboat'] = '14'

df1.loc[df1.Lifeboat == '15?', 'Lifeboat'] = '15'

df1.loc[df1.Lifeboat == 'A[64]', 'Lifeboat'] = 'A'



df1 = pd.get_dummies(df1, columns=['Sex'], prefix='', prefix_sep='')



df1.head()
df1 = df1.groupby('Lifeboat', as_index=False).sum()



df1.head()
order = ['7', '5', '3', '8', '1', '6', '16', '14', '12', '9',

         '11', '13', '15', '2', '10', '4', 'C', 'D', 'B', 'A']

df1 = df1.set_index('Lifeboat').reindex(order).reset_index()



df1.head()
df1['female_per'] = df1['female'] / (df1['female'] + df1['male']) * 100

df1['male_per'] = df1['male'] / (df1['female'] + df1['male']) * 100

df1[['female_per', 'male_per']] = df1[['female_per', 'male_per']].round(1)



lifeboat_odd = ['1', '3', '5', '7', '9', '11', '13', '15', 'A', 'C']

df1.loc[df1.Lifeboat.isin(lifeboat_odd), 'Side'] = 'starboard'

df1.loc[~df1.Lifeboat.isin(lifeboat_odd), 'Side'] = 'port'



df1.head()
# Create the ColumnDataSource object "s1"

s1 = ColumnDataSource(df1)



# Create the figure object "p1"

p1 = figure(title='Click on a column to display more information',

            plot_width=500, plot_height=325, x_range = s1.data['Lifeboat'],

            toolbar_location=None, tools=['hover', 'tap'], tooltips='@$name')



# Add stacked vertical bars to "p1"

p1.vbar_stack(['female', 'male'], x='Lifeboat', width=0.8, source=s1,

              fill_color=['#66c2a5', '#fc8d62'], line_color=None, legend=['Female', 'Male'])



# Change parameters of "p1"

p1.title.align = 'center'

p1.xaxis.axis_label = 'Lifeboat (in launch order)'

p1.yaxis.axis_label = 'Count'

p1.y_range.start = 0

p1.x_range.range_padding = 0.05

p1.xgrid.grid_line_color = None

p1.legend.orientation = 'horizontal'

p1.legend.location = 'top_left'



# Create the Div object "div1"

div1 = Div()



# Create the custom JavaScript callback

callback1 = CustomJS(args=dict(s1=s1, div1=div1), code='''

    var ind = s1.selected.indices;

    if (String(ind) != '') {

        lifeboat = s1.data['Lifeboat'][ind];

        female = s1.data['female'][ind];

        male = s1.data['male'][ind];

        female_per = s1.data['female_per'][ind];

        male_per = s1.data['male_per'][ind];

        side = s1.data['Side'][ind];

        message = '<b>Lifeboat: ' + String(lifeboat) + ' (' + String(side) + ' side)' + '</b><br>Females: ' + String(female) + ' (' + String(female_per) +  '%)' + '<br>Males: ' + String(male) + ' (' + String(male_per) +  '%)' + '<br>Total: ' + String(female+male);

        div1.text = message;

    }

    else {

        div1.text = '';

    }

''')        



# When tapping the plot "p1" execute the "callback1"

p1.js_on_event('tap', callback1)



# Display "p1" and "div1" as a row

show(row(p1, div1))
df2 = full[['Lifeboat', 'Pclass']].copy()



df2 = df2[(df2.Lifeboat.notna()) & (df2.Lifeboat != '?')]



df2.loc[df2.Lifeboat == '14?', 'Lifeboat'] = '14'

df2.loc[df2.Lifeboat == '15?', 'Lifeboat'] = '15'

df2.loc[df2.Lifeboat == 'A[64]', 'Lifeboat'] = 'A'



df2 = pd.get_dummies(df2, columns=['Pclass'], prefix='', prefix_sep='')



df2 = df2.groupby('Lifeboat', as_index=False).sum()



order = ['7', '5', '3', '8', '1', '6', '16', '14', '12', '9',

         '11', '13', '15', '2', '10', '4', 'C', 'D', 'B', 'A']

df2 = df2.set_index('Lifeboat').reindex(order).reset_index()



df2.head()
df2['1_per'] = df2['1'] / (df2['1'] + df2['2'] + df2['3']) * 100

df2['2_per'] = df2['2'] / (df2['1'] + df2['2'] + df2['3']) * 100

df2['3_per'] = df2['3'] / (df2['1'] + df2['2'] + df2['3']) * 100





df2['1_ang'] = df2['1_per'] / 100 * 2 * np.pi

df2['2_ang'] = df2['2_per'] / 100 * 2 * np.pi

df2['3_ang'] = df2['3_per'] / 100 * 2 * np.pi



df2.head()
df2_plot=pd.DataFrame({'class': ['Class 1', 'Class 2', 'Class 3'],

                       'percent': [float('nan'), float('nan'), float('nan')],

                       'angle': [float('nan'), float('nan'), float('nan')],

                       'color': ['#c9d9d3', '#718dbf', '#e84d60']})

df2_plot
# Create the ColumnDataSource objects "s2" and "s2_plot"

s2 = ColumnDataSource(df2)

s2_plot = ColumnDataSource(df2_plot)



# Create the Figure object "p2"

p2 = figure(plot_width=275, plot_height=350, y_range=(-0.5, 0.7),

            toolbar_location=None, tools=['hover'], tooltips='@percent{0.0}%')



# Add circular sectors to "p2"

p2.wedge(x=0, y=0, radius=0.8, source=s2_plot,

         start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),

         fill_color='color', line_color=None, legend='class')



# Change parameters of "p2"

p2.axis.visible = False

p2.grid.grid_line_color = None

p2.legend.orientation = 'horizontal'

p2.legend.location = 'top_center'



# Create the custom JavaScript callback

callback2 = CustomJS(args=dict(s2=s2, s2_plot=s2_plot), code='''

    var ang = ['1_ang', '2_ang', '3_ang'];

    var per = ['1_per', '2_per', '3_per'];

    if (cb_obj.value != 'Please choose...') {

        var boat = s2.data['Lifeboat'];

        var ind = boat.indexOf(cb_obj.value);

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = s2.data[ang[i]][ind];

            s2_plot.data['percent'][i] = s2.data[per[i]][ind];

        }

    }

    else {

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = undefined;

            s2_plot.data['percent'][i] = undefined;

        }



    }

    s2_plot.change.emit();

''')



# When changing the value of the dropdown menu execute "callback2"

options = ['Please choose...'] + list(s2.data['Lifeboat'])

select = Select(title='Lifeboat (in launch order)', value=options[0], options=options)

select.js_on_change('value', callback2)



# Display "select" and "p2" as a column

show(column(select, p2))
gdf = gpd.read_file('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson')

gdf.head()
gdf = gdf[gdf.NAME != 'Antarctica']

gdf.plot(figsize=(10, 5));
xs = []

ys = []

for obj in gdf.geometry.boundary:

    if obj.type == 'LineString':

        obj_x, obj_y = obj.xy

        xs.append([[list(obj_x)]])

        ys.append([[list(obj_y)]])

    elif obj.type == 'MultiLineString':

        obj_x = []

        obj_y = []

        for line in obj:

            line_x, line_y = line.xy

            obj_x.append([list(line_x)])

            obj_y.append([list(line_y)])

        xs.append(obj_x)

        ys.append(obj_y)



country = gdf['NAME'].values        



df3_plot = pd.DataFrame({'country': country, 'xs': xs, 'ys': ys, 'count': float('nan')})



df3_plot.head()
# Create the ColumnDataSource object "s3_plot"

s3_plot = ColumnDataSource(df3_plot)



# Create the Figure object "p3_test"

p3_test = figure(plot_width=775, plot_height=350,

                 toolbar_location=None, tools=['hover', 'pan', 'wheel_zoom'],

                 active_scroll='wheel_zoom', tooltips='@country')



# Add multipolygons to "p3_test"

p3_test.multi_polygons(xs='xs', ys='ys', fill_color='count', source=s3_plot)



# Change parameters of "p3_test"

p3_test.axis.visible = False

p3_test.grid.grid_line_color = None



# Create the custom JavaScript callback

callback3_test = CustomJS(args=dict(p3_test=p3_test), code='''

    p3_test.reset.emit();

''')    



# When clicking on the button execute "callback3_test"

button = Button(label='Reset view')

button.js_on_click(callback3_test)



# Display "p3_test" and "button" as a column

show(column(p3_test, button))
df3 = full[['Lifeboat', 'Hometown']].copy()

df3.head()
temp = df3.Hometown.str.extract(r'(?P<Town>.*)\, (?P<Country>.*$)')

df3['Home_country'] = temp['Country']

df3 = df3.drop('Hometown', axis=1, errors='ignore')



df3.head()
df3 = df3[(df3.Lifeboat.notna()) & (df3.Lifeboat != '?')]



df3.loc[df3.Lifeboat == '14?', 'Lifeboat'] = '14'

df3.loc[df3.Lifeboat == '15?', 'Lifeboat'] = '15'

df3.loc[df3.Lifeboat == 'A[64]', 'Lifeboat'] = 'A'



df3.head()
df3.Home_country.unique()
to_replace = [('US', 'United States of America'), ('UK[note 3]', 'India'),

              ('England', 'United Kingdom'), ('UK', 'United Kingdom'),

              ('Channel Islands', 'United Kingdom'), ('Siam', 'Thailand'),

              ('Syria[81]', 'Syria'), ('Scotland', 'United Kingdom'),

              ('British India', 'India'), ('Ireland[note 1]', 'Ireland'),

              ('Russian Empire', 'Russia'), ('Russian Empire[note 6]', 'Finland'),

              ('Siam[note 5]', 'Thailand'), ('German Empire[note 2]', 'Germany'),

              ('British India[note 3]', 'India')]



for old, new in to_replace:

    df3.loc[df3.Home_country == old, 'Home_country'] = new

    

df3.Home_country.unique()
df3 = pd.get_dummies(df3, columns=['Home_country'], prefix='', prefix_sep='')



df3 = df3.groupby('Lifeboat', as_index=False).sum()



order = ['7', '5', '3', '8', '1', '6', '16', '14', '12', '9',

         '11', '13', '15', '2', '10', '4', 'C', 'D', 'B', 'A']

df3 = df3.set_index('Lifeboat').reindex(order).reset_index()



df3.head()
country = df3_plot['country']

diff = country[~country.isin(df3.columns)].values

df3 = pd.concat([df3, pd.DataFrame(columns=diff)], axis=1).fillna(0)

df3 = df3.loc[:, np.append(['Lifeboat'], country.values)]



df3.head()
# Create the ColumnDataSource objects "s3_plot" and "s3"

s3_plot = ColumnDataSource(df3_plot)

s3 = ColumnDataSource(df3)



# Reverse the palette and create a linear color map

Blues8.reverse()

cmap = linear_cmap('count', palette=Blues8, low=0, high=1)



# Create the Figure object "p3"

p3 = figure(plot_width=775, plot_height=350,

            toolbar_location=None, tools=['hover', 'pan', 'wheel_zoom'],

            active_scroll='wheel_zoom', tooltips='@country: @count')



# Add multipolygons to "p3"

p3.multi_polygons(xs='xs', ys='ys', fill_color=cmap, source=s3_plot)



# Change parameters of "p3"

p3.axis.visible = False

p3.grid.grid_line_color = None



# Create the custom JavaScript callbacks

callback3_select = CustomJS(args=dict(s3=s3, s3_plot=s3_plot), code='''

    var country = s3_plot.data['country'];

    if (cb_obj.value != 'Please choose...') {

        var boat = s3.data['Lifeboat'];

        var ind = boat.indexOf(cb_obj.value);

        for (i = 0; i < country.length; i++) {

            s3_plot.data['count'][i] = s3.data[country[i]][ind];

        }

    }

    else {

        for (i = 0; i < country.length; i++) {

            s3_plot.data['count'][i] = undefined;

        }

    }

    s3_plot.change.emit();

''')



callback3_button = CustomJS(args=dict(p3=p3), code='''

    p3.reset.emit();

''')

    

# When changing the value of the dropdown menu execute "callback3_select"

options = ['Please choose...'] + list(s3.data['Lifeboat'])

select = Select(title='Lifeboat (in launch order)', value=options[0], options=options)

select.js_on_change('value', callback3_select)



# When clicking on the reset button execute "callback3_button"

button = Button(label='Reset view')

button.js_on_click(callback3_button)



# Display "select", "p3", and "button" as a column

show(column(select, p3, button))
# Create the custom JavaScript callback

callback4 = CustomJS(args=dict(s1=s1, s2=s2, s3=s3, s2_plot=s2_plot, s3_plot=s3_plot), code='''

    var ind = s1.selected.indices;

    var ang = ['1_ang', '2_ang', '3_ang'];

    var per = ['1_per', '2_per', '3_per'];

    var country = s3_plot.data['country'];

    if (String(ind) != '') {

        for (i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = s2.data[ang[i]][ind];

            s2_plot.data['percent'][i] = s2.data[per[i]][ind];

        }

        for (i = 0; i < country.length; i++) {

            s3_plot.data['count'][i] = s3.data[country[i]][ind];

        }

    }

    else {

        for (i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = undefined;

            s2_plot.data['percent'][i] = undefined;

        }

        for (i = 0; i < country.length; i++) {

            s3_plot.data['count'][i] = undefined;

        }

    }

    s2_plot.change.emit();

    s3_plot.change.emit();

''')    

    

# When tapping the plot "p1" execute "callback4"

p1.js_on_event('tap', callback4)



# Display "p1","p2", "p3" and "button" in the specified layout

l = layout([[p1, p2], [p3], [button]])

show(l)