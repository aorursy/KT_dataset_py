import pandas as pd

from bokeh.io import curdoc, output_notebook, push_notebook, show

from bokeh.plotting import figure

from bokeh.layouts import column, widgetbox

from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper

from bokeh.palettes import plasma, Category10

from ipywidgets import interact



output_notebook()



datafile1 = '../input/rk_data_cleaned.csv'

rk_data = pd.read_csv(datafile1, parse_dates=True, index_col='Date')



rk_data.loc[:, 'Duration'] = pd.to_datetime(rk_data['Duration'], format='%Y-%m-%d %H:%M:%S')

rk_data.loc[:, 'Average Pace'] = pd.to_datetime(rk_data['Average Pace'], format='%Y-%m-%d %H:%M:%S')



src_nm = ColumnDataSource()

src_nm.data = {

    'x_ax': rk_data.index,

    'y_ax': rk_data['Average Speed (km/h)'],

    'average_speed': rk_data['Average Speed (km/h)'],

    'duration': rk_data['Duration'],

    'average_pace': rk_data['Average Pace'],

    'distance': rk_data['Distance (km)'],

    'climb': rk_data['Climb (m)'],

    'ahr': rk_data['Average Heart Rate (bpm)'],

    'type': rk_data['Type'],

}



atypes = list(rk_data['Type'].unique())

my_num_axes = ['Distance (km)', 'Average Speed (km/h)', 'Climb (m)', 'Average Heart Rate (bpm)']



mapper = CategoricalColorMapper(

    factors=atypes,

    # palette=plasma(len(atypes)),

    palette=Category10[len(atypes)],

)

hvr = HoverTool()

hvr.tooltips = [

    ('Activity', '@type'),

    ('Date', '@x_ax{%F}'),

    ('Distance', '@distance{0.00} km'),

    ('Duration', '@duration{%H:%M:%S}'),

    ('Average Speed', '@average_speed{0.00} km/h')

]



hvr.formatters = {

    'x_ax': 'datetime',

    'duration': 'datetime',

    'average_pace': 'datetime',

}



toolbox = ['pan', 'box_zoom', 'reset', 'crosshair', hvr]



plot1 = figure(plot_width=700,

               x_axis_label='Date',

               x_axis_type='datetime',

               tools=toolbox,

               toolbar_location='above',

               background_fill_color='#bbbbbb',

               )



plot1.yaxis.axis_label = 'Average speed (km/h)'

title_main = 'Training activities historical data: '

plot1.title.text = title_main + 'Average speed (km/h)'



plot1.circle(x='x_ax', y='y_ax', source=src_nm,

             size=5,

             color={'field': 'type', 'transform': mapper},

             # fill_color='white',

             legend='type'

             )

plot1.legend.location = 'top_left'

plot1.legend.click_policy = "hide"





def update_plot2(ax='Average Speed (km/h)'):

    """

    Define the callback: update_plot

    """

    y = ax

    plot1.yaxis.axis_label = y



    # Set new_data

    new_data1 = {

        'x_ax': rk_data.index,

        'y_ax': rk_data[y],

        'duration': rk_data['Duration'],

        'average_speed': rk_data['Average Speed (km/h)'],

        'average_pace': rk_data['Average Pace'],

        'distance': rk_data['Distance (km)'],

        'climb': rk_data['Climb (m)'],

        'ahr': rk_data['Average Heart Rate (bpm)'],

        'type': rk_data['Type'],

    }



    src_nm.data = new_data1  # Assign new_data to source.data



    # Set the range of all axes

    plot1.y_range.start = min(rk_data[y])*1.03

    plot1.y_range.end = max(rk_data[y])*1.03



    plot1.title.text = title_main + y  # Add title to plot



    #  Set new tooltips with selected parameter at the end, except distance, which is displayed always

    if y != "Distance (km)":

        hvr.tooltips = [

            ('Activity', '@type'),

            ('Date', '@x_ax{%F}'),

            ('Distance', '@distance{0.00} km'),

            ('Duration', '@duration{%H:%M:%S}'),

            (y, '@y_ax{0.00}')

        ]

    else:

        hvr.tooltips = [

            ('Activity', '@type'),

            ('Date', '@x_ax{%F}'),

            ('Distance', '@distance{0.00} km'),

            ('Duration', '@duration{%H:%M:%S}'),

        ]

    push_notebook()



# layout = column(widgetbox(menu1), plot1)

layout = column(plot1)



# show(layout)

s = show(layout, notebook_handle=True)

menu = interact(update_plot2, ax=my_num_axes)
