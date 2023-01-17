import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from bokeh.plotting import figure, show, output_notebook

from bokeh.models import ColumnDataSource, HoverTool



output_notebook()



# import data

data = pd.read_csv("../input/restaurant-scores-lives-standard.csv")



# convert 'inspection date' to datetime format

data['inspection_date'] = pd.to_datetime(data['inspection_date'])



# sort

data = data.sort_values(by=['inspection_date'])
# get the newest data on number of restaurants with high risks violations

new_high_risk = data[(data['inspection_date'] == data['inspection_date'].iloc[-1]) & (data['risk_category'] == 'High Risk')]



# number of inspections

print("Number of inspections overall:", data['inspection_id'].nunique())



# newest inpsection with high risks

print("Latest inspections:", data['inspection_date'].iloc[-1], ", Number of restaurants with high risk violations: ", new_high_risk['business_id'].nunique())
# number of violations in each category by day



# create dataframes for each type of risks

low_risk = data[data['risk_category'] == 'Low Risk']

low_risk = low_risk[['inspection_date', 'risk_category']].groupby('inspection_date')['risk_category'].count().reset_index()



moderate_risk = data[data['risk_category'] == 'Moderate Risk']

moderate_risk = moderate_risk[['inspection_date', 'risk_category']].groupby('inspection_date')['risk_category'].count().reset_index()



high_risk = data[data['risk_category'] == 'High Risk']

high_risk = high_risk[['inspection_date', 'risk_category']].groupby('inspection_date')['risk_category'].count().reset_index()



# plot graph

TOOLS = 'crosshair,save,pan,box_zoom,reset,wheel_zoom'

p = figure(title="Number of violations by category per day", y_axis_type="linear", plot_height = 400, tools = TOOLS, plot_width = 800,

          x_axis_type="datetime")

p.xaxis.axis_label = 'Date'

p.yaxis.axis_label = 'Total violations by category'



p.line(low_risk['inspection_date'], low_risk['risk_category'], color='green', alpha=0.5, legend='Low Risk')



p.line(moderate_risk['inspection_date'], moderate_risk['risk_category'], color='orange', alpha=0.5, legend='Moderate Risk')



p.line(high_risk['inspection_date'], high_risk['risk_category'], color='red', alpha=0.5, legend='High Risk')



p.legend.location = "top_left"



p.add_tools(HoverTool(

    tooltips=[

        ('date', '@x{%F}'),

        ('Number of violations', '@y')

    ],

    formatters={

        'x': 'datetime', # use 'datetime' formatter for 'date' field

                                  # use default 'numeral' formatter for other fields

    },

    # display a tooltip whenever the cursor is vertically in line with a glyph

    mode='vline'

))



show(p)
# number of inspections by years/month/day

inspection = data[['inspection_date', 'inspection_id']].drop_duplicates()



x = inspection.groupby('inspection_date')['inspection_id'].nunique().reset_index()



# plot graph

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(title="Number of inspections per day", y_axis_type="linear", plot_height = 400, tools = TOOLS, plot_width = 800,

          x_axis_type="datetime")

p.xaxis.axis_label = 'Date'

p.yaxis.axis_label = 'Total Inspections'



p.line(x['inspection_date'], x['inspection_id'], color='navy', alpha=0.5)



p.add_tools(HoverTool(

    tooltips=[

        ('date', '@x{%F}'),

        ('Number of inspections', '@y')

    ],

    formatters={

        'x': 'datetime', # use 'datetime' formatter for 'date' field

                                  # use default 'numeral' formatter for other fields

    },

    # display a tooltip whenever the cursor is vertically in line with a glyph

    mode='vline'

))



show(p)
# inspection_score: median/mean overall/by year/by month/by day/by neighborhood



score = data[['inspection_date', 'inspection_score']].drop_duplicates().dropna()



y = score.groupby('inspection_date')['inspection_score'].mean().reset_index()



# plot graph

TOOLS = 'save,pan,box_zoom,reset,wheel_zoom'

p = figure(title="Average inspection score per day", y_axis_type="linear", plot_height = 400, tools = TOOLS, plot_width = 800,

          x_axis_type="datetime")

p.xaxis.axis_label = 'Date'

p.yaxis.axis_label = 'Average score'



p.line(y['inspection_date'], y['inspection_score'], color='navy', alpha=0.5)



p.add_tools(HoverTool(

    tooltips=[

        ('date', '@x{%F}'),

        ('Average score', '@y')

    ],

    formatters={

        'x': 'datetime', # use 'datetime' formatter for 'date' field

                                  # use default 'numeral' formatter for other fields

    },

    # display a tooltip whenever the cursor is vertically in line with a glyph

    mode='vline'

))



show(p)
# Bins will be 5 scores, so the number of bins is (length of interval / 5). Min and max scores are used for the range

#max score always will be 100, min score could change



min_score = data['inspection_score'].min()

score_hist, edges = np.histogram(data['inspection_score'].dropna(), 

                                 bins = int((100 - min_score) / 5),

                                 range = [min_score, 100])



# Put the information in a dataframe



scores = pd.DataFrame({'scores': score_hist,

                       'left': edges[:-1],

                       'right': edges[1:]})



# Add a column showing the extent of each interval

scores['s_interval'] = ['%d to %d' % (left, right) for left, right in zip(scores['left'], scores['right'])]



# Convert dataframe to a ColumnDataSource

src = ColumnDataSource(scores)



# Create histogram using Bokeh

p = figure(plot_height = 600, plot_width = 600, 

           title = 'Histogram of Inspection Score',

           x_axis_label = 'Inspection score', 

           y_axis_label = 'Number of inspections')



# Add a quad glyph

p.quad(source=src, bottom=0, top='scores', 

       left='left', right='right', 

       fill_color='red', line_color='black', fill_alpha = 0.75,

       hover_fill_alpha = 1.0, hover_fill_color = 'navy')



# add HoverTool

hover = HoverTool(tooltips=[('Scores', '@s_interval'),

                        ('Num of Restaurants', '@scores')])



p.add_tools(hover)



# Show the plot

show(p)