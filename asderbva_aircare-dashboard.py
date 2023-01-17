import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from bokeh.plotting import figure

from bokeh.models import Span, Band, Label

from bokeh.io import output_file, show, output_notebook
def dummy_df():

    # today's datetime

    month_today = datetime.now()

    months = pd.date_range(month_today, month_today + timedelta(720), freq='M')



    #randomising the sensor data

    np.random.seed(seed=2020)

    data = np.random.randint(1, high=80, size=len(months))

    

    #creating dataframe out of the random data

    df = pd.DataFrame({'time':months, 'pm2.5':data})

    return df
def create_pm25_cat_hlines():

    # common data for the categories

    data = [

        {'cat':'Good', 'min':0, 'max':12.0, 'line_color':'green'},

        {'cat':'Moderate', 'min':12.1, 'max':35.4, 'line_color':'yellowgreen'},

        {'cat':'Unhealthy (S)', 'min':35.5, 'max':55.4, 'line_color':'gold'},

        {'cat':'Unhealthy', 'min':55.5, 'max':150.4, 'line_color':'green'}

        

    ]

    

    spans = []

    labels = []

    

    for v in data:

        # Code for the spans

        spans.append(Span(location=v['max'], dimension='width', 

                              line_color=v['line_color'], line_width=3, 

                              line_dash='dashed'))

        # Code for the labels

        labels.append(Label, x=0, y=v['max'], text = v['cat'], 

                          text_color = v['line_color'])

            

        p.renderers.extend(spans)

        

        for label in labels:

            p.add_layout(label)

            
def readings_plot(df):

    #configuring the figure

    p = figure(x_axis_type='datetime', x_axis_label='Time', y_axis_label = 'PM2.5', 

               title='PM2.5 vs Month-Year', title_location = 'right')



    #Scatter Plots for the individual sensor points

    p.circle(df['Time'], df['PM2.5'], size=10, color='blue', alpha=0.75)



    #line plots for the continuity

    p.line(df['Time'], df['PM2.5'], color='blue', alpha=0.75, legend='PM2.5_Sensor1')

    

    return output_file('plot.html') 