# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Data Handling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Bokeh Libraries

from bokeh.plotting import figure, show

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.layouts import row, column, gridplot

from bokeh.io import output_file, output_notebook



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
output_notebook()
# prepare the data

path_to_data = '/kaggle/input/heart-disease-uci/heart.csv'

heart_data = pd.read_csv(path_to_data)

print(type(heart_data))

heart_data.head()
heart_cds = ColumnDataSource(heart_data)
# set-up the figure

fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            toolbar_location=None)
# Connect to and draw the data

fig.circle(x='age', y='chol', source=heart_cds)
# Preview and Save the Figure

show(fig)
# Prepare the data

# Create the cds for each outcome

disease_data = heart_data[heart_data['target']==1]

healthy_data = heart_data[heart_data['target']==0]
# set-up the figure

fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            toolbar_location=None)
# Connect to and draw the data

fig.circle(x='age', y='chol', source=disease_data, color='red', size=7, legend_label='Heart Disease')

fig.circle(x='age', y='chol', source=healthy_data, color='green', size=7, legend_label='False Alarm')
# Preview and Save the Figure

show(fig)
# Prepare the data

# Create the cds for each outcome

female_disease_data = disease_data[disease_data['sex']==0]

male_disease_data = disease_data[disease_data['sex']==1]

female_healthy_data = healthy_data[healthy_data['sex']==0]

male_healthy_data = healthy_data[healthy_data['sex']==1]
# set-up the figure

fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            toolbar_location=None)
# Connect to and draw the data

fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease')

fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease')

fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm')

fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm')
# Preview and Save the Figure

show(fig)
# Prepare the data

# Create the cds for each outcome

female_disease_data = disease_data[disease_data['sex']==0]

male_disease_data = disease_data[disease_data['sex']==1]

female_healthy_data = healthy_data[healthy_data['sex']==0]

male_healthy_data = healthy_data[healthy_data['sex']==1]



# set-up the figure

fig = figure(title="Bokeh Visualisation of UCI Heart Disease Data",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            toolbar_location=None)
# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



# add a policy to the legend what should happen when it is clicked

fig.legend.click_policy = 'mute' #'hide'
# Preview and Save the Figure

show(fig)
fig_hover = figure(title="Bokeh Visualisation of UCI Heart Disease Data",

                    plot_height=400, plot_width=600,

                    x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

                    toolbar_location=None)

# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

fig_hover.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

fig_hover.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

fig_hover.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

fig_hover.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



# add a policy to the legend what should happen when it is clicked

fig_hover.legend.click_policy = 'mute' #'hide'
# define information to display

tool_tips = [

             ('Age', '@age'),

             ('Sex', '@sex'),

             ('Pain Type', '@cp'),

             ('Max Heartrate', '@thalach')

            ]



# connect to the figure

fig_hover.add_tools(HoverTool(tooltips=tool_tips))
# Preview and Save the Figure

show(fig_hover)
# Creating the plots we actualy already have

# set-up the figure

cholesterol_fig = figure(title="Correlation between age and cholesterol",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            toolbar_location=None)

# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

cholesterol_fig.circle(x='age', y='chol', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

cholesterol_fig.diamond(x='age', y='chol', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

cholesterol_fig.circle(x='age', y='chol', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

cholesterol_fig.diamond(x='age', y='chol', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



cholesterol_fig.legend.click_policy = 'mute'
# Create the new plot

# set-up the figure

bloodpres_fig = figure(title="Correlation between age and blood preasure",

            plot_height=400, plot_width=600,

            x_axis_label='age [y]', y_axis_label='blood preasure [mm Hg]',

            toolbar_location=None)

# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

bloodpres_fig.circle(x='age', y='trestbps', source=female_disease_data, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

bloodpres_fig.diamond(x='age', y='trestbps', source=male_disease_data, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

bloodpres_fig.circle(x='age', y='trestbps', source=female_healthy_data, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

bloodpres_fig.diamond(x='age', y='trestbps', source=male_healthy_data, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



bloodpres_fig.legend.click_policy = 'mute'
# Organize the layout

row_layout = row([cholesterol_fig, bloodpres_fig])

# Preview and Save the Figure

show(row_layout)
# Prepare the data

# Create the cds for each outcome

female_disease_cds = ColumnDataSource(disease_data[disease_data['sex']==0])

male_disease_cds = ColumnDataSource(disease_data[disease_data['sex']==1])

female_healthy_cds = ColumnDataSource(healthy_data[healthy_data['sex']==0])

male_healthy_cds = ColumnDataSource(healthy_data[healthy_data['sex']==1])
# Specify the tools

tool_list = ['pan', 'wheel_zoom', 'box_select', 'reset']



# Creating the plots we actualy already have

# set-up the figure

cholesterol_fig = figure(title="Correlation between age and cholesterol",

            plot_height=300, plot_width=900,

            x_axis_label='age [y]', y_axis_label='cholesterol [mg/dl]',

            tools=tool_list)



# set-up the figure

bloodpres_fig = figure(title="Correlation between age and blood preasure",

            plot_height=300, plot_width=900,

            x_axis_label='age [y]', y_axis_label='blood preasure [mm Hg]',

            tools=tool_list)


# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

cholesterol_fig.circle(x='age', y='chol', source=female_disease_cds, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

cholesterol_fig.diamond(x='age', y='chol', source=male_disease_cds, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

cholesterol_fig.circle(x='age', y='chol', source=female_healthy_cds, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

cholesterol_fig.diamond(x='age', y='chol', source=male_healthy_cds, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



cholesterol_fig.legend.click_policy = 'mute'



# Create the new plot



# adding interactivity in Bokeh is quiet simple. For now, let's mute the data points if we click on the legend

# Connect to and draw the data

bloodpres_fig.circle(x='age', y='trestbps', source=female_disease_cds, color='red', size=7, legend_label='Female Heart Disease', muted_alpha=0.1)

bloodpres_fig.diamond(x='age', y='trestbps', source=male_disease_cds, color='red', size=10, legend_label='Male Heart Disease', muted_alpha=0.1)

bloodpres_fig.circle(x='age', y='trestbps', source=female_healthy_cds, color='green', size=7, legend_label='Female False Alarm', muted_alpha=0.1)

bloodpres_fig.diamond(x='age', y='trestbps', source=male_healthy_cds, color='green', size=10, legend_label='Male False Alarm', muted_alpha=0.1)



bloodpres_fig.legend.click_policy = 'mute'

#now we link together the x-axes

cholesterol_fig.x_range = bloodpres_fig.x_range
# Aligne the figures column wise

column_layout = column([cholesterol_fig, bloodpres_fig])

# Preview and Save the Figure

show( column_layout )