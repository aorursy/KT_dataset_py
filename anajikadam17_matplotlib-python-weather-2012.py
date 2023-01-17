# Import the required Libraries

from matplotlib import pyplot as plt

import numpy as np



# Set it up to work with Jupyter Notebook

# (this is a Jupyter command, not Python code)

%matplotlib inline
# Create data



def create_sample_chart():

    x_series = np.arange(10)

    y_series = x_series ** 2



    #Plotting to our canvas in memory

    plt.plot(x_series, y_series)

    #Title of our canvas

    plt.title('String Title Here')

    #X-axis label

    plt.xlabel('X Axis Title Here')

    #Y_axis label

    plt.ylabel('Y Axis Title Here')



    #Showing what we plotted

    plt.show();

    

create_sample_chart()
plt.figure()  # create a plot figure

x = np.linspace(0, 10, 100)

# create the first of two panels and set current axis

plt.subplot(2, 1, 1) # (rows, columns, panel number)

plt.plot(x, np.sin(x))



# create the second panel and set current axis

plt.subplot(2, 1, 2)

plt.plot(x, np.cos(x));

plt.show()
# First create a grid of plots

# ax will be an array of two Axes objects

fig, ax = plt.subplots(2)



# Call plot() method on the appropriate object

ax[0].plot(x, np.sin(x))

ax[1].plot(x, np.cos(x));
x = np.linspace(0, 10, 30)

y = np.sin(x)



plt.plot(x, y, 'o', color='black');

plt.show()
plt.plot(x, y, '-ok');

plt.show()
# Data Introduction

from matplotlib import pyplot as plt

import numpy as np



import pandas as pd

weather_df = pd.read_csv('../input/weather_2012.csv', parse_dates=True, index_col='Date/Time')

weather_df.head(5)
monthly_data = weather_df.groupby(weather_df.index.month).mean()

monthly_data.head(3)

x_series = monthly_data.index

y_series = monthly_data['Temp (C)']
plt.plot(x_series, y_series)



plt.title('Temperature Trend, 2012')

plt.xlabel('Month')

plt.ylabel('Temp (C)')

plt.show()
# First, get calendar month names

import calendar

calendar_months = calendar.month_name[1:]





print(calendar_months)
x=calendar_months

y=monthly_data['Temp (C)']

plt.plot(x,y)

plt.title('Temperature Trend, 2012')

plt.xlabel('Month')

plt.xticks(rotation=90)

plt.ylabel('Temp (C)')

plt.show()
import matplotlib.pyplot as plt

import calendar



def bar_plot():

    weekly_data = weather_df.groupby(weather_df.index.dayofweek).mean()

    

    plt.bar(weekly_data.index, weekly_data['Visibility (km)'])



    plt.title('Visibility by week, 2012')

    plt.xlabel('Day of week')

    plt.ylabel('Visibility (km)')



    plt.xticks(weekly_data.index, calendar.day_abbr, rotation=45)



    plt.show()



    

bar_plot()
# Sample histogram 



x = np.arange(0, 10, 0.1)

y1 = (((x - 3) ** 3 ) - 100) + np.random.randint(-20, 20, size=len(x))



plt.hist(y1)

plt.show()
def hist_plot():

    plt.hist(weather_df['Wind Spd (km/h)'])

    plt.xlabel('Wind Spd (km/h)')

    plt.ylabel('Frequency');

    

hist_plot()
# Sample boxplot 



x= np.arange(0, 10, 0.1)

y = np.exp(x)



plt.boxplot(y)

plt.show()
def box_plot():

    plt.boxplot(weather_df['Wind Spd (km/h)']);



box_plot()
# Sample scatter plot



x= np.arange(0, 10, 0.1)

y1 = (((x - 3) ** 3 ) - 100) + np.random.randint(-20, 20, size=len(x))

y2 = (((3 - x) ** 3 ) + 50) + np.random.randint(-20, 20, size=len(x))



plt.scatter(x, y1, c='r')

plt.scatter(x, y2, c='b')

plt.show()
jan_df = weather_df['2012-01']



def scatter_plot():

    plt.scatter(x=jan_df['Temp (C)'],y=jan_df['Stn Press (kPa)'])   

    plt.xlabel('Temp (C)')

    plt.ylabel('Pressure')

    plt.title('Corelation between Temperature and Pressure for January')

    

scatter_plot() # Scatter plot below shows a negative correlation.
fig, ax = plt.subplots()

ax.plot(x, x**2, 'b.-') # blue line with dots

ax.plot(x, x**2.5, 'g--') # green dashed line

ax.plot(x, x**3, c='r') # red line color

fig.show()
def two_plots():

    x = np.array([0, 1, 2, 3, 4, 5])

    y = x ** 2



    # Create Figure (empty canvas)

    fig = plt.figure()



    # Add set of axes to figure

    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

    axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes 

    # (0.2x left, 0.5x bottom) to (0.4x width, 0.3x height)



    # Larger Figure Axes 1

    axes1.plot(x, y, 'b')

    axes1.set_xlabel('X_label_axes1')

    axes1.set_ylabel('Y_label_axes1')

    axes1.set_title('Axes 1 Title')



    # Insert Figure Axes 2

    axes2.plot(y, x, 'r')

    axes2.set_xlabel('X_label_axes2')

    axes2.set_ylabel('Y_label_axes2')

    axes2.set_title('Axes 2 Title');  

    

two_plots()
# Canvas of 2 by 2 subplots

fig, axes = plt.subplots(nrows=2, ncols=2)



# axes is an array of shape (2, 2)
x = np.arange(0,10,0.1)



fig,ax = plt.subplots(nrows=2,ncols=2)

plt.subplot(2,2,1)

plt.plot(x,x**2,c='r')

plt.subplot(2,2,4)

plt.plot(x,x**2,'g--');