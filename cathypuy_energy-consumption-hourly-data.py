import pandas as pd

import matplotlib as plt

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Y = pd.read_csv('/kaggle/input/hourly-energy-consumption/DAYTON_hourly.csv', index_col=0)

Y = Y.sort_values(by=['Datetime'])

Y.index = pd.to_datetime(Y.index)

Y['Month'] = Y.index.month

Y['Year'] = Y.index.year
def by_hour(Y) :

    # Acquire distributions of energy consumption for each hour of day

    midnite=Y.at_time('00:00') 

    one_am=Y.at_time('01:00') 

    two_am=Y.at_time('02:00') 

    three_am=Y.at_time('03:00') 

    four_am=Y.at_time('04:00') 

    five_am=Y.at_time('05:00') 

    six_am=Y.at_time('06:00') 

    seven_am=Y.at_time('07:00') 

    eight_am=Y.at_time('08:00') 

    nine_am=Y.at_time('09:00') 

    ten_am=Y.at_time('10:00') 

    eleven_am=Y.at_time('11:00') 

    noon=Y.at_time('12:00') 

    one_pm=Y.at_time('13:00') 

    two_pm=Y.at_time('14:00') 

    three_pm=Y.at_time('15:00') 

    four_pm=Y.at_time('16:00') 

    five_pm=Y.at_time('17:00') 

    six_pm=Y.at_time('18:00') 

    seven_pm=Y.at_time('19:00') 

    eight_pm=Y.at_time('20:00') 

    nine_pm=Y.at_time('21:00') 

    ten_pm=Y.at_time('22:00') 

    eleven_pm=Y.at_time('23:00')

    

    #specify plot will contain subplots, here there are 24 subplots

    fig, ax = plt.pyplot.subplots(figsize=(13, 10)) 

    data_to_plot = [midnite['DAYTON_MW'], one_am['DAYTON_MW'],

                    two_am['DAYTON_MW'], three_am['DAYTON_MW'],

                    four_am['DAYTON_MW'], five_am['DAYTON_MW'],

                    six_am['DAYTON_MW'], seven_am['DAYTON_MW'],

                    eight_am['DAYTON_MW'], nine_am['DAYTON_MW'],

                    ten_am['DAYTON_MW'], eleven_am['DAYTON_MW'],

                    noon['DAYTON_MW'], one_pm['DAYTON_MW'],

                    two_pm['DAYTON_MW'], three_pm['DAYTON_MW'],

                    four_pm['DAYTON_MW'], five_pm['DAYTON_MW'],

                    six_pm['DAYTON_MW'], seven_pm['DAYTON_MW'],

                    eight_pm['DAYTON_MW'], nine_pm['DAYTON_MW'],

                    ten_pm['DAYTON_MW'], eleven_pm['DAYTON_MW']]

    labels=['12AM', '1AM','2AM','3AM','4AM','5AM','6AM','7AM','8AM','9AM',

            '10AM','11AM','12PM', '1PM','2PM','3PM','4PM','5PM','6PM','7PM',

            '8PM','9PM','10PM','11PM']

    # plot data as box and whisker plot

    ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=labels)

    plt.pyplot.xlabel('Hour of Day')

    plt.pyplot.ylabel('Energy Use (MW)')

    plt.pyplot.title('Average Energy Use by Hour of Day')

    plt.pyplot.show()



by_hour(Y)
Y['Day'] = Y.index.dayofweek

mon = Y[Y['Day']== 0]

tue = Y[Y['Day']== 1]

wed = Y[Y['Day']== 2]

thu = Y[Y['Day']== 3]

fri = Y[Y['Day']== 4]

sat = Y[Y['Day']== 5]

sun = Y[Y['Day']== 6]

fig, ax = plt.pyplot.subplots(figsize=(13, 10))

data_to_plot=[mon['DAYTON_MW'], tue['DAYTON_MW'], wed['DAYTON_MW'],

            thu['DAYTON_MW'], fri['DAYTON_MW'], sat['DAYTON_MW'],

            sun['DAYTON_MW']]

labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=labels)

plt.pyplot.xlabel('Weekday')

plt.pyplot.ylabel('Energy Use (MW)')

plt.pyplot.title('Average Energy Use by Days of the Week')

plt.pyplot.show()



monavg = np.mean(mon['DAYTON_MW'])

tueavg = np.mean(tue['DAYTON_MW'])

wedavg = np.mean(wed['DAYTON_MW'])

thuavg = np.mean(thu['DAYTON_MW'])

friavg = np.mean(fri['DAYTON_MW'])

satavg = np.mean(sat['DAYTON_MW'])

sunavg = np.mean(sun['DAYTON_MW'])
stddev = np.std([monavg, tueavg, wedavg, thuavg, friavg])

print('Standard Deviation of Mon-Fri: ', stddev)

stddev = np.std([satavg, sunavg])

print('Standard Deviation of Sat-Sun: ', stddev)
stddev = np.std([monavg, tueavg, wedavg, thuavg, friavg, satavg, sunavg])

print('Standard Deviation of Mon-Sun: ', stddev)
def by_month(Y):

    jan = Y[Y['Month']== 1]

    feb = Y[Y['Month']== 2]

    mar = Y[Y['Month']== 3]

    apr = Y[Y['Month']== 4]

    may = Y[Y['Month']== 5]

    jun = Y[Y['Month']== 6]

    jul = Y[Y['Month']== 7]

    aug = Y[Y['Month']== 8]

    sep = Y[Y['Month']== 9]

    octo = Y[Y['Month']== 10]

    nov = Y[Y['Month']== 11]

    dec = Y[Y['Month']== 12]

    fig, ax = plt.pyplot.subplots(figsize=(13, 10))

    data_to_plot=[jan['DAYTON_MW'], feb['DAYTON_MW'], mar['DAYTON_MW'],

                  apr['DAYTON_MW'], may['DAYTON_MW'], jun['DAYTON_MW'],

                  jul['DAYTON_MW'], aug['DAYTON_MW'], sep['DAYTON_MW'],

                  octo['DAYTON_MW'], nov['DAYTON_MW'], dec['DAYTON_MW']]

    labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',

            'Oct', 'Nov', 'Dec']

    ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=labels)

    plt.pyplot.xlabel('Month')

    plt.pyplot.ylabel('Energy Use (MW)')

    plt.pyplot.title('Average Energy Use by Month')

    plt.pyplot.show()



by_month(Y)
def by_year(Y):

    Y['Year'] = Y.index.year

    y2004 = Y[Y['Year']== 2004]

    y2005 = Y[Y['Year']== 2005]

    y2006 = Y[Y['Year']== 2006]

    y2007 = Y[Y['Year']== 2007]

    y2008 = Y[Y['Year']== 2008]

    y2009 = Y[Y['Year']== 2009]

    y2010 = Y[Y['Year']== 2010]

    y2011 = Y[Y['Year']== 2011]

    y2012 = Y[Y['Year']== 2012]

    y2013 = Y[Y['Year']== 2013]

    y2014 = Y[Y['Year']== 2014]

    y2015 = Y[Y['Year']== 2015]

    y2016 = Y[Y['Year']== 2016]

    y2017 = Y[Y['Year']== 2017]

    y2018 = Y[Y['Year']== 2018]

    fig, ax = plt.pyplot.subplots(figsize=(13, 10))

    data_to_plot=[y2004['DAYTON_MW'], y2005['DAYTON_MW'], y2006['DAYTON_MW'], 

                  y2007['DAYTON_MW'], y2008['DAYTON_MW'], y2009['DAYTON_MW'], 

                  y2010['DAYTON_MW'], y2011['DAYTON_MW'], y2012['DAYTON_MW'], 

                  y2013['DAYTON_MW'], y2014['DAYTON_MW'], y2015['DAYTON_MW'], 

                  y2016['DAYTON_MW'], y2017['DAYTON_MW'], y2018['DAYTON_MW']]

    labels=['2004','2005','2006','2007','2008','2009','2010','2011','2012',

            '2013','2014','2015','2016','2017','2018']

    ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=labels)

    plt.pyplot.xlabel('Year')

    plt.pyplot.ylabel('Energy Use (MW)')

    plt.pyplot.title('Average Energy Use by Year')

    plt.pyplot.show()



by_year(Y)