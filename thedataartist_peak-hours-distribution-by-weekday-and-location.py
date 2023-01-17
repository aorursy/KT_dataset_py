# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

from matplotlib import cm

import matplotlib.animation as animation

from IPython.display import HTML
udata = pd.read_csv('../input/uber-raw-data-aug14.csv')

udata.head()
udata.info()
udata['Date/Time'] = pd.to_datetime(udata['Date/Time'],format = "%m/%d/%Y %H:%M:%S")

udata['DayOfWeekNum'] = udata['Date/Time'].dt.dayofweek

udata['DayOfWeek'] = udata['Date/Time'].dt.weekday_name

udata['HourOfDay'] = udata['Date/Time'].dt.hour

udata.head()
#Compare number of journeys by day of the week

weekday_journeys = udata.pivot_table(index = ['DayOfWeekNum','DayOfWeek'],values = 'Base', aggfunc = 'count')





weekday_journeys.plot(kind = 'bar', figsize=(7,5))

plt.ylabel('Total no.of journeys')

plt.title('Journeys by Day of Week')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Compare number of journeys by hour of day

uber_hour = udata.pivot_table(index=['HourOfDay'],values='Base',aggfunc='count')

uber_hour.plot(kind='bar',figsize=(7,5))

plt.title('Journeys by Hour of Day')
#Identification of peak hours in each day of the week

fig = plt.figure(figsize = (8,6))

ax = fig.add_subplot(111)  

sns.violinplot('DayOfWeekNum','HourOfDay',data = udata)

ax.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

ax.set(xlabel = 'Day of Week',yticks = np.arange(0,26,2),title = 'Journeys by hour of the day per day of the week')

plt.show()





#Insights: 

#Peak hours seem to be 16.00-22.00 hrs on mondays through saturdays, with the bracket increasing in the same order.

  #There is moderate to high traffic during early morning hours(6.00-8.00 hrs)



#On weekends, there is a surge in traffic during mid-night hours (00.00-00.01 hrs),when compared to weekdays, 

#which accounts for moderate to peak volume.

#On sundays, there is also a tendency for traffic to peak during 16.00-18.00 hrs.



   



#Visualizing Uber traffic by hour across locations in NYC

def update(frame_num):

    

    

    x, y = m(udata.Lon[udata['HourOfDay']==frame_num].values,udata.Lat[udata['HourOfDay']==frame_num].values)

    hex_.set_offsets(np.dstack((x, y)));

    hour_text.set_text('hour {}'.format(str(frame_num)))

   

  

    

    

    





fig = plt.figure(figsize=(18, 16))

fig.suptitle('Journeys through the day over New York City',fontsize = 40,fontweight='bold')

m = Basemap(projection = 'merc',llcrnrlat=40.50, urcrnrlat=40.92,llcrnrlon=-74.26, urcrnrlon=-73.70, lat_ts=40.50, resolution='i')

m.drawmapboundary(fill_color='blue')

x, y = m(udata.Lon[udata['HourOfDay']==0].values,udata.Lat[udata['HourOfDay']==0].values)

hex_ = m.hexbin(x, y,gridsize = 350, bins='log',cmap=cm.viridis);

hour_text = plt.text(-170, 50, 'hour {}'.format(str(0)),fontsize=40,color='white')

plt.close()

a = animation.FuncAnimation(fig, update, interval=100, frames = 24) 

a.save('animation.gif', writer='imagemagick', fps=2)
import io

import base64



filename = 'animation.gif'



video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))