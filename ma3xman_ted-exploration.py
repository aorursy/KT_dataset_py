# Setup



import numpy as np

import pandas as pd



metadata = pd.read_csv('../input/ted-talks/ted_main.csv')

transcripts = pd.read_csv('../input/ted-talks/transcripts.csv')



metadata.describe()
# Get timedeltas from the end point

from datetime import datetime



end_date = datetime(year=2017, month=9, day=21)

metadata['days_online'] = end_date - metadata['published_date'].map(datetime.fromtimestamp)

metadata['days_online'] = metadata['days_online'].map(lambda x: x.days)

metadata[['days_online']].describe()
end_date = datetime(year=2017, month=9, day=23)

metadata['days_online'] = end_date - metadata['published_date'].map(datetime.fromtimestamp)

metadata['days_online'] = metadata['days_online'].map(lambda x: x.days)

metadata[['days_online']].describe()
metadata['views'].hist()
metadata['comments'].hist()
metadata['log_views'] = np.log2(metadata['views'])

metadata['log_comments'] = np.log2(metadata['comments'])

metadata['log_views'].hist()
metadata['log_comments'].hist()
metadata[['log_views', 'log_comments', 'days_online', 'languages', 'duration']].corr()
import matplotlib.pyplot as plt



# made a little helper

def scatter_trend(*, x, y, color=None):

    #plt.ylim(min(y), max(y))

    plt.scatter(x, y, c=color)

    # calc the trendline

    z = np.polyfit(x, y, 2)

    p = np.poly1d(z)

    # ensure the x is ordered for the line plot

    sorted_data = sorted(list(zip(x, p(x))))

    new_x = [i[0] for i in sorted_data]

    new_p = [i[1] for i in sorted_data]

    plt.plot(new_x, new_p, "r--")

    plt.show()



scatter_trend(x=metadata['log_views'], y=metadata['log_comments'], color=metadata['days_online'])
scatter_trend(x=metadata['log_views'], color=metadata['log_comments'], y=metadata['days_online'])
!pip install ipyvolume
import ipyvolume as ipv



def plot3D(*, x, y, z):

    def scale(s):

        return s.apply(lambda i: i/s.max())

    _x, _y, _z = (scale(metadata[x]), scale(metadata[y]), scale(metadata[z]))

    fig = ipv.figure()

    scatter = ipv.quickscatter(_x, _y, _z, marker='sphere', color='green')

    ipv.pylab.xlim(_x.min(), _x.max())

    ipv.pylab.ylim(_y.min(), _y.max())

    ipv.pylab.zlim(_z.min(), _z.max())

    ipv.pylab.xlabel(x)

    ipv.pylab.ylabel(y)

    ipv.pylab.zlabel(z)

    ipv.show()

    

plot3D(x='log_views', y='log_comments', z='days_online')
plot3D(x='log_views', y='log_comments', z='languages')