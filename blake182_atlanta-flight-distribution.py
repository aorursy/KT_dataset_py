from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.artist import setp
import pandas.core.common as com
from pandas.compat import range, lrange, lmap, map, zip
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd
import numpy as np
"""
This module provides helper methods to carry out data distribution
analysis on flight data found on https://www.kaggle.com/usdot/flight-delays.

These methods are specific to the flight dataset and is not meant to be 
generic functions for other datasets.
"""

def scatter_matrix_all(frame, alpha=0.5, figsize=None, grid=False, diagonal='hist', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05, **kwds):
    
    df = frame
    num_cols = frame._get_numeric_data().columns.values
    n = df.columns.size
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=figsize, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = com.notnull(df)
    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # workaround because `c='b'` is hardcoded in matplotlibs scatter method
    kwds.setdefault('c', plt.rcParams['patch.facecolor'])

    boundaries_list = []
    for a in df.columns:
        if a in num_cols:
            values = df[a].values[mask[a].values]
        else:
            values = df[a].value_counts()
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_+ rdelta_ext))

    for i, a in zip(lrange(n), df.columns):
        for j, b in zip(lrange(n), df.columns):
            ax = axes[i, j]

            if i == j:
                if a in num_cols:    # numerical variable
                    values = df[a].values[mask[a].values]
                    # Deal with the diagonal by drawing a histogram there.
                    if diagonal == 'hist':
                        ax.hist(values, **hist_kwds)
                    elif diagonal in ('kde', 'density'):
                        from scipy.stats import gaussian_kde
                        y = values
                        gkde = gaussian_kde(y)
                        ind = np.linspace(y.min(), y.max(), 1000)
                        ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                    ax.set_xlim(boundaries_list[i])
                else:                # categorical variable
                    values = df[a].value_counts()
                    ax.bar(list(range(df[a].nunique())), values)
            else:
                common = (mask[a] & mask[b]).values
                # two numerical variables
                if a in num_cols and b in num_cols:
                    if i > j:
                        ax.scatter(df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds)
                        # The following 2 lines add the lowess smoothing
                        ys = lowess(df[a][common], df[b][common])
                        ax.plot(ys[:,0], ys[:,1], 'red')
                    else:
                        pearR = df[[a, b]].corr()
                        ax.text(df[b].min(), df[a].min(), 'r = %.4f' % (pearR.iloc[0][1]))
                    ax.set_xlim(boundaries_list[j])
                    ax.set_ylim(boundaries_list[i])
                # two categorical variables
                elif a not in num_cols and b not in num_cols:
                    if i > j:
                        from statsmodels.graphics import mosaicplot
                        mosaicplot.mosaic(df, [b, a], ax, labelizer=lambda k:'')
                # one numerical variable and one categorical variable
                else:
                    if i > j:
                        tol = pd.DataFrame(df[[a, b]])
                        if a in num_cols:
                            label = [ k for k, v in tol.groupby(b) ]
                            values = [ v[a].tolist() for k, v in tol.groupby(b) ]
                            ax.boxplot(values, labels=label)
                        else:
                            label = [ k for k, v in tol.groupby(a) ]
                            values = [ v[b].tolist() for k, v in tol.groupby(a) ]
                            ax.boxplot(values, labels=label, vert=False)

            ax.set_xlabel('')
            ax.set_ylabel('')

            _label_axis(ax, kind='x', label=b, position='bottom', rotate=True)
            _label_axis(ax, kind='y', label=a, position='left')

            if j!= 0:
                ax.yaxis.set_visible(False)
            if i != n-1:
                ax.xaxis.set_visible(False)

    for ax in axes.flat:
        setp(ax.get_xticklabels(), fontsize=8)
        setp(ax.get_yticklabels(), fontsize=8)
    return fig
    

def _label_axis(ax, kind='x', label='', position='top', ticks=True, rotate=False):
    from matplotlib.artist import setp
    if kind == 'x':
        ax.set_xlabel(label, visible=True)
        ax.xaxis.set_visible(True)
        ax.xaxis.set_ticks_position(position)
        ax.xaxis.set_label_position(position)
        if rotate:
            setp(ax.get_xticklabels(), rotation=90)
    elif kind == 'y':
        ax.yaxis.set_visible(True)
        ax.set_ylabel(label, visible=True)
        #ax.set_ylabel(a)
        ax.yaxis.set_ticks_position(position)
        ax.yaxis.set_label_position(position)
    return

def _get_marker_compat(marker):
    import matplotlib.lines as mlines
    import matplotlib as mpl
    if mpl.__version__ < '1.1.0' and marker == '.':
        return 'o'
    if marker not in mlines.lineMarkers:
        return 'o'
    return marker

def plotBarPercentage(data, groupAttr, dependencyAttr, axAttr, condition, filter=0):
    totaldf = data.groupby([groupAttr])[dependencyAttr].count()
    denomdf = data.loc[condition]
    denomdf = denomdf.groupby([groupAttr])[dependencyAttr].count()
    df  = denomdf/totaldf*100
    df  = df[df > filter]
    if len(df) > 0:
        ax = df.plot.bar(figsize=(14, 6), ax = axAttr)
        ax.set_title(dependencyAttr)
        ax.set_ylabel('Percentage')

def plotBar(data, groupAttr, dependencyAttr, axAttr, condition):
    df = data.loc[condition]
    df = df.groupby([groupAttr])[dependencyAttr].count()
    ax = df.plot.bar(figsize=(14, 6), ax = axAttr)
    ax.set_ylabel(dependencyAttr)

def plotBars(data, groupAttr, dependencyAttrs, rows, cols, conditions):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    r = 0
    c = 0
    for i in range(len(dependencyAttrs)):
       plotBar(data, groupAttr, dependencyAttrs[i], axes[r,c], conditions[i])
       if c == cols-1:
           c = -1
           r = r + 1
       c = c + 1
        
def plotBarsPercentage(data, groupAttr, dependencyAttrs, rows, cols, conditions, filter = 0):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    r = 0
    c = 0
    for i in range(len(dependencyAttrs)):
       if rows > 1:
          plotBarPercentage(data, groupAttr, dependencyAttrs[i], axes[r,c], conditions[i], filter)
       else:
          plotBarPercentage(data, groupAttr, dependencyAttrs[i], axes[c], conditions[i], filter)

       if c == cols-1:
           c = -1
           r = r + 1
       c = c + 1

def plotMapData(df, longAttr, latAttr, valAttr, figw=8, figh=8, initmarksize= 0.5):
    # setup Lambert Conformal basemap.
    plt.figure(figsize=(figw,figh))
    m = Basemap(width=12000000,height=9000000,projection='lcc',
                resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
    # draw a boundary around the map, fill the background.
    # this background will end up being the ocean color, since
    # the continents will be drawn on top.
    m.drawmapboundary(fill_color='aqua')
    # fill continents, set lake color same as ocean color.
    m.fillcontinents(color='coral',lake_color='aqua')
    # draw parallels and meridians.
    # label parallels on right and top
    # meridians on bottom and left
    parallels = np.arange(0.,81,10.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[False,True,True,False])
    meridians = np.arange(10.,351.,20.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    # plot blue dot on Boulder, colorado and label it as such.

    for lon, lat, mag in zip(df[longAttr].values, df[latAttr].values, df[valAttr].values):
        xpt,ypt = m(lon, lat)
        lonpt, latpt = m(xpt,ypt,inverse=True)
        msize = mag * initmarksize
        #map.plot(x, y, marker_string, markersize=msize)
        m.plot(xpt,ypt,'bo', markersize=msize)  # plot a blue dot there  

    plt.show()

def plotJointPlotSplice0_10_240_By(x, y, delayAttr, data):
    # Create dataset based on splice conditions
    flights_greater_than_0_and_less_than_10 = data.loc[
        (data[delayAttr] > 0)
        & (data[delayAttr] <= 10)
    ]
    flights_greater_than_10_and_less_than_240 = data.loc[
        (data[delayAttr] > 10)
        & (data[delayAttr] <= 240)
    ]

    flights_greater_than_240 = data.loc[
        (data[delayAttr] > 240)
    ]
    sns.jointplot(x=x, y=y, kind="kde", data=flights_greater_than_0_and_less_than_10, size=4)
    sns.jointplot(x=x, y=y, kind="kde", data=flights_greater_than_10_and_less_than_240, size=4)
    sns.jointplot(x=x, y=y, kind="kde", data=flights_greater_than_240, size=4)

def plotJointPlot(x, y, delayAttr, data, title):
    df = data
    datasetSize = len(df)
    g = sns.jointplot(x=x, y=y, kind="kde", data=df, size=4)
    txt = plt.title(title + ",\n Dataset Size = " + str(datasetSize), fontsize = 24, y = 0.5, x = 6)
    
def plotJointPlotSplice(x, y, delayAttr, data, cond, title):
    df = data.loc[cond]
    datasetSize = len(df)
    g = sns.jointplot(x=x, y=y, kind="kde", data=df, size=4)
    txt = plt.title(title + ",\n Dataset Size = " + str(datasetSize), fontsize = 24, y = 0.5, x = 6)
    
def generateDistributionDF(data, timeAttr, monthAttr, delayAttr, aggfunc= np.sum):
    pivot = pd.pivot_table(data,index=[monthAttr, timeAttr],values=[delayAttr],aggfunc=aggfunc)
    pivot.reset_index(level=0, inplace=True)
    pivot.reset_index(level=0, inplace=True)
    return pivot

def plot3D(data, x, y, z):
    distdf = generateDistributionDF(data, y, x, z)
    distdf_avg = generateDistributionDF(data, y, x, z, np.mean)    

    fig = plt.figure(figsize=(16, 6), dpi=80)

    #---- First subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    surf = ax.plot_trisurf(distdf[x], distdf[y], distdf[z], cmap=plt.cm.jet, linewidth=0.03)
    fig.colorbar(surf)

    #---- Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_trisurf(distdf_avg[x], distdf_avg[y], distdf_avg[z], cmap=plt.cm.jet, linewidth=0.03)
    fig.colorbar(surf)

    plt.show()

import Geohash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import importlib
import warnings
import seaborn as sns
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')
#Read data
flights = pd.read_csv('../input/atlantaairportdata/ATL_')
# Clean up data.  Fill NA with 0.  Assume 0 delay in cases where information is not entered
flights = flights.fillna(0)

# Let's create a category for Departure time based on hour (0 to 2400 hr clock)
flights['DEPARTURE_TIME_BIN'] = pd.cut(flights['DEPARTURE_TIME'], bins=np.arange(0,2400, 100), labels=np.arange(23))
# Let's create a category for Schedule Departure time based on hour (0 to 2400 hr clock)
flights['SCHEDULED_DEPARTURE_BIN'] = pd.cut(flights['SCHEDULED_DEPARTURE'], bins=np.arange(0,2400, 100), labels=np.arange(23))
axbox = flights.plot.box(figsize=(16, 6), rot=90)
# Note we are only interested in flights that have a delay greater than 0
fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(10, 4))
df = flights.loc[flights['SECURITY_DELAY'] > 0]
axbox = df[['SECURITY_DELAY']].plot.box(ax = axes[0], rot=90)
df = flights.loc[flights['WEATHER_DELAY'] > 0]
axbox = flights[['WEATHER_DELAY']].plot.box(ax = axes[1], rot=90)
df = flights.loc[flights['TAXI_OUT'] > 0]
axbox = flights[['TAXI_OUT']].plot.box(ax = axes[2], rot=90)
df = flights.loc[flights['DEPARTURE_DELAY'] > 0]
axbox = flights[['DEPARTURE_DELAY']].plot.box(ax = axes[3], rot=90)
# Note we are only interested in flights that have a delay greater than 0
overalDF = pd.DataFrame()
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(16, 8))
df = flights.loc[flights['SECURITY_DELAY'] > 0]
overalDF['SECUIRTY_DELAY'] = df['SECURITY_DELAY'].describe()
axbox = df[['SECURITY_DELAY']].plot.hist(ax = axes[0, 0], rot=90)
df = flights.loc[flights['WEATHER_DELAY'] > 0]
overalDF['WEATHER_DELAY'] = df['WEATHER_DELAY'].describe()
axbox = flights[['WEATHER_DELAY']].plot.hist(ax = axes[0, 1], rot=90)
df = flights.loc[flights['TAXI_OUT'] > 0]
overalDF['TAXI_OUT'] = df['TAXI_OUT'].describe()
axbox = flights[['TAXI_OUT']].plot.hist(ax = axes[1, 0], rot=90)
df = flights.loc[flights['DEPARTURE_DELAY'] > 0]
overalDF['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].describe()
axbox = flights[['DEPARTURE_DELAY']].plot.hist(ax = axes[1, 1], rot=90)
fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(16, 8))
df = flights.loc[flights['SECURITY_DELAY'] > 0]
axbox = df[['SECURITY_DELAY']].plot.box(ax = axes[0], rot=90)
df = flights.loc[(flights['WEATHER_DELAY'] > 0) & (flights['WEATHER_DELAY'] < 200)]
axbox = df[['WEATHER_DELAY']].plot.box(ax = axes[1], rot=90)
df = flights.loc[flights['TAXI_OUT'] > 0]
axbox = flights[['TAXI_OUT']].plot.box(ax = axes[2], rot=90)
df = flights.loc[(flights['DEPARTURE_DELAY'] > 0) & (flights['DEPARTURE_DELAY'] < 240)]
axbox = df[['DEPARTURE_DELAY']].plot.box(ax = axes[3], rot=90)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(16, 8))
overallDF = pd.DataFrame()
df = flights.loc[flights['SECURITY_DELAY'] > 0]
overallDF['SECUIRTY_DELAY'] = df['SECURITY_DELAY'].describe()
axbox = df[['SECURITY_DELAY']].plot.hist(ax = axes[0, 0], rot=90)
df = flights.loc[(flights['WEATHER_DELAY'] > 0) & (flights['WEATHER_DELAY'] < 200)]
overallDF['WEATHER_DELAY'] = df['WEATHER_DELAY'].describe()
axbox = df[['WEATHER_DELAY']].plot.hist(ax = axes[0, 1], rot=90)
df = flights.loc[flights['TAXI_OUT'] > 0]
overallDF['TAXI_OUT'] = df['TAXI_OUT'].describe()
axbox = df[['TAXI_OUT']].plot.hist(ax = axes[1, 0], rot=90)
df = flights.loc[(flights['DEPARTURE_DELAY'] > 0) & (flights['DEPARTURE_DELAY'] < 240)]
overallDF['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].describe()
axbox = df[['DEPARTURE_DELAY']].plot.hist(ax = axes[1, 1], rot=90)
overallDF
plotJointPlot('DEPARTURE_TIME','MONTH', 'DEPARTURE_DELAY', 
                             flights, 'All Flights')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'DEPARTURE_DELAY', 
                             flights, (flights['DEPARTURE_DELAY'] > 0) & (flights['DEPARTURE_DELAY'] <= 10),
                             'Depature Delay > 0 \n and Depature Delay <= 10')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'DEPARTURE_DELAY', 
                             flights, (flights['DEPARTURE_DELAY'] > 10) & (flights['DEPARTURE_DELAY'] <= 240),
                             'Depature Delay > 10 \n and Departure Delay <= 240')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'DEPARTURE_DELAY', flights, flights['DEPARTURE_DELAY'] > 240, 'Depature Delay > 240')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 0) & (flights['WEATHER_DELAY'] <= 10),
                             'Weather Delay > 0 \n and Weather Delay <= 10')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 10) & (flights['WEATHER_DELAY'] <= 240),
                             'Weather Delay > 10 \n and Weather Delay <= 240')
plotJointPlotSplice('DEPARTURE_TIME','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 240),
                             'Weather Delay > 240')
plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'DEPARTURE_DELAY', 
                             flights, (flights['DEPARTURE_DELAY'] > 0) & (flights['DEPARTURE_DELAY'] <= 10),
                             'Departure Delay > 0 \n and Departure Delay <= 10')
plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'DEPARTURE_DELAY', 
                             flights, (flights['DEPARTURE_DELAY'] > 0) & (flights['DEPARTURE_DELAY'] <= 240),
                             'Departure Delay > 0 \n and Departure Delay <= 240')
plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'DEPARTURE_DELAY', flights, flights['DEPARTURE_DELAY'] > 240, 'Departure Delay > 240')
plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 0) & (flights['WEATHER_DELAY'] <= 10),
                             'Weather Delay > 0 \n and Weather Delay <= 10')

plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 10) & (flights['WEATHER_DELAY'] <= 240),
                             'Weather Delay > 10 \n and Weather Delay <= 240')
plotJointPlotSplice('DAY_OF_WEEK','MONTH', 'WEATHER_DELAY', 
                             flights, (flights['WEATHER_DELAY'] > 240),
                             'Weather Delay > 240')
# only consider delays greater than 0
df = flights.loc[flights['DEPARTURE_DELAY'] > 0]
df = df[['DEPARTURE_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
       'WEATHER_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'AIR_SYSTEM_DELAY']]
fig = scatter_matrix_all(df, alpha=0.4, figsize=(14,14))      
# Read weather information for atlanta international airport
weather = pd.read_csv('../input/airport-weather/Weather.csv')
# Merge in weather data
mergeddf = pd.merge(flights, weather, on=['MONTH', "DAY"])
x = mergeddf[['WEATHER_DELAY', 'PRCP', 'TAVG', 'WSF2', 'WSF5']]
df = x.loc[(x['WEATHER_DELAY'] > 0) & (x['WEATHER_DELAY'] < 240)]
df = df.fillna(0)
fig = scatter_matrix_all(df, alpha=0.4, figsize=(12,12))   
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import scipy.stats                  # for pearson correlation

seaborn.regplot(x="WEATHER_DELAY", y="PRCP", fit_reg=True, data=df, line_kws={"color": "red"});
plt.xlabel('Precipitation');
plt.ylabel('Weather Delay');

fig.tight_layout()
plt.show()
seaborn.regplot(x="WEATHER_DELAY", y="TAVG", fit_reg=True, data=df, line_kws={"color": "red"});
plt.xlabel('Average Tempurature');
plt.ylabel('Weather Delay');

fig.tight_layout()
plt.show()
seaborn.regplot(x="WEATHER_DELAY", y="WSF2", fit_reg=True, data=df, line_kws={"color": "red"});
plt.xlabel('Wind Speed');
plt.ylabel('Weather Delay');

fig.tight_layout()
plt.show()
ax = flights['AIRLINE'].value_counts().plot(kind='bar', figsize=(16,5))
flights['AIRLINE'].value_counts()
