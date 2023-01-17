import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import scipy as sp



%matplotlib inline



df = pd.read_csv('../input/uber-raw-data-apr14.csv')





df.head()
#! /usr/bin/env python3



from xml.etree import ElementTree

import xml.etree

import requests

import pandas as pd

import numpy as np



def getdata(lat,lon):

  page = requests.get("http://data.fcc.gov/api/block/2010/find?latitude=%f&longitude=%f"%(lat,lon))  

  root = ElementTree.fromstring(page.content)



  status = root.attrib['status']

  block = root[0].attrib['FIPS']  

  county = root[1].attrib['name']

  state = root[2].attrib['code']



  data= {'lat':lat,'lon':lon,'status':status,'block':block,'county':county,'state':state}



  return data



def read_uber_data(inf='uber-raw-data-apr14.csv',outf='uber_apr14_blocks.csv'):



  df = pd.read_csv(inf)

  length = df.shape[0]

  lats = df.Lat.as_matrix()

  lons = df.Lon.as_matrix()

  f = open(outf,'w')

  f.write('status,block,county,state\n')

  count = 0

  for lat,lon in zip(lats,lons):

    if (count%10 ==0 ):

      print(count)

    count += 1

    #print(lat,lon)

    data = getdata(lat,lon)

    f.write( str(data['status'])+','+str(data['block'])+','+str(data['county'])+','+str(data['state'])+'\n')

  f.close()
def TransformTimestamp(df):

    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    df['DayOfWeek'] = [x.dayofweek for x in df['Date/Time']]

    df['Day'] = [x.day for x in df['Date/Time']]

    df['Hour'] = [x.hour for x in df['Date/Time']]

    df['DayTime'] = [x.hour+x.minute/60. for x in df['Date/Time']]

    df['MonthTime'] = [x.day + x.hour/24. + x.minute/(24.*60) for x in df['Date/Time']]

    df['IsWeekend'] = (df.DayOfWeek == 0) | (df.DayOfWeek == 6)

    return df

df = TransformTimestamp(df)
## Create a figure since we'll have several subfigures

fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(221)



## First, total rides by day of the week.

ax.hist(df.DayOfWeek,bins=7,range=[0,7],alpha=0.8)

ax.set_xlabel('Day of Week')

ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])

ax.set_xticklabels(['Su','Mo','Tu','We','Th','Fr','Sa'])

ax.set_ylabel('# of Rides in April 2014')

ax.set_xlim([0,7])



## Next, let's look at the distributions every 15 min throughout the day

ax = fig.add_subplot(222)

ax.hist(df.DayTime,bins=96,range=[0,24],alpha=0.8)

ax.set_xlabel('Time of Day [hrs]')

ax.set_ylabel('# of Rides in April 2014')

ax.set_xlim([0,24])



## Finally, let's look at the # of rides every 12 hours for the whole month

ax = fig.add_subplot(212)

ax.hist(df.MonthTime,bins=30*12,range=[1,31],alpha=0.8)

ax.set_xlabel('Time in Month [days]')

ax.set_ylabel('# of Rides in April 2014')

ax.set_xlim([1,31])



fig = plt.figure(figsize=[12,6])

plt.hist(df.MonthTime,bins=30,range=[1,31],alpha=0.8)

plt.xlabel('Day [days]')

plt.ylabel('# of Rides in April 2014')

plt.xlim([1,31])

plt.show()
fg = sns.FacetGrid(df,col='IsWeekend',size=5,sharex=True,xlim=[0,24])

fg.map(plt.hist,'DayTime',alpha=0.8,normed=True,bins=96,range=[0,24])

for ax in fg.axes.flat:

    ax.set_xlabel('Time of Day [hr]')



plt.show()
### If we're going to make any histogram type plots, it would be good to have square bins

### Away from the equator, degrees in latitude and longitude are different

### (i.e. the Jacobian on a spherical surface is sin(theta) )



latitude_nyc = 40.75 * np.pi/180

aspect = 1/np.sin(0.5*np.pi - latitude_nyc)

aspect
def WholeCityPlot(df,ax=None,xbins=250):

    if ax is None:

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

    ax.grid(b=False)



    xlims = [-74.2,-73.7]

    ylims = [40.6,40.9]

    ybins = xbins * aspect*(ylims[1]-ylims[0] )/(xlims[1]-xlims[0])

    hist = ax.hist2d(df.Lon,df.Lat,bins=[xbins,ybins],range=[xlims,ylims],cmap='inferno',norm=mpl.colors.LogNorm())

    

WholeCityPlot(df)

plt.show()
fig = plt.figure(figsize=[10,5])

ax = fig.add_subplot(121)

WholeCityPlot(df[df['IsWeekend'] == True ],ax=ax)

ax = fig.add_subplot(122)

WholeCityPlot(df[df['IsWeekend'] == False],ax=ax)
def MapNoAirports(df,ax=None,xbins=200):

    if ax is None:

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

    ax.grid(b=False)



    xlims = [-74.05,-73.9]

    ylims = [40.6,40.9]

    ybins = xbins * aspect*(ylims[1]-ylims[0] )/(xlims[1]-xlims[0])

    hist = ax.hist2d(df.Lon,df.Lat,bins=[xbins,ybins],range=[xlims,ylims],cmap='inferno',norm=mpl.colors.LogNorm())

    

MapNoAirports(df)
def ManhattanMap(df,ax=None,xbins=150):

    if ax is None:

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

    ax.grid(b=False)



    xlims = [-74.03,-73.92]

    ylims = [40.69,40.88]

    ybins = xbins * aspect*(ylims[1]-ylims[0] )/(xlims[1]-xlims[0])

    hist = ax.hist2d(df.Lon,df.Lat,bins=[xbins,ybins],range=[xlims,ylims],cmap='inferno',norm=mpl.colors.LogNorm())

    

fig = plt.figure(figsize=[10,10])

ax = fig.add_subplot(221)

ax.set_title('Weekend')

ManhattanMap(df[df['IsWeekend'] == True ],ax=ax,xbins=100)

ax = fig.add_subplot(222)

ax.set_title('Weekday')

ManhattanMap(df[df['IsWeekend'] == False],ax=ax,xbins=100)

ax = fig.add_subplot(223)

ax.set_title('Late Night')

ManhattanMap(df[(df.DayTime<5)],ax=ax,xbins=100)

ax = fig.add_subplot(224)

ax.set_title('Not Late Night')

ManhattanMap(df[(df.DayTime>=5)],ax=ax,xbins=100)



fig = plt.figure(figsize=[10,5])

ax = fig.add_subplot(121)

ax.set_title('Morning Commute')

ManhattanMap(df[ (df.DayTime > 6)&(df.DayTime<9) ],ax=ax,xbins=80)

ax = fig.add_subplot(122)

ax.set_title('Afternoon Commute')

ManhattanMap(df[(df.DayTime>16)&(df.DayTime<19)],ax=ax,xbins=80)
YankeeStadium = [40.8294,-73.9267]

## -73.9319, -73.9213

## 40.8247 40.834

df_ys = df[(df.Lat > 40.824 ) & (df.Lat < 40.837) & (df.Lon > -73.9319) & (df.Lon < -73.92)]

fig = plt.figure(figsize=[10,10])

ax = fig.add_subplot(221)

ax.scatter(df_ys.Lon,df_ys.Lat)

ax.set_xlim([-73.9319,-73.92])

ax.set_ylim([40.824,40.837])

ax.set_xlabel('Lon [deg]')

ax.set_ylabel('Lat [deg]')



ax = fig.add_subplot(222)

ax.hist(df_ys.MonthTime,range=[1,31],bins=30)

ax.set_xlim([1,31])

ax.set_xlabel('Day')

ax.set_ylabel('Count')



ax = fig.add_subplot(212)

ax.hist(df_ys.MonthTime,range=[7,17],bins=10*24)

ax.set_xlim([7,17])

ax.set_xlabel('Day')

ax.set_ylabel('Count')



plt.show()
def YankeesNightGame(df,day):

    fig = plt.figure(figsize=[10,5])

    ax = fig.add_subplot(121)

    ax.hist( (df.MonthTime-day) * 24 ,range=[18,25],bins=40)

    ax = fig.add_subplot(122)

    df_game = df[(df_ys.MonthTime>(day+0.75)) & (df.MonthTime<(day+0.95))]

    ax.scatter(df_game.Lon,df_game.Lat,c=df_game.DayTime,cmap='jet',marker='.')

    ax.set_xlim([-73.9319,-73.92])

    ax.set_ylim([40.824,40.837])

    

YankeesNightGame(df_ys,9)

YankeesNightGame(df_ys,10)

YankeesNightGame(df_ys,11)

YankeesNightGame(df_ys,16)

YankeesNightGame(df_ys,25)

YankeesNightGame(df_ys,27)

YankeesNightGame(df_ys,29)



plt.show()
def UESHarlemMap(df,ax=None,xbins=50):

    if ax is None:

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

    ax.grid(b=False)



    xlims = [-73.96,-73.93]

    ylims = [40.75,40.85]

    ybins = xbins * aspect*(ylims[1]-ylims[0] )/(xlims[1]-xlims[0])

    hist = ax.hist2d(df.Lon,df.Lat,bins=[xbins,ybins],range=[xlims,ylims],cmap='inferno',norm=mpl.colors.LogNorm(vmin=1))

    

UESHarlemMap(df,None,25)

#plt.show()
df_ues = df[ (df.Lat > 40.77) & (df.Lat < 40.78) & (df.Lon >-73.95) & (df.Lon < -73.946)]

df_sh = df[ (df.Lat > 40.79) & (df.Lat < 40.81) & (df.Lon > -73.95 ) & (df.Lon < -73.946)]



fig = plt.figure()

plt.hist(df_ues.DayTime,alpha=0.5,normed=True,label='Upper East Side',range=[0,24],bins=24)

plt.hist(df_sh.DayTime,alpha=0.5,color='r',normed=True,label='Harlem',range=[0,24],bins=24)

plt.legend(loc='upper right')

plt.xlim([0,24])





#plt.show()
### Note: This does not work in all browsers, seems like it doesn't work in Kaggle due to missing libraries.

## Uncomment the last two lines if running on your own computer.



class ManhattanAnimator:

    def __init__(self,df,nbins):

        self._df = df

        self._nbins = nbins

        self._fig = plt.figure(figsize=(8,8))

        self._ax = self._fig.add_subplot(111)

        self._xlims = [-74.03,-73.94]

        self._ylims = [40.69,40.825]



        aspect =  1.3200187714761737



        ybins = self._nbins * aspect*(self._ylims[1]-self._ylims[0] )/(self._xlims[1]-self._xlims[0])

        self._hists = []

        self._max = 0

        

        for i in range(48):

            hist,xrange,yrange = np.histogram2d(df[(df.DayTime>=i*0.5) & (df.DayTime <(i+1)*0.5)].Lon,df[(df.DayTime>=i*0.5) & (df.DayTime<(i+1)*0.5)].Lat,range=[self._xlims,self._ylims],bins=[self._nbins,ybins])

            maxval = np.max(hist)

            if maxval > self._max:

                self._max = maxval

            self._hists.append( hist )

        self._im = self._ax.imshow(self._hists[0].transpose()[::-1],interpolation='nearest',cmap='inferno',norm=mpl.colors.LogNorm(vmin=1,vmax=maxval),extent=[self._xlims[0],self._xlims[1],self._ylims[0],self._ylims[1]],animated=True)

        self._ax.set_xlim(self._xlims)

        self._ax.set_ylim(self._ylims)

        self._ax.set_xlabel('Longitude [deg]')

        self._ax.set_ylabel('Latitude [deg]')

        self._ax.set_title("Uber Pickups between 0:00 and 0:30")

        self._ax.grid(b=False)



    def init_anim(self):

        #self._line.set_data([], [])

        return (self._im,)

    def animate(self,i):



        self._im.set_array(self._hists[i].transpose()[::-1])

        self._ax.set_title("Uber Pickups between %02i:%02i and %02i:%02i"%(int(i*0.5),int( 60*((i*0.5)%1) ),int((i+1)*0.5),int(60*(((i+1)*0.5)%1)+0.0001))) 



        return (self._im,)

    def run_animation(self):

        from matplotlib import animation, rc

        from IPython.display import HTML, Image

        mpl.rcParams['animation.writer'] = 'avconv'

        mpl.rcParams['animation.html'] = 'html5'

   # First set up the figure, the axis, and the plot element we want to animate

        anim = animation.FuncAnimation(self._fig, self.animate, init_func=self.init_anim,frames=48, interval=1000, blit=True)

        plt.close(anim._fig)

        #anim.save('manhattan_uber.gif', writer='imagemagick', fps=1)

        #Image(url='manhattan_uber.gif')

        return HTML(anim.to_html5_video()) 



    

#a = ManhattanAnimator(df,25)

#a.run_animation()
fig = plt.figure(figsize=[10,5])

ax = fig.add_subplot(121)

ax.set_title('April 1-6,8-29')

ManhattanMap(df[ (df.Day != 7)&(df.Day!=30) ],ax=ax,xbins=40)



ax = fig.add_subplot(122)

ax.set_title('April 7')

ManhattanMap(df[(df.Day == 7)],ax=ax,xbins=40)



fig = plt.figure(figsize=[10,5])

ax = fig.add_subplot(121)

ax.set_title('April 1-6,8-29')

ManhattanMap(df[ (df.Day != 7)&(df.Day!=30) ],ax=ax,xbins=40)



ax = fig.add_subplot(122)

ax.set_title('April 30')

ManhattanMap(df[(df.Day == 30)],ax=ax,xbins=40)
