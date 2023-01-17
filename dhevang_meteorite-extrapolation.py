import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

import scipy as sp

from scipy import stats

import scipy.optimize as optimization



meteorites = pd.read_csv('../input/meteorite-landings.csv')

meteorites = meteorites.groupby('nametype').get_group('Valid') # only investigate 'Valid' types

meteorites = meteorites.dropna(how='any')# Remove ill-defined values

meteorites = meteorites[(meteorites.reclat!=0.0) & (meteorites.reclong!=0.0)]# Remove ill-defined locations

meteorites = meteorites[(meteorites.year>=860) & (meteorites.year<=2016)]# Remove known illegitimate years

print(meteorites.describe())
lat = np.array(meteorites['reclat'])

lon = np.array(meteorites['reclong'])

lat2 = np.arange(-90,92,2)

lon2 = np.arange(-180,182,2)

H,lat2,lon2 = np.histogram2d(lat,lon,bins=(lat2,lon2))# H is the histogram in 2D-array form

FellorFound = meteorites['fall'].isin(['Fell'])# Series which tells us if its fall was observed(True)

H2,lat2,lon2 = np.histogram2d(lat,lon,bins=(lat2,lon2),weights=FellorFound)



Hits=[]



colors = ['MidnightBlue','MediumBlue','Blue','Cyan','SpringGreen','Yellow','Salmon','Orange','Red','DeepPink']

    

for i in range(len(lat2)-1):

    for j in range(len(lon2)-1):

        if H[i][j]<1: continue

        

        color_num = int(H[i][j]/10.)

        if color_num >=10: color_num=9

        symboltype='circle-open'

        if float(H2[i][j])/H[i][j] > 0.5: symboltype='diamond'

        hit = dict(

            type = 'scattergeo',

            lat = np.array([lat2[i]+1]),

            lon = np.array([lon2[j]+1]),

            marker = dict(

                symbol=symboltype,

                size = 5 + 2*np.log(H[i][j]),

                opacity = 0.7,

                color = colors[color_num],

                sizemode = 'area'           

            ),

            name = str(int(H2[i][j]))+'Fell,'+str(int(H[i][j]-H2[i][j]))+'Found'

        )

        Hits.append(hit)   





maplayout = go.Layout(

    title='Map of meteorite impacts.  Solid diamonds for sites where more than 50% seen falling',

    showlegend=False,

    geo = dict(

        projection = dict(

            type='orthographic',

            rotation=dict(lat=10,lon=30)

        ),

        showland=True,

        showocean=True,

        showcountries=True,

        landcolor='rgb(240,240,240)',

        oceancolor='rgb(0,255,255)',

        lonaxis=dict(showgrid=True, gridcolor='rgb(10,10,10)'),

        lataxis=dict(showgrid=True, gridcolor='rgb(10,10,10)')

        )       

)



fig = dict(data=Hits, layout=maplayout)

py.iplot(fig)
# Meteoroids observed falling per region per decade

Europe_part1 = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>37) & (meteorites.reclat<66) 

                    & (meteorites.reclong>-25) & (meteorites.reclong<26)]

Europe_part2 = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>42) & (meteorites.reclat<66) 

                    & (meteorites.reclong>26) & (meteorites.reclong<50)]

frames = [Europe_part1,Europe_part2] # 2 frames needed to go around Turkey

Europe = pd.concat(frames)

Europe = Europe['year']

India = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>8) & (meteorites.reclat<30) 

                    & (meteorites.reclong>70) & (meteorites.reclong<88)]['year']

Japan = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>29) & (meteorites.reclat<41) 

                    & (meteorites.reclong>128) & (meteorites.reclong<145)]['year']

China = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>20) & (meteorites.reclat<120) 

                    & (meteorites.reclong>100) & (meteorites.reclong<125)]['year']

Africa = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>-36) & (meteorites.reclat<36) 

                    & (meteorites.reclong>-17) & (meteorites.reclong<44)]['year']

USA = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>25) & (meteorites.reclat<49) 

                    & (meteorites.reclong>-125) & (meteorites.reclong<-67)]['year']

SouthAmerica = meteorites[(meteorites.fall=='Fell') & (meteorites.reclat>-56) & (meteorites.reclat<12) 

                    & (meteorites.reclong>-81) & (meteorites.reclong<-34)]['year']

#

regionlist=[Japan,India,Europe,USA,China,Africa,SouthAmerica]

namelist=['Japan','India','Europe','USA','China','Africa','SouthAmerica']

colorlist=['blue','orange','pink','yellow','green','cyan','red']

regionsize=[np.ones(Japan.size)/0.38, np.ones(India.size)/3.28, np.ones(Europe.size)/10.38, np.ones(USA.size)/9.83, 

            np.ones(China.size)/9.6, np.ones(Africa.size)/30.37,  np.ones(SouthAmerica.size)/17.84]



plt.figure(figsize=(15,8))

decades=np.arange(1750,2020,10)

plt.hist(regionlist,decades,weights=regionsize,histtype='bar',log=True, color=colorlist)

plt.xlabel('Decade',fontsize=20)

plt.ylabel('Normalized meteorite count',fontsize=20)

plt.title('Number of meteorites seen falling per decade per 1 M km^2',fontsize=20)

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)





plt.legend(namelist,fontsize=18)
Felltype = meteorites[(meteorites.fall=='Fell')]

Mass = Felltype['mass']



scale = 100 # bin width in grams

Mass_binned = [int(int(x)/scale) + 1 for x in Mass]

Mass_list = [int(x) for x in set(Mass_binned)]



Mass_entries = [Mass_binned.count(x) for x in Mass_list]

Mass_error = [math.sqrt(x) for x in Mass_entries]

Mass_list = [float(scale*(x-0.5)) for x in Mass_list]



def func(x,a,b):

    return a/pow(x,b)



xarr1 = np.array(Mass_list)

yarr1 = np.array(Mass_entries)

yarr1_e = np.array(Mass_error)

xarr2 = np.array(Mass_list[:10])

yarr2 = np.array(Mass_entries[:10])

yarr2_e = np.array(Mass_error[:10])

x0 = np.array([1e5,1])



bnds=([100,0.2],[1e6,1.5])

params1, cov = optimization.curve_fit(func, xarr1, yarr1, x0, yarr1_e, bounds=bnds)

params2, cov = optimization.curve_fit(func, xarr2, yarr2, x0, yarr2_e)



xsmooth = np.array([scale*(x-0.5) for x in range(1,10000)])

curve1 = params1[0]/pow(xsmooth,params1[1])

curve2 = params2[0]/pow(xsmooth,params2[1])



trace0 = go.Histogram(

        x=Mass,

        histnorm='probability',

        name='Data',

        opacity=0.6,

        autobinx=False,

        xbins = dict(start=0, end=8, size=scale)

)

trace1 = go.Scatter(

        x=xsmooth,

        y=curve1/Mass.size,

        mode='lines',

        marker=go.Marker(color='rgb(255, 0, 0)'),

        name='Entire fit'

)

trace2 = go.Scatter(

        x=xsmooth,

        y=curve2/Mass.size,

        mode='lines',

        marker=go.Marker(color='rgb(0, 255, 0)'),

        name='Partial fit'

)



layout = go.Layout(

        xaxis=dict(

            title='Meteorite mass (g)',

            type='log',

            range=[0,6]

            ),

        yaxis=dict(

            title='Meteorite conditional probabiltiy',

            type='log',

            autorange=True

            )

)

data = [trace0, trace1, trace2]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)
m=1e5 # assumed mass where P_om can be approximated

M=1e18 # size of K-Pg extinction asteroid 66 M years ago

P_om=0.29*0.03*(6/24.) # ~29% of Earth is land, ~3% of land is urban, ~6h of a day people are awake at night.

P_oM=1 # There's no missing it!

P_mo=params1[0]/m**(params1[1]) / Mass.size

P_Mo=params1[0]/M**(params1[1]) / Mass.size

P_ratio1 = P_Mo/P_mo * (P_om/P_oM)

P_mo=params2[0]/m**(params2[1]) / Mass.size

P_Mo=params2[0]/M**(params2[1]) / Mass.size

P_ratio2 = P_Mo/P_mo * (P_om/P_oM)

print('P_M / P_m:\nestimate 1 = {0}, estimate 2 = {1}'.format(P_ratio1, P_ratio2))
