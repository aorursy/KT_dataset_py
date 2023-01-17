import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!conda install r -y
!conda install -c r rpy2=2.9.4 -y
!pip install thresholdmodeling
import sys

sys.path.insert(1, '../input/newlmoment')
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

import seaborn

from scipy import stats

import lmoment



from pylab import rcParams

rcParams['figure.figsize'] = 15, 8



from rpy2.robjects.packages import importr

import rpy2.robjects.packages as rpackages



base = importr('base')

utils = importr('utils')

utils.chooseCRANmirror(ind=1)

utils.install_packages('POT')



from thresholdmodeling import thresh_modeling
df = pd.read_csv("/kaggle/input/windanalysisjulian/WIND_ANALYSIS.csv", index_col=0)
df.drop(np.nan,axis=0,inplace=True)
df[['Date','Time']] = df.Calender_datetime.str.split(" ",expand=True)
df.drop(['Calender_datetime','CCYYMM','DDHHmm','Date'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Julian.1'] + ' ' + df['Time'])
df['YYYY'] = pd.to_datetime(df['Julian.1']).dt.year

df['MM'] = pd.to_datetime(df['Julian.1']).dt.month

df['DD'] = pd.to_datetime(df['Julian.1']).dt.day
df = df[['Date','Julian.1','YYYY','MM','DD','Time', 'WD', 'WS', 'ETOT', 'TP', 'VMD', 'ETTSea', 'TPSea', 'VMDSea','ETTSw', 'TPSw', 'VMDSw', 'MO1', 'MO2', 'HS', 'DMDIR', 'ANGSPR','INLINE', 'HSURt', 'CSt', 'CDt']]
df.drop(['Julian.1'],axis=1,inplace=True)
df.drop(['ETOT', 'TP', 'VMD', 'ETTSea', 'TPSea', 'VMDSea',

       'ETTSw', 'TPSw', 'VMDSw', 'MO1', 'MO2', 'HS', 'DMDIR', 'ANGSPR',

       'INLINE', 'HSURt', 'CSt', 'CDt'],axis=1, inplace=True)
df.isnull().sum()
df['WD'] = df['WD'].astype(float)

df['WS'] = df['WS'].astype(float)
df1 = df.groupby("YYYY").WS.max()

plt.plot(df1)

plt.title("Maximum Wind Speed in different years(1983-2009)")
!pip install git+https://github.com/OpenHydrology/lmoments3.git

!pip install git+https://github.com/kikocorreoso/scikit-extremes.git
import skextremes as ske
print("MRL PLOT")

thresh_modeling.MRL(df['WS'], 0.05)
print("Shape Stability Plot")

thresh_modeling.Parameter_Stability_plot(df['WS'], 0.05)
threshold_value = 8

values_per_year = 8760
thresh_modeling.gpdfit(df['WS'], threshold_value, 'mle')
thresh_modeling.gpdpdf(df['WS'], threshold_value, 'mle', 'sturges', 0.05)
thresh_modeling.gpdcdf(df['WS'], threshold_value, 'mle', 0.05)
thresh_modeling.qqplot(df['WS'],threshold_value, 'mle', 0.05)
thresh_modeling.ppplot(df['WS'], threshold_value, 'mle', 0.05)
return_period = [1,5,10,20,25,50,75,100,200,500,1000]

for i in return_period:

    print("Return Period:",i)

    thresh_modeling.return_value(df['WS'], threshold_value, 0.05, values_per_year, values_per_year*i, 'mle')
model = ske.models.classic.GEV(df['WS'], fit_method='lmoments', ci=0, ci_method=None, frec = 8760,return_periods=np.array([1,10,20,25,50,100,200]).all())
model.plot_summary()
modelGum = ske.models.classic.Gumbel(df['WS'], fit_method='lmoments', ci=0, ci_method=None,frec = 8760, return_periods=np.array([1,10,20,25,50,100,200]).all())
modelGum.plot_summary()
LMU = lmoment.samlmu(df['WS'])

LMU
weifit = lmoment.pelwei(LMU)
weifit
gevfit = lmoment.pelgev(LMU)
gumfit = lmoment.pelgum(LMU)
T = np.array([1,5,10,20,25,50,75,100,200,500,1000])

gevres = []

weires = []

gumres = []

for i in T:       

        gevST = lmoment.quagev(1.0-(1./i), gevfit)

        gevres.append(gevST)

        weiST = lmoment.quawei(1.0-(1./i), weifit)

        weires.append(weiST)

        gumST = lmoment.quagum(1.0-(1./i), gumfit)

        gumres.append(gumST)
plt.xscale('log')

plt.xlabel('Average Return Interval (Year)')

plt.ylabel('Wind Speed')



 



# draw extreme values from GEV distribution

line1, = plt.plot(T, gevres, 'red',linewidth=5, label='GEV')

line2, = plt.plot(T, weires, 'yellow', linewidth=5, label='WEI')

line3, = plt.plot(T, gumres, 'green', linewidth=5, label='GUM')



labels = [1,5,10,20,25,50,75,100,200,500,1000]

plt.xticks(labels) 



# draw extreme values from observations(empirical distribution)

N    = np.r_[1:len(df['WS'].index)+1]*1.0 #must *1.0 to convert int to float

Nmax = max(N) 



plt.scatter(Nmax/N, sorted(df['WS'])[::-1], color = 'black', facecolors='none', label='Empirical')

plt.legend(handler_map={line2: HandlerLine2D(numpoints=4)})
# prepare probabilites according to observations

N    = np.r_[1:len(df['WS'].index)+1]*1.0 #must *1.0 to convert int to float

Nmax = max(N)   

P0   = (N-1.)/Nmax

P    = np.delete(P0,0)



obs = sorted(df1)[1:]



gevres2 = []

weires2 = []

gumres2 = []

for i in P:

    gevSTo = lmoment.quagev(i, gevfit)

    gevres2.append(gevSTo)

    weiSTo = lmoment.quawei(i, weifit)

    weires2.append(weiSTo)

    gumSTo = lmoment.quagum(i, gumfit)

    gumres2.append(gumSTo)



# do ks test

ks = [('GEV', stats.ks_2samp(obs, gevres2)), ('GUM', stats.ks_2samp(obs, gumres2)), ('WEI', stats.ks_2samp(obs, weires2))]



labels = ['Distribution', 'KS (statistics, pvalue)']

pd.DataFrame(ks, columns=labels)
plt.xscale('log')

plt.xlabel('Average Return Interval (Year)')

plt.ylabel('Wind Speed')

line1, = plt.plot(T, weires, 'yellow', label='WEI',linewidth=3)

labels = [1,5,10,20,25,50,75,100,200,500,1000]

plt.xticks(labels)



# draw extreme values from observations(empirical distribution)

N    = np.r_[1:len(df['WS'].index)+1]*1.0 #must *1.0 to convert int to float

Nmax = max(N)



plt.scatter(Nmax/N, sorted(df['WS'])[::-1], color = 'black', facecolors='none', label='Empirical')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.xscale('log')

plt.xlabel('Average Return Interval (Year)')

plt.ylabel('Wind Speed')

line1, = plt.plot(T, gumres, 'g', label='GUM',linewidth=3)



# draw extreme values from observations(empirical distribution)

N    = np.r_[1:len(df['WS'].index)+1]*1.0 #must *1.0 to convert int to float

Nmax = max(N)



labels = [1,5,10,20,25,50,75,100,200,500,1000]

plt.xticks(labels)



plt.scatter(Nmax/N, sorted(df['WS'])[::-1], color = 'black', facecolors='none', label='Empirical')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.xscale('log')

plt.xlabel('Average Return Interval (Year)')

plt.ylabel('Wind Speed')

line1, = plt.plot(T, gevres, 'red', label='GEV',linewidth=3)



# draw extreme values from observations(empirical distribution)

N    = np.r_[1:len(df['WS'].index)+1]*1.0 #must *1.0 to convert int to float

Nmax = max(N)



labels = [1,5,10,20,25,50,75,100,200,500,1000]

plt.xticks(labels)



plt.scatter(Nmax/N, sorted(df['WS'])[::-1], color = 'black', facecolors='none', label='Empirical')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
return_weibull = pd.DataFrame({"Return Period":T,"Weibull Fit(all)":weires})
return_weibull = return_weibull.set_index("Return Period")
wind_directions = [22.5,67.5,112.5,157.5,202.5,247.5,292.5]

for i in wind_directions:

    if((i-45)<0):

        result = df[(df['WD']<=i) | (df['WD']>=337.5)]

        a = 337.5

    else:

        result = df[(df['WD']<=i+45) & (df['WD']>=i)] 

        a = i+45

    LMU = lmoment.samlmu(result['WS'])

    weifit = lmoment.pelwei(LMU)

    T = np.array([1,5,10,20,25,50,75,100,200,500,1000])

    weires = []

    for j in T:

        weiST = lmoment.quawei(1.0-(1./j), weifit)

        weires.append(weiST)

    if((i-45)<0):

        name = "Direction:"+ str(a)+ "-"+str(i)

    else:

        name =  "Direction:"+ str(i)+ "-"+str(a)

        #name = str(i)+"-"+str(a)             

    return_weibull[name] = weires
return_weibull