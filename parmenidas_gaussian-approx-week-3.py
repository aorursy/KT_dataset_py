# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import datetime

from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error as mse

def gaussian(x,a,mu,sigma):

    return a*np.exp(-(x-mu)**2./(2.*sigma**2.))

def GetRegion(regional,f="Fatalities"):

    y=regional[f].values

    y=np.diff(y,prepend=[0])

    x=regional["Date"].values.astype(np.datetime64)

    x0=x[0].copy()

    dx=x-x0

    max_y=np.max(y)

    y=y/np.max(y)

    return dx,y,x0,max_y

def SetRegion(regional,x0):

    x=regional["Date"].values.astype(np.datetime64)

    f_id=regional["ForecastId"].values.astype(np.int)

    dx=x-x0

    return dx,f_id

def doFit(region,typef="ConfirmedCases"):

    dx,y,x0,max_y=GetRegion(region,f=typef)

    n=dx.shape[0]

    mu=np.sum(dx.astype(np.float)*y)/n

    sigma=np.sum(y*np.square(dx.astype(np.float)-mu))/n

    popt,pcov=curve_fit(gaussian,dx,y,p0=[1.,mu,sigma],maxfev=1000000)

    return popt,pcov,x0,max_y

def AppendPopt(list_par,regional,f="ConfirmedCases",cutoff=1000):

    if regional[f].max()>cutoff:

        popt,*_,max_y=doFit(regional,f)

        if popt[0]<100:

            list_par.append(np.append(popt,max_y))

def ShiftDistr(region,distr,typef):

    dx,y,x0,max_y=GetRegion(region,f=typef)

    shift=dx.shape[0]

    min_mse=1000000.

    best_popt=distr[0]  #just to initialize, no real meaning

    for i in distr:

        #print(i)

        for mu in np.arange(-shift,shift):

            curve=gaussian(dx.astype(np.int),i[0],i[1]+mu,i[2])

            m=mse(y,curve)

            if m<min_mse:

                min_mse=m

                best_popt=i

                best_popt[1]=i[1]+mu



    return best_popt,x0,max_y

def PredictDistr(regional,test_dx,par,f):

    if regional[f].max()>0:                    

        popt,_,max_y=ShiftDistr(regional,par,typef=f)            

        curve=max_y*gaussian(test_dx.astype(np.int),*popt[0:3])

        print(popt,end=' ')            

    else:

        curve=np.zeros(test_dx.shape[0])

        print('[0,0,0]',end=' ') 



    return curve

 



df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv").fillna(0)

countries=df["Country_Region"].unique()



list_par=[]

list_fat=[]

for i in countries: 



    tmp=df.loc[df["Country_Region"]==i]

    states=tmp["Province_State"].unique()



    if states.size<=1:

        AppendPopt(list_par,tmp,f="ConfirmedCases",cutoff=10000)

        AppendPopt(list_fat,tmp,f="Fatalities",cutoff=1000)

    else:

        for k in states:

            tmpk=tmp.loc[tmp["Province_State"]==k]

            AppendPopt(list_par,tmpk,f="ConfirmedCases",cutoff=10000)

            AppendPopt(list_fat,tmpk,f="Fatalities",cutoff=1000)



print(list_par)

print(list_fat)





test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv").fillna(0)

fout=open("submission.csv","w")

print("ForecastId,ConfirmedCases,Fatalities",file=fout)



startDate=datetime.date(2020,3,26)

for i in countries: 



    tmp=df.loc[df["Country_Region"]==i]

    states=tmp["Province_State"].unique()

    x0=tmp["Date"].values.astype(np.datetime64)[0]





    if states.size<=1:

        print(i,end=' ')

        scc=tmp.loc[tmp["Date"].values.astype(np.datetime64)==startDate]["ConfirmedCases"].values

        sfat=tmp.loc[tmp["Date"].values.astype(np.datetime64)==startDate]["Fatalities"].values

        test_tmp=test.loc[test["Country_Region"]==i] 

        test_dx,test_id=SetRegion(test_tmp,x0)

        curve1=PredictDistr(tmp,test_dx,list_par,"ConfirmedCases")

        curve2=PredictDistr(tmp,test_dx,list_fat,"Fatalities")

        print('')



        for i_id,i_c1,i_c2 in zip(test_id,scc+np.cumsum(curve1),sfat+np.cumsum(curve2)):

            print("{:d},{:d},{:d}".format(i_id,np.round(i_c1).astype(np.int),np.round(i_c2).astype(np.int)),file=fout)



    else:

        for k in states:

            print(i,",",k,end=' ')

            tmpk=tmp.loc[tmp["Province_State"]==k]

            test_tmp=test.loc[test["Country_Region"]==i]

            test_tmpk=test_tmp.loc[test_tmp["Province_State"]==k]

            scc=tmpk.loc[tmpk["Date"].values.astype(np.datetime64)==startDate]["ConfirmedCases"].values

            sfat=tmpk.loc[tmpk["Date"].values.astype(np.datetime64)==startDate]["Fatalities"].values

            test_dx,test_id=SetRegion(test_tmpk,x0)

            curve1=PredictDistr(tmpk,test_dx,list_par,"ConfirmedCases")

            curve2=PredictDistr(tmpk,test_dx,list_fat,"Fatalities")

            print('')

                

            for i_id,i_c1,i_c2 in zip(test_id,scc+np.cumsum(curve1),sfat+np.cumsum(curve2)):

                print("{:d},{:d},{:d}".format(i_id,np.round(i_c1).astype(np.int),np.round(i_c2).astype(np.int)),file=fout)



fout.close()