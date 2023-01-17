# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import datetime
import statistics
import scipy as sp
import scipy.optimize
import scipy.integrate
import logging
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def dSCRdt(t,y,beta,gama):
    """
    Parameters
    ----------
    t: double
        time
    y: ndarray double
        numpy nd array double like [S,C,R]
    beta: double
        contagion propagation coefficient
    gama: double
        recovery coefficient
    Returns
    -------
    dydt: ndarray double
        time derivative of S,C,R, in that order

    This is the implementation of the SIR compartmentalized model,
    with a slight modification : we sum the I equation to the R one
    to get the derivative for "C", the total (C)ases.
    
    This is to adapt the current measurements done by governments :
    they measure the discovered (C)ases, and Deaths + Survivals
    or (R)ecovered
    """

    S=y[0]
    C=y[1]
    I=y[1]-y[2]
    R=y[2]
    N=y[0]+y[1]
    
    
    dSdt = - beta*I*S/N
    dCdt = beta*I*S/N
    dRdt = gama*I
    
    return [dSdt,dCdt,dRdt]

def SRCTau(t,t0,S0,O0,param):
    """
    Parameters
    ----------
    t :list Double
        vector especificando os instantes de tempo a devolver,
        assume-se que haja um intervalo de 1
    t0: double
        instante de tempo da primeira observação
    S0: double
        População sã inicial, antes de haver a doença
    O0: double
        nr de casos observados no caso da primeira observação 
    param: ndarray double
        o array numpy dos coeficientes, na forma [beta,gamma,Tau]
        beta é a ritmo de contagio
        gamma o ritmo de recuperação
        Tau o periodo de atraso das observações
    
    Returns
    -------
    Y: ndarray double
        as colunas correspondem aos instantes do tempo
        as linhas correspondem a SIRO:
        Susceptible
        Cases
        Recovered
        Observed

    Implementa o modelo SIR
    """
    y0 = np.array([S0-O0,O0,0])
    if t[0]-param[2]!=0:
        T = [t[0]-param[2]]
    else:
        T=[]
    T.extend(t)
    logging.debug(f'(SRCTau) Time instants {T}')
    tInt = (T[0],T[len(T)-1])
    sol = sp.integrate.solve_ivp(dSCRdt,tInt,y0,t_eval=T,args=(param[0],param[1]))
    Y = sol.y


    if t[0]-param[2]==0:
        y1 = Y
        y2 = Y[1,:]
    else:
        y2 = Y[1,0:len(t)]
        y1 = Y[:,len(T)-len(t)-1:len(T)-1]
    
    logging.debug(f'(SRCTau) outputted solution {sol}')
    logging.debug(f'(SRCTau) Original time interval length {len(t)}, vector is {t}')
    logging.debug(f'(SRCTau) Extended time interval length {len(T)}, vector is {T}')
    logging.debug(f'(SRCTau) SRC Values {y1}')
    logging.debug(f'(SRCTau) O Values (observed with delay) {y2}')
    Y = np.vstack((y1,y2))

    return Y

def errOD(param,t,t0,S0,O0,O,D):
    Y=SRCTau(t,t0,S0,O0,[param[0],param[1],param[2]])
    alpha = param[3]
    out=0
    for i in range(0,len(O)):
        SSR = (Y[3,i]-O[i])**2+(alpha*Y[2,i]-D[i])**2
    return SSR
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
print(train)
popStat_NorthQuay = pd.read_csv("/kaggle/input/popdatacsv/Data Join - RELEASE.csv")
print(popStat_NorthQuay)
sStat = popStat_NorthQuay.loc[:,['Province_State','Country_Region','TRUE POPULATION']] 
simpleTrain = pd.merge(train,sStat,on=['Province_State','Country_Region']).drop_duplicates(subset=['Province_State','Country_Region','Date']).fillna(0) #tenho que forçar os NaN a 0 senão torna-se complicado recuperar os grupos com group by
simpleTrain= simpleTrain[simpleTrain['ConfirmedCases']>0]
simpleTrain['pdDate'] = pd.to_datetime(simpleTrain['Date'])
simpleTrain['dayYear'] = simpleTrain['pdDate'].dt.dayofyear #need to convert to pandas date time, and then use DT to access day of year
simpleTrain['TRUE POPULATION']= pd.to_numeric(simpleTrain['TRUE POPULATION'].str.replace(',', ''))
# declaração das variaveis para calculo do modelo

print(simpleTrain)
regionList = simpleTrain.loc[:,['Province_State','Country_Region']].drop_duplicates()
regionList['beta']=0.0
regionList['gama']=0.0
regionList['tau']=0.0
regionList['alpha']=0.0
regionList['SSTOT']=0.0
regionList['SSRES']=0.0
regionList['t0']=0.0
regionList['R']=0.0
regionList['Cf']=0.0
regionList['Sf']=0.0
ssTot = 0.0
for row_index, row in regionList.iterrows():
    #exemplo de como modificar valores.. regionList.at[row_index,'beta']=i
    #Linha a implementar res = sp.optimize.least_squares(errOD,[1.0,1.0,14.1,.02], bounds=(0,np.inf),args=(daysofYear,t0,okPop,first observarion,Obs,Deat))
    ssTot = 0.0
    provCountry = regionList.loc[row_index,'Province_State':'Country_Region'].tolist()
    timePeriod = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])].loc[:,'dayYear'].tolist()
    S0 = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])].loc[:,'TRUE POPULATION'].tolist()
    S0=S0[0]
    Obs = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])].loc[:,'ConfirmedCases'].tolist()
    Dead = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])].loc[:,'Fatalities'].tolist()
    
    res = sp.optimize.least_squares(errOD,[1.0,1.0,14.1,.02], bounds=(0,np.inf),args=(timePeriod,timePeriod[0],S0,Obs[0],Obs,Dead))
    for item in Obs:
        ssTot = ssTot + (item-statistics.mean(Obs))**2
    for item in Dead:
        ssTot = ssTot + (item-statistics.mean(Dead))**2
        
    regionList.at[row_index,'beta']=res.x[0]
    regionList.at[row_index,'gama']=res.x[1]
    regionList.at[row_index,'tau']=res.x[2]
    regionList.at[row_index,'alpha']=res.x[3]
    regionList.at[row_index,'t0']=timePeriod[0]
    regionList.at[row_index,'SSRES']=res.fun
    regionList.at[row_index,'SSTOT']=ssTot
    Y=SRCTau(timePeriod,timePeriod[0],S0,int(Obs[0]),[res.x[0],res.x[1],res.x[2]])
    regionList.at[row_index,'Cf']=max(Y[1,:])
    if 1-res.fun/ssTot >= 0:
        regionList.at[row_index,'R']=math.sqrt(1-res.fun/ssTot)
    else:
        regionList.at[row_index,'R']=0
print(regionList)
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
test = test.fillna(0)
test['pdDate'] = pd.to_datetime(test['Date'])
test['dayYear'] = test['pdDate'].dt.dayofyear #need to convert to pandas date time, and then use DT to access day of year
test['ConfirmedCases'] = 0.0
test['Fatalities'] = 0.0
print(test)
for row_index, row in regionList.iterrows():
    #exemplo de como modificar valores.. regionList.at[row_index,'beta']=i
    #Linha a implementar res = sp.optimize.least_squares(errOD,[1.0,1.0,14.1,.02], bounds=(0,np.inf),args=(daysofYear,t0,okPop,first observarion,Obs,Deat))
    ssTot = 0.0
    provCountry = regionList.loc[row_index,'Province_State':'Country_Region'].tolist()
    t0 = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])&(simpleTrain['ConfirmedCases']>0)].loc[:,'dayYear'].tolist()
    t0 = max([min(t0),93])
    Cf = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])&(simpleTrain['dayYear']>=t0)].loc[:,'ConfirmedCases'].tolist()
    Cf = min(Cf)

    
    print(f't0 is {t0} and Cf is {Cf}')
    
    timePeriod = test[(test['Province_State']==provCountry[0])&(test['Country_Region']==provCountry[1])&(test['dayYear']>=t0)].loc[:,'dayYear'].tolist()
    S0 = simpleTrain[(simpleTrain['Province_State']==provCountry[0])&(simpleTrain['Country_Region']==provCountry[1])].loc[:,'TRUE POPULATION'].tolist()
    S0=S0[0]
    
    beta = regionList.loc[row_index,'beta']
    gama = regionList.loc[row_index,'gama']
    tau = regionList.loc[row_index,'tau']
    alpha = regionList.loc[row_index,'alpha']
    
    Y=SRCTau(timePeriod,timePeriod[0],S0-Cf,Cf,[beta,gama,tau])
    
    test.at[(test['Province_State']==provCountry[0])&(test['Country_Region']==provCountry[1])&(test['dayYear']>=t0),'ConfirmedCases']= Y[3,:]
    test.at[(test['Province_State']==provCountry[0])&(test['Country_Region']==provCountry[1])&(test['dayYear']>=t0),'Fatalities']=alpha*Y[2,:]
submission = test.drop(['Province_State','Country_Region','Date','pdDate','dayYear'],axis=1)
submission.to_csv("submission.csv")