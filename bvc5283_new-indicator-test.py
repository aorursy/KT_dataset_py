import numpy as np

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from datetime import datetime



import plotly.plotly as py



print(os.listdir("../input"))

ba=pd.read_csv("../input/BA.csv")

ba.info()

ba.head()
ba.describe()
ba=ba[ba.Volume!=0]
# Change data format

ba['Date']=pd.to_datetime(ba['Date'],infer_datetime_format=True)

ba.set_index('Date',inplace=True)
# Agg time to weekly

logic = {'Open'  : 'first',

        'High'  : 'max',

        'Low'   : 'min',

        'Close' : 'last',

        'Volume': 'mean'}

offset=pd.offsets.timedelta(days=-6)

ba_w=ba.resample('W',loffset=offset,how=logic)



ba_w=ba['2010':'2015']
ba_w.tail()
hold=ba_w['Close'].iloc[-1]/ba_w['Close'].iloc[0]-1

hold
# Value setup

span=14

n=1

SF=1.01

volume=ba_w['Volume']

# Velocity

velocity=ba_w.Close.diff()

# Acceleration

acceleration_t=velocity.diff()

acceleration=acceleration_t.ewm(span=span,min_periods=n).mean()

# Force

Force=volume*acceleration

# Momentum

Momentum=volume*velocity

# Range

Range=ba_w.High-ba_w.Low

# RealBody

RealBody=abs(ba_w['Close']-ba_w['Open'])
# LinearForceToday= volume*Acceleration

LFI_T=acceleration*volume

# LinearForce= ema(LinearForceToday)

ba_w['LFI']=LFI_T.ewm(span=span,min_periods=n).mean()





# IntegralForceToday= Momentum

IFI_T=Momentum

# IntegralForce= ema(IntegralForceToday)

ba_w['IFI']=IFI_T.ewm(span=span,min_periods=n).mean()

plt.figure(figsize = (25,7))

plt.plot(ba_w['LFI'])

plt.plot(ba_w['IFI'])

plt.legend()
# PressureToday= Force/Range

PRI_T=Force/Range

# Pressure= ema(PressureToday)

ba_w['PRI']=PRI_T.ewm(span=span,min_periods=n).mean()



# IntegralPressureToday= Momentum/Range

IPRI_T=Momentum/Range

# IntegralPressure= ema(IntegralPressureToday)

ba_w['IPRI']=IPRI_T.ewm(span=span,min_periods=n).mean()

plt.figure(figsize = (25,7))

plt.plot(ba_w['PRI'])

plt.plot(ba_w['IPRI'])

plt.legend()
# StrengthToday= Force/(SF*Range-RealBody)

SI_T=Force/(SF*Range-RealBody)

# Strength= ema(StrengthToday,n)

ba_w['SI']=SI_T.ewm(span=span,min_periods=n).mean()





# IntegralStrengthToday= Momentum/((SF*Range)-RealBody)

ISI_T=Momentum/((SF*Range)-RealBody)

# IntegralStrength= ema(IntegralStrengthToday,n)

ba_w['ISI']=ISI_T.ewm(span=span,min_periods=n).mean()

plt.figure(figsize = (25,7))

plt.plot(ba_w['SI'])

plt.plot(ba_w['ISI'])

plt.legend()
# PowerEquation= Force*Velocity

PWRI_T=Force*velocity

ba_w['PWRI']=np.where(velocity>0,abs(PWRI_T),-abs(PWRI_T))

# Power= ema(PowerToday)

ba_w['PWRI']=ba_w['PWRI'].ewm(span=span,min_periods=n).mean()





# IntegralPowerEquation= Momentum*Velocity

IPWRI_T=Momentum*velocity

ba_w['IPWRI']=np.where(velocity>0,abs(IPWRI_T),-abs(IPWRI_T))

# IntegralPower= ema(IntegralPowerToday)

ba_w['IPWRI']=ba_w['IPWRI'].ewm(span=span,min_periods=n).mean()

plt.figure(figsize = (25,7))

plt.plot(ba_w['PWRI'])

plt.plot(ba_w['IPWRI'])

plt.legend()
# IntensityEquation = Force*Velocity/Range

II_T=Force*velocity/Range

ba_w['II']=np.where(velocity>0,abs(II_T),-abs(II_T))

# Intensity= ema(IntensityToday)

ba_w['II']=ba_w['II'].ewm(span=span,min_periods=n).mean()



# IntegralIntensityEquation= Momentum*Velocity/Range

III_T=Momentum*velocity/Range

ba_w['III']=np.where(velocity>0,abs(III_T),-abs(III_T))

# IntegralIntensity= ema(IntegralIntensityToday)

ba_w['III']=ba_w['III'].ewm(span=span,min_periods=n).mean()
plt.figure(figsize = (25,7))

plt.plot(ba_w['II'])

plt.plot(ba_w['III'])

plt.legend()
# DynamicStrengthEquation= Force*Velocity/(SF*Range-RealBody)

DSI_T=Force*velocity/(SF*Range-RealBody)

ba_w['DSI']=np.where(velocity>0,abs(DSI_T),-abs(DSI_T))

# DynamicStrength= ema(DynamicStrengthToday)

ba_w['DSI']=ba_w['DSI'].ewm(span=span,min_periods=n).mean()



# IntegralDynamicStrengthEquation= Momentum*Velocity/(SF*Range-RealBody)

IDSI_T=Momentum*velocity/(SF*Range-RealBody)

ba_w['IDSI']=np.where(velocity>0,abs(IDSI_T),-abs(IDSI_T))

#IntegralDynamicStrength= ema(IntegralDynamicStrengthToday)

ba_w['IDSI']=ba_w['IDSI'].ewm(span=span,min_periods=n).mean()
plt.figure(figsize = (25,7))

plt.plot(ba_w['DSI'])

plt.plot(ba_w['IDSI'])

plt.legend()
# 1,-1,0 represent buy, sell, no action signal

a=ba_w[['LFI','PRI','SI','PWRI','II','DSI']]

b=ba_w[['IFI','IPRI','ISI','IPWRI','III','IDSI']]

ba_w['Signal_L']=np.where((a > 0).all(axis=1),1,np.where((a < 0).all(axis=1),-1,0))

ba_w['Signal_I']=np.where((b > 0).all(axis=1),1,np.where((b < 0).all(axis=1),-1,0))

ba_w['Signal']=np.where((ba_w[['Signal_I','Signal_L']]==1).all(axis=1),1,

                        np.where((ba_w[['Signal_I','Signal_L']]==-1).all(axis=1),-1,0))
Strength_I=ba_w.mask(ba_w.Signal_I != 1)

Weak_I=ba_w.mask(ba_w.Signal_I != -1)

Neutral_I=ba_w.mask(ba_w.Signal_I != 0)



Strength_L=ba_w.mask(ba_w.Signal_L != 1)

Weak_L=ba_w.mask(ba_w.Signal_L != -1)

Neutral_L=ba_w.mask(ba_w.Signal_L != 0)
plt.figure(figsize = (10,15))

plt.subplot(2,1,1)

plt.bar(Strength_I.index, Strength_I.Close,color='g')

plt.bar(Weak_I.index, Weak_I.Close,color='r')

plt.bar(Neutral_I.index, Neutral_I.Close,color='black')



plt.subplot(2,1,2)

plt.bar(Strength_L.index, Strength_L.Close,color='g')

plt.bar(Weak_L.index, Weak_L.Close,color='r')

plt.bar(Neutral_L.index, Neutral_L.Close,color='black')



plt.show()
def calc_MACD(indata):



    indata['26 ema'] = indata['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()

    indata['12 ema'] = indata['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()

    indata['MACD'] = (indata['12 ema'] - indata['26 ema'])

    indata['Ind_MACD'] = indata['MACD'].ewm( span=9).mean()

    indata['M'] = indata['MACD'] - indata['Ind_MACD']

    outdata = indata[['Close','Open','High','Low','Volume','MACD','Ind_MACD','M']]

    return outdata
outdata=calc_MACD(ba_w)
columns=['LFI','IFI','PRI','SI','ISI','PWRI','IPWRI','II','III','DSI','IDSI','M']

for name in columns:

    ba_w['Signal_'+name]=np.where(ba_w[name]> 0,1,np.where(ba_w[name]< 0,-1,0))
def signal_pair(col_name):

    ba_w['trade_price']=ba_w['Close'].shift(-1)*ba_w[col_name]

    is_add=False

    pair=[]

    trade=[]

    for i in ba_w['trade_price']:

        if i >0 and is_add==False:

            pair.append(i)

            is_add=True

        elif i <0 and is_add==True:

            pair.append(-i)

            trade.append(pair)

            pair=[]

            is_add=False

    trade = pd.DataFrame.from_records(trade,columns=['buy','sell'])

    trade['return']=trade.sell/trade.buy-1

    return trade
Return=[]

Count=[]

Name=[]

col_name=[col for col in ba_w.columns if col.startswith('Signal')]

for name in col_name:

    df=signal_pair(name)



    Return.append(df['return'].sum())    

    Count.append(df['return'].count())

    Name.append(name)

result=pd.DataFrame({'Return' : Return,

                                'Count' : Count,

                                'Name' : Name })
result.set_index('Name',inplace=True)

result.sort_index()
ba_w.tail(1)
result.to_csv('New_Indicator.csv')