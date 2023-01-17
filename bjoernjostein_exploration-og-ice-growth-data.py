import pandas as pd

import seaborn as sns

import numpy as np
!pip install pixiedust
import pixiedust
ConsolidationTcLong=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/2019.08.05_consolidation_Tc_long.csv")

ConsolidationTcShort=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/2019.08.05_consolidation_Tc_short.csv")

ConsolidationTi=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/2019.08.05_consolidation_Ti.csv")

DirectMeasurement=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/directMeasurement.csv")

Shortplasticstringcalibration=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/Shortplasticstringcalibration.csv")
DirectMeasurement.head()
Shortplasticstringcalibration.head()
ConsolidationTi.head()
ConsolidationTcLong.head()
ConsolidationTcShort.head()
def moveRow0toHeader(dataset):

    new_header =dataset.iloc[0]

    dataset = dataset[1:] 

    dataset.columns = new_header 

    return dataset
ConsolidationTcLong=moveRow0toHeader(ConsolidationTcLong)

ConsolidationTcShort=moveRow0toHeader(ConsolidationTcShort)

ConsolidationTi=moveRow0toHeader(ConsolidationTi)
ConsolidationTi.describe()
ConsolidationTi=ConsolidationTi[2:]
ConsolidationTi
import matplotlib.pyplot as plt
def objectToFloat(dataset,startcol):

    for column in dataset.columns[startcol:]:

        dataset[column]=dataset[column].astype(float)

    return dataset
ConsolidationTi=objectToFloat(ConsolidationTi,1)
ConsolidationTi['TIMESTAMP'] = pd.to_datetime(ConsolidationTi['TIMESTAMP'])
ConsolidationTi.dtypes
ConsolidationTcShort.head()
ConsolidationTcShort.keys()
ConsolidationTcShort.rename(columns={'      Time': 'Time','      #3;Dig.Temp;oC': 'Sensor1', '      #4;Dig.Temp;oC': 'Sensor2', '      #5;Dig.Temp;oC': 'Sensor3','      #6;Dig.Temp;oC': 'Sensor4','      #7;Dig.Temp;oC': 'Sensor5', '      #8;Dig.Temp;oC': 'Sensor6','      #9;Dig.Temp;oC': 'Sensor7','      #10;Dig.Temp;oC': 'Sensor8','      #11;Dig.Temp;oC': 'Sensor9','      #12;Dig.Temp;oC': 'Sensor10', '      #HK-Bat;V': 'Battery'}, inplace=True)

ConsolidationTcShort.head()
ConsolidationTcShort=objectToFloat(ConsolidationTcShort,2)

sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor1)

sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor2)

sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor3)

sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor4)

sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor5)

#sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor6)

#sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor7)

#sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor8)

#sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor9)

#sns.lineplot(x=ConsolidationTcShort.Time,y=ConsolidationTcShort.Sensor10)



ConsolidationTcLong.keys()
ConsolidationTcLong.rename(columns={'#1:oC': 'Sensor1','#2:oC': 'Sensor2','#3:oC': 'Sensor3','#4:oC': 'Sensor4','#5:oC': 'Sensor5','#6:oC': 'Sensor6','#7:oC': 'Sensor7','#8:oC': 'Sensor8','#9:oC': 'Sensor9','#10:oC': 'Sensor10','#11:oC': 'Sensor11','#12:oC': 'Sensor12','#13:oC': 'Sensor13','#14:oC': 'Sensor14','#15:oC': 'Sensor15','#16:oC': 'Sensor16','#17:oC': 'Sensor17','#18:oC': 'Sensor18','#19:oC': 'Sensor19','#20:oC': 'Sensor20','#21:oC': 'Sensor21','#22:oC': 'Sensor22','#23:oC': 'Sensor23','#24:oC': 'Sensor24','#25:oC': 'Sensor25','#26:oC': 'Sensor26','#27:oC': 'Sensor27','#28:oC': 'Sensor28','#HK-Bat:V': 'Battery','#HK-Temp:oC': 'HK-Temp','#HK-rH:%': 'HK-Humidity'},inplace=True)
ConsolidationTcLong.head()
ConsolidationTcLong=objectToFloat(ConsolidationTcLong,2)

#display(ConsolidationTcLong)
x=ConsolidationTcShort['Sensor1'].values

freezeBool=x< - 2

freezeBool=freezeBool[12:]

freezingRow = np.where(freezeBool)

freezingNumpy=np.asarray(freezingRow)

freezingPointTcShortSensor1 = freezingNumpy[0][0]

freezingPointTcShortSensor1
x=ConsolidationTcShort['Sensor2'].values

freezeBool=x< - 2

freezeBool=freezeBool[12:]

freezingRow = np.where(freezeBool)

freezingNumpy=np.asarray(freezingRow)

freezingPointTcShortSensor2 = freezingNumpy[0][0]

freezingPointTcShortSensor2
x=ConsolidationTcShort['Sensor3'].values

freezeBool=x< -2

freezeBool=freezeBool[12:]

freezingRow = np.where(freezeBool)

freezingNumpy=np.asarray(freezingRow)

freezingPointTcShortSensor3 = freezingNumpy[0][0]

freezingPointTcShortSensor3
x=ConsolidationTcShort['Sensor4'].values

freezeBool=x< -2

freezeBool=freezeBool[12:]

freezingRow = np.where(freezeBool)

freezingNumpy=np.asarray(freezingRow)

freezingPointTcShortSensor4 = freezingNumpy[0][0]

freezingPointTcShortSensor4
x=ConsolidationTcShort['Sensor5'].values

freezeBool=x< -2

freezeBool=freezeBool[12:]

freezingRow = np.where(freezeBool)

freezingNumpy=np.asarray(freezingRow)

freezingPointTcShortSensor5 = freezingNumpy[0][0]

freezingPointTcShortSensor5
sensorH=[[1,0],[2,-3],[3,-6],[4,-9],[5,-12],[6,-15],[7,-18],[8,-28.8],[9,-39],[10,-48]]
SensorShortThermistor = pd.DataFrame(sensorH, columns = ['Sensor', 'cm from top']) 
SensorShortThermistor
SensorShortThermistorBelowSurface=SensorShortThermistor['cm from top']+5.8
SensorShortThermistorBelowSurface
freezingFromShortThermistor=[freezingPointTcShortSensor2,freezingPointTcShortSensor3,freezingPointTcShortSensor4,freezingPointTcShortSensor5]
freezingFromShortThermistor
ConsolidationTcShort['Time']
height=SensorShortThermistorBelowSurface[2:6]

height
NumpyTime=ConsolidationTcShort['Nr'].values
#interpolated=np.interp(NumpyTime,freezingFromShortThermistor,height)
ConsolidationTcShort.shape
frozenPlot=np.zeros(shape=464,dtype=np.float)

frozenPlot.fill(np.nan)
frozenPlot[freezingFromShortThermistor[0]]=SensorShortThermistorBelowSurface[2]

frozenPlot[freezingFromShortThermistor[1]]=SensorShortThermistorBelowSurface[3]

frozenPlot[freezingFromShortThermistor[2]]=SensorShortThermistorBelowSurface[4]

frozenPlot[freezingFromShortThermistor[3]]=SensorShortThermistorBelowSurface[5]

frozenPlot
from scipy import interpolate

f = interpolate.interp1d(ConsolidationTcShort['Time'], frozenPlot)
len(frozenPlot)
ConsolidationTcShort['Nr']
plt.plot_date(ConsolidationTcShort['Time'],frozenPlot)
ConsolidationTcShort['Time']
def freezingPoint(dataset,startcol,endcol):

    freezingPoint=[]

    for column in dataset.columns[startcol:endcol]:

        x=dataset[column].values

        freezeBool=x< -2

        freezingRow = np.where(freezeBool)

        freezingNumpy=np.asarray(freezingRow)

        freezingPoint = freezingNumpy[0]

        freezingPoint.append(column)

    return freezingPoint
ConsolidationTcShort['EstimatedFreezingShortThermistor']=frozenPlot
ConsolidationTcShort['Time'] = pd.to_datetime(ConsolidationTcShort['Time'])
DirectMeasurement=DirectMeasurement.rename(columns={"ice level": "level_ice"})

DirectMeasurement
DirectMeasurement['IceThickness']=[0,-1.2,-2.1,-2.5,-4.4,-5,-5.7,-7.5,-8]
DirectMeasurement
plt.plot(ConsolidationTcShort['Time'],ConsolidationTcShort['EstimatedFreezingShortThermistor'],'ro')
ConsolidationTcShort['Time']=ConsolidationTcShort['Time'].astype(str)
ConsolidationTcShort['Time'][402]
DirectMeasurement['Time']
ConsolidationTcShort['Time'].head()
ConsolidationTcShort['Time'] = pd.to_datetime(ConsolidationTcShort['Time'])
DirectMeasurement['Time']= pd.to_datetime(DirectMeasurement['Time'])
len(DirectMeasurement)
Timestamp1=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][0])

Timestamp2=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][1])

Timestamp3=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][2])

Timestamp4=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][3])

Timestamp5=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][4])

Timestamp6=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][5])

Timestamp7=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][6])

Timestamp8=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][7])

Timestamp9=np.where(ConsolidationTcShort['Time']==DirectMeasurement['Time'][8])
Timestamp1=Timestamp1[0]

Timestamp2=Timestamp2[0]

Timestamp3=Timestamp3[0]

Timestamp4=Timestamp4[0]

Timestamp5=Timestamp5[0]

Timestamp6=Timestamp6[0]

Timestamp7=Timestamp7[0]

Timestamp8=Timestamp8[0]

Timestamp9=Timestamp9[0]
DirectMeasurement
ConsolidationTcShort.shape
measureArray=np.zeros(shape=464)

measureArray.fill(np.nan)
measureArray[Timestamp1]=DirectMeasurement['IceThickness'][0]

measureArray[Timestamp2]=DirectMeasurement['IceThickness'][1]

measureArray[Timestamp3]=DirectMeasurement['IceThickness'][2]

measureArray[Timestamp4]=DirectMeasurement['IceThickness'][4]

measureArray[Timestamp5]=DirectMeasurement['IceThickness'][5]

measureArray[Timestamp6]=DirectMeasurement['IceThickness'][5]

measureArray[Timestamp7]=DirectMeasurement['IceThickness'][6]

measureArray[Timestamp8]=DirectMeasurement['IceThickness'][7]

measureArray[Timestamp9]=DirectMeasurement['IceThickness'][8]
ConsolidationTcShort['MeasuredIceThickness']=measureArray
ConsolidationTcShort.head()
plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['EstimatedFreezingShortThermistor'], 'bx')

#plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['MeasuredIceThickness'],'ro')

plt.show()
ConsolidationTcShort
ConsolidationTcShort['Time'][462]
len(ConsolidationTcShort)
MeasuredConsolidatedIce=np.zeros(464)

MeasuredConsolidatedIce.fill(np.nan)

MeasuredConsolidatedIce[460]=-15
MeasuredConsolidatedIce
ConsolidationTcShort['MeasuredConsolidatedIce']=MeasuredConsolidatedIce
plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['MeasuredConsolidatedIce'], 'bx')

plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['MeasuredIceThickness'], 'ro')
ConsolidationTcShort['EstimatedFreezingShortThermistor'].head()
type(ConsolidationTcShort['Nr'])
TestConsolidationTcShort=ConsolidationTcShort

TestConsolidationTcShort['EstimatedFreezingShortThermistor'].dropna().index
TestMIT=TestConsolidationTcShort['EstimatedFreezingShortThermistor'].dropna().values.reshape(-1, 1)

TestNR=TestConsolidationTcShort['EstimatedFreezingShortThermistor'].dropna().index.values.reshape(-1, 1)
len(TestMIT)
len(TestNR)
testtid=np.arange(1,464)
from scipy import interpolate
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(TestNR,TestMIT)
lm.coef_
lm.intercept_
y=lm.coef_* x +lm.intercept_
x = np.array(range(464))

y=y.reshape(-1,1)
y=y.ravel()
y.shape
x.shape
y
plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['EstimatedFreezingShortThermistor'], 'bx')



plt.plot(ConsolidationTcShort['Nr'],ConsolidationTcShort['MeasuredIceThickness'],'ro')

plt.show()
#sns.lineplot(ConsolidationTcShort.EstimatedFreezingShortThermistor)
plt.plot(x,y)

plt.plot(x, ConsolidationTcShort['MeasuredConsolidatedIce'],'gx')

plt.plot(x, ConsolidationTcShort['EstimatedFreezingShortThermistor'],'ro')
lm.coef_* 464 +lm.intercept_
x
ConsolidationTcShort['Time'].tail()
# #############################################################################

# Display results

plt.plot(x, y , "r-", color='C0',label=r'$h_{c-thermistor}$') #r- lager en kontinuerlig "strek" av xnew og ypredict

plt.plot(x, ConsolidationTcShort['EstimatedFreezingShortThermistor'],'ro', color='C5',label=r'$T_{thermistor}>-2℃$')

plt.plot(x, ConsolidationTcShort['MeasuredConsolidatedIce'],'gx',label=r'$h_{c-measured}$')

#plt.axis([0,1.0,0, -12.5]) #Setter aksene på plottet

plt.xlabel('Time') #Setter navn på x-akse

plt.xticks([])

plt.ylabel('Under ice surface (cm)') #Setter navn på y-akse

plt.title(r'Estimated ice thickness using short thermistor - group 2') #Setter navn på plottet

plt.legend(loc='upper right')

plt.savefig('EstimatedIceThicknessShortThermistorGroup2.png')

display()
# The code was removed by Watson Studio for sharing.
consolidation_Tc_short_all=pd.read_csv("/kaggle/input/unis-ice-measurements-2019/consolidation_Tc_short_all.csv")
consolidation_Tc_short_all.head()
consolidation_Tc_short_all=moveRow0toHeader(consolidation_Tc_short_all)
consolidation_Tc_short_all.head()
consolidation_Tc_short_all=consolidation_Tc_short_all.replace(r'<NO VALUE>', np.nan, regex=True)
consolidation_Tc_short_all=objectToFloat(consolidation_Tc_short_all,2)
consolidation_Tc_short_all['      Time']
consolidation_Tc_short_all.rename(columns={'      Time': 'Time','      #3;Dig.Temp;oC': 'Sensor1', '      #4;Dig.Temp;oC': 'Sensor2', '      #5;Dig.Temp;oC': 'Sensor3','      #6;Dig.Temp;oC': 'Sensor4','      #7;Dig.Temp;oC': 'Sensor5', '      #8;Dig.Temp;oC': 'Sensor6','      #9;Dig.Temp;oC': 'Sensor7','      #10;Dig.Temp;oC': 'Sensor8','      #11;Dig.Temp;oC': 'Sensor9','      #12;Dig.Temp;oC': 'Sensor10', '      #HK-Bat;V': 'Battery'}, inplace=True)

consolidation_Tc_short_all['Time'] = pd.to_datetime(consolidation_Tc_short_all['Time'])
type(consolidation_Tc_short_all)
wrongtextindex=consolidation_Tc_short_all['Nr']=="<Parameters changed ('A70396_0265_310719.par')>"
np.where(wrongtextindex)
consolidation_Tc_short_all['Nr'][2]='NaN'
consolidation_Tc_short_all['Nr'][29]='NaN'
consolidation_Tc_short_all['Nr'][36]='NaN'
consolidation_Tc_short_all['Nr']
consolidation_Tc_short_all.dtypes
np.where(consolidation_Tc_short_all['Nr']=="<Parameters changed ('A70396_0267_020819.par')>")
consolidation_Tc_short_all['Nr']=consolidation_Tc_short_all['Nr'].astype(str).astype(float)
consolidation_Tc_short_all.dtypes
#display(consolidation_Tc_short_all)
plt.plot_date(consolidation_Tc_short_all['Time'],consolidation_Tc_short_all['Sensor1'])
sns.scatterplot(consolidation_Tc_short_all['Time'],consolidation_Tc_short_all['Sensor1'])
consolidation_Tc_short_all['Time']
#sns.lineplot([consolidation_Tc_short_all.Sensor1, consolidation_Tc_short_all.Sensor3])
np.where(consolidation_Tc_short_all['Time']=='2019-05-08 15:00:00')
len(consolidation_Tc_short_all['Time'])
OtherGroups=consolidation_Tc_short_all[499:]
#display(OtherGroups)
len(ConsolidationTcShort['Sensor1'])
#freezeP=np.ones(464)*-1.6
FDD_a=np.trapz(ConsolidationTcShort['Sensor5']-ConsolidationTcShort['Sensor1'])
FDD_a_cumsum=np.cumsum(ConsolidationTcShort['Sensor5']-ConsolidationTcShort['Sensor1'])

FDD_a_cumsum.tail()
plt.plot(ConsolidationTcShort['Sensor1'],color='blue')

plt.plot(ConsolidationTcShort['Sensor5'], color='maroon', label='Freezing point')

plt.fill_between(x,ConsolidationTcShort['Sensor5'],ConsolidationTcShort['Sensor1'],color='lightgrey')

plt.legend(loc='upper right')

plt.title(r'$FDD_a$')

plt.show()
FDD_i = np.trapz(ConsolidationTcShort['Sensor5']-ConsolidationTcShort['Sensor3'])
FDD_i_cumsum=np.cumsum(ConsolidationTcShort['Sensor5']-ConsolidationTcShort['Sensor3'])

FDD_i_cumsum.tail()
FDD_i
plt.plot(ConsolidationTcShort['Sensor3'], color='blue')

plt.plot(ConsolidationTcShort['Sensor5'], color='maroon', label='Freezing point')

plt.fill_between(x,ConsolidationTcShort['Sensor5'],ConsolidationTcShort['Sensor3'],color='lightgrey')

plt.legend(loc='upper right')

plt.title(r'$FDD_i$')

plt.show()
import math
k_i = 2.1 #Thermal conductivity of ice

rho_i=916.7 #

l_i=333.4*(10**3) #Latent heat

del1=math.sqrt((2 * k_i)/(rho_i*l_i))
stephan=del1*FDD_i_cumsum*(60/10)*24*(3+5/24)
#plt.plot(ConsolidationTcShort['Sensor5'])

#plt.plot(ConsolidationTcShort['Sensor3'])

plt.plot(stephan,label='Stefan´s law')

plt.legend()
max(stephan)
consolidation_Tc_short_all.dropna(subset=['Nr'], inplace=true)
consolidation_Tc_short_all=consolidation_Tc_short_all[23:]

consolidation_Tc_short_all
plt.plot(consolidation_Tc_short_all['Sensor3'])
#display(ConsolidationTcShort[5:])
plt.plot(ConsolidationTcShort['Sensor1'][17:])
ConsolidationTcShort[['Time','MeasuredIceThickness','EstimatedFreezingShortThermistor']][16:].to_csv(r'MeasuredIceAndShortThermistor.csv')

#ConsolidationTcShort[['Time','MeasuredIceThickness','EstimatedFreezingShortThermistor']][16:]
plt.plot(ConsolidationTcShort['EstimatedFreezingShortThermistor'],'ro')

#plt.plot(ConsolidationTcShort['EstimatedFreezingShortThermistor'],'bx')

plt.plot(ConsolidationTcShort['MeasuredIceThickness'],'bx')
Fix=pd.read_csv("MeasuredIceAndShortThermistor.csv")
Fix2=pd.read_csv("MeasuredIceAndShortThermistor.csv")
Fix.dropna(subset=['MeasuredIceThickness'], inplace=true)
Fix=Fix[['Time','MeasuredIceThickness']]
Fix.to_csv('measuredIceThickness.csv')
Fix2.dropna(subset=['EstimatedFreezingShortThermistor'],inplace=true)
Fix2.to_csv('ShortThermistor.csv')
Fix2=Fix2[['Time','EstimatedFreezingShortThermistor']]
Fix2.to_csv('ShortThermistor.csv')
Fix
ConsolidationTcLong.head()