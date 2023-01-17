import numpy as np

import matplotlib.pyplot as plt

import differint.differint as df



from sklearn import preprocessing as pr

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestRegressor



import csv

from os import listdir

from os.path import isfile, join
#General plot style

def PlotStyle(Axes,Title,x_label,y_label):

    

    Axes.spines['top'].set_visible(False)

    Axes.spines['right'].set_visible(False)

    Axes.spines['bottom'].set_visible(True)

    Axes.spines['left'].set_visible(True)

    Axes.xaxis.set_tick_params(labelsize=12)

    Axes.yaxis.set_tick_params(labelsize=12)

    Axes.set_ylabel(y_label,fontsize=14)

    Axes.set_xlabel(x_label,fontsize=14)

    Axes.set_title(Title)



def MinimalLoader(filename, delimiter=',', dtype=float):

  

  """

  modified from SO answer by Joe Kington

  """

  def IterFunc():

    with open(filename, 'r') as infile:

      for line in infile:

        line = line.rstrip().split(delimiter)

        for item in line:

          yield dtype(item)

    MinimalLoader.rowlength = len(line)



  data = np.fromiter(IterFunc(), dtype=dtype)

  data = data.reshape((-1, MinimalLoader.rowlength))



  return data



#Mean and standard deviation of a time series 

def GetScalerParameters(TimeSeries):

  return np.mean(TimeSeries),np.std(TimeSeries)



#Generates a Zero mean and unit variance signal 

def MakeScaledSeries(Signal,MeanValue,StdValue):

  StandardSignal=[(val-MeanValue)/StdValue for val in Signal]

  return StandardSignal



#Makes a matrix of time series samples 

def MakeSamplesMatrix(TimeSeries,TimeToFailure,FragmentSize,delay):

    

  """

  TimeSeries -> Data to be sampled

  TimeToFailure-> Time to failure 

  FragmentSize-> Size of the time series sample

  delay-> Number of steps to wait to get a new sample, manages the amount of overlay between samples

  """

  

  cData=TimeSeries

  cTim=TimeToFailure

  cFrag=FragmentSize

  container=[]

  time=[]

  nData=len(cData)

  counter=0

  

  for k in range(nData-cFrag):

    

    if counter==delay:

      

      cSample=list(cData[k:k+cFrag])

      container.append(cSample)

      time.append(cTim[k+cFrag])

      counter=0

      

    else:

      counter=counter+1



  return np.array(container),np.array(time)



#Data features

def MakeFeaturesRF(DataMatrix):

  

  cont=[]

  featuresList=[np.mean,np.std,np.ptp,np.min,np.sum]

  

  for sample in DataMatrix:

    sampCont=[]

    for feature in featuresList:

      sampCont.append(feature(sample))

      sampCont.append(np.abs(feature(sample)))

    

    sampCont.append(np.mean(np.diff(sample)))

    sampCont.append(np.mean(np.abs(np.diff(sample))))

    cont.append(sampCont)

    

  return np.array(cont)
Data=MinimalLoader(r'../input/lanl15/train15.csv',delimiter=',')



AcData=Data[:,0]

TimeData=Data[:,1]



GlobalMean,GlobalStd=GetScalerParameters(AcData)

ScaledData=MakeScaledSeries(AcData,GlobalMean,GlobalStd)



del Data,AcData
SamplesData,TimeTE=MakeSamplesMatrix(ScaledData,TimeData,10000,10000)

FeaturesData=MakeFeaturesRF(SamplesData)

RFScaler=pr.MinMaxScaler()

RFScaler.fit(FeaturesData)

FeaturesData=RFScaler.transform(FeaturesData)



RFR=RandomForestRegressor(n_estimators=100)

RFR.fit(FeaturesData,TimeTE)
plt.figure(1)

plt.plot(RFR.feature_importances_)

ax=plt.gca()

PlotStyle(ax,'','Features','Importance')
plt.figure(2,figsize=(15,5))

plt.subplot(131)

plt.plot(FeaturesData[:,-1])

ax=plt.gca()

PlotStyle(ax,'','Samples','Abs Sum Of Changes')

plt.subplot(132)

plt.plot(FeaturesData[:,2])

ax=plt.gca()

PlotStyle(ax,'','Samples','Standard Deviation')

plt.subplot(133)

plt.plot(FeaturesData[:,-2])

ax=plt.gca()

PlotStyle(ax,'','Samples','Mean')
Corr=[]

order=[]

df1=ScaledData[0:1500000]



for d in np.linspace(-1,1,20): 

  df2=df.GL(d,df1,num_points=len(df1)) 

  df2=MakeScaledSeries(df2,df2.mean(),df2.std())

  corr=np.corrcoef(df1,df2)[0,1] 

  Corr.append(corr)

  order.append(d)



plt.figure(3)

plt.plot(order,Corr)

ax=plt.gca()

PlotStyle(ax,'','Differintegration order','Correlation')
plt.figure(4,figsize=(15,5))



for order in [-0.1,0,0.1,0.2]:

    Der0=[np.sum(np.abs(df.GL(order,df1[k:k+10000],num_points=10000))) for k in range(300000,len(df1),10000)]

    Der1=[np.mean(df.GL(order,df1[k:k+10000],num_points=10000)) for k in range(300000,len(df1),10000)]

    

    plt.subplot(121)

    plt.plot(Der0,label='Order ='+str(order))

    plt.legend(loc=3)

    ax=plt.gca()

    PlotStyle(ax,'','Time','Abs Sum Of Changes')

    

    plt.subplot(122)

    plt.plot(Der1,label='Order ='+str(order))

    plt.legend(loc=3)

    ax=plt.gca()

    PlotStyle(ax,'','Time','Mean')

    

del df1,Der0,FeaturesData,SamplesData
SeriesFragment=10000

Delay=int(0.1*SeriesFragment)

DerOrders=np.linspace(-0.1, 0.25, 6, endpoint=True)
#Location function 

def GetSampleLoc(SampleTime,boundaries):

  

  """

  

  Returns the bin index of a time to the next eartquake sample 

  

  SampleTime: Time To the next eartquake sample

  boundaries: list of the boundaries of the bined time to the next earquake distribution 

  

  """

  

  for k in range(len(boundaries)-1):

      

    if SampleTime>=boundaries[k] and SampleTime<=boundaries[k+1]:

        

      cLoc=k

      break

      

  return cLoc



#Equalizes the samples over the range of time to the next earthquake

def MakeEqualizedSamples(DataSamples,TimeSamples):

  

  """

  

  DataSamples:  Matrix of size (SampleSize,NumberOfSamples), contains the time 

                series samples

  Time Samples: Array of size (NumberOfSamples), contains the time to the next 

                earthquake

  

  """

  

  cData=DataSamples

  cTime=TimeSamples

  nData=len(cTime)

  nBins=1000

  

  cMin,cMax=np.min(cTime),np.max(cTime)

  bins=np.linspace(cMin,cMax,num=nBins+1)

  

  SamplesCount=[0 for k in range(nBins)]

  

  Xcont=[]

  Ycont=[]

  

  index=[k for k in range(len(cTime))]

  np.random.shuffle(index)

  

  for k in range(nData):

    

    cXval=cData[index[k]]

    cYval=cTime[index[k]]

    

    cLoc=GetSampleLoc(cYval,bins)

    

    if SamplesCount[cLoc]<=15:

      

      Xcont.append(list(cXval))

      Ycont.append(cYval)

      SamplesCount[cLoc]=SamplesCount[cLoc]+1

      

  return np.array(Xcont),np.array(Ycont)

Samples,Times=MakeSamplesMatrix(ScaledData,TimeData,SeriesFragment,Delay)

SamplesE,TimesE=MakeEqualizedSamples(Samples,Times)



del Samples,Times
plt.figure(5,figsize=(15,5))

plt.subplot(121)

n, bins, patches=plt.hist(TimeTE,bins=1000)

ax=plt.gca()

PlotStyle(ax,'Normal Sampling','','')

plt.subplot(122)

n, bins, patches=plt.hist(TimesE,bins=1000)

ax=plt.gca()

PlotStyle(ax,'Random Sampling','','')
#Calculate the features for each sample 

def CalculateFeatures(Sample,Orders):

  

  """

  Sample: Time series fragment

  Orders: Array of non integer differentiation orders 

  """



  container=[]

  nSample=len(Sample)

  

  for order in Orders:

      

    derSample=df.GL(order,Sample,num_points=nSample)

    absSample=np.abs(derSample)



    container.append(np.log(1+np.mean(absSample)))

    container.append(np.mean(derSample))



  return container



#A brief description 

def MakeDataMatrix(Samples,Orders):

  

  """

  Samples: Matrix of time series samples 

  Orders: Array of non integer differentiation orders

  """

  

  container=[]

  

  for samp in Samples:

    

    container.append(CalculateFeatures(samp,Orders))

    

  return np.array(container)
Xtrain0=MakeDataMatrix(SamplesE,DerOrders)

ToMinMax=pr.MinMaxScaler()

ToMinMax.fit(Xtrain0)

MMData=ToMinMax.transform(Xtrain0)



Xtrain,Xtest,Ytrain,Ytest=train_test_split(MMData,TimesE, train_size = 0.9,test_size=0.1,shuffle=True)



del Xtrain0,MMData
params={'n_estimators':[10,100,150,200],

        'max_depth':[2,4,8,16,32,None],

        'min_samples_split':[0.1,0.5,1.0],

        'min_samples_leaf':[1,2,4],

        'bootstrap':[True,False]}





RFR=RandomForestRegressor() 

FinalModel=GridSearchCV(RFR,params,cv=2,verbose=1,n_jobs=2)

FinalModel.fit(Xtrain,Ytrain)

preds4 = FinalModel.predict(Xtest)

MAE='Mean Absolute Error = ' +str(sum(np.abs(preds4-Ytest))/len(Ytest))



plt.figure(3)

plt.plot(preds4,Ytest,'bo',alpha=0.15)

plt.plot([0,17],[0,17],'r')

plt.xlim([0,17])

plt.ylim([0,17])

ax=plt.gca()

PlotStyle(ax,MAE,'Predicted','Real')

TestSamples=np.genfromtxt(r'../input/tested/test.csv',delimiter=',')

TestIds=np.genfromtxt(r'../input/tested/ord.csv',delimiter=',')



SamplesFeatures=MakeDataMatrix(TestSamples,DerOrders)

ScaledTest=ToMinMax.transform(SamplesFeatures)

final=FinalModel.predict(ScaledTest)

PredictionDir=r'../predictions.csv'

firstRow=['seg_id','time_to_failure']



with open(PredictionDir,'w',newline='') as output:

        

    writer=csv.writer(output)

    nData=len(final)

    writer.writerow(firstRow)

            

    for k in range(nData):

      cRow=[TestIds[k],final[k]]

      writer.writerow(cRow)

        