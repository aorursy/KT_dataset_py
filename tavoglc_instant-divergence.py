import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from scipy.spatial import distance as ds

from sklearn.covariance import LedoitWolf



from sklearn.svm import NuSVC

from sklearn.linear_model import LogisticRegression

#General PlotStyle

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



#MeshGeneration    

def MeshData(TargetData,Model,MeshSteps):

  

  nVals=MeshSteps

  xMin,xMax=1.2*min(TargetData[:,0]),1.2*max(TargetData[:,0])

  yMin,yMax=1.2*min(TargetData[:,1]),1.2*max(TargetData[:,1])

    

  xx, yy = np.meshgrid(np.linspace(xMin, xMax, nVals), np.linspace(yMin, yMax, nVals))

  Z = Model.decision_function(np.c_[xx.ravel(), yy.ravel()])



  return xx,yy,Z

 

#Decision function plot

def PlotDecisionFunction(TargetData,Targetlabels,Model,MeshSteps,Axis):

  

  localMesh=MeshData(TargetData,Model,MeshSteps)

  

  Class0=[k for k in range(len(labls)) if labls[k]==0]

  Class1=[k for k in range(len(labls)) if labls[k]==1]

  

  Axis.contourf(localMesh[0],localMesh[1],localMesh[2].reshape(localMesh[0].shape),cmap=plt.cm.bwr)

  Axis.plot(TargetData[Class0,0],TargetData[Class0,1],'bo',alpha=0.25)

  Axis.plot(TargetData[Class1,0],TargetData[Class1,1],'ro',alpha=0.25)

  
#Calculate the covariance matrix for each label

def GetModelParams(DataFrame,ColumnIndex):

  

  cDataSet=DataFrame

  

  cData0=cDataSet[cDataSet['target']==0]

  cData1=cDataSet[cDataSet['target']==1]

  

  bData0=np.array(cData0[ColumnIndex])

  bData1=np.array(cData1[ColumnIndex])

  

  Cov0=LedoitWolf(assume_centered=False).fit(bData0)

  Cov1=LedoitWolf(assume_centered=False).fit(bData1)

  

  Mean0=np.mean(bData0,axis=0)

  Mean1=np.mean(bData1,axis=0)

  

  RegCov0=Cov0.covariance_

  RegCov1=Cov1.covariance_

  

  return RegCov0,RegCov1,Mean0,Mean1



#Calculation of the coefficients 

def KullbackLeiberCoefficients(CovarianceA,CovarianceB):

  

  invA=np.linalg.inv(CovarianceA)

  coef1a=np.dot(CovarianceB,invA)

  _,coefa=np.linalg.slogdet(coef1a)

  coef1b=np.dot(invA,CovarianceB)

  coefb=np.trace(coef1b)

  

  return invA,coefa,coefb



#Kullback Leiber divergence for each sample

def KullbackLeiberDivergence(CoefficientA,CoefficientB,CoefficientC,Mean,Sample):

  

  distance=(ds.mahalanobis(Mean,Sample,CoefficientA))**2

  divergence=CoefficientC+distance-CoefficientB-len(Mean)

  

  return divergence/2



#Wrapper function for the Kullback Leiber Coefficients 

def MakeModelParams(Data,ColumnIndex):

  

  cData=Data

  Cov0,Cov1,Mean0,Mean1=GetModelParams(cData,ColumnIndex)

  Inv0,CoefA0,CoefB0=KullbackLeiberCoefficients(Cov0,Cov1)

  Inv1,CoefA1,CoefB1=KullbackLeiberCoefficients(Cov1,Cov0)

  

  return Mean0,Mean1,Inv0,CoefA0,CoefB0,Inv1,CoefA1,CoefB1



#Calculation of the divergence ratios

def SampleFeatures(Sample,Params):

  

  cSample=Sample

  Mean0,Mean1=Params[0],Params[1]

  Inv0,CoefA0,CoefB0=Params[2],Params[3],Params[4]

  Inv1,CoefA1,CoefB1=Params[5],Params[6],Params[7]

  

  div00=KullbackLeiberDivergence(Inv0,CoefA0,CoefB0,np.array(Mean0),cSample)

  div01=KullbackLeiberDivergence(Inv0,CoefA0,CoefB0,np.array(Mean1),cSample)

  div10=KullbackLeiberDivergence(Inv1,CoefA1,CoefB1,np.array(Mean0),cSample)

  div11=KullbackLeiberDivergence(Inv1,CoefA1,CoefB1,np.array(Mean1),cSample)

  

  return [(div00-div10)/(div10+div00),(div01-div11)/(div11+div01)]



#Model Features 

def ModelFeatures(Data,Params,ColumnIndex):

  

  cData=Data

  trainData=np.array(cData[ColumnIndex])  

  container=[]

  

  for k in range(len(trainData)):

    

    cSample=trainData[k]

    container.append(SampleFeatures(cSample,Params))

    

  return np.array(container)

    
Xtrain=pd.read_csv('../input/train.csv')

DataColumns=[c for c in Xtrain.columns if c not in ['id','target','wheezy-copper-turtle-magic']]



ModelNumber=np.random.randint(0,512)

train2 = Xtrain[Xtrain['wheezy-copper-turtle-magic']==ModelNumber]

Vars=train2.std(axis=0)

VarColumns=[Vars.index[k] for k in range(len(Vars)) if Vars.iloc[k]>1.5]

Params=MakeModelParams(train2,VarColumns)

mfeat=ModelFeatures(train2,Params,VarColumns)

labls=np.array(train2['target'])



bls=[k for k in range(len(labls)) if labls[k]==0]

rds=[k for k in range(len(labls)) if labls[k]==1]
plt.plot(mfeat[bls,0],mfeat[bls,1],'bo',alpha=0.15)

plt.plot(mfeat[rds,0],mfeat[rds,1],'ro',alpha=0.15)



ax=plt.gca()

PlotStyle(ax,'','MAPE-divergence 01','MAPE-divergence 10')
reg=LogisticRegression(solver='lbfgs',tol=1e-4,random_state =256)

reg.fit(mfeat,labls)



svcLinear=NuSVC(kernel='linear',probability=True,random_state=256)

svcLinear.fit(mfeat,labls)



svcPoly=NuSVC(gamma='scale',kernel='poly',degree=3,probability=True,random_state=256)

svcPoly.fit(mfeat,labls)
f, (ax0, ax1,ax2) = plt.subplots(1, 3,figsize=(15,5))



PlotDecisionFunction(mfeat,labls,reg,200,ax0)

PlotStyle(ax0,'Logistic Regression','MAPE-divergence 01','MAPE-divergence 10')

PlotDecisionFunction(mfeat,labls,svcLinear,200,ax1)

PlotStyle(ax1,'SVC Linear','MAPE-divergence 01','MAPE-divergence 10')

PlotDecisionFunction(mfeat,labls,svcPoly,200,ax2)

PlotStyle(ax2,'SVC Poly','MAPE-divergence 01','MAPE-divergence 10')

plt.tight_layout()