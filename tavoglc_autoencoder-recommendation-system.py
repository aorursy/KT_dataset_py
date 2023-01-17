#importing the libraries to use 

import os

import re

import nltk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from mpl_toolkits.mplot3d import Axes3D

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn import preprocessing as pr

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split





from keras import initializers

from keras.layers import Dense

from keras.optimizers import Adam

from keras.models import Sequential, Model

from keras.callbacks import ReduceLROnPlateau



globalSeed=50

from numpy.random import seed

seed(globalSeed)

from tensorflow import set_random_seed

set_random_seed(globalSeed)
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

    
AnswersData=pd.read_csv(r"../input/answers.csv")

QuestionsData=pd.read_csv(r"../input/questions.csv")



AnswersData=AnswersData.drop(AnswersData.index[48625]) #While exploring the data that answer body is equal to 'Nan'
AnswersBody=np.array(AnswersData['answers_body'])

QuestionsBody=np.array(QuestionsData['questions_body'])

QuestionsTitle=np.array(QuestionsData['questions_title'])
def TextHTMLRemoval(TargetString):

  #Removes HTML tags from the text'

  characterList=['<p>','</p>','<br>','<ol>','<li>','</li>','h1&gt','&lt;','\n','\r','href','html','http','https']

  cTarg=TargetString

  

  for val in characterList:

    cTarg=re.sub(val,' ',cTarg)

    

  return cTarg

  

def GetTextOnly(TargetString):

  #Removes numbers and punctuation from the text

  cTarg=TargetString

  cTarg=re.sub(r'[^\w\s]',' ',cTarg)

  cTarg=re.sub(r'[0-9]+', ' ', cTarg)

  

  return cTarg

  

def MakeTextTransform(TargetString):

  #Wrapper function 

  cTarg=TargetString

  cTarg=TextHTMLRemoval(cTarg)

  cTarg=GetTextOnly(cTarg)

  

  return cTarg.lower()
FiltAnswers=[MakeTextTransform(val) for val in AnswersBody]

FiltQuestionsB=[MakeTextTransform(val) for val in QuestionsBody]



del AnswersBody,QuestionsBody,QuestionsTitle
def UniqueDataSetTokens(TextData):

  

  cData=TextData

  nData=len(cData)

    

  def SplitAndReduce(TargetString):

    return list(set(TargetString.split()))#returns the unique text elements

  

  container=SplitAndReduce(cData[0])

  

  for k in range(1,nData):

    container=container+SplitAndReduce(cData[k])

    if k%100==0:  #only each 100 steps the container is transformed to a set to eliminate duplicates

      container=list(set(container))

  

  return container

def TokenProcessing(TokenData):

  

  cToken=TokenData

  s=stopwords.words('english')

  cToken=list(set(cToken)-set(s)) #Eliminates stop words

   

  PosTagging=nltk.pos_tag(cToken) #Selecting only nouns using part of speech tagging

  lToken=[]

  for val in  PosTagging:

    if val[1]=='NN':

      lToken.append(val[0])

  

  localps=PorterStemmer()

  localToken=[]

  for val in lToken:

    localToken.append(localps.stem(val))#stem of each token 

  

  stemS=[]

  for val in s:

    stemS.append(localps.stem(val))

  

  localToken=list(set(localToken))

  localTok=[val for val in localToken if 16>len(val)>4]#eliminating words of small size and those greater than 16 characters

  

  return localTok
def TokenFrequencies(TargetString,Token):

  

  cST0=TargetString.split()

  localPS=PorterStemmer()

  cST=[]

  for val in cST0:

    cST.append(localPS.stem(val))#changes each word in the text to the stem word

  

  cTok=Token

  nToken=len(cTok)

  Container=[0 for k in range(nToken)]#frequencies of each token 

  cDict={}

  

  for k in range(nToken):

    cDict[cTok[k]]=k #returns the index over the container 

    

  for val in cST:

    try: 

      cval=cDict[val]

      Container[cval]=Container[cval]+1 

    except KeyError: 

      pass

  

  return np.array(Container)*(1/(sum(Container)+1))#adding one to avoid division by zero error 

TargetDataSet=FiltQuestionsB #Data set used to generate the autoencoder can be changed to QuestionsTitle or AnswersBody 

 

TargetTokens=UniqueDataSetTokens(TargetDataSet)   

stemToken=TokenProcessing(TargetTokens)



TargetFreq=[TokenFrequencies(val,stemToken) for val in TargetDataSet]

TargetFreq=np.reshape(TargetFreq,(len(TargetFreq),len(TargetFreq[0])))

sumFreqs=TargetFreq.sum(axis=0)



plt.figure(1)

plt.hist(sumFreqs,bins=200)

plt.yscale('log',nonposy='clip')

ax=plt.gca()

PlotStyle(ax,'','Token Frequency','Frequency Counts')
CropFactor=15

SelectedIndex=[j for j in range(sumFreqs.size) if sumFreqs[j]>CropFactor*sumFreqs.min()]

FinalToken=[stemToken[val] for val in SelectedIndex]
TargetFreq=[TokenFrequencies(val,FinalToken) for val in TargetDataSet]

TargetSc=pr.MinMaxScaler()#MinMaxScaling of the target data

TargetSc.fit(TargetFreq)  

normData=TargetSc.transform(TargetFreq)
fixedSeed=2017



inputShape=normData[0].size



ae = Sequential()

ae.add(Dense(int(inputShape/3),  activation='elu',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed),

            input_shape=(inputShape,)))



ae.add(Dense(380,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(190,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(45,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(12,  activation='elu',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(3,    activation='linear', name="bottleneck",

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(12,  activation='elu',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(45,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(190,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(380,  activation='sigmoid',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(int(inputShape/3),  activation='elu',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



ae.add(Dense(inputShape,  activation='linear',

            kernel_initializer=initializers.glorot_uniform(seed=fixedSeed),

            bias_initializer=initializers.glorot_uniform(seed=fixedSeed)))



decay=0.001

reduce_lr=ReduceLROnPlateau(monitor='loss',factor=0.1,patience=5,min_lr=decay)    



ae.compile(loss='mean_squared_error', optimizer = Adam(lr=0.01,decay=decay))

ae.fit(normData, normData, batch_size=3024, epochs=50, verbose=1,callbacks=[reduce_lr])



encoder=Model(ae.input, ae.get_layer('bottleneck').output)

Zenc=encoder.predict(normData)

EncScaler=pr.MinMaxScaler() #Min Max scaling of the encoded data

EncScaler.fit(Zenc)

EncoderScaled=EncScaler.transform(Zenc)



fig = plt.figure(2,figsize=(8,8))



ax0=fig.add_subplot(2, 2, 1)

ax0.plot(EncoderScaled[:,0],EncoderScaled[:,1],'bo',alpha=0.0125)

PlotStyle(ax0,'','Encoded Dimension 1','Encoded Dimension 2')



ax1=fig.add_subplot(2, 2, 2)

ax1.plot(EncoderScaled[:,0],EncoderScaled[:,2],'bo',alpha=0.0125)

PlotStyle(ax1,'','Encoded Dimension 1','Encoded Dimension 3')



ax2=fig.add_subplot(2, 2, 3)

ax2.plot(EncoderScaled[:,1],EncoderScaled[:,2],'bo',alpha=0.0125)

PlotStyle(ax2,'','Encoded Dimension 2','Encoded Dimension 3')



ax3=fig.add_subplot(2, 2, 4,projection='3d')

ax3.scatter(EncoderScaled[:,0],EncoderScaled[:,1],EncoderScaled[:,2],color='b',alpha=0.05)



plt.tight_layout()
kmeansInertia=[]

fitness=[]

xv=[]



for k in range(5,60,3):

  

  cluster=KMeans(n_clusters=k)

  cluster.fit(EncoderScaled)

  labels =cluster.labels_

  kmeansInertia.append(cluster.inertia_)

  fitness.append(metrics.silhouette_score(EncoderScaled, labels) )

  xv.append(k)



plt.figure(3)

plt.plot(xv,kmeansInertia/max(kmeansInertia),label='SSE/max(SSE)')

plt.plot(xv,fitness/max(fitness),label='Silhouette Score')

plt.legend(loc=0)

ax=plt.gca()

PlotStyle(ax,'','k-clusters','Normalized performance')



optnClusters=xv[np.argmax(fitness[3:len(fitness)])]

OptimalKMeans=KMeans(n_clusters=optnClusters)

OptimalKMeans.fit(EncoderScaled)

YVals=OptimalKMeans.labels_
RandomQuestTSample=np.random.randint(0,len(YVals),5)



for val in RandomQuestTSample:

  

  print('Cluster number = ' + str(YVals[val]))

  print(FiltQuestionsB[val])
Xtrain,Xtest,Ytrain,Ytest=train_test_split(EncoderScaled, YVals, train_size = 0.75,test_size=0.25, random_state = 23)



RFC=RandomForestClassifier(n_estimators=100)

RFC.fit(Xtrain,Ytrain)



yPred=RFC.predict(Xtest)

YTest=np.reshape(np.array(Ytest),(yPred.shape))



cm=metrics.confusion_matrix(YTest,yPred)

ncm=cm/cm.sum(axis=1)



del Xtrain, Xtest, Ytrain, Ytest, yPred



plt.figure(4)

ax=plt.gca()

im=ax.imshow(ncm,interpolation='nearest',cmap=plt.cm.Blues)

ax.figure.colorbar(im,ax=ax)
AnswersTokFreq=[TokenFrequencies(val,FinalToken) for val in FiltAnswers]

normAnswerData=TargetSc.transform(AnswersTokFreq)

EncAnswers=encoder.predict(normAnswerData)

EncScaledAnswers=EncScaler.transform(EncAnswers)

AnswersClasses=RFC.predict(EncScaledAnswers)
RandomClusters=np.random.randint(0,optnClusters,2)



for val in RandomClusters:

  

  print('---------------------------------------------')

  print('Cluster Number = ' + str(val))

  for j in range(len(YVals)):

    if YVals[j]==val:

      print('----------------Question----------------')

      print(FiltQuestionsB[j])

      break

    

  for j in range(len(AnswersClasses)):

    if AnswersClasses[j]==val:

      print('----------------Answer----------------')

      print(FiltAnswers[j])

      break
def GetAuthorFrequency(Authors,UniqueAuthors):

  

  cAuth=Authors

  uAuth=UniqueAuthors

  nAuthors=len(uAuth)

  

  localFreq=[0 for k in range(nAuthors)]#frequencies of each author

  localDict={}



  for j in range(nAuthors):

    localDict[uAuth[j]]=j#returns the index over localFreq

  

  for val in cAuth:

    localFreq[localDict[val]]=localFreq[localDict[val]]+1

  

  return np.array(localFreq)/len(cAuth)



def AuthorClusterCorrespondance(AnswerID,UniqueAuthors,AnswerClasses):

  

  cAnswID=AnswerID

  uAuth=UniqueAuthors  

  cClass=AnswerClasses

  nAuthors=len(uAuth)

  nData=len(cClass)

  

  localMatrix=np.zeros((nAuthors,optnClusters))#frequencies of each author over each cluster

  localDict={}



  for j in range(nAuthors):

  

    localDict[uAuth[j]]=j#returns the index over localMatrix

    

  for j in range(nData):

    

    AuthLoc=localDict[cAnswID[j]]

    ClassLoc=cClass[j]

    localMatrix[AuthLoc,ClassLoc]=localMatrix[AuthLoc,ClassLoc]+1

    

  return localMatrix/len(YVals)



AnswersAuthorID=np.array(AnswersData['answers_author_id'])

UniqueAuthors=np.unique(AnswersAuthorID)



AuthorFreq=GetAuthorFrequency(AnswersAuthorID,UniqueAuthors)

AuthorMatrix=AuthorClusterCorrespondance(AnswersAuthorID,UniqueAuthors,AnswersClasses)



plt.figure(6)

fig,axes=plt.subplots(1,2,figsize=(10,4))



axes[0].plot(AuthorFreq)

PlotStyle(axes[0],'','Unique Authors','Answer Probability')



axes[1].hist(AuthorFreq,bins=40)

axes[1].set_yscale('log')

PlotStyle(axes[1],'','Answer Probability','Probability Counts')

def GetMostLikelyAuthors(ClusterNumber,UniqueAuthors,AuthorMatrix):

  #Wrapper function for the first recomendation

  cData=AuthorMatrix[:,ClusterNumber]

  ind=np.argsort(cData) #sorting of the data

  lAuthors=[UniqueAuthors[ind[-1]],UniqueAuthors[ind[-2]],UniqueAuthors[ind[-3]],UniqueAuthors[ind[-4]],UniqueAuthors[ind[-5]]]

  

  return lAuthors



def FindLocations(value,List):

  

  cval=value

  clis=List

  cont=[]

    

  for k in range(len(clis)):

    if cval==clis[k]:

      cont.append(k)

      

  return cont
QuestionID=np.array(QuestionsData['questions_id'])

AnswerQuestionID=np.array(AnswersData['answers_question_id'])

AnswProb=[]



for k in range(len(YVals)):

  

  cClust=YVals[k]

  

  cRec=GetMostLikelyAuthors(cClust,UniqueAuthors,AuthorMatrix)

  locs=FindLocations(QuestionID[k],AnswerQuestionID)

  kL=[0,0,0,0,0]

  

  for val in locs:

    

   cAid=AnswersAuthorID[val]

   

   if cAid==cRec[0]:

     kL=[1,0,0,0,0]

   elif cAid==cRec[1]:

      kL=[0,1,0,0,0]

   elif cAid==cRec[2]:

      kL=[0,0,1,0,0]

   elif cAid==cRec[3]:

      kL=[0,0,0,1,0]

   elif cAid==cRec[4]:

      kL=[0,0,0,0,1]

       

  AnswProb.append(kL)



AnswProb=np.array(AnswProb)



fig,axes=plt.subplots(1,2,figsize=(10,4))



axes[0].plot(AnswProb.sum(axis=0)/len(YVals))

PlotStyle(axes[0],'','Recomendations','Answer Probability')



axes[1].plot(np.cumsum(AnswProb.sum(axis=0))/len(YVals))

PlotStyle(axes[1],'','Recomendations','Cumulative Answer Probability')
ProfesionalsData=pd.read_csv(r"../input/professionals.csv")

VolunteersData=ProfesionalsData.dropna()

VolunteersData['variance'] = VolunteersData['professionals_industry'] +' ' +VolunteersData['professionals_headline']



VolunteersID=np.array(VolunteersData['professionals_id'])

VolunteersText=np.array(VolunteersData['variance'])
FiltVolunteers=[MakeTextTransform(val) for val in VolunteersText]
VolunteersTokFreq=[TokenFrequencies(val,FinalToken) for val in FiltVolunteers]

normVolunteersData=TargetSc.transform(VolunteersTokFreq)

EncVolunteers=encoder.predict(normVolunteersData)

EncScaledVolunteers=EncScaler.transform(EncVolunteers)

VolunteersClasses=RFC.predict(EncScaledVolunteers)
def GetLikelyVolunteers(cluster,VolunteersID,VolunteersClasses):

  

  VolID=VolunteersID

  VolClas=VolunteersClasses

  cClust=cluster

  

  nData=len(VolID)

  cont=[]

  for j in range(nData):  



    cVol=VolClas[j]

    if cVol==cClust:

      

      cont.append(j)

  

  if len(cont)<=20:  

    inxs=cont

  else:  

    inxs=np.random.choice(cont,20)

    

  vol=VolunteersID[tuple([inxs])]

  

  return list(vol)
def FinalRecomendation(cluster,UniqueAuthors,AuthorMatrix,VolunteersID,VolunteersClasses):

  #Wrapper function for the final recomendation 

  cClus=cluster

  UA=UniqueAuthors

  AM=AuthorMatrix

  VolID=VolunteersID

  VolC=VolunteersClasses

  

  Rec0=GetMostLikelyAuthors(cClus,UA,AM)

  Rec1=GetLikelyVolunteers(cClus,VolID,VolC)

  

  NonOverlapping=list(set(Rec1)-set(Rec0))

  

  for k in range(5):

    Rec0.append(NonOverlapping[k])

    

  return Rec0
print(FinalRecomendation(3,UniqueAuthors,AuthorMatrix,VolunteersID,VolunteersClasses))