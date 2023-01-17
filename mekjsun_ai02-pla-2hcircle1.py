

import numpy as np

import pandas as pd

import sys 

from Package.ixFileP import fileP

fp = fileP()

#Agrithm for preception linear regression

#substract





#introduction





#method





#result





#concolution

#1. discussion for program convergency speed by 

#    1.1 increase data

#    1.2 distance in decreasing or increasing punish value 

#    1.3 

#2. 
#program

#limitation

#    dtad : 2 Dimension and 2 category parts

#running before: input data in initial block

#plot after running:

#    change function label

#    fit range of x

#input data

fdExamplesFinal = 'Z:\\out\\tmp\\dataset\\Final\\' #dir of data

fnTrain = '2Hcircle1.txt' #name of train data 

fnTest = '' #name of test data if exist

fname = fdExamplesFinal+fnTrain

fnameTest = fdExamplesFinal+fnTest



slicePart = 1  # slice data of 2/3 to purpose of train 





#initial value

#y(N) = sum(w(N) . x(N)) +b

#    here (x,y) stands for some points in 2d plane 

#    and according to decribe method one need threshhold value b to inhibat divergent when program running

bThold = -1

# learning step eTar = 0.8

eTar = 0.8

# weight(w(0) = [0,1,0,1] ;

w =[]

#w.append( [bThold , 0 , 0,0,0] 

x =[]



#predict value stand for 

C1 = 1 #upper side

C2 = 2 #lower side



testPS = 100000 #maximum count for upper limit of recursive



def read_data(fn):

    try:

        df = pd.read_csv(fdExamplesFinal+fn, sep=" ", header=None)

        colNames = ['X' + str(f) for f in df.columns]

        df.columns = colNames

        print('fname',fn,'columns',colNames,'data shape',df.shape)

        #df = df.set_index(0)

        #df.to_csv(fdExamplesFinal+fn, sep=" ")

    except :

        print('error fn',fn) 

        return None

    return df

def preDataWithoutTestData(df, slicePart):  # df: data of pd dataframe , slicepart : slice data of 2/3

    df1 =   df.copy()

    dfData =  pd.DataFrame(df1, columns=df1.columns[0:-1])

    dfPred =  pd.DataFrame(df1, columns=[df1.columns[-1]])

    lenn = int(len(df1)* slicePart)

    dfTrain = dfData[0:lenn]

    dftrainPred = dfPred[0:lenn]

    dfTest =  dfData[lenn:]    

    dftestPred = dfPred[lenn:]    

    trainData = dfTrain.values.tolist()

    trainPred = dftrainPred.values.tolist()

    trainPred =  [f[0] for f in trainPred ]

    testData = dfTest.values.tolist()

    testPred = dftestPred.values.tolist()

    testPred =  [f[0] for f in testPred ]

    #for plot

    dfC1 = df[df[df.columns[-1]] == C1]

    dfC2 = df[df[df.columns[-1]] == C2]



    print('dfTrain',dfTrain.shape,'train predict',dftrainPred.shape ,'dfTest',dfTest.shape,'test predict',dftestPred.shape)

    return trainData,testData,trainPred,testPred,df1.columns[0:-1],dfC1,dfC2

def gAns(columns, w):

    ans = ''

    for idx in range(0,len(columns)):

        if(len(ans)>0): ans = ans + ' + '+   str(w[idx+1]) + ' * ' + columns[idx] + ' '

        else :  ans =  str(w[idx+1]) + ' ' + columns[idx] + ' '

    ans = ans + ' + '+ str(bThold* w[0])

    return ans
# condition for convergen 

# this program will check if all weight value unchange i.e (w(N+1) = w(N) )

# in addition, it's collect if phi value >0 and wright will be kept(case 1) else will punish weight value 

#  like 

#     case 2 w(N+1) = w(N) + eTar * x(N)  if predict value =1 but  phi value < 0

#     case 3 w(N+1) = w(N) - eTar * x(N)  if predict value =-1 but  phi value >= 0





# read data 

#field[x,y,expect value]

#points = read_hw1_18() #[ [0,0,1] ,  [0,1,1] ,  [1,0,-1] , [1,1,1]]

df = read_data(fnTrain)

if( df is None): sys.exit()     

trainData,testData,trainPred,testPred,columns,dfC1,dfC2 = preDataWithoutTestData(df,slicePart)

points = trainData

#RecordMega =[1,1,1,1] # done if all value in RecordMega is zero. however, all value will be one if any weight value changed

RecordMega = [1] * len(points)



#initial connect value

nD = len(trainData[0]) #numbers of how many dimension do you want to research

keyVal = [ 0 ] * nD  #by constant of nD

   



w.append( [bThold]+  keyVal)

print ('init w',w)

step = 0



#check dimensiod wether  its value is more than 2 or not    

if(len(trainData[0] ) >2): 

    print('error at data dimension is more than 2')

    sys.exit()

#check category wether  its value is more than 2 or not    

if(len(set(trainPred) ) >2): 

    print('error at data dimension is more than 2')

    sys.exit()



#sys.exit() 

print('processing...')

#while(True):

for ii in range(testPS):

    #print('step' , step, 'w[step]', w[step])

    iStep = step % len(points) # find position at points

    x =   [bThold ] + trainData[iStep]

    

    phiValue = np.sum( np.multiply(w[step], x) )

    predictValue = trainPred[iStep]

    #print('step',step,'position',iStep,'phiValue',phiValue ,'predictValue', predictValue)

    

    #part to check weight value if need to punsih

    wP = None

    if( (predictValue == C1 and phiValue > 0) or (predictValue == C2 and phiValue <= 0)) : #case 1 

        wP =  w[step]  # w(N+1) = w(N) 

        RecordMega[iStep] = 0  #set monitor value to zero i.e weight is kept

    else:

        #print('step' , step, 'w[step]', w[step])

        #print('step',step,'position',iStep,'phiValue',phiValue ,'predictValue', predictValue)

        RecordMega = [1] * len(RecordMega) # reset all monitor value to 1

        if(predictValue ==C2 and phiValue>=0):   #case 3 

            punX = np.multiply(x, eTar)  #wP =  w(N) - eTar * x(N)  

            wP =    np.subtract(w[step]  , punX)

        else:

            if  (predictValue == C1 and phiValue < 0)   :  #case 2

                punX = np.multiply(x, eTar)  #wP =  w(N) + eTar * x(N)  

                wP =  np.add(w[step]  , punX)

            

    #check if all weight value unchange

    if(wP is None) :

        print ('error at wP(next weight) value is null ')

        print ('step',step, 'weight', w[step])

        break

        

    wV = np.sum(RecordMega)

    if(wV==0):

        print('congraduation !! the answer is followed as')

        print ('step',step, 'weight', w[step])

        ans = gAns(columns, w[step])

        print (ans)

        break

    if(step > testPS ): 

        print ('program stop at more than ' + testPS)

        print ('step',step, 'weight',  w[step])

        print('step',step,'position',iStep,'phiValue',phiValue ,'predictValue', predictValue)

        break

    if( ((step %  len(points)) == 0) or (len(points)<50) ): 

        print ('step',step, 'weight',  w[step])

        print('step',step,'position',iStep,'phiValue',phiValue ,'predictValue', predictValue)

        #break

    step = step +1

    w.append(wP)



#run result

print('end')

#save w data

wAns = w[step][1:]

bValue = w[step][0] * bThold

#test

wx = w[step] # -0.8 x  + 1.0 y   +  -0.2 





testData = [(-2.5, 3), (4.6,1), (-2.3,-12),(12,-34)]

for pt in testData:

    testValue = wx[1]* pt[0] + wx[2]* pt[1]  +  -1 * wx[0]

    print('point =',pt , 'testValue = ', testValue )
#plot

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm

from numpy.random import randn

%matplotlib inline

x1 = np.linspace(-5,5,10)

y1 = np.add( np.multiply(x1,wAns[0])  , bValue ) / (wAns[1] *-1)

#1.5584 X0  + -2.3464 * X1  + 20.2

#0.6    7.72   0.816

#points = [ [0,0,1] ,  [0,1,1] ,  [1,0,-1] , [1,1,1]]



#data of train

#get C1

px1 = dfC1[dfC1.columns[0]].values.tolist()

py1 = dfC1[dfC1.columns[1]].values.tolist()



#get C2

px2 = dfC2[dfC2.columns[0]].values.tolist()

py2 = dfC2[dfC2.columns[1]].values.tolist()

#data of test

#get C1

px3 = [-2.5 ]

py3 = [3]

#get C2

px4 = [ 4.6, -2.3, 12]

py4 = [1,-12,-34]
pla =  plt.plot(x1,y1, label = "-0.66648 X0  + -2.91728 * X1  + 13.0", color='blue',linewidth = 0.5)

plb =  plt.plot(px1,py1,'go',1)

plb2 =  plt.plot(px2,py2,'ro',1)

#plb3 =  plt.plot(px3,py3,'go',1)

#plb4 =  plt.plot(px4,py4,'ro',1)



plt.show()