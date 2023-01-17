# Use inline matlib plots

%matplotlib inline



# Import python libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# Get specific functions from some other python libraries

from math import floor, log

from scipy.stats import skew, kurtosis

from scipy.io import loadmat   # For loading MATLAB data (.dat) files



import numpy as np

import pandas as pd

from sklearn import tree

import random



from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel





def convertMatToDictionary(path):

    

    try: 

        mat = loadmat(path)

        names = mat['dataStruct'].dtype.names

        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

        

    except ValueError:     # Catches corrupted MAT files (e.g. train_1/1_45_1.mat)

        print('File ' + path + ' is corrupted. Will skip this file in the analysis.')

        ndata = None

    

    return ndata



def calcNormalizedFFT(epoch, lvl, nt, fs):

    

    lseg = np.round(nt/fs*lvl).astype('int')

    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))

    D[0,:]=0                                # set the DC component to zero

    D /= D.sum()                      # Normalize each channel               



    return D



def defineEEGFreqs():

    return (np.array([0.1, 4, 8, 14, 30, 45, 70, 180])) 



def calcDSpect(epoch, lvl, nt, nc,  fs):

    

    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    lseg = np.round(nt/fs*lvl).astype('int')

    

    dspect = np.zeros((len(lvl)-1,nc))

    for j in range(len(dspect)):

        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)

        

    return dspect



def calcShannonEntropy(epoch, lvl, nt, nc, fs):

    

    # compute Shannon's entropy, spectral edge and correlation matrix

    # segments corresponding to frequency bands

    dspect = calcDSpect(epoch, lvl, nt, nc, fs)



    # Find the shannon's entropy

    spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

    

    return spentropy



def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):

    

    # Find the spectral edge frequency

    sfreq = fs

    tfreq = 40

    ppow = 0.5



    topfreq = int(round(nt/sfreq*tfreq))+1

    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    A = np.cumsum(D[:topfreq,:], axis=0)

    B = A - (A.max()*ppow)    

    spedge = np.min(np.abs(B), axis=0)

    spedge = (spedge - 1)/(topfreq-1)*tfreq

    

    return spedge



def corr(data, type_corr):

    

    C = np.array(data.corr(type_corr))

    C[np.isnan(C)] = 0  # Replace any NaN with 0

    C[np.isinf(C)] = 0  # Replace any Infinite values with 0

    w,v = np.linalg.eig(C)

    #print(w)

    x = np.sort(w)

    x = np.real(x)

    return x



def calcCorrelationMatrixChan(epoch):

    

    # Calculate correlation matrix and its eigenvalues (b/w channels)

    data = pd.DataFrame(data=epoch)

    type_corr = 'pearson'

    

    lxchannels = corr(data, type_corr)

    

    return lxchannels



def calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs):

    

        # Calculate correlation matrix and its eigenvalues (b/w freq)

        dspect = calcDSpect(epoch, lvl, nt, nc, fs)

        data = pd.DataFrame(data=dspect)

        

        type_corr = 'pearson'

        

        lxfreqbands = corr(data, type_corr)

        

        return lxfreqbands



def calcSkewness(epoch):

    '''

    Calculate skewness

    '''

    # Statistical properties

    # Skewness

    sk = skew(epoch)

        

    return sk



def calcKurtosis(epoch):

    

    '''

    Calculate kurtosis

    '''

    # Kurtosis

    kurt = kurtosis(epoch)

    

    return kurt



def calcMean(epoch):

    

    '''

    Calculate mean

    '''

    # Mean

    meanV = np.mean(epoch,axis=0)

    

    return meanV



def calcMedian(epoch):

    

    '''

    Calculate median

    '''

    # Mdian

    medianV = np.median(epoch,axis=0)

    

    return medianV



def calculate_features(file_name,className):

    

    #file_name='C:\\Users\\CUDALAB2\\Desktop\\EEG DATA\\P2\\train_1\\train_1\\1_10_0.mat'

    f = convertMatToDictionary(file_name)



    fs = f['iEEGsamplingRate'][0,0]

    

    eegData = f['data']

    [nt, nc] = eegData.shape

    #print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))



    lvl = defineEEGFreqs()



    subsampLen = int(floor(fs * 60))  # Grabbing 60-second epochs from within the time series

    numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples

    sampIdx = range(0,(numSamps+1)*subsampLen,int(subsampLen))



    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'

                 , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'

                 , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'

                 , 'skewness' : 'calcSkewness(epoch)'

                 , 'kurtosis' : 'calcKurtosis(epoch)'

                 }



    # Initialize a dictionary of pandas dataframes with the features as keys

    feat = {key[0]: pd.DataFrame() for key in functions.items()}  



    for i in range(1, numSamps+1):



        #print('processing file {} epoch {}'.format(file_name,i))

        epoch = eegData[sampIdx[i-1]:sampIdx[i], :] 



        for key in functions.items():

            feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)



    for key in functions.items():

            #feat[key[0]]['Minutes to Seizure'] = np.subtract(range(numSamps), 70-10*f['sequence'][0][0] + 5)

            feat[key[0]]['ClassName']=np.subtract(int(className)+2,2)

            #feat[key[0]] = feat[key[0]].set_index('Minutes to Seizure')

    

    return feat

        



# Prediction



def initFeat():

    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'

                     , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'

                     , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'

                     , 'skewness' : 'calcSkewness(epoch)'

                     , 'kurtosis' : 'calcKurtosis(epoch)'

                     }



    feat = {key: pd.DataFrame() for key in functions.keys()}  

    return feat



def attribRed():

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)

    model = SelectFromModel(lsvc, prefit=True)

    xnew= model.transform(X)



def loadFeat(feat,inter,pre):

    classData=0

    for i in range(1,inter+1):

            index=random.randint(1,1000)

            temp=calculate_features('../input/train_1/1_'+str(index)+'_'+str(classData)+'.mat',classData)

            for key in temp.keys():

                feat[key]=feat[key].append(temp[key])

            print("Inter "+str(i))

                

    classData=1

    for i in range(1,pre+1):

            index=random.randint(1,40)

            temp=calculate_features('../input/train_1/1_'+str(index)+'_'+str(classData)+'.mat',classData)

            for key in temp.keys():

                feat[key]=feat[key].append(temp[key])

            print("pre "+str(i))

                

    return feat







def loadVerifTest(size):

    

    

    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'

                     , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'

                     , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'

                     , 'skewness' : 'calcSkewness(epoch)'

                     , 'kurtosis' : 'calcKurtosis(epoch)'

                     , 'mean' : 'calcMean(epoch)'

                     , 'median' : 'calcMedian(epoch)'

                     }



    feat = {key: pd.DataFrame() for key in functions.keys()}  

    

    for i in range(1,size+1):

        index=random.randint(1,150)

        classData=random.randint(0,1)

        temp=calculate_features('../input/train_1/1_'+str(index)+'_'+str(classData)+'.mat',classData)

        for key in temp.keys():

            feat[key]=feat[key].append(temp[key])

        print("Test "+str(i))



                    

    return feat





#NORMAL TEST DATA

def normalTest(result):

    finalResult={}



    for prop in result:

        for index,data in enumerate(prop):

            if(finalResult.get(index,False)):

                finalResult[index]+=data

            else:

                finalResult[index]=data



    for key in finalResult:

        if finalResult[key]>4:

            finalResult[key]=1

        else:

            finalResult[key]=0

        

    return finalResult



#ACCURACY VERIFICATION

def accuracyVerification(result,actual):

    finalResult={}

    finalResultList=[]

    

    actual=list(actual[0])



    for prop in result:

        for index,data in enumerate(prop):

            if(finalResult.get(index,False)):

                finalResult[index]+=data

            else:

                finalResult[index]=data



    for index,key in enumerate(finalResult):

        if finalResult[key]>4:

            finalResultList.append(str(actual[index])+",1")

        else:

            finalResultList.append(str(actual[index])+",0")



    return finalResultList



def calcDSpect(epoch, lvl, nt, nc,  fs):

    

    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    lseg = np.round(nt/fs*lvl).astype('int')

    

    dspect = np.zeros((len(lvl)-1,nc))

    for j in range(len(dspect)):

        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)

        

    return dspect



'''

Compute spectral edge frequency

'''

def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):

    

    # Find the spectral edge frequency

    sfreq = fs

    tfreq = 40

    ppow = 0.5



    topfreq = int(round(nt/sfreq*tfreq))+1

    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    A = np.cumsum(D[:topfreq,:], axis=0)

    B = A - (A.max()*ppow)    

    spedge = np.min(np.abs(B), axis=0)

    spedge = (spedge - 1)/(topfreq-1)*tfreq

    

    return spedge



def calcActivity(epoch):

    '''

    Calculate Hjorth activity over epoch

    '''

    

    # Activity

    activity = np.nanvar(epoch, axis=0)

    

    return activity



def calcMobility(epoch):

    '''

    Calculate the Hjorth mobility parameter over epoch

    '''

      

    # Mobility

    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)

    mobility = np.divide(

                        np.nanstd(np.diff(epoch, axis=0)), 

                        np.nanstd(epoch, axis=0))

    

    return mobility



def calcComplexity(epoch):

    '''

    Calculate Hjorth complexity over epoch

    '''

    

    # Complexity

    complexity = np.divide(

        calcMobility(np.diff(epoch, axis=0)), 

        calcMobility(epoch))

        

    return complexity  



'''

Computes Shannon Entropy for the Dyads

'''



def calcShannonEntropyDyad(epoch, lvl, nt, nc, fs):

    

    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)

                           

    # Find the Shannon's entropy

    spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

        

    return spentropyDyd



def calcDSpectDyad(epoch, lvl, nt, nc, fs):

    

    # Spectral entropy for dyadic bands

    # Find number of dyadic levels

    ldat = int(floor(nt/2.0))

    no_levels = int(floor(log(ldat,2.0)))

    seg = floor(ldat/pow(2.0, no_levels-1))



    D = calcNormalizedFFT(epoch, lvl, nt, fs)

    

    # Find the power spectrum at each dyadic level

    dspect = np.zeros((no_levels,nc))

    for j in range(no_levels-1,-1,-1):

        dspect[j,:] = 2*np.sum(D[int(floor(ldat/2.0))+1:ldat,:], axis=0)

        ldat = int(floor(ldat/2.0))



    return dspect



def calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs):

    

    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)

    

    # Find correlation between channels

    data = pd.DataFrame(data=dspect)

    type_corr = 'pearson'

    lxchannelsDyd = corr(data, type_corr)

    

    return lxchannelsDyd



def removeDropoutsFromEpoch(epoch):

    

    '''

    Return only the non-zero values for the epoch.

    It's a big assumption, but in general 0 should be a very unlikely value for the EEG.

    '''

    return epoch[np.nonzero(epoch)]
feat
feat= initFeat()

#loadFeat(inter,pre)

feat=loadFeat(feat,15,15)
#TRAINING 

features = list(feat['kurtosis'].columns[:16])





functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'

                     , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'

                     , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'

                     , 'skewness' : 'calcSkewness(epoch)'

                     , 'kurtosis' : 'calcKurtosis(epoch)'

                     

                     }



from sklearn import ensemble

# Fit classifier with out-of-bag estimates

params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,

          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}



clf={}

for key in functions.keys():

    #print(key)

    y = feat[key]['ClassName']

    X = feat[key][features].fillna(0)

    clf[key] = ensemble.GradientBoostingClassifier(**params)

    clf[key] = clf[key].fit(X,y)
#TESTING

f=loadVerifTest(10)
result=[]

actual=[]



actual.append(list(f[list(functions.keys())[0]]['ClassName']))



for key in functions.items():

    test=f[key[0]][features].fillna(0)

    result.append(clf[key[0]].predict(test))

     

#print(result)



finalResult=normalTest(result)



accuracyResult=accuracyVerification(result,actual)



print(accuracyResult)



acc=0

for x in accuracyResult:

    i,j=x.split(',')

    if(i==j):

        acc+=1

print(acc*100/len(accuracyResult))