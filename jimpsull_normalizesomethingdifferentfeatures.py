from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd

def smoteAdataset(Xig, yig, test_size=0.2, random_state=0):
    
    Xig_train, Xig_test, yig_train, yig_test = train_test_split(Xig, yig, test_size=test_size, random_state=random_state)
    print("Number transactions X_train dataset: ", Xig_train.shape)
    print("Number transactions y_train dataset: ", yig_train.shape)
    print("Number transactions X_test dataset: ", Xig_test.shape)
    print("Number transactions y_test dataset: ", yig_test.shape)

    classes=[]
    for i in np.unique(yig):
        classes.append(i)
        print("Before OverSampling, counts of label " + str(i) + ": {}".format(sum(yig_train==i)))
        
    sm=SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(Xig_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(yig_train_res.shape))
    
    for eachClass in classes:
        print("After OverSampling, counts of label " + str(eachClass) + ": {}".format(sum(yig_train_res==eachClass)))
        
    return Xig_train_res, yig_train_res, Xig_test, yig_test

#!pip install modin

import os
print(os.listdir("../input"))
print(os.listdir("../input/assemblecustomtestsetrevb"))
print(os.listdir("../input/something-different"))
# Any results you write to the current directory are saved as output.
import pandas as pd
traindf=pd.read_csv("../input/something-different/newTrainFeatureOutputUnprocessed.csv")
if 'Unnamed: 0' in traindf.columns:
    traindf=traindf.drop('Unnamed: 0', axis=1)
#traindf = traindf.round(8)
traindf.shape
testdf=pd.read_csv("../input/assemblecustomtestsetrevb/completeTestSetRevB.csv")
if 'Unnamed: 0' in testdf.columns:
    testdf=testdf.drop('Unnamed: 0', axis=1)

print(traindf.shape)
#traindf.head()
print(testdf.shape)
testdf.columns
def repeatPerBand(df, prefix):
    
    df.loc[:,prefix + 'frsq']=df.loc[:,prefix+'med']**2 / df.loc[:,prefix+'mfl']**2
    df.loc[:,prefix + 'frsqxf']=df.loc[:,prefix+'frsq'] * df.loc[:,prefix+'med']
    
    return df

prefixes=['le','lm','ll','he','hm','hl']
for prefix in prefixes:
    
    traindf=repeatPerBand(traindf, prefix)
    print(traindf.shape)
    
#traindf.head()
for prefix in prefixes:
    
    testdf=repeatPerBand(testdf, prefix)
    print(testdf.shape)
    
#testdf.head()
def removePerBand(df, prefixes=['le','lm','ll','he','hm','hl']):
    for prefix in prefixes:
        df=df.drop([prefix + 'mxd', prefix + 'mnd'], axis=1)
        print(df.shape)
    

    return df

traindf=removePerBand(traindf)  
#traindf.head()
testdf=removePerBand(testdf)  
#testdf.head()
#distmod
traindf['distmod'] = traindf['distmod'].fillna(value=0)
#traindf.info()
#for column in testdf.columns:
#    print(column)
#    joe = testdf[column].isna().sum()
#    if joe>0:
#        print(joe)
testdf['distmod'] = testdf['distmod'].fillna(value=0)
print(testdf['distmod'].isna().sum())
#hostgal_specz
#distmod
traindf=traindf.drop('hostgal_specz', axis=1)
testdf=testdf.drop('hostgal_specz', axis=1)
print(traindf.shape)
print(testdf.shape)
def photoztodist(df) :
    dfhpz=(data["hostgal_photoz"])
    return ((((((np.log(((dfhpz + (np.log(((dfhpz + (np.sqrt((np.log((np.maximum(((3.0)), (dfhpz * 2.0))))))))))))))) + (12.99870681762695312))) + (1.17613816261291504))) * (3.0))
#from Helgi's comment on Scirpus' Photoz2Distance kernel
#np.log(meta _train.hostgal _photoz)
#https://www.kaggle.com/scirpus/photoz2distance
#traindf['photozDist']=0
#distmodnonzero=traindf.loc[:,'distmod']>0
#traindf.loc[distmodnonzero,'photozDist']=#np.log(traindf.loc[distmodnonzero,'hostgal_photoz'])
#traindf['photozDist'] = traindf['photozDist'].fillna(value=0)
#print(traindf.shape)
#testdf['photozDist']=np.log(testdf['hostgal_photoz'])
#testdf['photozDist'] = testdf['photozDist'].fillna(value=0)
#print(testdf.shape)
#traindf.describe()
print(traindf.shape)
print(testdf.shape)

#trobjids=traindf.loc[:,'object_id']
#print(trobjids.shape)

#traindf=traindf.drop('object_id',axis=1)
#print(traindf.shape)


#teobjids=testdf.loc[:,'object_id']
#print(teobjids.shape)
#testdf=testdf.drop('object_id',axis=1)
#print(testdf.shape)

#trobjdf=pd.DataFrame(columns=['object_id'], data=trobjids)
#print(trobjdf.shape)
#teobjdf=pd.DataFrame(columns=['object_id'], data=teobjids)
#print(teobjdf.shape)
def convertAllToLogBase(df, excludeLogCols=['ra', 'decl', 'gal_l', 'gal_b', 'hostgal_photoz', 'target',
                                            'hostgal_photoz_err', 'distmod', 'outlierScore', 'object_id']):
    
    for cindex in df.columns:
        if (cindex not in excludeLogCols) & (len(df.loc[:,cindex].unique()) > 2):
            #zeroFilter=df.loc[:,cindex]==0 (stays zero)
            negFilter=df.loc[:,cindex]<0
            posFilter=df.loc[:,cindex]>0
            
            df.loc[negFilter,cindex]=-1.0*np.log(-1.0*df.loc[negFilter,cindex])
            df.loc[posFilter,cindex]=np.log(df.loc[posFilter,cindex])
            #print(cindex)
    return df

traindf=convertAllToLogBase(traindf)

testdf=convertAllToLogBase(testdf)
def reduceExtremeOutliers(df,targCol='target', logStdRng=8.0):
    
    #possibleLog=[]
    #nonLog=[]
    for cindex in df.columns:
        
        if (cindex==targCol) | (len(df.loc[:,cindex].unique())<=2) | (cindex=='object_id'):
            
            print('doing nothing for ' + str(cindex))
            
        else:
            stdev=np.std(df.loc[:,cindex])
            minVal=np.min(df.loc[:,cindex])
            theRange=np.max(df.loc[:,cindex])-minVal
            stdRange=theRange / stdev
            if logStdRng < stdRange:
                #df.loc[:,cindex]=df.loc[:,cindex]-minVal
                #df.loc[:,cindex]=log(df.loc[:,cindex])
                print('flagged ' + str(cindex) + ' for high variability')
                print(stdRange)
                #possibleLog.append(cindex)
            
                
            newStdev=np.std(df.loc[:,cindex])
            med=np.median(df.loc[:,cindex])
            tooBig=med+newStdev*logStdRng/2
            tooLittle=med-newStdev*logStdRng/2
            tooBigFilter=df.loc[:,cindex]>tooBig
            tooLittleFilter=df.loc[:,cindex]<tooLittle
            df.loc[tooBigFilter,cindex]=tooBig
            df.loc[tooLittleFilter,cindex]=tooLittle
            
    return df #, possibleLog, nonLog
            
traindf=reduceExtremeOutliers(traindf)


testdf=reduceExtremeOutliers(testdf)

def featureScaleAllExcept(df, targCol='target'):
    
    for cindex in df.columns:
        if '_TF' not in cindex:
            if (cindex != targCol) & (cindex !='object_id'):
                minval=np.min(df.loc[:,cindex])
                theRange=np.max(df.loc[:,cindex])-minval
                if theRange==0:
                    #this feature contains no information
                    df=df.drop(cindex,axis=1)
                    print('dropped ' + str(cindex) + ' from dataFrame')

                elif type(df[cindex])==bool:
                    print(str(cindex) + ' is a boolean, doing nothing')

                elif theRange==1:
                    print('the range for ' + str(cindex) + ' is already 1, doing nothing')

                else:
                    #feature scale
                    df.loc[:,cindex]=(df.loc[:,cindex]-minval)/theRange
                
    return df

print(traindf.shape)
traindf=featureScaleAllExcept(traindf)
print(traindf.shape)
print(traindf.loc[:,'target'].unique())

print(testdf.shape)
testdf=featureScaleAllExcept(testdf)
print(testdf.shape)
print(traindf.shape)

print(testdf.shape)

#trobjdf.to_csv('trainObjectIds.csv',sep='\t', encoding='utf-8', index=False)
traindf = traindf.round(8)
traindf.to_csv('traindfNormal.csv', index=False)
#teobjdf.to_csv('testObjectIds.csv',sep='\t', encoding='utf-8', index=False)
testdf = testdf.round(8)
testdf.to_csv('testdfNormal.csv', index=False)