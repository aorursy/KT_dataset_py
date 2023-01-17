# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import gc,sys
gc.enable()
def preprocessing(is_train=True):
    if is_train:
        #read file
        print("Preprocessing train_V2.csv")
        print("Reading the data...",end='')
        df = pd.read_csv('../input/train_V2.csv')
        print("Done!")
        #No Loner
        print("Kicking out the loner...",end='')
        df = df[df['maxPlace'] > 1] 
        print("Done!")
    else:
        #process testfile
        print("Preprocessing test_V2.csv",)
        print("Reading the data...",end='')
        df = pd.read_csv('../input/test_V2.csv')
        print('Done!')
    #print("Dropping unnesscary data")
    
    #Drop useless data
    #df = df.drop(['Id','groupId','matchId'],axis=1)
    #print("Id Drop---Done")
    
    #Get Target
    if is_train:
        print("Getting the target...",end='')
        target = df['winPlacePerc']
        print("Done!")
        print('Remove target from dataframe...',end='')
        df = df.drop (['winPlacePerc'],axis=1)
        print("Done!")
    
    #Return
    print("Dataframe Exported.")
    if is_train:
        return df,target
    return df

def feature_managing(df,is_train=True,lowfeature=True,valuematchtype=True):
    err = 0
    print("Feature Management Initiated")
    #Player's Id has 0 correlation
    if is_train:
        print("Dropping Player's Id")
        try:
            df = df.drop(['Id'],axis=1)
        except:
            print("Warning: 'Id' not found.")
            err += 1
    else:
        #we need player's Id to sent as result though.
        print('Retrive Player\'s Id...',end='')
        test_id = df['Id']
        df = df.drop(['Id'],axis=1)
        print('Done!')
    #Cuz im too lazy to do aggregated feature
    if lowfeature:
        print("Dropping MatchId")
        try:
            df = df.drop(['matchId'],axis=1)
            print('Dropping GroupId')
            df = df.drop(['groupId'],axis=1)
        except:
            print('Warning: MatchId and GroupId not found')
            err += 1
    #Shut this off if you want to aggregated feature thing right away.
    if valuematchtype:
        mtype = ['solo','solo-fpp','duo','duo-fpp','squad','squad-fpp']
        print('Determine Value for MatchType',end='')
        df.loc[df.matchType == 'solo','matchType'] = 1
        print('.',end='')
        df.loc[df.matchType == 'normal-solo','matchType'] = 1
        print('.',end='')
        df.loc[df.matchType == 'solo-fpp','matchType'] = 0.5
        print('.',end='')
        df.loc[df.matchType == 'normal-solo-fpp','matchType'] = 0.5
        print('.',end='')
        df.loc[df.matchType == 'duo','matchType'] = 2
        print('.',end='')
        df.loc[df.matchType == 'normal-duo','matchType'] = 2
        print('.',end='')
        df.loc[df.matchType == 'duo-fpp','matchType'] = 2.5
        print('.',end='')
        df.loc[df.matchType == 'normal-duo-fpp','matchType'] = 2.5
        print('.',end='')
        df.loc[df.matchType == 'squad','matchType'] = 4
        print('.',end='')
        df.loc[df.matchType == 'normal-squad','matchType'] = 4
        print('.',end='')
        df.loc[df.matchType == 'squad-fpp','matchType'] = 3.5
        print('.',end='')
        df.loc[df.matchType == 'normal-squad-fpp','matchType'] = 4
        print('.',end='')
        df.loc[df.matchType == 'crashfpp','matchType'] = 5
        print('.',end='')
        df.loc[df.matchType == 'crashtpp','matchType'] = 5.5
        print('.',end='')
        df.loc[df.matchType == 'flarefpp','matchType'] = 6
        print('.',end='')
        df.loc[df.matchType == 'flaretpp','matchType'] = 6.5
        print('Done!')
    #Drop things off the board cuz we gonna do aggregated feature!
    else:
        df = df.drop['matchType']
        
    ##BEGIN FEATURE ENGINEERING!
    #Part of Idea from harshit kernel
    print('Generating Feature')
    print('Generate KillKnockRatio')  
    df['KillKnockRatio'] = df['kills']/df['DBNOs'] #May Produce NaN or inf ##Determine Efficiency of Killing and knocking ppl
    print('Generate HealthItemsUsed') 
    df['HealthItemsUsed'] = df['heals']+df['boosts'] #Determine Healthitems used
    print('Generate HeadshotRatio')
    df['HeadshotRatio'] = df['headshotKills']/df['kills'] #May Produce Nan or inf ##Determine Headaimming Skills
    print('Generate AverageDamagePerkill')
    df['AverageDamagePerKill'] = df['damageDealt']/df['kills'] #May Produce Nan or inf ##Determine Efficiency
    print('Generate KillstreakRate')
    df['KillstreakRate'] = df['killStreaks']/df['kills'] #May Produce Nan or inf ##Determine Efficiency when handling large group of enemy
    print('Generate TotalDistance')
    df['TotalDistance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
    print('Generate WeaponRetriveOverDistance')
    df['WROD'] = df['TotalDistance']/df['weaponsAcquired'] #Determine Weapon Pickup along the way
    print('Generate WalkingHeal')
    df['walkHeal'] = df['walkDistance']/df['heals'] #Heals Per Meter (Does this even related? IDK, someone visualize this for me pls.)
    print('Generate WalkingKills')
    df['walkkills'] = df['walkDistance']/df['kills'] #same as above need correlation map!
    print('Generate KillPerMeters')
    df['KPM'] = df['kills']/df['walkDistance'] #same as above...
    #Purging NaN
    print('Purging NaN from Dataframe...',end='') #U know m8
    try:
        df[df == np.Inf] = np.NaN #set all inf to NaN
        df[df == np.NINF] =np.NaN #set all Ninf to NaN
        df.fillna(0,inplace=True) #Replace all NaN with 0
        print('Done!')
    except:
        print('Nothing to Purge')
        err += 1
    
    if is_train:
        print('End of Operation')
        return df
    #Retrive maxplace to deal with an edge case.
    else:
        print('Retrive maxPlace...',end='')
        test_edge = df['maxPlace']
        test_edge = pd.DataFrame(test_edge)
        print('Done!')
        print('End of Operation')
        return df,test_id,test_edge
# Thanks and credited to https://www.kaggle.com/gemartin creator of this mem reducer.
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
#Run this line if mem gone wild (Guarantee 75% mem reduction.)
df_train,target = preprocessing(is_train=True)
df_train = feature_managing(df_train,is_train=True,lowfeature=True,valuematchtype=True)
df_train = reduce_mem_usage(df_train)
df_train[1:2] #Debuging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
print('scaling the dataset')
scaler = StandardScaler()
scaler.fit(df_train)
scaled_train = scaler.transform(df_train)
print('scaler ready!')
mlpreg = MLPRegressor(hidden_layer_sizes=(35,20,15),max_iter=500) #Reduce complexity cuz it's sucks and literally improve nothing.
print("begin fitting...",end='')
mlpreg.fit(scaled_train,target)
print("Done!")
def prediction(reducemem = True,returnall = True,smoothedge=True,csvwrite=False):
    df_test = preprocessing(is_train=False)
    df_test,test_id,test_edge = feature_managing(df_test,is_train=False,lowfeature=True,valuematchtype=True)
    if reducemem:
        df_test = reduce_mem_usage(df_test) #Reduce Test Dataframe Memory.
    print('scaling test dataframe')
    scaled_test = scaler.transform(df_test)
    print('scaled!')
    print('Begin Prediction...',end='')
    res = mlpreg.predict(scaled_test)
    print('Done!')
    print('Managing Output')
    print('Transform to np array to dataframe')
    res = pd.DataFrame(res)
    res.columns = ['winPlacePerc']
    print('Preparing Id')
    test_id = pd.DataFrame(test_id)
    print('Initial Result Dataframe')
    submis = pd.DataFrame({'Id':[] , 'winPlacePerc':[], 'maxPlace':[]})
    print('Record Result to Dataframe')
    submis['Id']=test_id['Id']
    submis['winPlacePerc']=res['winPlacePerc']
    print('Edge Smoothie')
    if smoothedge:
        #Dealing With Edge
        submis['maxPlace'] = test_edge['maxPlace']
        submis.loc[submis.maxPlace == 0, 'winPlacePerc'] = 0
        submis.loc[submis.maxPlace == 1, 'winPlacePerc'] = 1
        #Edge Smoothing
        submis.loc[submis.winPlacePerc <= 0,'winPlacePerc'] = 0
        submis.loc[submis.winPlacePerc > 1,'winPlacePerc'] = 1
        print('Edge Smoothie Ready to Serve!')
    else:
        print('Skipping Edge case smoothing')
    print('Finalizing ... ',end='')
    #End of Operation
    submis = submis.drop(['maxPlace'],axis=1)
    print('Done!')
    if csvwrite:
        print('exporting to csv...',end='')
        submis.to_csv('submission.csv',index=False)
    if returnall:
        print('Returned All')
        return res,submis
    print('Returned Result')
    return submis
res,submis = prediction(reducemem=True,returnall=True,smoothedge=True,csvwrite=True)
import seaborn as sns
targetframe = pd.DataFrame(target)
plt.figure()
sns.distplot(targetframe['winPlacePerc'])
plt.show()
targetframe.plot()
resframe = pd.DataFrame(res)
resframe.plot()
plt.figure()
sns.distplot(resframe['winPlacePerc'])
plt.show()
submis['winPlacePerc'].plot()