# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression

# from sklearn.ensemble import RandomForestRegressor

import catboost

from catboost import CatBoostRegressor



import pickle 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



def analiseModData(df):

    df.dropna(axis=0, inplace=True)

    df = df.drop(['Id', 'matchId','matchType', 'matchDuration',

                  'teamKills', 'rankPoints', 'winPoints','roadKills',

                  'swimDistance', 'kills', 'killPoints', 'maxPlace'], axis=1)

    

    dictValueCounts ={}

    GroupValueCount = df['groupId'].value_counts()

    dictValueCounts = dict(zip(GroupValueCount.index.tolist(), GroupValueCount.tolist()))

    df['groupId'].update(df['groupId'].map(dictValueCounts))   

    del dictValueCounts, GroupValueCount     

    return df  



def linModel(dataTrain):

    modTrain = analiseModData(dataTrain)

    (trainData,

     testData,

     trainLabel,

     testLabel) = train_test_split(modTrain.drop('winPlacePerc', axis=1),

                              modTrain["winPlacePerc"],

                              test_size=0.3,

                              random_state=1234126)

    

#     linReg = LinearRegression()

#     rfr = RandomForestRegressor()

#     rfr.fit(trainData, trainLabel)

    cbr = CatBoostRegressor()

    cbr.fit(trainData, trainLabel)





#     linReg.fit(trainData, trainLabel)

    del trainData, trainLabel, testData, testLabel, modTrain  

    with open("pickle_lin_model.pkl", 'wb') as file:  

        pickle.dump(cbr, file)



    return cbr





def prdict(test):

    import pickle

    testd = analiseModData(test)

    with open('pickle_lin_model.pkl', 'rb') as file:  

        pickle_lin_model = pickle.load(file)

    

    preds = pickle_lin_model.predict(testd)

    del testd

    return preds

dataTrain = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

# train = pd.read_csv('../input/mytrain/my_train_6670.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

id_value = test.Id

myModel = linModel(dataTrain)

preds = prdict(test)



submission = pd.DataFrame(

    {'id': id_value, 'winPlacePerc': preds},

    columns=['id', 'winPlacePerc'])

del id_value, preds, dataTrain, test, myModel

submission.to_csv('submission3.csv', index=False)



    



# Any results you write to the current directory are saved as output.