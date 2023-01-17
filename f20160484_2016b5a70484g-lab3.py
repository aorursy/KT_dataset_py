import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

# from sklearn import cross_validation, metrics

from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.cluster import SpectralClustering



from sklearn.preprocessing import MinMaxScaler



%matplotlib inline
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df.PaymentMethod.unique()
#

def adjust(df):

    SubMap = {'Monthly' : 1, 'Biannually' : 6, 'Annually' : 12}

    df1 = df[df.columns]

    df1.Subscription = df1.Subscription.map(SubMap)

    #

    df1['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric, errors = 'coerce')

    df1 = df1[df1['TotalCharges'].notnull()]

    df2 = df1.reset_index(drop = True)

    #df2['TotalCharges'].isnull().any()

    #

    HSmap = {'No internet' : 0, 'No' : 1, 'Yes' : 2}

    df2['HighSpeed'] = df2['HighSpeed'].map(HSmap)

    #df2['HighSpeed'].head()

    #

    Gmap = {'Male' : 0, 'Female' : 1}

    df2['gender'] = df2['gender'].map(Gmap) 

    #

    YNmap = {'No' : 0, 'Yes' : 1}

    df2['Married'] = df2['Married'].map(YNmap)

    df2['Children'] = df2['Children'].map(YNmap)

    df2['AddedServices'] = df2['AddedServices'].map(YNmap)

    df2['Internet'] = df2['Internet'].map(YNmap)

    #dropping rows

#     df2.drop('Internet', axis = 1, inplace = True)

    #

    df2 = pd.get_dummies(data = df2, columns = ['TVConnection', 'PaymentMethod'])

    

    #

    chMap = {'No tv connection' : 0, 'No' : 1, 'Yes' : 2}

    df2[ ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'] ] = df2[ ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'] ].replace(chMap) 

#     df2['SumCh'] = df2['Channel1'] + df2['Channel2'] + df2['Channel3'] + df2['Channel4'] + df2['Channel5'] + df2['Channel6']

#     df2.drop(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'], axis = 1, inplace = True)



    return df2
#

def adjustTest(df, dd):

    SubMap = {'Monthly' : 1, 'Biannually' : 6, 'Annually' : 12}

    df1 = df[df.columns]

    df1.Subscription = df1.Subscription.map(SubMap)

    #

    df1['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric, errors = 'coerce')

    df1['TotalCharges'].fillna(dd['TotalCharges'].mean(), inplace = True)

    #df2['TotalCharges'].isnull().any()

    df2 = df1[df1.columns]

    #

    HSmap = {'No internet' : 0, 'No' : 1, 'Yes' : 2}

    df2['HighSpeed'] = df2['HighSpeed'].map(HSmap)

    #df2['HighSpeed'].head()

    #

    Gmap = {'Male' : 0, 'Female' : 1}

    df2['gender'] = df2['gender'].map(Gmap) 

    #

    YNmap = {'No' : 0, 'Yes' : 1}

    df2['Married'] = df2['Married'].map(YNmap)

    df2['Children'] = df2['Children'].map(YNmap)

    df2['AddedServices'] = df2['AddedServices'].map(YNmap)

    df2['Internet'] = df2['Internet'].map(YNmap)



    #dropping rows

#     df2.drop('Internet', axis = 1, inplace = True)

    #

    df2 = pd.get_dummies(data = df2, columns = ['TVConnection', 'PaymentMethod'])

    

    #

    chMap = {'No tv connection' : 0, 'No' : 1, 'Yes' : 2}

    df2[ ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'] ] = df2[ ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'] ].replace(chMap) 

#     df2['SumCh'] = df2['Channel1'] + df2['Channel2'] + df2['Channel3'] + df2['Channel4'] + df2['Channel5'] + df2['Channel6']

#     df2.drop(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'], axis = 1, inplace = True)



    return df2
df1 = df[df.columns]

df1 = adjust(df1)

df1.head()
def trainKM(df):



    X = df[df.columns]

    y = df['Satisfied']

    X.drop(['custId', 'Satisfied', 'Internet'], axis = 1, inplace = True)

#     X.drop(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6'], axis = 1, inplace = True)



    scaler = MinMaxScaler()

    cols = ['TotalCharges', 'tenure', 'MonthlyCharges']

    X[cols] = scaler.fit_transform(X[cols])

    nc = 10

    clustering = KMeans(nc, n_init = 100, random_state = 10).fit(X)

    y_pred = clustering.labels_



    zr = [0 for i in range(nc)]

    on = [0 for i in range(nc)]



    for i in range(df.shape[0]):

        if (y[i] == 0):

            zr[y_pred[i]] = zr[y_pred[i]] + 1

        else:

            on[y_pred[i]] = on[y_pred[i]] + 1

        

    cl_count = {}



    for i in range(nc):

        print('on : ', on[i], 'zr : ', zr[i])

        if (on[i] >= 0.75*(on[i] + zr[i])):

            cl_count[i] = 1

        else:

            cl_count[i] = 0

            

    for i in range(df.shape[0]):

        y_pred[i] = cl_count[y_pred[i]]

        

    acc = accuracy_score(y, y_pred)



    

    return (acc, scaler, clustering, cl_count)
def testKM(test, scaler, clus, c_map):

    

#     test = adjust(test)

    test.drop(['custId', 'Internet'], axis = 1, inplace = True)

    

    cols = ['TotalCharges', 'tenure', 'MonthlyCharges']

    test[cols] = scaler.transform(test[cols])

    

    y_pred = clus.predict(test)

    

    for i in range(test.shape[0]):

        y_pred[i] = c_map[y_pred[i]]

    

    return y_pred
acc, scaler, clus, c_map = trainKM(df1)

acc
test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

Xtest = test[test.columns]

Xtest = adjustTest(Xtest, df1)

Xtest.head()
pred = testKM(Xtest, scaler, clus, c_map)
c_map
pred = pd.DataFrame(pred)

pred[0].value_counts() 
test_ans = pd.concat([test['custId'], pred], axis = 1)

test_ans.columns = ['custId', 'Satisfied']

test_ans.head()
test_ans.to_csv('submission6.csv', index = False)