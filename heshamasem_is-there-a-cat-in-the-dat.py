import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb

train = pd.read_csv('../input/cat-in-the-dat/train.csv')  

test = pd.read_csv('../input/cat-in-the-dat/test.csv')  



print(f'Train data Shape is {train.shape}')

print(f'Test data Shape is {test.shape}')
def Drop(feature) :

    global data

    data.drop([feature],axis=1, inplace=True)

    data.head()

    

def UniqueAll(show_value = True) : 

    global data

    for col in data.columns : 

        print(f'Length of unique data for   {col}   is    {len(data[col].unique())} ')

        if show_value == True  : 

            print(f'unique values ae {data[col].unique()}' )

            print('-----------------------------')

            

def Encoder(feature , new_feature, drop = True) : 

    global data

    enc  = LabelEncoder()

    enc.fit(data[feature])

    data[new_feature] = enc.transform(data[feature])

    if drop == True : 

        data.drop([feature],axis=1, inplace=True)

        

def CPlot(feature) : 

    global data

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))

    

def Mapp(feature , new_feature ,f_dict, drop_feature = True) : 

    global data

    data[new_feature] = data[feature].map(f_dict)

    if drop_feature == True : 

        data.drop([feature],axis=1, inplace=True)

    else :

        data.head()

def Unique(feature) : 

    global data

    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')
train.head()
test.head()
X_train = train.drop(['id' , 'target'], axis=1, inplace=False)

X_test = test.drop(['id'], axis=1, inplace=False)



X_train.shape , X_test.shape
X = pd.concat([X_train , X_test])

X.shape
X.head()
data = X
CPlot('bin_0')
CPlot('bin_1')
CPlot('bin_2')
CPlot('bin_3')
Mapp('bin_3' , 'bin_03' , {'T':1 , 'F':0} , True)
data.head()
CPlot('bin_03')
Mapp('bin_4' , 'bin_04' , {'Y':1 , 'N':0} , True)
CPlot('bin_04')
UniqueAll(False)
for C in ['nom_0' , 'nom_1' , 'nom_2' , 'nom_3' , 'nom_4'] : 

    enc  = LabelEncoder()

    enc.fit(X[C])

    X[C] = enc.transform(X[C])
data.head()
CPlot('nom_0')
CPlot('nom_1')
CPlot('nom_2')
CPlot('nom_3')
CPlot('nom_4')
Unique('nom_5')
Unique('nom_6')
for C in ['nom_5' , 'nom_6' , 'nom_7' , 'nom_8' , 'nom_9']: 

    enc  = LabelEncoder()

    enc.fit(X[C])

    X[C] = enc.transform(X[C])
data.head()
CPlot('ord_0')
CPlot('ord_1')
CPlot('ord_2')
CPlot('ord_3')
CPlot('ord_4')
CPlot('ord_5')
for C in ['ord_0' , 'ord_1' , 'ord_2' , 'ord_3' , 'ord_4' , 'ord_5']: 

    enc  = LabelEncoder()

    enc.fit(X[C])

    X[C] = enc.transform(X[C])



data.head()
train_data = data.iloc[:train.shape[0],:]

test_data=  data.iloc[train.shape[0]:,:]

train_data.shape , test_data.shape
X = train_data

y = train['target']

X.shape , y.shape
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
num_round = 25000



parameters = {'num_leaves': 128,

             'min_data_in_leaf': 20, 

             'objective':'binary',

             'max_depth': 8,

             'learning_rate': 0.001,

             "min_child_samples": 20,

             "boosting": "gbdt",

             "feature_fraction": 0.9,

             "bagging_freq": 1,

             "bagging_fraction": 0.9 ,

             "bagging_seed": 44,

             "metric": 'auc',

             "verbosity": -1}





traindata = lgb.Dataset(X_train, label=y_train)

testdata = lgb.Dataset(X_test, label=y_test)



LGBModel = lgb.train(parameters, traindata, num_round, valid_sets = [traindata, testdata],

                     verbose_eval=50, early_stopping_rounds = 600)

test = scaler.transform(test_data)

test.shape
y_pred = LGBModel.predict(test)

y_pred.shape
y_pred[:10]
data = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')  



print(f'Test data Shape is {data.shape}')

data.head()
idd = data['id']

FinalResults = pd.DataFrame(y_pred,columns= ['target'])

FinalResults.insert(0,'id',idd)

FinalResults.head()
FinalResults.to_csv("sample_submission.csv",index=False)