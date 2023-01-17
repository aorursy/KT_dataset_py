import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
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

del X_train

del X_test

X.shape
X.head()
data = X
CPlot('bin_0')
CPlot('bin_1')
CPlot('bin_2')
CPlot('bin_3')
CPlot('bin_4')
CPlot('nom_0')
CPlot('nom_1')
CPlot('nom_2')
CPlot('nom_3')
CPlot('nom_4')
CPlot('ord_0')
CPlot('ord_1')
CPlot('ord_2')
CPlot('ord_3')
CPlot('ord_4')
CPlot('ord_5')
data.head()
OHE  = OneHotEncoder()

data_dummies = OHE.fit_transform(data)
data_dummies.shape
train_data = data_dummies[:train.shape[0],:]

test_data=  data_dummies[train.shape[0]:,:]

train_data.shape , test_data.shape
X = train_data

y = train['target']

X.shape , y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
LogisticRegressionModel = LogisticRegression(penalty='l2',solver='lbfgs',C=1.0,random_state=33)

LogisticRegressionModel.fit(X_train, y_train)
print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))

print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))
y_pred = LogisticRegressionModel.predict_proba(test_data)

y_pred.shape
y_pred[:,1]
data = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')  



print(f'Test data Shape is {data.shape}')

data.head()
idd = data['id']

FinalResults = pd.DataFrame(y_pred[:,1],columns= ['target'])

FinalResults.insert(0,'id',idd)

FinalResults.head()
FinalResults.to_csv("sample_submission.csv",index=False)