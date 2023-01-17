import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split



%matplotlib inline
data=pd.read_csv('../input/airline-delay/DelayedFlights.csv')
data[['a','Year','Month','DayofMonth','DayOfWeek','CRSDepTime','CRSArrTime','FlightNum','Distance','Cancelled']]= data[['a','Year','Month','DayofMonth','DayOfWeek','CRSDepTime','CRSArrTime','FlightNum','Distance','Cancelled']].astype('int32')
data[['DepTime','ArrTime','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','TaxiIn','TaxiOut']]=data[['DepTime','ArrTime','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','TaxiIn','TaxiOut']].astype('float32')
data_drop_uni = data.drop(columns=['a','Year'],axis=1)
data_drop_uni.isna().sum()
data_drop_uni.drop(index = data_drop_uni[data_drop_uni.ArrDelay.isna()].index, inplace = True)

data_drop_uni.drop(index = data_drop_uni[data_drop_uni.TailNum.isna()].index, inplace = True)

data_drop_uni.drop(columns=['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'],inplace=True)
data_drop_uni.isnull().sum()
data_drop_uni['Month'] = data_drop_uni['Month'].astype('object')

data_drop_uni['DayofMonth'] = data_drop_uni['DayofMonth'].astype('object')

data_drop_uni['DayOfWeek'] = data_drop_uni['DayOfWeek'].astype('object')
data_drop_uni.drop(columns = ['Diverted'], inplace = True) 
data_describe = data_drop_uni.describe(percentiles = [.001,.01,.25,.75,.95,.99])

data_describe
outlier_feature = ['ArrDelay','DepDelay','TaxiIn','TaxiOut']

for i in outlier_feature:

    q1 = data_describe[i]['25%']

    q3 = data_describe[i]['75%']

    iqr = q3-q1

    oulier1 = data_drop_uni[data_drop_uni[i]> q3 + 1.5*iqr].index

    oulier2 = data_drop_uni[data_drop_uni[i] < q1-1.5*iqr].index

    oulier = np.concatenate((oulier1, oulier2), axis=0)

    data_drop_uni.drop(oulier,inplace=True)

data_drop_uni.describe(percentiles = [.01,.75,.99])
def classify(x):

    if x > 30:

        return 'Yes'

    else:

        return 'No'

data_drop_uni['Late'] = data_drop_uni.ArrDelay.apply(lambda x: classify(x))

data_classify = data_drop_uni.drop(columns='ArrDelay')
data_classify
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

data_classify['UniqueCarrier']=le1.fit_transform(data_classify['UniqueCarrier'])

le2 = LabelEncoder()

data_classify['TailNum']=le2.fit_transform(data_classify['TailNum'])

le3 = LabelEncoder()

data_classify['Origin']=le3.fit_transform(data_classify['Origin'])

le4 = LabelEncoder()

data_classify['Dest']=le4.fit_transform(data_classify['Dest'])

le5 = LabelEncoder()

data_classify['CancellationCode']=le5.fit_transform(data_classify['CancellationCode'])
X = data_classify.iloc[:,:-1]

Y = data_classify.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 ,random_state=1)
DecisionTree = DecisionTreeClassifier()

DecisionTree.fit(X_train, Y_train)
Y_pred=DecisionTree.predict(X_test)

Y_pred
test_result = np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.values.reshape(len(Y_test),1)), axis=1)

test_result = pd.DataFrame(data = test_result, columns =['Y_Predict','Y_test'] )

test_result.head(30)
test_result.Y_Predict.value_counts()
test_result.Y_test.value_counts()
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))