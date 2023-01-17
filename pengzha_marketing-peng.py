# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_selection import RFE, f_regression

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import matthews_corrcoef

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



#head of train data

dataframe_train = pd.read_csv('../input/marketing2/train.csv', sep=';')

dataframe_train.head()



#drop_list=['default', 'duration']

#drop_list=['duration']

drop_list=[]

dataframe_train = dataframe_train.drop( drop_list, axis = 1 )

dataframe_train.head()
'''jobDict = {k: v for v, k in enumerate(dataframe_train['job'].unique())}

eduDict = {k: v for v, k in enumerate(dataframe_train['education'].unique())}

monthDict = {k: v for v, k in enumerate(dataframe_train['month'].unique())}

dayOfWeekDict = {k: v for v, k in enumerate(dataframe_train['day_of_week'].unique())}

defaultDict = {k: v for v, k in enumerate(dataframe_train['default'].unique())}

'''

#for test case

#eduDict['illiterate']=6

#defaultDict['yes']=2

dataframe_train = dataframe_train.assign(education_illiterate=pd.Series(np.zeros(dataframe_train.shape[0], dtype=np.int)))

dataframe_train = dataframe_train.assign(default_yes=pd.Series(np.zeros(dataframe_train.shape[0], dtype=np.int)))

'''

dataframe_train['job'] = dataframe_train['job'].map(jobDict)

dataframe_train['education'] = dataframe_train['education'].map(eduDict)

dataframe_train['month'] = dataframe_train['month'].map(monthDict)

dataframe_train['day_of_week'] = dataframe_train['day_of_week'].map(dayOfWeekDict)

dataframe_train['default'] = dataframe_train['default'].map(defaultDict)

'''

dataframe_train['y'] = dataframe_train['y'].map({'no':0,'yes':1})

dummy_fields = ['marital', 'housing', 'loan', 'contact', 'poutcome','job','education','month','day_of_week','default']

#dummy_fields = ['marital', 'housing', 'loan', 'contact', 'poutcome']

for each in dummy_fields:

    dummies = pd.get_dummies( dataframe_train.loc[:, each], prefix=each ) 

    dataframe_train = pd.concat( [dataframe_train, dummies], axis = 1 )

dataframe_train = dataframe_train.drop( dummy_fields, axis = 1 )

dataframe_train.head()
# Looking for nulls

#print(dataframe_train.isnull().any())

# Inspecting type

#print(dataframe_train.dtypes)



#dataframe_train.describe().transpose()

# get train set , test set

debug=True

X = dataframe_train.drop('y',axis=1)

y = dataframe_train['y']



scaler = StandardScaler()

if debug:

    X_train, X_test, y_train, y_test = train_test_split(X, y)

else:

    X_train = X

    y_train = y



scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:

X_train = scaler.transform(X_train)

if debug:

    X_test = scaler.transform(X_test)

clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(31,),max_iter=800)

#clf =  RandomForestClassifier(n_jobs=-1, n_estimators=2000)

#clf = SVC()

#clf = BaggingClassifier(mlp, n_estimators=20,max_samples=0.7,max_features=1.0, random_state=1,bootstrap=True)#, n_jobs = -1)



clf.fit(X_train,y_train)
if debug:  

    predictions = clf.predict(X_test)

    print(confusion_matrix(y_test,predictions))

    print(classification_report(y_test,predictions))

    print(matthews_corrcoef(y_test, predictions) )

else:

    predictions = clf.predict(X_train)

    print(confusion_matrix(y_train,predictions))

    print(classification_report(y_train,predictions))

    print(matthews_corrcoef(y_train, predictions) )
fileds =['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration','campaign','pdays', 'previous','poutcome','emp.var.rate', 'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']

dataframe_test = pd.read_csv('../input/marketing2/test.csv', sep=';',names = fileds)

dataframe_test.head()
#dataframe_test['default'].unique()

dataframe_test = dataframe_test.drop(drop_list, axis = 1 )



'''

dataframe_test['job'] = dataframe_test['job'].map(jobDict)

dataframe_test['education'] = dataframe_test['education'].map(eduDict)

dataframe_test['month'] = dataframe_test['month'].map(monthDict)

dataframe_test['day_of_week'] = dataframe_test['day_of_week'].map(dayOfWeekDict)

dataframe_test['default'] = dataframe_test['default'].map(defaultDict)

'''

for each in dummy_fields:

    dummies = pd.get_dummies( dataframe_test.loc[:, each], prefix=each ) 

    dataframe_test = pd.concat( [dataframe_test, dummies], axis = 1 )

dataframe_test = dataframe_test.drop( dummy_fields, axis = 1 )

dataframe_test.head()

#print(dataframe_test[dataframe_test.isnull().any(1)])

#print(dataframe_test.isnull().any())

scaler = StandardScaler()

scaler.fit(dataframe_test)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:

dataframe_test = scaler.transform(dataframe_test)

#dataframe_test.head()
predictions = clf.predict(dataframe_test)
predictions.shape=(1120,1)

ids = np.array(range(1,1121), order='C')

ids.shape=(1120,1)

predictions_res = np.append(ids, predictions, axis=1)

dataframe_predictions = pd.DataFrame(data=predictions_res[:,:] ,index=range(0,len(predictions)),columns=['Id','prediction'])

#dataframe_predictions

#if not debug:

print(dataframe_predictions.to_string())

#dataframe_predictions.values

#with pd.option_context('display.max_rows', None, 'display.max_columns', 3):

    #print(dataframe_predictions)