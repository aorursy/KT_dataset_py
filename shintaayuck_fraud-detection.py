import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder
# load functions

def readCombineCsv(directory):

    train = []

    test = []

    train_label = []

    test_label = []

    for filename in os.listdir(directory):

        data_type, data_func = filename.split('.')[3], filename.split('.')[4]

        if data_func == 'Inputs':

            data = pd.read_csv(directory + '/' + filename)

            train.append(data) if data_type == 'Train' else test.append(data)

        else:

            data = pd.read_csv(directory + '/' + filename, names=['label'])

            train_label.append(data) if data_type == 'Train' else test_label.append(data)

    train = pd.concat(train, axis=0, ignore_index=True)

    train_label = pd.concat(train_label, axis=0, ignore_index=True)

    test = pd.concat(test, axis=0, ignore_index=True)

    test_label = pd.concat(test_label, axis=0, ignore_index=True)

    return train, test, train_label, test_label
# load data

X_train, X_test, y_train, y_test = readCombineCsv('/kaggle/input/dataset/datasets')



display(X_train.head())
# describe data



X_train.describe()
import pandas_profiling 



pandas_profiling.ProfileReport(X_train)
print(y_train.label.value_counts())

y_train.label.hist(bins=2)
import plotly.offline as py

py.init_notebook_mode(connected=True)

total_transact=X_train['amount'].groupby(X_train['state1']).sum().reset_index().sort_values('amount', ascending=False)

total_transact.head()



aggregations = {

   # 'amount':[count,sum],

    'amount': lambda x: sum(x)

}

X_train.groupby('state1',as_index=False).agg(aggregations).sort_values('amount', ascending=False).head()





data_plotly = [ dict(

        type='choropleth',

        autocolorscale = True,

        locations = total_transact['state1'],

        z = total_transact['amount'].astype(float),

        locationmode = 'USA-states',

        text = total_transact['state1'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "USD")

        ) ]



layout = dict(

        title = 'Total amout of transactions<br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict( data=data_plotly, layout=layout )

py.iplot( fig, filename='transaction-map' )




fraud_by_state=X_train['label'].groupby(X_train['state1']).sum().reset_index().sort_values('label', ascending=False)

fraud_by_state.head()



aggregations = {

    'amount':'count',

    'label': 'sum'

#     'class' : lambda x: sum(x)

  }

fraud_by_state_devided_by_transactions=X_train.groupby('state1',as_index=False).agg(aggregations)

fraud_by_state_devided_by_transactions['fraud_per_totnum']=fraud_by_state_devided_by_transactions['label']/fraud_by_state_devided_by_transactions['amount']*100

fraud_by_state_devided_by_transactions.sort_values('fraud_per_totnum',ascending=False).head()



data_plotly = [ dict(

        type='choropleth',

        autocolorscale = True,

        locations = fraud_by_state_devided_by_transactions['state1'],

        z = fraud_by_state_devided_by_transactions['fraud_per_totnum'],

        locationmode = 'USA-states',

        text = fraud_by_state_devided_by_transactions['state1'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "No of fraud. Tran. per total number of transactions")

        ) ]



layout = dict(

        title = 'Total Number of fradulant transactions per total number of transactions in each state <br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict( data=data_plotly, layout=layout )

py.iplot( fig, filename='No_of_fraud_transaction_per_transaction-map' )
# Merge data & label for preprocessing

data_train = X_train.join(pd.get_dummies(y_train))

data_test = X_test.join(pd.get_dummies(y_test))

print('Original data train length: ', len(data_train))

print('Original data test length: ', len(data_test))
# drop rows with NaN values

data_train = data_train.dropna().copy()

data_test = data_test.dropna().copy()



print('Data train length without NaN: ', len(data_train))

print('Data test length without NaN: ', len(data_test))
# drop duplicated rows

# data_train.drop_duplicates(inplace=True)

# data_test.drop_duplicates(inplace=True)



# print('Data train length without duplicates: ', len(data_train))

# print('Data test length without duplicates: ', len(data_test))
# Label Encoding non-numerical feature

le = LabelEncoder()

data_train['state1'] = le.fit_transform(data_train['state1'])

data_train['domain1'] = le.fit_transform(data_train['domain1'])



data_test['state1'] = le.fit_transform(data_test['state1'])

data_test['domain1'] = le.fit_transform(data_test['domain1'])

 

display(data_train.head())
# data_train correlation

corr = data_train.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# data_test correlation

corr = data_test.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
import plotly.offline as py

py.init_notebook_mode(connected=True)

total_transact=X_train['amount'].groupby(X_train['state1']).sum().reset_index().sort_values('amount', ascending=False)

total_transact.head()



aggregations = {

   # 'amount':[count,sum],

    'amount': lambda x: sum(x)

}

X_train.groupby('state1',as_index=False).agg(aggregations).sort_values('amount', ascending=False).head()





data_plotly = [ dict(

        type='choropleth',

        autocolorscale = True,

        locations = total_transact['state1'],

        z = total_transact['amount'].astype(float),

        locationmode = 'USA-states',

        text = total_transact['state1'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "USD")

        ) ]



layout = dict(

        title = 'Total amout of transactions<br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict( data=data_plotly, layout=layout )

py.iplot( fig, filename='transaction-map' )
# remove columns with high correlation



## data train

corr = data_train.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            print(columns[j])

            if columns[j]:

                columns[j] = False

selected_train_columns = data_train.columns[columns]

data_train = data_train[selected_train_columns]



corr = data_train.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# remove columns with high correlation



## data test

corr = data_test.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_test_columns = data_test.columns[columns]

data_test = data_test[selected_test_columns]



corr = data_test.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# Split data & label after preprocessing



y_train = pd.DataFrame()

y_train['label'] = data_train.iloc[:,-1]



y_test = pd.DataFrame()

y_test['label'] = data_test.iloc[:,-1]



X_train = data_train.copy()

X_test = data_test.copy()
# normalized numerical data

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_data = []

for key in X_train.iteritems() :

    if (len(X_train[key[0]].unique()) > 6) :

        num_data.append(key[0])

        

X_train[num_data] = minmax_scale(X_train[num_data])

X_test[num_data] = minmax_scale(X_test[num_data])

X_test.describe()
# undersampling data train

from imblearn.under_sampling import NearMiss



nr = NearMiss()

train_data, train_label = nr.fit_sample(X_train, y_train)

np.bincount(train_label)
# Create model, fit, and score



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix



clf = RandomForestClassifier(n_estimators=100, max_depth=2)

clf.fit(train_data, train_label)



y_true = y_test

y_pred = clf.predict(X_test)

print('Confusion matrix : \n', confusion_matrix(y_true, y_pred))

print('Accuracy : ', clf.score(X_test, y_test))



print("F1 Score : ", f1_score(y_true, y_pred, average='macro'))
# Average score from 100 training



acc = []

f1score = []

falsePrediction = []



for i in range(100):

    clf.fit(train_data, train_label)



    y_pred = clf.predict(X_test)

    

    conf = confusion_matrix(y_true, y_pred)

    

    acc.append(clf.score(X_test, y_test))

    f1score.append(f1_score(y_true, y_pred, average='macro'))

    falsePrediction.append(conf[0,1])

    falsePrediction.append(conf[1,0])



    

print("Average accuracy : ", sum(acc) / len(acc))

print("F1-Score accuracy : ", sum(f1score) / len(f1score))

print("Total false prediction : ", sum(falsePrediction))

print("Average false prediction : ", sum(falsePrediction) / len(falsePrediction))
# Model score without undersampling the dataset



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100, max_depth=2)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)



y_true = y_test

y_pred = clf.predict(X_test)

print(confusion_matrix(y_true, y_pred))

print('Accuracy without undersampling: ', clf.score(X_test, y_test))



print('F1-Score without undersampling: ', f1_score(y_true, y_pred, average='macro'))