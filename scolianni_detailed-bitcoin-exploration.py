import pandas as pd 

import datetime



def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))



dataSetSize = 5000

data = pd.read_csv('../input/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv', parse_dates=True, date_parser=dateparse, index_col=[0])

data = data[['Close']].apply(pd.to_numeric)

data = data.dropna()

data = data.head(dataSetSize)



print(data.head())

print('\nDataset Size:\t' + str(len(data.index.tolist())))
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



layout = go.Layout(

    title='Original Time Series',

    xaxis=dict(title='Time (Minutes)'),

    yaxis=dict(title='USD')

)



figData = [{'x': [i for i in range(len(data['Close'].tolist()))], 'y': data['Close'].tolist()}]



fig = go.Figure(data=figData, layout=layout)



iplot(fig, show_link=False)
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import kpss



def testStationarity(inputData, alphs):

    

    # Augmented Dickey-Fuller Test

    # H0 is non-stationary

    results = adfuller(inputData)

    pValue = results[1]

    if pValue < alpha:

        print('ADF Result: \t Stationary' + '\t P-Value: \t' + str(pValue))

    else:

        print('ADF Result: \t Non-Stationary' + '\t P-Value: \t' + str(pValue))



    # Kwiatkowski-Phillips-Schmidt-Shin Test

    # H0 is stationary

    results = kpss(inputData)

    pValue = results[1]

    if pValue >= alpha:

        print('KPSS Result: \t Stationary' + '\t P-Value: \t' + str(pValue))

    else:

        print('KPSS Result: \t Non-Stationary' + '\t P-Value: \t' + str(pValue))



# Define alpha value for hypothesis testing

alpha = 0.05

testStationarity(data['Close'].values, alpha)
# Calculate different orders of difference in closing price

def calculateDifferences(data, numDifs):

    if numDifs == 0:

        keyColumn = 'Close'

    else:

        keyColumn = str(numDifs) + '_Dif'

    for i in range(1, numDifs+1):

        if i == 1:

            data[str(i) + '_Dif'] = data['Close'] - data['Close'].shift(1)

        else:

            data[str(i) + '_Dif'] = data[str(i-1) + '_Dif'] - data[str(i-1) + '_Dif'].shift(1)

    data = data.dropna()

    return data, keyColumn



numDifs = 1

data, keyColumn = calculateDifferences(data, numDifs)



print(data.head())
testStationarity(data[keyColumn].values, alpha)
layout = go.Layout(

    title='Differenced Time Series (' + str(keyColumn) + ')',

    xaxis=dict(title='Time (Minutes)'),

    yaxis=dict(title='USD')

)



figData = [{'x': [i for i in range(len(data[keyColumn].tolist()))], 'y': data[keyColumn].tolist()}]



fig = go.Figure(data=figData, layout=layout)



iplot(fig, show_link=False)
%matplotlib inline

hist = data[keyColumn].hist(bins=50)

hist.set_title("Histogram of Changes in Closing Price")

hist.set_xlabel("Difference in Closing Price (USD)")

hist.set_ylabel("Number of Minutes")
from scipy import stats

import pylab 

stats.probplot(data[keyColumn].values, dist='norm', plot=pylab)

pylab.show()
# Kolmogorov-Smirnov Test

# H0 is that both distributions are identical

results = stats.kstest(data[keyColumn].values, 'norm')

pValue = results[1]

if pValue >= alpha:

    print('KS Result: \t Gaussian' + '\t P-Value: \t' + str(pValue))

else:

    print('KS Result: \t Non-Gaussian' + '\t P-Value: \t' + str(pValue))
from statsmodels.graphics.tsaplots import plot_acf

import matplotlib.pyplot as pyplot

plot_acf(data[keyColumn])

pylab.xlim([0, 50])

pylab.ylim([-0.2, 0.2])

pyplot.show()
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data[keyColumn].head(100))

pylab.xlim([0, 100])

pylab.ylim([-0.8, 0.8])

pyplot.show()
from itertools import groupby



# Define the transaction fee (theta), and number of previous time steps to use (d)

theta = 0.000

d = 4



# Calculate the class at each timestamp

data['Class'] = data[keyColumn].apply(lambda x: 1 if x>theta else -1 if x<(-1*theta) else 0)



# Extract the class at data[t-1]

data['Previous_Class'] = data['Class'].shift(1)



# Extract the tally count for each class in the past d time steps

data['Class_-1_Tally'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: sum([1 for i in x if i == -1]))

data['Class_0_Tally'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: sum([1 for i in x if i == 0]))

data['Class_1_Tally'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: sum([1 for i in x if i == 1]))



# Extract the maximum consecutive run-length for each classin the past d time steps

def getLongestRun(x, val):

    groupedX = [sum(1 for i in g) for k, g in groupby(x) if k == val]

    if groupedX == []:

        longestRun = 0

    else:

        longestRun = max(groupedX)

    return longestRun

data['Class_-1_Consec'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: getLongestRun(x, -1))

data['Class_0_Consec'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: getLongestRun(x, 0))

data['Class_1_Consec'] = data['Class'].shift(1).rolling(window=d).apply(lambda x: getLongestRun(x, 1))



# Clean the data

data = data[['Class', 'Previous_Class', 'Class_-1_Tally', 'Class_0_Tally', 'Class_1_Tally', 'Class_-1_Consec', 'Class_0_Consec', 'Class_1_Consec']]

data = data.dropna()



print(data.head())
timestamps = data.index.tolist()



trainValSplit = timestamps[round(len(timestamps) * 0.6)]

valTestSplit = timestamps[round(len(timestamps) * 0.8)]



train = data.loc[:trainValSplit]

validation = data.loc[trainValSplit:valTestSplit]

test = data.loc[valTestSplit:]



print('Training Set Size: \t' + str(len(train)))

print('Validation Set Size: \t' + str(len(validation)))

print('Test Size: \t\t' + str(len(test)))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



y_train = train['Class'].values

X_train = train.drop(['Class'], axis=1).values



lr = LogisticRegression(class_weight='balanced')

lr.fit(X_train, y_train)



rf = RandomForestClassifier()

rf.fit(X_train, y_train)
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support



def getGammaByAccuracy(clf, clfName, X_val, y_val):

    y_pp = clf.predict_proba(X_val)

    results = []

    gammaRange = np.linspace(0, 1, num=100)

    for gamma in gammaRange:

        acceptedPredictionIndicies = [i for i in range(len(y_pp)) if (max(y_pp[i]) > gamma)]

            

        if acceptedPredictionIndicies != []:

            subX_val = [X_val[i] for i in acceptedPredictionIndicies]

            subY_val = [y_val[i] for i in acceptedPredictionIndicies]

                        

            # Parse out the instances where 0 was predicted

            subY_pred = clf.predict(subX_val)

            keepIndicies = [i for i in range(len(subY_pred)) if subY_pred[i] != 0]

            subY_pred_parsed = [subY_pred[i] for i in keepIndicies]

            subY_val_parsed = [subY_val[i] for i in keepIndicies]

            

            # Calculate Metrics

            accuracy = accuracy_score(subY_val_parsed, subY_pred_parsed)

            precision, recall, fscore, support = precision_recall_fscore_support(subY_val_parsed, subY_pred_parsed, labels=[-1, 0, 1])

            product = accuracy * len(acceptedPredictionIndicies)

            results.append((product, gamma, accuracy, len(acceptedPredictionIndicies), precision, recall, fscore, support))

            

        else:

            break

    product, gamma, accuracy, numPredictions, precision, recall, fscore, support = max(results)



    print('\n' + clfName + ' Results')

    print('Product: \t\t' + str(product))

    print('Gamma: \t\t\t' + str(gamma))

    print('Accuracy: \t\t' + str(accuracy)) 

    print('Num Predictions: \t' + str(numPredictions))

    print('Precision: \t\t' + str(precision)) 

    print('Recall: \t\t' + str(recall)) 

    print('F-Score: \t\t' + str(fscore)) 

    print('support: \t\t' + str(support)) 

    

    return gamma

    

y_val = validation['Class'].values

X_val = validation.drop(['Class'], axis=1).values



lrGamma = getGammaByAccuracy(lr, 'Logistic Regression', X_val, y_val)

rfGamma = getGammaByAccuracy(rf, 'Random Forest', X_val, y_val)
def getGammaByPrecision(clf, clfName, X_val, y_val):

    y_pp = clf.predict_proba(X_val)

    results = []

    gammaRange = np.linspace(0, 1, num=100)

    for gamma in gammaRange:

        acceptedPredictionIndicies = [i for i in range(len(y_pp)) if (max(y_pp[i]) > gamma)]

        if acceptedPredictionIndicies != []:

            subX_val = [X_val[i] for i in acceptedPredictionIndicies]

            subY_val = [y_val[i] for i in acceptedPredictionIndicies]

                        

            # Parse out the instances where 0 was predicted

            subY_pred = clf.predict(subX_val)

            keepIndicies = [i for i in range(len(subY_pred)) if subY_pred[i] != 0]

            subY_pred_parsed = [subY_pred[i] for i in keepIndicies]

            subY_val_parsed = [subY_val[i] for i in keepIndicies]

            

            # Calculate Metrics

            accuracy = accuracy_score(subY_val_parsed, subY_pred_parsed)

            precision, recall, fscore, support = precision_recall_fscore_support(subY_val_parsed, subY_pred_parsed, labels=[-1, 0, 1])

            product = precision[0] * precision[2] * len(acceptedPredictionIndicies)

            results.append((product, gamma, accuracy, len(acceptedPredictionIndicies), precision, recall, fscore, support))

            

        else:

            break

    product, gamma, accuracy, numPredictions, precision, recall, fscore, support = max(results)

    

    print('\n' + clfName + ' Results')

    print('Product: \t\t' + str(product))

    print('Gamma: \t\t\t' + str(gamma))

    print('Accuracy: \t\t' + str(accuracy)) 

    print('Num Predictions: \t' + str(numPredictions))

    print('Precision: \t\t' + str(precision)) 

    print('Recall: \t\t' + str(recall)) 

    print('F-Score: \t\t' + str(fscore)) 

    print('support: \t\t' + str(support)) 

    

    return gamma

    

y_val = validation['Class'].values

X_val = validation.drop(['Class'], axis=1).values



lrGamma = getGammaByPrecision(lr, 'Logistic Regression', X_val, y_val)

rfGamma = getGammaByPrecision(rf, 'Random Forest', X_val, y_val)
def testPerformance(clf, gamma, clfName, X_test, y_test, timestamps):

    

    # Parse out predicitons that don't meet gamma value

    y_pp = clf.predict_proba(X_test)

    acceptedPredictionIndicies = [i for i in range(len(y_pp)) if (max(y_pp[i]) > gamma)]

    subX_test = [X_test[i] for i in acceptedPredictionIndicies]

    subY_test = [y_test[i] for i in acceptedPredictionIndicies]

    subTimestamps = [timestamps[i] for i in acceptedPredictionIndicies]

    

    # Parse out the instances where 0 was predicted

    subY_pred = clf.predict(subX_test)

    keepIndicies = [i for i in range(len(subY_pred)) if subY_pred[i] != 0]

    subY_pred_parsed = [subY_pred[i] for i in keepIndicies]

    subY_test_parsed = [subY_test[i] for i in keepIndicies]

    subTimestamps_parsed = [subTimestamps[i] for i in keepIndicies]

    

    # Calculate Metrics

    accuracy = accuracy_score(subY_test_parsed, subY_pred_parsed)

    precision, recall, fscore, support = precision_recall_fscore_support(subY_test_parsed, subY_pred_parsed)



    # Print Metrics

    print('\n' + clfName + ' Results')

    print('Accuracy: \t\t' + str(accuracy))

    print('Precision: \t\t' + str(precision))

    print('Recall: \t\t' + str(recall)) 

    print('F-Score: \t\t' + str(recall))

    print('Support: \t\t' + str(support))

    

    # Return predictions, true classes, and timestamps

    return (subY_pred_parsed, subY_test_parsed, subTimestamps_parsed)



timestamps = test.index.tolist()

y_test = test['Class'].values

X_test = test.drop(['Class'], axis=1).values



lrResults = testPerformance(lr, lrGamma, 'Logistic Regression', X_test, y_test, timestamps)

rfResults = testPerformance(rf, rfGamma, 'Random Forest', X_test, y_test, timestamps)