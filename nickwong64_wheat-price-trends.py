# Import library
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
# load dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')
series = read_csv('../input/wheat_200910-201803.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# check the head and tail rows
print(series.head(10))
print(series.tail())
#resample to weekly Friday level, last one is excluded as not actual Friday
weekly = series.resample('W-FRI').last()
weekly = weekly[:-1]
#Weekly plot of wheat price
pyplot.rcParams["figure.figsize"] = [16,9]
weekly.plot()
pyplot.show()
#resample to monthly level, last one is excluded as not actual monthly end
monthly = series.resample('1M').last()
monthly = monthly[:-1]
monthly.tail()
#monthly plot
monthly.plot()
pyplot.show()
#from this notebook, we can see that wheat price is very choppy. It is hard to predict for humans.
#Our next challenge is to build a prediction model that can beat baselines. 
# Calculate baseline (RMSE, Correct Trend Predictions) for the predictions by shifting the predicted as the last observed price
# split data into train and test
X = weekly["close"].values
train, test = X[0:-12], X[-12:]
# walk-forward validation, this get the baseline prediction base on the last observed price
history = [x for x in train]
predictions = list()
nb_correct_predict = 0
for i in range(len(test)):
    # get the history last row as predictions
    predictions.append(history[-1])
    # append the test set to the history
    history.append(test[i])
    # expected price
    expected = history[-1]
    #predicted price
    yhat = predictions[-1]
    #calculate number of correct trend predictions
    if i != 0:
        if (expected > old_expected) and (yhat > old_yhat):
            nb_correct_predict = nb_correct_predict+1
        elif (expected < old_expected) and (yhat < old_yhat):
            nb_correct_predict = nb_correct_predict+1
        elif (expected == old_expected) and (yhat == old_yhat):
            nb_correct_predict = nb_correct_predict+1
    print('Date=%s, Predicted=%.2f, Expected=%.2f' % (weekly.index[-12+i], yhat, expected))
    old_yhat = yhat
    old_expected = expected
# calculate rmse
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# print correct number of trend predictions
p_correct_predict = nb_correct_predict/(len(test)-1) * 100
print('Number of correct trend predictions: %d, percentage: %.1f' % (nb_correct_predict, p_correct_predict))
# line plot of observed vs predicted
pyplot.plot(test, label = 'Expected Value')
pyplot.plot(predictions, label = 'Predicted Value')
pyplot.legend()
pyplot.show()
#Our model need to have RMSE lower than 15.715 and no. of correct trend predictions > 50%
