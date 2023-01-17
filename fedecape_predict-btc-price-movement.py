import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from subprocess import check_output



print("Available data:\n")

print(check_output(["ls", "../input"]).decode("utf8"))
available_data = {

    'bitstamp': pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv"),

    'coinbase': pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv"),

    'btce': pd.read_csv("../input/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv"),

    'kraken': pd.read_csv("../input/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv")

}
print("Bitstamp data shape: {0}\nCoinbase data shape: {1}\nBTCe data shape: {2}\nKraken data shape: {3}".format(

    available_data['bitstamp'].shape,

    available_data['coinbase'].shape,

    available_data['btce'].shape,

    available_data['kraken'].shape))
btc = available_data['bitstamp']
# Show how the data is structured



btc.head()
# Fill the value gaps forward



btc[btc.columns.values] = btc[btc.columns.values].ffill()
# Plot how the Open prices look



btc['Open'].plot()
btc['Delta'] = btc['Close'] - btc['Open']
# And we plot the per-minute movements



btc['Delta'].plot(kind='line')
btc[abs(btc['Delta']) >= 100]
def digitize(n):

    if n > 0:

        return 1

    return 0

    

btc['to_predict'] = btc['Delta'].apply(lambda d: digitize(d))
# Show the last 5 elements of the btc dataframe



btc.tail()
btc_mat = btc.as_matrix()
def rolling_window(a, window):

    """

        Takes np.array 'a' and size 'window' as parameters

        Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'

        e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )

             Output: 

                     array([[1, 2, 3, 4],

                           [2, 3, 4, 5],

                           [3, 4, 5, 6]])

    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)

    strides = a.strides + (a.strides[-1],)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)





WINDOW_SIZE = 22
# Generate the X dataset (the 'Delta' column is the 8th)

# Remove the last row since it can't have its Y value



X = rolling_window(btc_mat[:,8], WINDOW_SIZE)[:-1,:]
# Let's see how it looks



btc['Delta'].tail(10)
# And now let's compare the above with the X matrix



print("{0}\n\nShape: {1}".format(X, X.shape))
# We generate the corresponding Y array and check if X and Y shapes are compatible



Y = btc['to_predict'].as_matrix()[WINDOW_SIZE:]

print("{0}\n\nShape: {1}".format(Y, Y.shape))
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4284, stratify=Y)
# Let's see how it looks



y_test[:100]
clf = RandomForestClassifier(random_state=4284, n_estimators=50)

clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(classification_report(y_test, predicted))
conf_mat = confusion_matrix(y_test, predicted)



# Confusion matrix in percentages

pct_conf_mat = conf_mat/np.sum(conf_mat) * 100



print("Pred:  0\t\t1\n{}".format(pct_conf_mat))