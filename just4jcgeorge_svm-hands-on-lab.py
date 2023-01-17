# Machine learning

#Type Your Code here to - Import 'SVC' from 'sklearn.svm'

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



# For data manipulation

import pandas as pd

import numpy as np



# To plot

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# To ignore warnings

import warnings

warnings.filterwarnings("ignore")
# Fetch the data 

# Type your code here - Read data of 'SPY' from CSV file. Store this in a DataFrame 'Df'

Df = pd.read_csv('../input/week4data/SPY.csv')

Df= Df.dropna()

Df = Df.set_index(Df.Date)

Df = Df.drop(columns='Date')

Df.head()
# Predictor variables

Df['Open-Close'] = Df.Open - Df.Close

Df['High-Low'] = Df.High - Df.Low

X= Df[['Open-Close','High-Low']]

X.head()
# Target variables

y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)
split_percentage = 0.8

split = int(split_percentage*len(Df))



# Train data set

X_train = X[:split]

y_train = y[:split] 



# Test data set

X_test = X[split:]

y_test = y[split:]
# Support vector classifier

cls = SVC().fit(X_train, y_train)
# train and test accuracy

accuracy_train = accuracy_score(y_train, cls.predict(X_train))

accuracy_test = accuracy_score(y_test, cls.predict(X_test))



print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))

print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))
# Predicted Signal

Df['Predicted_Signal'] = cls.predict(X)



# Calculate daily returns

Df['Return'] = Df.Close.pct_change()



# Calculate strategy returns

Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal



# Calculate geometric returns

geometric_returns = (Df.Strategy_Return.iloc[split:]+1).cumprod()



# Plot geometric returns

geometric_returns.plot(figsize=(10,5))

plt.ylabel("Strategy Returns (%)")

plt.xlabel("Date")

plt.show()