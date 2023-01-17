import pandas as pd
import math
import quandl
data = quandl.get("WIKI/GOOGL")
data.head()
data= data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
data.head()
data['HL_PCT'] = (data['Adj. High']-data['Adj. Close'])/data['Adj. Close']*100.00
data['PCT_change'] = (data['Adj. Close']-data['Adj. Open'])/data['Adj. Open']*100.00
data.head()
data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forcast_col = 'Adj. Close'
data.fillna(-99999, inplace = True)
forcast_out = int(math.ceil(0.1*len(data)))
data['LABEL'] = data[forcast_col].shift(-forcast_out)
data.dropna(inplace=True)
data.head()
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
X = np.array(data.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]
y = np.array(data['label'][:-forcast_out])
#y = np.array(data['label'])
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
clf = LinearRegression()
clf.fit(X_train, y_train)
Accuracy  = clf.score(X_test, y_test)
Forcast_set = clf.predict(X_lately)
#print(Forcast_set, Accuracy, forcast_out)
data['Forcast']= np.nan
import datetime
import matplotlib.pyplot as plt
import numpy as np

x_axis = data.index
x_axis = x_axis[-forcast_out:]
#print(x_axis)
y_axis = Forcast_set
#print(y_axis)
plt.plot(x_axis, y_axis)
plt.xlable = "Date"
plt.xlable = "Forcasted Value"
plt.show()
Last_Date = data.iloc[-1].name
Last_Unix = Last_Date.timestamp()
one_day = 86400
Next_Unix = Last_Unix + one_day

for i in Forcast_set:
    Next_date = datetime.datetime.fromtimestamp(Next_Unix)
    Next_Unix += one_day
    data.loc[Next_date]= [np.nan for _ in range(len(data.columns)-1)]+ [i]
                                               
print(data.tail())
data['Adj. Close'].plot()
data['Forcast'].plot()
plt.legend(loc=4)
plt.xlable = 'Date'
plt.ylable = 'price'
plt.show()

                                               
                                               
    