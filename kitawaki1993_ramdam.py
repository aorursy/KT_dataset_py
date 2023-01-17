import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

test = pd.read_csv("../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv")



test = test.iloc[::-1]



x_test = test['Open']

y_test = test['Close']



x_test = x_test.as_matrix()

x_test = np.reshape(x_test, (x_test.size, 1))

y_test = y_test.as_matrix()

y_test = np.reshape(y_test, (y_test.size, 1))





train = train.iloc[::-1]

x_train = train['Open']

y_train = train['Close']







x_train = x_train.as_matrix()

x_train = np.reshape(x_train, (x_train.size, 1))

#print(b)

#xdata_n[0] = x_data

y_train = y_train.as_matrix()

y_train = np.reshape(y_train, (y_train.size, 1))



from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor()

model.fit(x_train, y_train)





prices = model.predict(x_test)



for i , p in enumerate(prices):

	print("predict:%s, Target:%s" % (p, y_test[i]))



score = model.score(x_test, y_test)

print(score)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

test = pd.read_csv("../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv")



test = test.iloc[::-1]



x_test = test.loc[:, ['Open', 'High', 'Low', 'Close']]

y_test = test['Close']



x_test = x_test.as_matrix()

x_test = x_test[0:int(x_test.size / 4) - 1]



y_test = y_test.as_matrix()

y_test = np.reshape(y_test, (y_test.size, 1))

#print(y_train)

y_test = y_test[1:]







train = train.iloc[::-1]

#print(train)

x_train = train.loc[:, ['Open', 'High', 'Low', 'Close']]

y_train = train['Close']





x_train = x_train.as_matrix()

x_train = x_train[0:int(x_train.size / 4) - 1]

#print(x_train)

#x_train = np.reshape(x_train, (x_train.size, 1))



y_train = y_train.as_matrix()

y_train = np.reshape(y_train, (y_train.size, 1))

#print(y_train)

y_train = y_train[1:]

#print(y_train)



from sklearn import linear_model

clf = linear_model.LinearRegression()



clf.fit(x_train, y_train)



prices = clf.predict(x_test)



for i , p in enumerate(prices):

	print("predict:%s, Target:%s" % (p, y_test[i]))



score = clf.score(x_test, y_test)

print(score)
