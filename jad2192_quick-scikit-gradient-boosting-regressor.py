import pandas as pd

import sklearn
data = pd.read_csv('../input/train.csv')
def dummify(s):

    return 1*(s == 'B')
for k in range(1,117):

    data['cat'+str(k)] = data['cat'+str(k)].apply(dummify)
Y = data['loss'].as_matrix()

X = data.ix[:,'cat1':'cont14'].as_matrix()
from sklearn.ensemble import GradientBoostingRegressor



reg1 = GradientBoostingRegressor(loss='huber',n_estimators=500)



reg1.fit(X,Y)
data_test = pd.read_csv('../input/test.csv')
for k in range(1,117):

    data_test['cat'+str(k)] = data_test['cat'+str(k)].apply(dummify)
test_data = data_test.ix[:,'cat1':].as_matrix()



test_loss = reg1.predict(test_data)



data_test['loss'] = pd.Series(test_loss)
output = data_test[['id','loss']]



#output.to_csv('../input/results.csv',index=False)