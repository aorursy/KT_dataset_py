import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
million1 = np.where (train.price_doc==1e6, 1, 0)

million2 = np.where (train.price_doc==2e6, 1, 0)

million3 = np.where (train.price_doc==3e6, 1, 0)

basedate = pd.to_datetime("2010-08-19").toordinal()

time = pd.to_datetime(train.timestamp).apply(lambda x: x.toordinal()) - basedate

times = pd.DataFrame({"time":time })

testtime = pd.to_datetime(test.timestamp).apply(lambda x: x.toordinal()) - basedate

testtimes = pd.DataFrame({"time":testtime })
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(max_iter=1000)



logit.fit(times, million1)

p1train = logit.predict_proba(times)

p1test = logit.predict_proba(testtimes)



logit.fit(times, million2)

p2train = logit.predict_proba(times)

p2test = logit.predict_proba(testtimes)



logit.fit(times, million3)

p3train = logit.predict_proba(times)

p3test = logit.predict_proba(testtimes)
print(p1train[:,1].mean())

print(p1test[:,1].mean())



print(p2train[:,1].mean())

print(p2test[:,1].mean())



print(p3train[:,1].mean())

print(p3test[:,1].mean())