# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/covtype.csv')
import numpy as np

np.random.seed(42)



msk = np.random.rand(len(data)) < 0.8



train = data[msk]

test = data[~msk]
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(train[['Elevation', 'Slope']], train['Cover_Type'])

train_pred = model.predict(train[['Elevation', 'Slope']])

test_pred = model.predict(test[['Elevation', 'Slope']])





print("Train %d points : %d" % (train[['Elevation', 'Slope']].shape[0],(train['Cover_Type'] != train_pred).sum()))

print("Test %d points : %d" % (test[['Elevation', 'Slope']].shape[0],(test['Cover_Type'] != test_pred).sum()))
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

cm = confusion_matrix(test['Cover_Type'], test_pred)

plt.imshow(cm)

plt.colorbar()
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

model = gnb.fit(train[['Elevation', 'Slope']], train['Cover_Type'])



cols = ['Elevation', 'Slope', '']



train_pred = model.predict(train[cols])

test_pred = model.predict(test[cols])





print("Train %d points : %d" % (train[cols].shape[0],(train['Cover_Type'] != train_pred).sum()))

print("Test %d points : %d" % (test[cols].shape[0],(test['Cover_Type'] != test_pred).sum()))



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

cm = confusion_matrix(test['Cover_Type'], test_pred)

plt.imshow(cm)

plt.colorbar()
copy = data.corr()['Cover_Type'][:]

abs(copy).sort_values()
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

cols = ['Elevation', 'Wilderness_Area4', 'Soil_Type10', 'Wilderness_Area1']



model = gnb.fit(train[cols], train['Cover_Type'])



train_pred = model.predict(train[cols])

test_pred = model.predict(test[cols])





print("Train %d points : %d" % (train[cols].shape[0],(train['Cover_Type'] != train_pred).sum()))

print("Test %d points : %d" % (test[cols].shape[0],(test['Cover_Type'] != test_pred).sum()))



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

cm = confusion_matrix(test['Cover_Type'], test_pred)

plt.imshow(cm)

plt.colorbar()

plt.title((test['Cover_Type'] == test_pred).sum() / len(test))
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

cols = ['Elevation']



model = gnb.fit(train[cols], train['Cover_Type'])



train_pred = model.predict(train[cols])

test_pred = model.predict(test[cols])



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

cm = confusion_matrix(test['Cover_Type'], test_pred)



cm = (cm / len(test))*100



plt.imshow(cm)

plt.colorbar()

plt.title((test['Cover_Type'] == test_pred).sum() / len(test))
df = pd.DataFrame()

df['act'] = test['Cover_Type']

df['pred'] = test_pred
correct_preds = df[df['act']==df['pred']]['act'].value_counts()

totals = test['Cover_Type'].value_counts()



combine = pd.DataFrame(correct_preds).join(pd.DataFrame(totals))



combine.plot.bar()
from sklearn import datasets

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

cols = ['Elevation']



model = gnb.fit(train[cols], train['Cover_Type'])



train_pred = model.predict(train[cols])

test_pred = model.predict(test[cols])



from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

cm = confusion_matrix(test['Cover_Type'], test_pred)



plt.imshow(cm)

plt.colorbar()

plt.title((test['Cover_Type'] == test_pred).sum() / len(test))



df = pd.DataFrame()

df['act'] = test['Cover_Type']

df['pred'] = test_pred



correct_preds = df[df['act']==df['pred']]['act'].value_counts()

totals = test['Cover_Type'].value_counts()



combine = pd.DataFrame(correct_preds).join(pd.DataFrame(totals))



combine.plot.bar()
cm