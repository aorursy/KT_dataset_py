# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_y = pd.read_csv('../input/data_csv_0608_y_notime_c5.csv')

data_x = pd.read_csv('../input/data_csv_0608_x_notime.csv')

print(data_y.shape)

print(list(data_y.columns))

print(list(data_x.columns))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

#logit cluster_2



y_field="cluster_2"



data_x_sel=data_x.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]

data_y_sel=data_y.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]



train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

logreg.fit(train_x_scaled, train_y)

pred_y = logreg.predict(test_x_scaled)



print(y_field+'_logit: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)
#logit cluster_1



y_field="cluster_1"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)

logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

logreg.fit(train_x_scaled, train_y)

pred_y = logreg.predict(test_x_scaled)



print(y_field+'_logit: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)
#logit cluster_0



y_field="cluster_0"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

logreg.fit(train_x_scaled, train_y)

pred_y = logreg.predict(test_x_scaled)



print(y_field+'_logit: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)
#svm cluster_2

y_field="cluster_2"



data_x_sel=data_x.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]

data_y_sel=data_y.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]

#data_x_sel=data_x

#data_y_sel=data_y



train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



C = 1.

kernel = 'linear'

gamma  = 0.01



estimator = SVC(C=C, kernel=kernel, gamma=gamma)

classifier = OneVsRestClassifier(estimator)

classifier.fit(train_x_scaled, train_y)

pred_y_svm = classifier.predict(test_x_scaled)



print(y_field+'_SVM: {:.5f}'.format(accuracy_score(test_y, pred_y_svm)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y_svm)

print(cf_matrix)
#svm cluster_1



y_field="cluster_1"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

#data_x_sel=data_x

#data_y_sel=data_y



train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



C = 1.

kernel = 'linear'

gamma  = 0.01



estimator = SVC(C=C, kernel=kernel, gamma=gamma)

classifier = OneVsRestClassifier(estimator)

classifier.fit(train_x_scaled, train_y)

pred_y_svm = classifier.predict(test_x_scaled)



print(y_field+'_SVM: {:.5f}'.format(accuracy_score(test_y, pred_y_svm)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y_svm)

print(cf_matrix)
#svm cluster_0



y_field="cluster_0"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



C = 1.

kernel = 'linear'

gamma  = 0.01



estimator = SVC(C=C, kernel=kernel, gamma=gamma)

classifier = OneVsRestClassifier(estimator)

classifier.fit(train_x_scaled, train_y)

pred_y_svm = classifier.predict(test_x_scaled)



print(y_field+'_SVM: {:.5f}'.format(accuracy_score(test_y, pred_y_svm)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y_svm)

print(cf_matrix)
#RF cluster_2



y_field="cluster_2"



data_x_sel=data_x.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]

data_y_sel=data_y.loc[(data_y[y_field] == 2) | (data_y[y_field] == 3)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)

classifier.fit(train_x_scaled, train_y)

pred_y = classifier.predict(test_x_scaled)



print(y_field+'_RF: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)
#RF cluster_1



y_field="cluster_1"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)

classifier.fit(train_x_scaled, train_y)

pred_y = classifier.predict(test_x_scaled)



print(y_field+'_RF: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)
#RF cluster_0



y_field="cluster_0"



data_x_sel=data_x.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]

data_y_sel=data_y.loc[(data_y[y_field] == 1) | (data_y[y_field] == 5)]





train_x, test_x, train_y, test_y  = train_test_split(data_x_sel,data_y_sel[y_field],random_state=3)

scaler = StandardScaler()

train_x_scaled = scaler.fit_transform(train_x)

test_x_scaled = scaler.transform(test_x)



classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 0)

classifier.fit(train_x_scaled, train_y)

pred_y = classifier.predict(test_x_scaled)



print(y_field+'_RF: {:.5f}'.format(accuracy_score(test_y, pred_y)))



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

%matplotlib inline



cf_matrix = confusion_matrix(test_y, pred_y)

print(cf_matrix)