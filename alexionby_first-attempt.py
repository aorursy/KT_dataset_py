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
data = pd.read_csv("../input/train.csv")
data['species'].describe()
data.columns
labels = data['species'].unique()

labels.sort()

labels
for d in data.columns[2:]:

    data[d] = (data[d] - data[d].mean()) / data[d].std()
data = data.drop('id', axis=1)

data
temp = np.zeros(len(data['species']))

for i in range(len(data['species'])):

    temp[i] = np.where(labels == data['species'][i])[0]



temp
data['species'] = temp

data['species']
data['species'] = data['species'].astype(int)

data['species']
X = data.drop((['species','margin8', 'margin16', 'margin34', 'margin61']), axis=1)

Y = data['species']

feature_names = X.columns
feature_names
from sklearn.cross_validation import train_test_split
from sklearn import ensemble

rf = ensemble.RandomForestClassifier(n_estimators=1000, random_state=11)

rf.fit(X, Y)
err_train = np.mean(Y != rf.predict(X))

print(err_train)
a = X[:1]

print(rf.predict_proba(a))
test = pd.read_csv("../input/test.csv")

id_col = test['id']

test = test.drop(['id','margin8', 'margin16', 'margin34', 'margin61'], axis=1)
for t in test.columns:

    test[t] = (test[t] - test[t].mean()) / test[t].std()
test.head()
corr_df = test.corr() > 0.7

corr_df
np_corr = corr_df.as_matrix()

np.fill_diagonal(np_corr,False)
k = np.where(np_corr == True)

k = np.vstack(k)

k = np.stack((k[0],k[1]), axis = -1)

k[20:50]
err_test = rf.predict_proba(test)
temp = np.zeros(err_test.shape[0])

temp.shape
result = pd.DataFrame(err_test, columns=labels)

result = pd.concat((id_col,result), axis=1)

result
result.to_csv('result.csv', index=False)
important = rf.feature_importances_

#important.sort()
for i in zip(test.columns, important):

    if i[1] < 0.002 :

        print(i)