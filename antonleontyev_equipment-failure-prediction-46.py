import pandas as pd

import os 

import numpy as np
arr = os.listdir()

print(arr)
df_train = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')

df_test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
df_train
list(df_train)
df_train.dtypes
df_train1 = df_train.apply(pd.to_numeric, errors='coerce')

df_train1.dtypes
df_train1.fillna(0) # we can consider no reading as reading with value 0
sensor7 = ['sensor7_histogram_bin0',

 'sensor7_histogram_bin1',

 'sensor7_histogram_bin2',

 'sensor7_histogram_bin3',

 'sensor7_histogram_bin4',

 'sensor7_histogram_bin5',

 'sensor7_histogram_bin6',

 'sensor7_histogram_bin7',

 'sensor7_histogram_bin8',

 'sensor7_histogram_bin9']

sensor24 = ['sensor24_histogram_bin0',

 'sensor24_histogram_bin1',

 'sensor24_histogram_bin2',

 'sensor24_histogram_bin3',

 'sensor24_histogram_bin4',

 'sensor24_histogram_bin5',

 'sensor24_histogram_bin6',

 'sensor24_histogram_bin7',

 'sensor24_histogram_bin8',

 'sensor24_histogram_bin9']

sensor25 = ['sensor25_histogram_bin0',

 'sensor25_histogram_bin1',

 'sensor25_histogram_bin2',

 'sensor25_histogram_bin3',

 'sensor25_histogram_bin4',

 'sensor25_histogram_bin5',

 'sensor25_histogram_bin6',

 'sensor25_histogram_bin7',

 'sensor25_histogram_bin8',

 'sensor25_histogram_bin9']

sensor26 = ['sensor26_histogram_bin0',

 'sensor26_histogram_bin1',

 'sensor26_histogram_bin2',

 'sensor26_histogram_bin3',

 'sensor26_histogram_bin4',

 'sensor26_histogram_bin5',

 'sensor26_histogram_bin6',

 'sensor26_histogram_bin7',

 'sensor26_histogram_bin8',

 'sensor26_histogram_bin9']

sensor64 = ['sensor64_histogram_bin0',

 'sensor64_histogram_bin1',

 'sensor64_histogram_bin2',

 'sensor64_histogram_bin3',

 'sensor64_histogram_bin4',

 'sensor64_histogram_bin5',

 'sensor64_histogram_bin6',

 'sensor64_histogram_bin7',

 'sensor64_histogram_bin8',

 'sensor64_histogram_bin9']

sensor69 = ['sensor69_histogram_bin0',

 'sensor69_histogram_bin1',

 'sensor69_histogram_bin2',

 'sensor69_histogram_bin3',

 'sensor69_histogram_bin4',

 'sensor69_histogram_bin5',

 'sensor69_histogram_bin6',

 'sensor69_histogram_bin7',

 'sensor69_histogram_bin8',

 'sensor69_histogram_bin9']

sensor105 = ['sensor105_histogram_bin0',

 'sensor105_histogram_bin1',

 'sensor105_histogram_bin2',

 'sensor105_histogram_bin3',

 'sensor105_histogram_bin4',

 'sensor105_histogram_bin5',

 'sensor105_histogram_bin6',

 'sensor105_histogram_bin7',

 'sensor105_histogram_bin8',

 'sensor105_histogram_bin9']

df_train1['sensor7_average'] = df_train1[sensor7].mean(axis=1)

df_train1['sensor24_average'] = df_train1[sensor24].mean(axis=1)

df_train1['sensor25_average'] = df_train1[sensor25].mean(axis=1)

df_train1['sensor26_average'] = df_train1[sensor26].mean(axis=1)

df_train1['sensor64_average'] = df_train1[sensor64].mean(axis=1)

df_train1['sensor69_average'] = df_train1[sensor69].mean(axis=1)

df_train1['sensor105_average'] = df_train1[sensor105].mean(axis=1)

not_hists = [col for col in df_train1.columns if not  'histogram' in col]

df_train_2 = df_train1[not_hists]
df_train_2['target'].astype(bool).sum(axis=0)
df_majority = df_train_2[df_train_2.target==0]

df_minority = df_train_2[df_train_2.target==1]

from sklearn.utils import resample

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=59000)    # to match majority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df_upsampled.target.value_counts()
df_upsampled = df_upsampled.fillna(0)

y = df_upsampled.target

X = df_upsampled.drop(['target','id'], axis=1)
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

#from sklearn.svm import SVC
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

#models.append(('SVM', SVC()))

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10)

	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
import matplotlib.pyplot as plt

%matplotlib inline

fig = plt.figure()

fig.suptitle('Comparison of models')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#split dataset into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()

fitted_model = model.fit(X_train, y_train)

y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



print("Accuracy:",accuracy_score(y_test, y_predict))

print("F-1 score:", f1_score(y_test, y_predict))
import collections



print(collections.Counter(y_predict))

from sklearn.metrics import confusion_matrix



pd.DataFrame(

    confusion_matrix(y_test, y_predict),

    columns=['Predicted Surface Failure', 'Predicted Underground failure'],

    index=['True Surface failure', 'True Underground failure']

)
import pickle

pkl_filename = "pickled_decision_tree.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)
df_test2 = df_test



df_test2 = df_test2.apply(pd.to_numeric, errors='coerce')

df_test2.dtypes
sensor7 = ['sensor7_histogram_bin0',

 'sensor7_histogram_bin1',

 'sensor7_histogram_bin2',

 'sensor7_histogram_bin3',

 'sensor7_histogram_bin4',

 'sensor7_histogram_bin5',

 'sensor7_histogram_bin6',

 'sensor7_histogram_bin7',

 'sensor7_histogram_bin8',

 'sensor7_histogram_bin9']

sensor24 = ['sensor24_histogram_bin0',

 'sensor24_histogram_bin1',

 'sensor24_histogram_bin2',

 'sensor24_histogram_bin3',

 'sensor24_histogram_bin4',

 'sensor24_histogram_bin5',

 'sensor24_histogram_bin6',

 'sensor24_histogram_bin7',

 'sensor24_histogram_bin8',

 'sensor24_histogram_bin9']

sensor25 = ['sensor25_histogram_bin0',

 'sensor25_histogram_bin1',

 'sensor25_histogram_bin2',

 'sensor25_histogram_bin3',

 'sensor25_histogram_bin4',

 'sensor25_histogram_bin5',

 'sensor25_histogram_bin6',

 'sensor25_histogram_bin7',

 'sensor25_histogram_bin8',

 'sensor25_histogram_bin9']

sensor26 = ['sensor26_histogram_bin0',

 'sensor26_histogram_bin1',

 'sensor26_histogram_bin2',

 'sensor26_histogram_bin3',

 'sensor26_histogram_bin4',

 'sensor26_histogram_bin5',

 'sensor26_histogram_bin6',

 'sensor26_histogram_bin7',

 'sensor26_histogram_bin8',

 'sensor26_histogram_bin9']

sensor64 = ['sensor64_histogram_bin0',

 'sensor64_histogram_bin1',

 'sensor64_histogram_bin2',

 'sensor64_histogram_bin3',

 'sensor64_histogram_bin4',

 'sensor64_histogram_bin5',

 'sensor64_histogram_bin6',

 'sensor64_histogram_bin7',

 'sensor64_histogram_bin8',

 'sensor64_histogram_bin9']

sensor69 = ['sensor69_histogram_bin0',

 'sensor69_histogram_bin1',

 'sensor69_histogram_bin2',

 'sensor69_histogram_bin3',

 'sensor69_histogram_bin4',

 'sensor69_histogram_bin5',

 'sensor69_histogram_bin6',

 'sensor69_histogram_bin7',

 'sensor69_histogram_bin8',

 'sensor69_histogram_bin9']

sensor105 = ['sensor105_histogram_bin0',

 'sensor105_histogram_bin1',

 'sensor105_histogram_bin2',

 'sensor105_histogram_bin3',

 'sensor105_histogram_bin4',

 'sensor105_histogram_bin5',

 'sensor105_histogram_bin6',

 'sensor105_histogram_bin7',

 'sensor105_histogram_bin8',

 'sensor105_histogram_bin9']

df_test2['sensor7_average'] = df_test2[sensor7].mean(axis=1)

df_test2['sensor24_average'] = df_test2[sensor24].mean(axis=1)

df_test2['sensor25_average'] = df_test2[sensor25].mean(axis=1)

df_test2['sensor26_average'] = df_test2[sensor26].mean(axis=1)

df_test2['sensor64_average'] = df_test2[sensor64].mean(axis=1)

df_test2['sensor69_average'] = df_test2[sensor69].mean(axis=1)

df_test2['sensor105_average'] = df_test2[sensor105].mean(axis=1)

list(df_test2)
df_test3 = df_test2.fillna(0)

not_hists2 = [col for col in df_test3.columns if not  'histogram' in col]

df_test4 = df_test3[not_hists2]

df_test4
df_test5 = df_test4.drop(['id'], axis=1)

predictions = model.predict(df_test5)
import collections



collections.Counter(predictions)
#sanity check:

15709 + 292
predictions
res_series = pd.Series(predictions)

headers = ['target']

res_series.index += 1 

res_series.index.name = 'id'

res_series.to_csv('submission5a.csv', header = headers)