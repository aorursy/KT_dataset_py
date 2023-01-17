import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sorted(train['label'].unique())
train['label'].value_counts()
sns.countplot(train['label'])
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (100, ), random_state=1 ) 
y_train = train['label'].values
X_train = train.drop(columns = 'label', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)
%time MLPmodel = clf.fit(X_train/255.0, y_train)
MLPmodel.fit(X_train/255.0, y_train)
predicted = MLPmodel.predict(X_test/255.0)
print(classification_report(y_test, predicted)) # hidden_layer_sizes = Default, 100
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (1000, ), random_state=1 ) 
%time MLPmodel = clf.fit(X_train/255.0, y_train)
MLPmodel.fit(X_train/255.0, y_train)
predicted = MLPmodel.predict(X_test/255.0)
print(classification_report(y_test, predicted)) # hidden_layer_sizes = Default, 1000
train['label'].value_counts()
from sklearn.utils import shuffle
label_7 = shuffle(train[train['label'] == 7])[0:283]

label_3 = shuffle(train[train['label'] == 3])[0:333]

label_9 = shuffle(train[train['label'] == 9])[0:496]

label_2 = shuffle(train[train['label'] == 2])[0:507]

label_6 = shuffle(train[train['label'] == 6])[0:547]

label_0 = shuffle(train[train['label'] == 0])[0:552]

label_4 = shuffle(train[train['label'] == 4])[0:612]

label_8 = shuffle(train[train['label'] == 8])[0:621]

label_5 = shuffle(train[train['label'] == 5])[0:889]
train = train.append([label_7,label_3,label_9,label_2,label_6,label_0,label_4,label_8,label_5])
y_train = train['label'].values

X_train = train.drop(columns = 'label', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes = (1000, ), random_state=1 ) 
%time MLPmodel = clf.fit(X_train/255.0, y_train)
MLPmodel.fit(X_train/255.0, y_train)
predicted = MLPmodel.predict(X_test/255.0)
print(classification_report(y_test, predicted)) # hidden_layer_sizes = Default, 1000, UpSampling
test_predict = MLPmodel.predict(test/255.0)
submission = sample_submission
submission['Label'] = test_predict
submission.to_csv('MLP_submission.csv', index = False)