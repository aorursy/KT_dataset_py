import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/drugsComTrain_raw.csv")
test = pd.read_csv("../input/drugsComTest_raw.csv")
train.head()
train.drugName.unique()
from keras.utils import np_utils
map_dict = dict()
for i, j in enumerate(train.drugName.unique()):
#     print(i,j)
    map_dict[j] = i
    
train = (train).applymap(lambda s: map_dict.get(s) if s in map_dict else s)

# print(map)
# train.corr()
map_dict = dict()
for i, j in enumerate(train.condition.unique()):
#     print(i,j)
    map_dict[j] = i
    
train = (train).applymap(lambda s: map_dict.get(s) if s in map_dict else s)

train.drop(["uniqueID", 'review'], axis = 1, inplace = True)
train.head()
train.head()
train.head()
from datetime import datetime



year = lambda x: datetime.strptime(x, "%d-%b-%y" ).year
train['year'] = train['date'].map(year)
test['year'] = test['date'].map(year)

month = lambda x: datetime.strptime(x, "%d-%b-%y" ).month
train['month'] = train['date'].map(year)
test['month'] = test['date'].map(year)

day = lambda x: datetime.strptime(x, "%d-%b-%y" ).weekday()
train['day'] = train['date'].map(day)
test['day'] = test['date'].map(day)

train.drop(["date"], inplace = True, axis = 1)
train.head()
import seaborn as sns


import matplotlib.pyplot as plt


corr = train.corr()
f, ax = plt.subplots(figsize=(20, 20 ))
cmap = sns.diverging_palette(225, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train)

print(scaler.mean_)

scaler.transform(train)
# train = pd.concat([train, pd.get_dummies(train.condition), pd.get_dummies(train.drugName)], axis=1, sort=False)
# train.drop(['drugName','condition'], inplace = True, axis = 1)
f = plt.figure(figsize=(25,10))
f.add_subplot(221)
sns.boxplot(x=train['drugName'])
f.add_subplot(222)
sns.boxplot(x=train['condition'])
f.add_subplot(223)
sns.boxplot(x=train['year'])
f.add_subplot(224)
sns.boxplot(x=train['month'])
# from scipy import stats
# import numpy as np

# z = np.abs(stats.zscore(train))
# train = train[(z <= 2.5)]
label = train['rating']
train.drop(["rating"], inplace = True, axis = 1)
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers import Dropout, Conv2D
# from keras import regularizers

# from keras.utils import np_utils

# model = Sequential()

# model.add(Dense(units=100, activation='relu', input_dim= 4325 ) )

# model.add(Dense(units=100, activation='relu'  ))
# model.add(Dropout(0.1))
# model.add(Dense(units=100, activation='relu'  ))
# model.add(Dropout(0.1))

# #and now the output layer which will have 10 units to
# #output a 1-hot vector to detect one of the 10 classes
# model.add(Dense(units=1, activation='relu'))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20, random_state=0).fit(train)
clusters = kmeans.labels_
train['clusters'] = clusters
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train ,label , test_size=0.2, random_state=42)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

#uncomment the commented code and uncomment the commented to perform gridsearchCV
from xgboost import XGBClassifier

clf = ExtraTreesClassifier(n_estimators=300, random_state=0, n_jobs = -1)

clf.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(clf.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(clf.score(x_test, y_test) * 100))
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(x_train)
cnf_matrix = confusion_matrix(y_train, y_pred)
y_train.unique()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_train.unique(), normalize=True,
title='Normalized confusion matrix')

plt.show()
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import np_utils
# from keras import callbacks
# from keras import optimizers

# earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')
# optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
# model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=1000, batch_size=200, validation_split=0.2, verbose= 1 , callbacks=[earlystopping])
# import matplotlib.pyplot as plt

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])