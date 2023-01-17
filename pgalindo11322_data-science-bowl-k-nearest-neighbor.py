import pandas as pd

import numpy as np

import collections

from datetime import datetime

from datetime import timedelta



from tqdm import tqdm

tqdm.pandas()





from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt



from sklearn import linear_model

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import zero_one_loss

from sklearn.ensemble import RandomForestClassifier





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv",sep=',',decimal='.')

test=pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv",sep=',',decimal='.')

train_labels=pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv",sep=',',decimal='.')

specs=pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv",sep=',',decimal='.')

sample_submission=pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv",sep=',',decimal='.')
print('Evaluaciones: ', train_labels['title'].unique())

print('Total de Evaluaciones: ', len(train_labels['title'].unique()))
assessment=train_labels['title'].value_counts()

fig = plt.figure()

ax = assessment.plot(kind='barh',grid=False, color='blue')

plt.show()
train=train.drop(['timestamp','event_data'],axis=1)

test=test.drop(['timestamp','event_data'],axis=1)
train=train[train.installation_id.isin(train_labels.installation_id.unique())]

test_assess = test[test.type == 'Assessment'].copy()

test_labels = sample_submission.copy()

test_labels['title'] = test_labels['installation_id'].progress_apply(lambda install_id: test_assess[test_assess.installation_id == install_id].iloc[-1].title)
train=train.drop(['event_id','event_code'],axis=1)

test=test.drop(['event_id','event_code'],axis=1)



train_2=(pd.get_dummies(train.drop(columns=['game_session', 'event_count', 'game_time']),

            columns=['title', 'type', 'world']).groupby(['installation_id']).sum())



test_2=(pd.get_dummies(test.drop(columns=['game_session', 'event_count', 'game_time']),

            columns=['title', 'type', 'world']).groupby(['installation_id']).sum())



train_3=(train[['installation_id', 'event_count', 'game_time']].groupby(['installation_id'])

        .agg([np.sum, np.mean, np.std, np.min, np.max]))

            

test_3=(test[['installation_id', 'event_count', 'game_time']].groupby(['installation_id'])

        .agg([np.sum, np.mean, np.std, np.min, np.max]))
def parameters(group1, col):

    return group1[['installation_id', col, 'event_count', 'game_time']

                 ].groupby(['installation_id', col]).agg([np.mean, np.sum, np.std]).reset_index().pivot(

        columns=col,index='installation_id')





world_time_stats_train = parameters(train, 'world')

type_time_stats_train = parameters(train, 'type')

world_time_stats_test = parameters(test, 'world')

type_time_stats_test = parameters(test, 'type')
new_train=train_2.join(train_3).join(world_time_stats_train).join(type_time_stats_train).fillna(0)

new_test=test_2.join(test_3).join(world_time_stats_test).join(type_time_stats_test).fillna(0)
titles = train_labels.title.unique()

title2mode = {}



for title in titles:

    mode = train_labels[train_labels.title == title].accuracy_group.value_counts().index[0]

    title2mode[title] = mode



train_labels['title_mode'] = train_labels.title.apply(lambda title: title2mode[title])

test_labels['title_mode'] = test_labels.title.apply(lambda title: title2mode[title])
final_train = pd.get_dummies((train_labels.set_index('installation_id')

        .drop(columns=['num_correct', 'num_incorrect', 'accuracy', 'game_session'])

        .join(new_train)),columns=['title'])







final_train = final_train.reset_index().groupby('installation_id').apply(lambda x: x.iloc[-1])

final_train = final_train.drop(columns='installation_id')



print('Dimensión train_labels:', final_train.shape)



final_test = pd.get_dummies(test_labels.set_index('installation_id').join(new_test), columns=['title'])



print('Dimensión test_labels:',final_test.shape)
X = final_train.drop(columns='accuracy_group').values

y = final_train['accuracy_group'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



print('Dimensiones x_train: ',X_train.shape)

print('Dimensiones x_test: ',X_test.shape)

print('Dimensiones y_train: ',y_train.shape)

print('Dimensiones y_test: ',y_test.shape)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import matplotlib.patches as mpatches

import seaborn as sb
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


n_neighbors = 7



knn = KNeighborsClassifier(n_neighbors)

knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'

     .format(knn.score(X_train, y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'

     .format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
k_range = range(1, 20)

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    scores.append(knn.score(X_test, y_test))

plt.figure()

plt.xlabel('k')

plt.ylabel('accuracy')

plt.scatter(k_range, scores)

plt.xticks([0,5,10,15,20])
results=pd.DataFrame()

results['k']=list(range(1,20))

results['Scores']=scores
max_value_score=results['Scores'].max()

print('Max. Accuracy: ',max_value_score)

k_max=(results['k'][results['Scores']==max_value_score]).tolist()[0]

print('k value with max. accuracy: ', k_max)
n_neighbors = k_max



knn = KNeighborsClassifier(n_neighbors)

knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'

     .format(knn.score(X_train, y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'

     .format(knn.score(X_test, y_test)))
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
X2=final_train.drop(columns='accuracy_group').index

y2=final_train['accuracy_group'].index

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=1)
sub=pd.DataFrame()

sub['installation_id']=X_test2

sub['accuracy_group']=pred

sub.to_csv('submission.csv', index=False)