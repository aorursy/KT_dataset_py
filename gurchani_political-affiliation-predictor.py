import gc

gc.collect()



import pandas as pd

import numpy as np

 # Manually Annotated dataset from the above mentioned paper

df0 = pd.read_csv('../input/dataset-for-french/profilesmanualannotation.csv') 
import collections

df1 = df0[['UserId', 'party']] #Trimming down the first dataset

fr = pd.read_csv('../input/annotatedfriends/manualannotationFriends.csv', names=['id', 'friend']) #Dataset of Friends

fr.count
import matplotlib.pyplot as plt

GroupedData = fr[['friend', 'id']].groupby(['friend']).count().sort_values(['id'], ascending=False)

chunkSize = int(GroupedData.shape[0]/1000)

counter = 0

averageCounte = 0

Sum = 0

top = pd.DataFrame(columns=['Percentage', 'Number of Followers'])

for i in GroupedData['id']:

    counter = counter + 1

    Sum = Sum + i

    if (counter >= chunkSize):

        counter = 0

        averageCounte = averageCounte + 0.1

        average = Sum/chunkSize

        top = top.append({'Percentage':averageCounte, 'Number of Followers':average}, ignore_index=True)

        Sum = 0

        if average < 10:

            break

   
ax = top.plot(x='Percentage', y='Number of Followers', figsize = (10, 10))

ax.axhline(y=70, linewidth=0.5, color='r')

ax.set_xlabel('Percentage of Friends')
GroupedData.reset_index(inplace = True)

print(GroupedData.shape)

GroupedData = GroupedData.nlargest(int(GroupedData['friend'].shape[0]*0.0001), 'id', keep='first')

GroupedData.shape
print(GroupedData)
fr = fr[fr['friend'].isin(GroupedData['friend'])]
#Graph = nx.from_pandas_edgelist(fr, source = 'id', target = 'friend', edge_attr=None,  create_using=nx.DiGraph())

#print(Graph.size())

#nx.draw(Graph, node_size = 5)
#fr = fr.head(smallerTest)

DicList = []

for group, frame in fr.groupby('id'):

    ak = frame['friend'].tolist()

    dictOf = dict.fromkeys(ak , 1)

    DicList.append(dictOf)
from sklearn.feature_extraction import DictVectorizer

dictvectorizer = DictVectorizer(sparse = True)

features = dictvectorizer.fit_transform(DicList)

features.todense().shape
df1['party'].fillna(0, inplace = True)
parties = {'fi': 1,'ps': 1,'em': 2,'lr': 2,'fn': 2,'fi/ps': 4,'fi/em': 4, 'fi/lr': 4,'fi/fn': 5, 'ps/em': 6,

'ps/lr': 3, 'ps/fn': 4, 'em/lr': 4,'em/fn': 6, 'lr/fn': 6}

#print(df1['party'])



df1['party'] = df1['party'].map(parties)



#print(labels)

print(features.shape)
#dataFrame = pd.SparseDataFrame(features, columns = dictvectorizer.get_feature_names(), 

#                               index = fr['id'].unique())

dataFrame = pd.DataFrame.sparse.from_spmatrix(features, columns = dictvectorizer.get_feature_names(), 

                               index = fr['id'].unique())
df1 = df1.set_index('UserId')

df1.index
dataFrame.index.names = ['UserId']
#dataFrame.reset_index()

#df1
#df1['UserId'] = df1['UserId'].astype(int)

#dataFrame.reset_index()

#dataFrame

#print(dataFrame.index.type())



#df1.set_index('UserId')

#print(df1.index.type())

dataFrame = dataFrame.join(df1, how='inner')

print(dataFrame['party'])


dataFrame = dataFrame[(dataFrame['party']==1.0) | (dataFrame['party']==2.0)] 

                      #|(dataFrame['party']== 4.0) | (dataFrame['party']==5.0)]

#dataFrame
dataFrame['party'][dataFrame['party'] == 2].count()
from sklearn.ensemble import  RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

dataFrame.fillna(0, inplace = True)

featureSelector = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)

featureSelector.fit(dataFrame.iloc[:, :-1], dataFrame['party'])
model = SelectFromModel(featureSelector, prefit=True, max_features= 500)

X_new = model.transform(dataFrame.iloc[:, :-1].to_dense())

print(X_new.shape)

featureNames = []

for i in model.get_support(indices = True):

    featureNames.append(dataFrame.columns[i])

dataFrame2 = pd.DataFrame.from_dict(X_new)
#print(featureNames)

importancesOfAllFeatures = (sorted(zip(map(lambda x: round(x, 4), featureSelector.feature_importances_), dataFrame.columns), 

             reverse=True))

print(importancesOfAllFeatures[:10])
dataFrame2.index = dataFrame.index

dataFrame['party'] = dataFrame['party'].astype(float)

dataFrame2['party'] = dataFrame['party']

dataFrame2
#dataFrame['party'] = labels.head(dataFrame.shape[0]).values #This is the problem

from sklearn.model_selection import train_test_split

dataFrame2.fillna(0, inplace = True)

train, test = train_test_split(dataFrame2, test_size=0.1, shuffle = True)
test.shape
from sklearn.neural_network import MLPClassifier

#clf =  MLPClassifier(solver='lbfgs', alpha=50, hidden_layer_sizes=(100), random_state=1)

#from sklearn.neural_network import MLPClassifier

clf = MLPClassifier( solver='lbfgs', alpha=1e-5, random_state=1, max_iter=500)

clf.fit(train.iloc[:, :-1].to_dense(), train['party'].to_dense())

#clf.fit(train.iloc[:, :-1].to_dense(), train['party'].to_dense())

print(clf.score(test.iloc[:, :-1], test['party'].to_dense()))

print(clf.score(train.iloc[:, :-1].to_dense(), train['party'].to_dense()))
from scipy.stats import randint as sp_randint

from scipy.stats import expon

import scipy.stats

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

#"solver": ['lbfgs', 'sgd', 'adam'],

param_dist = { "solver": ['lbfgs', 'sgd', 'adam'],

               "hidden_layer_sizes": [(100), (100, 100), (50)],

              "activation": ['identity', 'logistic', 'tanh', 'relu'],

              "alpha": scipy.stats.expon(scale=.1),}



n_iter_search = 20

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=5, iid=False)

random_search.fit(train.iloc[:, :-1].to_dense(), train['party'].to_dense())
# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
report(random_search.cv_results_)
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0)

clf2.fit(train.iloc[:, :-1].to_dense(), train['party'].to_dense())
print(clf2.score(test.iloc[:, :-1], test['party'].to_dense() ))

print(clf2.score(train.iloc[:, :-1].to_dense(), train['party'].to_dense()))


#import sys

#np.set_printoptions(threshold=sys.maxsize)

#print(clf2.feature_importances_)
test
train.iloc[:, :-1]
from sklearn import linear_model

clf3 = linear_model.Lasso(alpha=.00001, max_iter=1500)

#clf3.fit(train.iloc[:, :-1], train['party'])

#print(clf3.score(train.iloc[:, :-1].to_dense(), train['party'].to_dense()))

#clf3.score(test.iloc[:, :-1], test['party'])
def get_score(model, X_train, Y_train, X_test, Y_test):

        model.fit(X_train, Y_train)

        return model.score(X_test, Y_test)

    
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

skf = StratifiedKFold(n_splits=4)

for i in [clf, clf2, clf3]:

    scores = cross_val_score(i, dataFrame.iloc[:, :-1].to_dense(), dataFrame['party'].to_dense(), cv=5)

    print(i)

    print(np.mean(scores))
print(train[train['party'] == None].size)

#from sklearn import linear_model

#clf3 = linear_model.Lasso(alpha=1)

#clf3.fit(train.iloc[:, :-1], train['party'].fillna(0))
from sklearn.metrics import confusion_matrix

y_pred = []

#print(np.transpose(test.iloc[i, :-1].reset_index().values.shape))



for i in range(0, test[0].count()):

 #'model' is the name of classifier from keras   

    y_pred.append(model.predict(np.transpose(test.iloc[i, :-1].reset_index().values)))

predicted_y = []

for i in y_pred:

    predicted_y.append(i[1])

print(predicted_y)

cm = confusion_matrix(test['party'].to_dense(), predicted_y)
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax


#print(len(predicted_y))

realValues = []

test['party'].fillna(0, inplace = True)

class_names = [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,15,16]

class_names = np.asarray(class_names)

#class_names = test['party'].unique()

#class_names = np.append(class_names , [0])

for i in test['party']:

    realValues.append(int(i))



#print(test.shape)

#print(test['party'])

#print(test['party'].tolist())

plot_confusion_matrix(realValues, predicted_y, classes=class_names,

                      title='Confusion matrix, without normalization')
import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()

model.add(Dense(100, activation='relu', input_dim=35220)) #input shape of 50

model.add(Dense(64, activation='relu')) #input shape of 50

model.add(Dense(28, activation='softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(train.iloc[:, :-1], train['party'], epochs=8)
#model.summary()

model.evaluate(test.iloc[:, :-1], test['party'])
from sklearn.dummy import DummyClassifier

for strategy in ['stratified', 'most_frequent', 'prior', 'uniform']:

    dummy = DummyClassifier(strategy=strategy)

    dummy.fit(train.iloc[:, :-1].to_dense(), train['party'].to_dense())

    print(dummy.score(test.iloc[:, :-1].to_dense(), test['party'].to_dense()))
fr['id'].nunique()

fr['friend'].count()



#fr['']