# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

songs = pd.read_csv("../input/data.csv")
# Drop the numberings, song_title and artist(there are 1343 unique artists out of 2017 songs, the size of the dataset is not large enough to analyze this user's preferences to different artists)
songs=songs.drop(columns=['Unnamed: 0', 'song_title', 'artist']).reset_index(drop=True)
from pandas.plotting import scatter_matrix
scatter_matrix(songs, alpha=0.2, figsize=(20,20), diagonal='kde')
songs.corr(method='pearson', min_periods=1)

# From the plot and the correlations we can see that the parameter "energy" is highly correlated with both "acousticness" and "loudness"
# Thus the latter two parameters should be removed in order to avoid the problem of overfitting
songs=songs.drop(columns=['acousticness','loudness'])

# The parameter "key", "mode" are presented as integer, however, they should be defined as unordered categories.
songs['key'] = songs['key'].astype('category')
songs['mode'] = songs['mode'].astype('category')
songs_tree = songs.copy()
# Method1 ANN
# we should encode the categorical variable as a series of dummy variables.
songs = pd.get_dummies(songs)

# split the dataset into train and test
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = songs.drop(columns=['target'])
Y = songs['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=2017)
colname=X_train.columns

# standardization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)
X_train.columns = colname
X_test.columns = colname
# use 5-fold cv to decide the number of hidden units in each layer (Let set alpha to be 0.06, in order to reduce the number of models we fit to the training dataset)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

kf = KFold(n_splits=5)
scores = pd.DataFrame(0, index=np.arange(5), columns=['(12,8)','(12,7)','(12,6)','(12,5)','(12,4)'])
fold=0
for train_index, test_index in kf.split(X_train):
    train = X_train.iloc[train_index]
    train_target = Y_train[train_index]
    test = X_train.iloc[test_index]
    test_target = Y_train[test_index]
    
    #Fit the models
    clf0 = MLPClassifier(hidden_layer_sizes=(12,8),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
    clf0.fit(train,train_target)
    scores.iloc[fold,0]=clf0.score(test,test_target)
    
    clf1 = MLPClassifier(hidden_layer_sizes=(12,7),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
    clf1.fit(train,train_target)
    scores.iloc[fold,1]=clf1.score(test,test_target)
    
    clf2 = MLPClassifier(hidden_layer_sizes=(12,6),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
    clf2.fit(train,train_target)
    scores.iloc[fold,2]=clf2.score(test,test_target)
    
    clf3 = MLPClassifier(hidden_layer_sizes=(12,5),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
    clf3.fit(train,train_target)
    scores.iloc[fold,3]=clf3.score(test,test_target)
    
    clf4 = MLPClassifier(hidden_layer_sizes=(12,4),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
    clf4.fit(train,train_target)
    scores.iloc[fold,4]=clf4.score(test,test_target)
    
    fold=fold+1

print(scores.mean(axis=0))

# From the scores we can see that the neural network with 2 layers, with 12 and 4 hidden units produce the highest prediction accuracy
# Now fit the model to the entire training dataset
clf = MLPClassifier(hidden_layer_sizes=(12,4),activation='logistic', solver='adam', alpha=0.06, max_iter=400)
clf.fit(X_train, Y_train)
predictions=clf.predict(X_test)
score = clf.score(X_test,Y_test)
print("The prediction accuracy is:", score)
print(confusion_matrix(Y_test,predictions))

#The prediction accuracy is relatively low, which might be due to the use of inappropriate number of hidden layers, hidden units and alpha, since only a few models were trained in order to save computational effort.
#In addition, all the models that were fitted to our dataset are fully connected, ANN with other structure might be able to produce more accurate predictions.
# Method 2 Random Forest
from sklearn.ensemble import RandomForestClassifier

# split the dataset into train and test
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X = songs_tree.drop(columns=['target'])
Y = songs_tree['target']
X_tree_train, X_tree_test, Y_tree_train, Y_tree_test = train_test_split(X,Y,test_size=0.3,random_state=2017)

clf_tree = RandomForestClassifier(n_estimators=50,criterion='gini',bootstrap=True)
clf_tree.fit(X_tree_train,Y_tree_train)
predictions_tree = clf_tree.predict(X_tree_test)

importances=pd.DataFrame(clf_tree.feature_importances_,index=X.columns,columns=['importances']).sort_values(by=['importances'],ascending=False)
print(importances)
print(confusion_matrix(Y_tree_test,predictions_tree))
print("The prediction accuracy is:", clf_tree.score(X_tree_test,Y_tree_test))
