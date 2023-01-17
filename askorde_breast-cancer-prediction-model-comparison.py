%matplotlib inline 

import matplotlib.pyplot as plt # side-stepping mpl backend

import matplotlib.gridspec as gridspec # subplots

#import mpld3 as mpl



import numpy as np

from sklearn import preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn import svm, neighbors, naive_bayes

from sklearn.model_selection import KFold, train_test_split, cross_val_score

import pandas as pd



from time import process_time as time
df = pd.read_csv("../input/data.csv",header = 0)

df.head()
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)

len(df)
df.describe()

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df.head()
features_mean=list(df.columns[1:31])

dfM=df[df['diagnosis'] ==1]

dfB=df[df['diagnosis'] ==0]
plt.rcParams.update({'font.size': 8})

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50

    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.tight_layout()

plt.show()
traindf, testdf = train_test_split(df, test_size = 0.3)
def classification_model(model, data, predictors, outcome):

  

  scores = cross_val_score(model, data[predictors],data[outcome],cv = 5, scoring = 'accuracy')

  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

  

  t = time()  

  model.fit(data[predictors],data[outcome])

  print("Time for learning:", time()-t)  

  

  t = time()   

  predictions = model.predict(data[predictors])

  print("Time for prediction:", time()-t)

  print("Confusion Matrix \n",metrics.confusion_matrix(data[outcome],predictions))
clfs = [LogisticRegression(), 

        neighbors.KNeighborsClassifier(algorithm='ball_tree', weights='uniform', leaf_size=22, n_neighbors=14), 

        svm.SVC(C=1, kernel = 'linear'),

        naive_bayes.GaussianNB()]
for clf in clfs:

    #predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

    features = list(df.columns[1:31])

    outcome_var='diagnosis'

    print(clf)

    classification_model(clf,traindf,features,outcome_var)

    print('\n')

    

    #classification_model(clf,traindf,predictor_var,outcome_var)

    print('\n\n')
features = list(df.columns[1:31])

outcome='diagnosis'

model= svm.SVC(C=1, kernel = 'linear')



t = time()

model.fit(traindf[features],traindf[outcome])

print("Time for learning:", time()-t)



t = time()

predictions = model.predict(testdf[features])

print("Time for learning:", time()-t)



accuracy = metrics.accuracy_score(predictions,testdf[outcome])

print("Accuracy of prediction on test data = ", accuracy)

print("Confusion Matrix \n",metrics.confusion_matrix(testdf[outcome],predictions))