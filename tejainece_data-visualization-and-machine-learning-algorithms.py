import sqlite3

import numpy as np

from sklearn import preprocessing





connection = sqlite3.connect('../input/database.sqlite') #Connect to the database



X = [] #Training feature set

Y = [] #Training class

X_test = [] #Testing feature set

Y_true = [] #Testing class

for i, tables in enumerate(['Dinner_audio_features','Party_audio_features','Sleep_audio_features','Workout_audio_features']):

    cursor = connection.execute("SELECT *  from "+tables) #select the data from the database

    result = list(cursor)

    for row in result[:21]: #select the first 21 for Training the learning algorithm

        row = list(row[2:]) #remove the non-numerical data selected from the database

        X.append(row) #Add the feature set to the training data 

        Y.append(i+1) #Add the class of each of the feature set to the class list

    for row in result[21:]:#select the remaning feature set for testing the learning model

        row = list(row[2:])#remove the non-numerical data selected from the database

        X_test.append(row)#Add the feature set to the testing data 

        Y_true.append(i+1)#Add the class of each of the feature set to the true class list



scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)

Y = np.array(Y)
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



pca = PCA(n_components=2) #Create a PCA object that reduces the dimentions to 2

X_r = pca.fit(X).transform(X) #fit the data

lda = LinearDiscriminantAnalysis(n_components=2)#Create a LDA object that reduces the dimentions to 2

X_r2 = lda.fit(X, Y).transform(X) #fit the data



colors = ['navy', 'turquoise', 'darkorange', 'red']

lw = 2

target_names = ['Dinner','Party','Sleep','Workout']

for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):

    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA of Audio feature dataset')



plt.figure()

for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):

    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of Audio Feature dataset')
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC



svmclf = SVC(C=1.55, decision_function_shape='ovr')

svmclf.fit(X, Y)



MLclf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(7,49), random_state=1, warm_start=True)

for i in range(5):

    MLclf.fit(X, Y)



Knnclf = KNeighborsClassifier(n_neighbors=3)

Knnclf.fit(X, Y)



eclf = VotingClassifier(estimators=[('svm', svmclf), ('nn', MLclf), ('knn', Knnclf)], voting='hard')

eclf.fit(X, Y)



svm_Predict = svmclf.predict(X_test)

MLP_Predict = MLclf.predict(X_test)

Knn_Predict = Knnclf.predict(X_test)

voting_Predict = eclf.predict(X_test)



print("kNN accuracy = {}% ".format(accuracy_score(Y_true, Knn_Predict) * 100))

print("SVM accuracy = {}% ".format(accuracy_score(Y_true, svm_Predict) * 100))

print("Neural Network accuracy = {}% ".format(accuracy_score(Y_true, MLP_Predict) * 100))

print("Votingaccuracy = {}%".format(accuracy_score(Y_true, voting_Predict) * 100))
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes=["dinner", "party", "sleep", "workout"], title="",

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title('Confusion matrix ' +title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



svm_cm = confusion_matrix(Y_true, svm_Predict)

MLP_cm = confusion_matrix(Y_true, MLP_Predict)

Knn_cm = confusion_matrix(Y_true, Knn_Predict)

voting_cm = confusion_matrix(Y_true, voting_Predict)



plt.figure()

plot_confusion_matrix(svm_cm, title = 'SVM')

plt.figure()

plot_confusion_matrix(MLP_cm, title = 'MLP')

plt.figure()

plot_confusion_matrix(Knn_cm, title = 'Knn')

plt.figure()

plot_confusion_matrix(voting_cm, title = 'Voting')
X = [] #Training feature set

Y = [] #Training class

X_test = [] #Testing feature set

Y_true = [] #Testing class

for i, tables in enumerate(['Dinner_spotify_features','Party_spotify_features','Sleep_spotify_features','Workout_spotify_features']):

    cursor = connection.execute("SELECT \"acousticness\",\"danceability\",\"energy\",\"instrumentalness\",\"speechiness\",\"tempo\",\"valence\",\"loudness\"  from "+tables) #select the data from the database

    result = list(cursor)

    for row in result[:150]: #select the first 21 for Training the learning algorithm

        row = list(row[4:]) #remove the non-numerical data selected from the database

        X.append(row) #Add the feature set to the training data 

        Y.append(i+1) #Add the class of each of the feature set to the class list

    for row in result[151:230]:#select the remaning feature set for testing the learning model

        row = list(row[4:])#remove the non-numerical data selected from the database

        X_test.append(row)#Add the feature set to the testing data 

        Y_true.append(i+1)#Add the class of each of the feature set to the true class list



scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)

Y = np.array(Y)

Y_true = np.array(Y_true)

pca = PCA(n_components=2) #Create a PCA object that reduces the dimentions to 2

X_r = pca.fit(X).transform(X) #fit the data

lda = LinearDiscriminantAnalysis(n_components=2)#Create a LDA object that reduces the dimentions to 2

X_r2 = lda.fit(X, Y).transform(X) #fit the data



colors = ['navy', 'turquoise', 'darkorange', 'red']

lw = 2

target_names = ['Dinner','Party','Sleep','Workout']

for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):

    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA of Spotify feature dataset')



plt.figure()

for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):

    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of Spotify Feature dataset');
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



svmclf = SVC(C=2.015, gamma=0.005, decision_function_shape='ovo')

svmclf.fit(X, Y)



MLclf = MLPClassifier(activation='tanh',solver='adam', alpha=1e-5,

                    hidden_layer_sizes=(25,25,45,35,45, 25), random_state=1, warm_start=True)

for i in range(5):

    MLclf.fit(X, Y)



rfclf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_features='auto', random_state=1)

rfclf.fit(X, Y)



etclf = ExtraTreesClassifier(criterion='entropy', n_estimators=100, max_features='auto')

etclf.fit(X, Y)



eclf = VotingClassifier(estimators=[('svm', svmclf), ('rf', rfclf), ('et', etclf)], voting='hard')

eclf.fit(X, Y)



svm_Predict = svmclf.predict(X_test)

MLP_Predict = MLclf.predict(X_test)

rfYpred = rfclf.predict(X_test)

etYpred = etclf.predict(X_test)

voting_Predict = eclf.predict(X_test)



print("RFaccuracy = {}% ".format(accuracy_score(Y_true, rfYpred) * 100))

print("ETaccuracy = {}% ".format(accuracy_score(Y_true, etYpred) * 100))

print("SVM accuracy = {}% ".format(accuracy_score(Y_true, svm_Predict) * 100))

print("Neural Network accuracy = {}% ".format(accuracy_score(Y_true, MLP_Predict) * 100))

print("Votingaccuracy = {}%".format(accuracy_score(Y_true, voting_Predict) * 100))