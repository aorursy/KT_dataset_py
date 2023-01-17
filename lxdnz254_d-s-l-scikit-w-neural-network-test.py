# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read csv (comma separated value) into data

train = pd.read_csv('../input/train.csv', header=None)

trainLabel = pd.read_csv('../input/trainLabels.csv', header=None)

test = pd.read_csv('../input/test.csv', header=None)

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
print('train shape:', train.shape)

print('test shape:', test.shape)

print('trainLabel shape:', trainLabel.shape)

train.head()
train.info()
train.describe()
# KNN with cross-validation

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, train_test_split



X, y = train, np.ravel(trainLabel)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Model complexity

neig = np.arange(1, 25)

kfold = 10

train_accuracy = []

val_accuracy = []

bestKnn = None

bestAcc = 0.0

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(X_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(X_train, y_train))

    # test accuracy

    val_accuracy.append(np.mean(cross_val_score(knn, X, y, cv=kfold)))

    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestAcc:

        bestAcc = np.mean(cross_val_score(knn, X, y, cv=10))

        bestKnn = knn



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, val_accuracy, label = 'Validation Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('k value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.show()



print('Best Accuracy without feature scaling:', bestAcc)

print(bestKnn)
# predict test

test_fill = np.nan_to_num(test)

submission = pd.DataFrame(bestKnn.predict(test_fill))

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
submission.to_csv('submission_no_normalization.csv', index=False)
print(check_output(["ls", "../working"]).decode("utf8"))
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer



std = StandardScaler()

X_std = std.fit_transform(X)

mms = MinMaxScaler()

X_mms = mms.fit_transform(X)

norm = Normalizer()

X_norm = norm.fit_transform(X)
# Model complexity

neig = np.arange(1, 30)

kfold = 10

val_accuracy = {'std':[], 'mms':[], 'norm':[]}

bestKnn = None

bestAcc = 0.0

bestScaling = None

# Loop over different values of k

for i, k in enumerate(neig):

    knn = KNeighborsClassifier(n_neighbors=k)

    # validation accuracy

    s1 = np.mean(cross_val_score(knn, X_std, y, cv=kfold))

    val_accuracy['std'].append(s1)

    s2 = np.mean(cross_val_score(knn, X_mms, y, cv=kfold))

    val_accuracy['mms'].append(s2)

    s3 = np.mean(cross_val_score(knn, X_norm, y, cv=kfold))

    val_accuracy['norm'].append(s3)

    if s1 > bestAcc:

        bestAcc = s1

        bestKnn = knn

        bestScaling = 'std'

    elif s2 > bestAcc:

        bestAcc = s2

        bestKnn = knn

        bestScaling = 'mms'

    elif s3 > bestAcc:

        bestAcc = s3

        bestKnn = knn

        bestScaling = 'norm'



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, val_accuracy['std'], label = 'CV Accuracy with std')

plt.plot(neig, val_accuracy['mms'], label = 'CV Accuracy with mms')

plt.plot(neig, val_accuracy['norm'], label = 'CV Accuracy with norm')

plt.legend()

plt.title('k value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.show()



print('Best Accuracy with feature scaling:', bestAcc)

print('Best kNN classifier:', bestKnn)

print('Best scaling:', bestScaling)
# predict on test

bestKnn.fit(X_norm, y)

submission = pd.DataFrame(bestKnn.predict(norm.transform(test_fill)))

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
submission.to_csv('submission_with_scaling.csv', index=False)
print(check_output(["ls", "../working"]).decode("utf8"))
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(pd.DataFrame(X_std).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



# split data train 70 % and val 30 %

X_train, X_val, y_train, y_val = train_test_split(X_std, y, test_size=0.3, random_state=42)



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(X_train,y_train)



ac = accuracy_score(y_val,clf_rf.predict(X_val))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_val,clf_rf.predict(X_val))

sns.heatmap(cm,annot=True,fmt="d")
from sklearn.svm import SVC

from sklearn.feature_selection import RFECV



kfold = 10

bestSVC = None

bestAcc = 0.0

val_accuracy = []

cv_range = np.arange(5, 11)

n_feature = []

for cv in cv_range:

    # Create the RFE object and compute a cross-validated score.

    svc = SVC(kernel="linear")

    # The "accuracy" scoring is proportional to the number of correct

    # classifications

    rfecv = RFECV(estimator=svc, step=1, cv=cv, scoring='accuracy')

    rfecv.fit(X_std, y)



    # print("Optimal number of features : %d" % rfecv.n_features_)

    # print('Best features :', pd.DataFrame(X_train).columns[rfecv.support_])



    # Model complexity

    val_accuracy += [np.mean(cross_val_score(svc, X_std[:, rfecv.support_], y, cv=kfold))]

    n_feature.append(rfecv.n_features_)

    if val_accuracy[-1] > bestAcc:

        bestAcc = val_accuracy[-1]



# Plot

plt.figure(figsize=[13,8])

plt.plot(cv_range, val_accuracy, label = 'CV Accuracy')

for i in range(len(cv_range)):

    plt.annotate(str(n_feature[i]), xy=(cv_range[i],val_accuracy[i]))

plt.legend()

plt.title('Cross Validation Accuracy')

plt.xlabel('k fold')

plt.ylabel('Accuracy')

plt.show()



print('Best Accuracy with feature scaling and RFECV:', bestAcc)
import numpy as np

import pandas as pd



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.ensemble import VotingClassifier



#### READING OUR GIVEN DATA INTO PANDAS DATAFRAME ####

x_train = train

y_train = trainLabel

x_test = test

x_train = np.asarray(x_train)

y_train = np.asarray(y_train)

x_test = np.asarray(x_test)

y_train = y_train.ravel()

print('training_x Shape:',x_train.shape,',training_y Shape:',y_train.shape, ',testing_x Shape:',x_test.shape)



#Checking the models

x_all = np.r_[x_train,x_test]

print('x_all shape :',x_all.shape)



#### USING THE GAUSSIAN MIXTURE MODEL ####

from sklearn.mixture import GaussianMixture

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        # Fit a mixture of Gaussians with EM

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(x_all)

        bic.append(gmm.aic(x_all))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(x_all)

x_train = best_gmm.predict_proba(x_train)

x_test = best_gmm.predict_proba(x_test)





#### TAKING ONLY TWO MODELS FOR KEEPING IT SIMPLE ####

knn = KNeighborsClassifier()

rf = RandomForestClassifier()



param_grid = dict( )

#### GRID SEARCH for BEST TUNING PARAMETERS FOR KNN #####

grid_search_knn = GridSearchCV(knn,param_grid=param_grid,cv=10,scoring='accuracy').fit(x_train,y_train)

print('best estimator KNN:',grid_search_knn.best_estimator_,'Best Score', grid_search_knn.best_estimator_.score(x_train,y_train))

knn_best = grid_search_knn.best_estimator_



#### GRID SEARCH for BEST TUNING PARAMETERS FOR RandomForest #####

grid_search_rf = GridSearchCV(rf, param_grid=dict( ), verbose=3,scoring='accuracy',cv=10).fit(x_train,y_train)

print('best estimator RandomForest:',grid_search_rf.best_estimator_,'Best Score', grid_search_rf.best_estimator_.score(x_train,y_train))

rf_best = grid_search_rf.best_estimator_





knn_best.fit(x_train,y_train)

print(knn_best.predict(x_test)[0:10])

rf_best.fit(x_train,y_train)

print(rf_best.predict(x_test)[0:10])



### IN CASE WE WERE USING MORE THAN ONE CLASSIFIERS THEN VOTING CLASSIFIER CAN BE USEFUL ###

clf = VotingClassifier(

        estimators=[('knn_best',knn_best),('rf_best',rf_best)],

        weights=[871856020222,0.907895269918]

    )

clf.fit(x_train,y_train)

print(clf.predict(x_test)[0:10])



#### SCORING THE MODELS ####

print('Score for KNN :',cross_val_score(knn_best,x_train,y_train,cv=10,scoring='accuracy').mean())

print('Score for Random Forest :',cross_val_score(rf_best,x_train,y_train,cv=10,scoring='accuracy').max())

print('Score for Voting Classifier :', cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy').max())





##### FRAMING OUR SOLUTION #####

knn_best_pred = pd.DataFrame(knn_best.predict(x_test))

rf_best_pred = pd.DataFrame(rf_best.predict(x_test))

voting_clf_pred = pd.DataFrame(clf.predict(x_test))



knn_best_pred.index += 1

rf_best_pred.index += 1

voting_clf_pred.index += 1



rf_best_pred.columns = ['Solution']

rf_best_pred['Id'] = np.arange(1,rf_best_pred.shape[0]+1)

rf_best_pred = rf_best_pred[['Id', 'Solution']]

#print(rf_best_pred)



voting_clf_pred.columns = ['Solution']

voting_clf_pred['Id'] = np.arange(1,voting_clf_pred.shape[0]+1)

voting_clf_pred = voting_clf_pred[['Id', 'Solution']]

print(voting_clf_pred)



#knn_best_pred.to_csv('knn_best_pred.csv')

#rf_best_pred.to_csv('Submission_rf.csv', index=False)

voting_clf_pred.to_csv('voting_clf_pred.csv', index=False)
# Attempt to implement a neural network to solve this solution

from keras.models import Sequential

from keras.layers import Dense, Activation

## For reproducibility

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

# Training split

X, y = train, np.ravel(trainLabel)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# create model

model = Sequential()

model.add(Dense(units = 80, kernel_initializer = 'uniform', input_dim=X_train.shape[1], activation='relu'))  # input layer

# Adding extra hidden layers

model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

model.add(Dense(1, activation='sigmoid'))  # output layer



# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])





# train model

model.fit(X_train, y_train, epochs=50, batch_size=32)



# evaluate the model

scores = model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions

predictions = model.predict(test)

# round predictions

rounded = [round(x[0]) for x in predictions]
# create output file

solution = np.vstack((test.index.astype('int')+1,

                      np.array(rounded).astype('int')))

solution = np.vstack((['Id', 'Solution'],

                       solution.transpose()))

solution = pd.DataFrame(solution)

print(solution[0:10])

solution.to_csv("Neural_Network_Solution.csv", index=False, header=False)
# Get dataframes to match

voting_clf_pred = pd.DataFrame(voting_clf_pred)

rf_best_pred = pd.DataFrame(rf_best_pred)

# readjusting solution DataFrame to match

solution.columns = ['Id', 'Solution']

solution = solution.iloc[1:,]
# Get solution columns

vSol = pd.Series(voting_clf_pred['Solution'])

rSol = pd.Series(rf_best_pred['Solution'])

sSol = pd.Series(solution['Solution']).astype('int')

# Compare the columns

vs = (vSol == sSol )

rs = (rSol == sSol)

vr = (vSol == rSol) 

# Output the results

print("Neural Network Score vs Voting Classifier : %.2f%%" % (np.count_nonzero(vs)/voting_clf_pred.shape[0]*100))

print("Neural Network Score vs Random Forest Classifier : %.2f%%" % (np.count_nonzero(rs)/rf_best_pred.shape[0]*100))

print("Voting Classifier vs Random Forest Classifier : %.2f%%" % (np.count_nonzero(vr)/rf_best_pred.shape[0]*100))