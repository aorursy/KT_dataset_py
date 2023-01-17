# This Python 3 environment comes with many helpful analytics libraries installed

#a)Importing the Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# b) Getting the dataset
bc =pd.read_table('../input/uci-breast-cancer-wisconsin-original/breast-cancer-wisconsin.data.txt', delimiter=',', names=('id number','clump_thickness','cell_size_uniformity','cell_chape_uniformity','marginal_adhesion','epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'))
bc.head(5)
bc.describe()
bc['bare_nuclei'].value_counts()
# c) FILLING THE MISSING VALUES
#we plan to replace the missing data denored by "?" with '-99999'. This is because most algorithm considers that as an 
#outlier. Instead, if we happen to drop all the rows containing missing vaules, we need to loose most of our data. 
bc.replace('?',-99999, inplace = True)
bc.dtypes
bc['bare_nuclei'] = pd.to_numeric(bc['bare_nuclei'])
bc.corr()['class']
#CREATING TRAIN-TEST SPLIT 
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.cross_validation import train_test_split
#Features
X = np.array(bc.drop(['class'], 1))
#Labels
y = np.array(bc['class'])
#Splitting the data set (X, y)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
print('X_train', X_train)
print('X_test', X_test)
print('y_train', y_train)
print('y_test', y_test)
#TRAIN THE CLASSIFIER (FIT THE ESTIMATOR) USING THE TRAINING DATA
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
#ESTIMATE THE ACCURACY OF THE CLASSIFIER ON FUTURE DATA, USING THE TEST DATA
accuracy = clf.score(X_test, y_test)
accuracy
sns.heatmap(bc.corr())
#We just drop the ID column so to build the model in a easy way. 
bc.drop(['id number'], 1, inplace = True)

#Features
X = np.array(bc.drop(['class'], 1))
#Labels
y = np.array(bc['class'])

#Splitting the data set (X, y)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
print('X_train', X_train)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

accuracy
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)
prediction = clf.predict(example_measures)
prediction
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=1)
#Using Gini Index Classification Metric
clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 1, max_depth =2, min_samples_leaf = 5)
clf_gini.fit(X_train, y_train)
predictions = clf_gini.predict(X_test)
predictions
print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_gini.predict(X_train), y_train))
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
degree = range(1,31)
train_scores, val_scores = validation_curve(DecisionTreeClassifier(criterion='gini',random_state=42), X_train, y_train,'max_depth', degree, cv=10)
plt.plot(degree, np.median(train_scores, 1),color='blue', label='training')
plt.plot(degree, np.median(val_scores, 1), color='red', label='cross validation')
plt.legend(loc='lower right')
plt.ylim(0.9, 1.01)
plt.title("Pruning by max_depth (Cancer)")
plt.xlabel('max_depth')
plt.ylabel('Accuracy');
plt.xticks(np.arange(0,31,5))
plt.show()
#Using  Entropy Function Metric
clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth=5, min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)
predictions = clf_entropy.predict(X_test)
print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_entropy.predict(X_train), y_train))
degree = range(1,31)
train_score=0
val_score=0
repeat=10
for i in range(repeat):
    train_scores, val_scores = validation_curve(DecisionTreeClassifier(criterion='entropy',min_samples_split = 9), X_train, y_train,'max_depth', degree, cv=10)
    train_score += np.mean(train_scores,1)
    val_score += np.mean(val_scores,1)
train_score /= repeat
val_score /= repeat
plt.plot(degree, train_score,color='blue', label='training')
plt.plot(degree, val_score, color='red', label='cross validation')

plt.legend(loc='lower right')
plt.ylim(0.9, 1.01)
plt.title("Pruning by max_depth (Cancer)")
plt.xlabel('max_depth')
plt.ylabel('Accuracy');
plt.xticks(np.arange(0,31,5))
plt.show()
from sklearn.model_selection import GridSearchCV
dtc_paras={"max_depth":[1,2,4,6,9,12,15,20,25,30]}
resTreepd = pd.DataFrame(dtc_paras)
resTreepd["train"] = resTreepd["cv"] = np.zeros(len(dtc_paras["max_depth"]))
repeat = 10
for _ in range(repeat):
    dtc=DecisionTreeClassifier(criterion = "entropy",min_samples_split = 9)
    gscvTree=GridSearchCV(dtc, dtc_paras,cv=5,n_jobs=2)
    gscvTree.fit(X_train, y_train)
    resTree = gscvTree.cv_results_
    resTreepd["train"] += 1-np.array(resTree['mean_train_score'])
    resTreepd["cv"] += 1-np.array(resTree['mean_test_score'])
resTreepd["train"] /= repeat
resTreepd["cv"] /= repeat
resTreepd.max_depth = resTreepd.max_depth.astype(str)
resTreepd=resTreepd.set_index(["max_depth"])
ax=resTreepd.plot(title="Performances of train and cross validation\n when changing max_depth for Decision Tree")
ax.set_ylabel("Mean Error")
idxmin = [i for i in range(resTreepd.shape[0]) if resTreepd.index[i] == resTreepd.idxmin()['cv']]
ax.axvline(x=idxmin[0],color='r',ls='--',alpha=0.5)
plt.show()
degree = range(2,100)
train_scores, val_scores = validation_curve(DecisionTreeClassifier(max_depth=5,random_state=42), X_train, y_train, 'min_samples_split', degree, cv=10)
plt.plot(degree, np.mean(train_scores, 1), color='blue', label='training score')
plt.plot(degree, np.mean(val_scores, 1), color='red', label='validation score')
plt.legend(loc='lower right')
plt.ylim(0.85, 1.01)
plt.title("Pruning by min_samples_split (Cancer)")
plt.xlabel('min_samples_split')
plt.ylabel('Accuracy');
plt.show()

# determine optimal pruning for the decision trees using cross-validation grid-search
# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2,3,4, 5,6, 10, 20],
              "max_depth": [2, 3,4,5,6,7, 10, 15, 20],
              }

tree_gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid=param_grid, cv=3)

tree_gs.fit(X_train, y_train)
tree_gs.best_params_
import timeit
tree_final_model = tree_gs.best_estimator_
start = timeit.default_timer()
clf = tree_final_model.fit(X_train, y_train)
stop = timeit.default_timer()
train_time = stop - start
treeObj = clf.tree_

# train
train_Y_predict = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_Y_predict)

# test
start = timeit.default_timer()
test_Y_predict = clf.predict(X_test)
stop = timeit.default_timer()
test_time = stop - start
test_acc = accuracy_score(y_test, test_Y_predict)

# print result
print ('tree size: ', treeObj.node_count)
print ('train acc: ', train_acc)
print ('test acc: ', test_acc)
print ('train time: ', train_time)
print ('test time: ', test_time)
clf = DecisionTreeClassifier()
start = timeit.default_timer()
clf = clf.fit(X_train, y_train)
stop = timeit.default_timer()
train_time = stop - start
treeObj = clf.tree_

# train
train_Y_predict = clf.predict(X_train)
train_acc = accuracy_score(y_train, train_Y_predict)

# test
start = timeit.default_timer()
test_Y_predict = clf.predict(X_test)
stop = timeit.default_timer()
test_time = stop - start
test_acc = accuracy_score(y_test, test_Y_predict)

# print result
print ('tree size: ', treeObj.node_count)
print ('train acc: ', train_acc)
print ('test acc: ', test_acc)
print ('train time: ', train_time)
print ('test time: ', test_time)
from sklearn.learning_curve import learning_curve
training_size, train_lc, val_lc = learning_curve(tree_final_model,
                                         X_train, y_train, cv=10,
                                         #train_sizes=np.arange(1, 409), n_jobs=1)
                                         train_sizes=np.linspace(0.001, 1, 50), n_jobs=1)
plt.plot(training_size, np.mean(train_lc, 1), color='blue', label='training')
plt.plot(training_size, np.mean(val_lc, 1), color='red', label='cross validation')
plt.hlines(np.mean([train_lc[-1], val_lc[-1]]), training_size[0], training_size[-1],
                 color='gray', linestyle='dashed')

plt.legend(loc='lower right')
plt.ylim(0.6, 1.05)
plt.xlim(training_size[0], training_size[-1])
plt.title("DT Learning curve (Cancer)")
plt.xlabel('training size')
plt.ylabel('Accuracy');
plt.show()
training_size, train_lc, val_lc = learning_curve(DecisionTreeClassifier(),
                                         X_train, y_train, cv=10,
                                         train_sizes=np.linspace(0.001, 1, 50), n_jobs=1)
plt.plot(training_size, np.mean(train_lc, 1), color='blue', label='training')
plt.plot(training_size, np.mean(val_lc, 1), color='red', label='cross validation')
plt.hlines(np.mean([train_lc[-1], val_lc[-1]]), training_size[0], training_size[-1],
                 color='gray', linestyle='dashed')

plt.legend(loc='lower right')
plt.ylim(0.5, 1.05)
plt.xlim(training_size[0], training_size[-1])
plt.title("DT Learning curve (Cancer)")
plt.xlabel('training size')
plt.ylabel('Accuracy');
plt.show()
plt.plot(degree, np.mean(train_scores, 1), color='blue', label='training')
plt.plot(degree, np.mean(val_scores, 1), color='red', label='cross validation')
plt.legend(loc='lower right')
plt.ylim(0.85, 1.1)
plt.title("Boosting of pruned tree (Cancer)")
plt.xlabel('# of iteration')
plt.ylabel('Accuracy');
plt.show()
from sklearn.ensemble import AdaBoostClassifier
degree = range(1,51)
train_scores, val_scores = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)), X_train, y_train,
                                          'n_estimators', degree, cv=10)
plt.plot(degree, np.mean(train_scores, 1), color='blue', label='training')
plt.plot(degree, np.mean(val_scores, 1), color='red', label='cross validation')
plt.legend(loc='lower right')
plt.ylim(0.85, 1.1)
plt.title("Boosting of pruned tree (Cancer)")
plt.xlabel('# of iteration (n_estimators)')
plt.ylabel('Accuracy');
plt.show()
#degree = range(1,51,5)
degree = [1,2,5,10,20,50,100,200]
train_scoresb1, val_scoresb1 = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=2)), X_train, y_train,
                                          'n_estimators', degree, cv=10)
train_scoresb2, val_scoresb2 = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,min_samples_split=2)), X_train, y_train,
                                          'n_estimators', degree, cv=10)
train_scoresb3, val_scoresb3 = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=10,min_samples_split=2)), X_train, y_train,
                                          'n_estimators', degree, cv=10)
train_scoresb4, val_scoresb4 = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=2)), X_train, y_train,
                                          'n_estimators', degree, cv=10)
plt.plot(degree, np.mean(val_scoresb1, 1), color='green', label='max_depth=2')
plt.plot(degree, np.mean(val_scoresb2, 1), color='blue', label='max_depth=5')
plt.plot(degree, np.mean(val_scoresb3, 1), color='purple', label='max_depth=10')
plt.plot(degree, np.mean(val_scoresb4, 1), color='red', label='unpruned')

plt.legend(loc='upper right')
plt.ylim(0.9, 1.05)
plt.title("Boosting: varying tree size (Cancer)")
plt.xlabel('# of iterations')
plt.ylabel('Cross-Validation Accuracy');
plt.show()
degree = np.arange(0.1,3,0.2)
train_scores, val_scores = validation_curve(AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=2)), X_train, y_train,
                                          'learning_rate', degree, cv=10)
plt.plot(degree, np.mean(train_scores, 1), color='blue', label='training')
plt.plot(degree, np.mean(val_scores, 1), color='red', label='cross validation')

plt.legend(loc='lower right')
plt.ylim(0.7, 1.02)
plt.title("Boosting: varying learning rate \n on pruned tree (Cancer)")
plt.xlabel('learning rate')
plt.ylabel('Accuracy');
#plt.xticks(np.arange(0,2.1,0.5))
plt.show()
from sklearn.ensemble import RandomForestClassifier
for md in [1,2,3,4,5,6,7,8,9,10,11]:
    clf_rf = RandomForestClassifier(max_depth = md, random_state =1, min_samples_leaf =4)
    clf_rf.fit(X_train, y_train)
    predictions  = clf_rf.predict(X_test)
    print(md, accuracy_score(predictions,y_test))
clf_rf = RandomForestClassifier(max_depth = 4, random_state =1, min_samples_leaf =4)
clf_rf.fit(X_train, y_train)
predictions  = clf_rf.predict(X_test)
print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_entropy.predict(X_train), y_train))
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Make Support Vector Classifier Pipeline
pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=9)),
                     ('clf', SVC(random_state=1))])
print('--> Made Pipeline')
#Fit Pipeline to Data
pipe_svc.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')
scores = cross_val_score(estimator=pipe_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
#Tune Hyperparameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
              {'clf__C': param_range,
               'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)
gs = gs.fit(X_train, y_train)
print('--> Tuned Parameters Best Score: ',gs.best_score_)
print('--> Best Parameters: \n',gs.best_params_)

#Use best parameters
clf_svc = gs.best_estimator_

#Get Final Scores
clf_svc.fit(X_train, y_train)
scores = cross_val_score(estimator=clf_svc,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)

print('--> Final Model Training Accuracy: %.6f +/- %.6f' %(np.mean(scores), np.std(scores)))

print('--> Final Accuracy on Test set: %.5f' % clf_svc.score(X_test,y_test))
from sklearn.linear_model import LogisticRegression

#Make Logistic Regression Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=9)),
                    ('clf', LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=1, max_iter=1000, n_jobs=-1))])
print('--> Made Logistic Regression Pipeline')

#Fit Pipeline to Data
pipe_lr.fit(X_train, y_train)
print('--> Fitted Pipeline to training Data')
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Model Training Accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

#Tune Hyperparameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_range_small = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
param_grid_lr = [{'clf__penalty': ['l1'],
               'clf__C': param_range,
               'clf__tol': param_range_small},
              {'clf__penalty': ['l2'],
               'clf__C': param_range,
               'clf__tol': param_range_small}]
gs_lr = GridSearchCV(estimator=pipe_lr,
                  param_grid=param_grid_lr,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)
gs_lr = gs_lr.fit(X_train, y_train)
print('--> Tuned Parameters Best Score: ',gs_lr.best_score_)
print('--> Best Parameters: \n',gs_lr.best_params_)

#Use best parameters
clf_lr = gs_lr.best_estimator_

#Get Final Scores
clf_lr.fit(X_train, y_train)
scores_lr = cross_val_score(estimator=clf_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('--> Final Model Training Accuracy: %.5f +/- %.5f' %(np.mean(scores_lr), np.std(scores_lr)))

print('--> Final Accuracy on Test set: %.5f' % clf_lr.score(X_test,y_test))
