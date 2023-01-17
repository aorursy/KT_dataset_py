import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') #shutting down warnings
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

iris = datasets.load_iris()
#Two classes are versicolor and virignica

#Two Features are Sepal Width and Petal length

X, y = iris.data[50:, [1,2]], iris.target[50:]
#let's convert two classes into numerical labels



le = LabelEncoder()

y = le.fit_transform(y)
np.unique(y)
#splitting the data

#Here I am using stratification it will make sure that training and test set have same proportion of classes



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 1, stratify = y)
#lets train three classifier, logistic regression, decision tree, k-nearest neighbors

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
clf1 = LogisticRegression(

        penalty='l2',

        C = 0.001,

        random_state = 1

)



clf2 = DecisionTreeClassifier(

            max_depth=1,

            criterion='entropy',

            random_state=0

)



clf3 = KNeighborsClassifier(

        n_neighbors=1,

        p = 2,

        metric='minkowski'

)
pipe1 = make_pipeline(StandardScaler(), clf1) #it is important to standarize data before LogisticRegression
pipe2 = make_pipeline(StandardScaler(), clf3) #it is important to standarize data before using KNN classification


clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']





for  label, clf in zip(clf_labels, [pipe1, clf2, pipe2]):

    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')

    

    print("ROC AUC: {:.2f} (+/- {:.2f}) {}".format(np.mean(scores), np.std(scores), label))

    
from sklearn.ensemble import VotingClassifier
en = VotingClassifier(

    estimators=[('m1',pipe1),('m2',clf2),('m3',pipe2)],

    voting='soft'

)
clf_labels  += ['Ensemble']
for label, clf in zip(clf_labels, [pipe1, clf1, pipe2, en]):

    scores = cross_val_score(estimator=clf, scoring='roc_auc', cv = 10, X = X_train, y = y_train)

    print("ROC AUC: {:.2f} (+/- {:.2f}) {}".format(np.mean(scores), np.std(scores), label))
from sklearn.metrics import roc_curve

from sklearn.metrics import auc
all_clf = [pipe1, clf2, pipe2, en]



colors = ['black', 'orange', 'blue', 'green']

linestyles = [':', '--', '-.', '-']



for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):

    #assuming the label of positive class is 1 remember that roc curve is tpr vs fpr

    

    probas = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]

    fpr, tpr, threshods = roc_curve(y_true=y_test, y_score=probas)

    

    roc_auc = auc(x=fpr, y = tpr)

    

    plt.plot(fpr, tpr, color = clr, linestyle=ls, label = '{} (auc = {:.2f})'.format(label, roc_auc))



plt.legend(loc = 'lower right')

plt.plot([0,1],[0,1], linestyle = '--', color = 'gray', linewidth = 2)

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.grid(alpha = 0.5)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
#we will standarize our data for visualization to be on same scale even though our pipeline have it for the two models.

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)



from itertools import product #it does cross product / cartesian product



x1_min, x1_max = X_train_std[:,0].min() - 1, X_train_std[:,0].max() + 1

x2_min, x2_max = X_train_std[:,1].min() - 1, X_train_std[:,1].max() + 1



xx1, xx2  = np.meshgrid(

            np.arange(x1_min, x1_max , 0.1),

            np.arange(x2_min, x2_max, 0.1)

)
#let's make subplots for our 2 X 2 figure first



fig, axis = plt.subplots(ncols=2, nrows=2, sharex='col', sharey='row' ,figsize = (7,5))



for idx, clf, tt in zip(product([0,1],[0,1]), all_clf, clf_labels):

    clf.fit(X_train_std, y_train) #fitting our model

    

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    #since tt is a tupe with (x, y)

    axis[idx[0], idx[1]].contourf(xx1, xx2, Z, alpha = 0.3)

    axis[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0], X_train_std[y_train==0, 1], c = 'blue', marker = '^', s = 50)

    axis[idx[0], idx[1]].scatter(X_train_std[y_train==1,0], X_train_std[y_train==1, 1], c = 'green', marker = 'o', s = 50)

    axis[idx[0], idx[1]].set_title(tt)



plt.text(-3.5, -4.7, s = 'Sepal width [standarized]', ha = 'center', va = 'center', fontsize = 12)

plt.text(-12.5, 4.5, s = 'Petal length [standarized]', ha = 'center', va = 'center', fontsize = 12, rotation = 90)

plt.show()
#let's tune this ensemble model, but for that first let's know what the variable get called

en.get_params()
from sklearn.model_selection import GridSearchCV
params = {

    'm1__logisticregression__C': [0.001, 0.1, 100.0],

    'm2__max_depth':[1,2]

}
grid = GridSearchCV(estimator=en, cv=10, scoring='roc_auc', param_grid=params)
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_
df_wine = pd.read_csv('../input/wine-data/wine.data', header = None)
df_wine.columns = [

    'class label',

    'Alcohol',

    'Malic Acid',

    'Ash',

    'Alcalinity of ash',

    'Magnesium',

    'Total phenols',

    'Flavanoids',

    'Nonflavanoid phenols',

    'Proanthocyanins',

    'Color intensity',

    'Hue',

    '0D280/0D315 of diluted wines',

    'Proline'

]
df_wine.head()
#here we will only consider wine class 2 and 3

df_wine = df_wine[df_wine['class label'] != 1]



#here we will only consider two features Alochol, OD280/0D315 of diluted wines

y = df_wine['class label'].values

X = df_wine[['Alcohol', '0D280/0D315 of diluted wines']].values
#let's label encode our class labels

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



y = le.fit_transform(y)
np.bincount(y), np.unique(y)
#let's split our data in 80:20 ration for training and testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
from sklearn.ensemble import BaggingClassifier
#we will be using decision tree with no pruning for our Bagging classifier

tree = DecisionTreeClassifier(

        criterion='entropy',

        random_state = 1,

        max_depth = None

)
'''

    Here, we will be making 500 estimator,

    each estimator will train for random sample with replacement (Bootstrap = True),

    each estimator will train for all feature without replacement (Bootstrap = False)

    

    See Bagging Classifier doc for understanding max_sample

    and max_feature, it is differnt for int value and float value

    for int, max_feature means max_feature for sampling.

    For float, max_feature means max_feature * X.shape[1] for sampling

    similary for max_sample

    for int; max_sample

    for Float; max_sample * X.shape[0]



'''





bag = BaggingClassifier(

        base_estimator=tree,

        n_estimators=500,

        max_samples=1.0,

        max_features=1.0,

        bootstrap=True,

        bootstrap_features=False,

        n_jobs=-1,

        random_state=1

)
from sklearn.metrics import accuracy_score
tree  = tree.fit(X_train, y_train) #fitting our decision tree first
y_train_pred = tree.predict(X_train) #how model perform on already seen data

y_test_pred = tree.predict(X_test)   #how model perform on unseen data



tree_training_set_accuracy = accuracy_score(y_pred=y_train_pred, y_true=y_train)

tree_testing_set_accuracy = accuracy_score(y_pred=y_test_pred, y_true=y_test)



print('Decision tree test/train score {:.3f}/{:.3f}'.format(tree_training_set_accuracy, tree_testing_set_accuracy))
bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)

y_test_pred = bag.predict(X_test)



bag_training_set_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)

bag_testing_set_accuracy  = accuracy_score(y_true=y_test, y_pred = y_test_pred)



print('Bagging train/test accuracies: {:.3f}/{:.3f}'.format(bag_training_set_accuracy, bag_testing_set_accuracy))
#let's see decision region for both the models



x1_min, x1_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1

x2_min, x2_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1



xx1, xx2 = np.meshgrid(

                np.arange(x1_min, x1_max, 0.1),

                np.arange(x2_min, x2_max, 0.1)

)
fig, ax = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize = (8,3))



for idx, clf, tt in zip([0,1], [tree, bag], ['Decision tree', 'Bagging']):

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    ax[idx].contourf(xx1, xx2, Z, alpha = 0.3)

    ax[idx].scatter(X_train[y_train == 0, 0], X_train[y_train==0, 1], c = 'blue', marker = '^')

    ax[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1,1], c = 'green', marker = 'o')

    ax[idx].set_title(tt)



ax[0].set_ylabel('Alcohol', fontsize = 12)

plt.text(10.2, -1.2, s = '0d280/0d315 of dilute wines', ha = 'center', va = 'center', fontsize = 12)

plt.show()
from sklearn.ensemble import AdaBoostClassifier
#let's use Decision tree for our base estimator, remeber decision stump

#creating decision tree stump having max_depth 1 (weak learner)



tree = DecisionTreeClassifier(

        criterion='entropy',

        random_state =1 ,

        max_depth = 1



)
ada = AdaBoostClassifier(

        base_estimator=tree,

        n_estimators=500,

        learning_rate=0.1,

        random_state=1

)
#let's train our tree model for comparison 

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train) #seeing how good model generalizes over seen data.

y_test_pred =  tree.predict(X_test)  #seeing how good model generalizes over unseen data.



tree_training_set_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)

tree_test_set_accuracy  = accuracy_score(y_true=y_test, y_pred=y_test_pred)



print('Decision Tree Stump Train/Test accuracies {:.3f}/{:.3f}'.format(tree_training_set_accuracy, tree_test_set_accuracy))
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train) #seeing how good model generalizes over seen data.

y_test_pred = ada.predict(X_test)   #seeing how good model generalizes over unseen data.



ada_training_set_accuracy = accuracy_score(y_train, y_train_pred)

ada_testing_set_accuracy = accuracy_score(y_test, y_test_pred)



print('AdaBoost Train/Test accuracies: {:.3f}/{:.3f}'.format(ada_training_set_accuracy, ada_testing_set_accuracy))
#let's check decision region for both the models



x1_min, x1_max = X_train[:,0].min() - 1, X_train[:,0].max() + 1

x2_min, x2_max = X_train[:,1].min() - 1, X_train[:,1].max() + 1



xx, yy = np.meshgrid(

            np.arange(x1_min, x1_max, 0.1),

            np.arange(x2_min, x2_max, 0.1)

)
fig, ax = plt.subplots(ncols=2, nrows=1, sharex='col', sharey='row', figsize = (8,3))



for idx, clf, tt in zip([0,1], [tree, ada], ['Decision Tree','Ada Boost']):

    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)

    Z = Z.reshape(xx.shape)

    

    ax[idx].contourf(xx, yy, Z, alpha = 0.3)

    ax[idx].scatter(X_train[y_train==0,0], X_train[y_train==0,1], c= 'blue', marker = '^')

    ax[idx].scatter(X_train[y_train==1,0], X_train[y_train==1, 1],  c= 'green', marker = 'o')

    ax[idx].set_title(tt)



ax[0].set_ylabel('Alchol', fontsize = 12)

plt.text(10.2, -0.5, s= '0D280/0D315 of diluted wines', ha='center', va='center', fontsize = 12)

plt.show()