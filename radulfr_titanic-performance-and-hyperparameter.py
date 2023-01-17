# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
traindf = pd.read_csv("../input/train.csv")
traindf.tail(10)

traindf.isnull().sum()
traindf['Age'].describe()
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
traindf['Age'] = imp.fit_transform(traindf[['Age']])

traindf['Age'].describe()
traindf[['Fare', 'Age']].describe()
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
traindf[['Fare', 'Age']] = mms.fit_transform(traindf[['Fare', 'Age']])
traindf[['Fare', 'Age']].describe()
traindf = pd.get_dummies(traindf, columns = ['Embarked', 'Sex'])
#Once encoded let's see. 
traindf.head()
#The most simple one :D
traindf['ratioFareClass'] = traindf['Fare']/traindf['Pclass']

traindf.head()
features = ['Age', 'Fare', 'Pclass', 'Embarked_C', 'Embarked_S', 'Embarked_Q', 'Sex_female', 'Sex_male', 'ratioFareClass']
from sklearn.model_selection import train_test_split
X_train, X_other, y_train, y_other = train_test_split(traindf[features],
                                                    traindf["Survived"],
                                                    test_size=0.4,
                                                    stratify=traindf['Survived'])

X_test, X_valid, y_test, y_valid = train_test_split(X_other,
                                                    y_other,
                                                    test_size=0.5,
                                                    stratify=y_other)
print("Length of datasets:\nTrain: %d\nTest: %d\nValidation: %d" % (len(X_train), len(X_test), len(X_valid)) )
from sklearn.model_selection import cross_val_score
def validate(model, X_train, y_train, k=10):
    result = 'K-fold cross validation:\n'
    scores = cross_val_score(estimator=model,
                             X=X_train,
                             y=y_train,
                             cv=k,
                             n_jobs=1)
    for i, score in enumerate(scores):
        result += "Iteration %d:\t%.3f\n" % (i, score)
    result += 'CV accuracy:\t%.3f +/- %.3f' % (np.mean(scores), np.std(scores))
    return result
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def learningCurve(model, X_train, y_train, k=10):
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=model,
                                   X=X_train,
                                   y=y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 10),
                                   cv=k,
                                   n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples', fontsize=fsize)
    plt.ylabel('Accuracy', fontsize=fsize)
    plt.legend(loc='lower right')
    plt.ylim([0.4, 1.03])
    plt.tight_layout()
    plt.show()
from sklearn.model_selection import validation_curve

def validationCurve(model, X_train, y_train,p_name, p_range, k=10, scale=False):
    train_scores, test_scores = validation_curve(
                    estimator=model, 
                    X=X_train, 
                    y=y_train, 
                    param_name=p_name,
                    param_range=p_range,
                    cv=k)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.rcParams["figure.figsize"] = [6,6]
    fsize=14
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)
    plt.plot(p_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')

    plt.fill_between(p_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(p_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')

    plt.fill_between(p_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')

    plt.grid()
    if scale:
        plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter %s' % p_name, fontsize=fsize)
    plt.ylabel('Accuracy', fontsize=fsize)
    plt.ylim([0.7, 1.0])
    plt.tight_layout()
    plt.show()
from sklearn.metrics import roc_curve, roc_auc_score

def rocCurve(model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = roc_auc_score(y_test, y_scores)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams["figure.figsize"] = [8,8]
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    

from sklearn.metrics import confusion_matrix

def confusionMatrix(model, X_train, y_train, X_test, y_test): 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)


print("Logistic Regression score (Train): {0:.2}".format(lr.score(X_train, y_train)))
print("Logistic Regression score (Test): {0:.2}".format(lr.score(X_test, y_test)))
print(validate(lr, X_train, y_train))
learningCurve(lr, X_train, y_train)
rocCurve(lr, X_test, y_test)
confusionMatrix(lr, X_train, y_train, X_test, y_test)
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("KNN score (Train): {0:.2}".format(neigh.score(X_train, y_train)))
print("KNN score (Test): {0:.2}".format(neigh.score(X_test, y_test)))
print(validate(neigh, X_train, y_train))
learningCurve(neigh, X_train, y_train)
rocCurve(neigh, X_train, y_train)
confusionMatrix(neigh, X_train, y_train, X_test, y_test)
#http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVC
svclass = SVC(probability=True)
svclass.fit(X_train, y_train) 
print("SVM score (Train): {0:.2}".format(svclass.score(X_train, y_train)))
print("SVM score (Test): {0:.2}".format(svclass.score(X_test, y_test)))
print(validate(svclass, X_train, y_train))

learningCurve(svclass, X_train, y_train)
rocCurve(svclass, X_test, y_test)
confusionMatrix(svclass, X_train, y_train, X_test, y_test)
#http://scikit-learn.org/stable/modules/tree.html
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
print("Decision Tree score (Train): {0:.2}".format(dt.score(X_train, y_train)))
print("Decision Tree score (Test): {0:.2}".format(dt.score(X_test, y_test)))
print(validate(dt, X_train, y_train))

learningCurve(dt, X_train, y_train)
rocCurve(dt, X_train, y_train)
confusionMatrix(dt, X_train, y_train, X_test, y_test)
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=3,
                                criterion='gini',
                                max_depth=3,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
X_train.head()
forest.fit(X_train, y_train)
print("Random Forest score (Train): {0:.2}".format(forest.score(X_train, y_train)))
print("Random Forest score (Test): {0:.2}".format(forest.score(X_test, y_test)))
print(validate(forest, X_train, y_train))

learningCurve(forest, X_train, y_train)
rocCurve(forest, X_test, y_test)
confusionMatrix(forest, X_train, y_train, X_test, y_test)
# C parameter: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# More info: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
validationCurve(lr, X_train, y_train, p_name='C', p_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], scale=True)
validationCurve(neigh, X_train, y_train, p_name='n_neighbors', p_range=[1,2,3,4,5,10,15,20])
validationCurve(dt, X_train, y_train, p_name='max_depth', p_range=[1,2,3,4,5,10,20])
validationCurve(forest, X_train, y_train, p_name='n_estimators', p_range=[1,2,3,4,5,10,20])
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[1,2,3,4,5,6,7,8,9,10],
              'max_depth': [1,2,3,4,5,10,20] }

# parameters = {'n_estimators':[1,2,3,4,5,6,7,8,9,10],
#               'max_depth': [1,2,3,4,5,10,20],
#               'min_samples_split': [2,3,4,5,6,7,8,9,10],
#               'min_samples_leaf': [2,3,4,5,6,7,8,9,10]
#              }
gs = GridSearchCV(estimator=forest,
                     param_grid=parameters,
                     scoring='accuracy',
                     cv=10)
gs.fit(X_train,y_train)                            
print(gs.best_score_)
print("Best parameters: " + str(gs.best_params_))

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Train accuracy: %.3f' % clf.score(X_train, y_train))
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
print('Validation accuracy: %.3f' % clf.score(X_valid, y_valid))