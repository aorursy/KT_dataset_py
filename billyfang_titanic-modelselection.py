# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

import warnings

data_train = pd.read_csv("../input/titanic-filter/train_filter.csv")
data_test = pd.read_csv("../input/titanic-filter/test_filter.csv")
from sklearn import metrics
from sklearn import linear_model

train_df = data_train.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title')
train_np = train_df.values

y = train_np[:, 0]

X = train_np[:, 1:]

modelL = linear_model.LogisticRegression()
modelL.fit(X, y)
    
modelL

test = data_test.filter(regex='Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title')
predictions = modelL.predict(test)
#result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#result.to_csv("...", index=False)
from sklearn import metrics
from sklearn.svm import SVC

train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]

X = train_np[:, 1:]

modelS = SVC()
modelS.fit(X,y)
    
modelS

test = data_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = modelS.predict(test)
#result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#result.to_csv("/Users/17728/Desktop/Python/JupyterNotebook/Titanic/SVM_predictions.csv", index=False)
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

train_df = data_train.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title')
train_np = train_df.values

y = train_np[:, 0]

X = train_np[:, 1:]

modelN = MLPClassifier(hidden_layer_sizes=(15,),learning_rate_init= 0.001,activation='relu',\
     solver='adam', alpha=0.0001,max_iter=30000)
modelN.fit(X,y)
    
modelN

test = data_test.filter(regex='Age|SibSp|Parch|Fare|Cabin|Embarked|Sex|Pclass|Title')
predictions = modelN.predict(test)
#result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#result.to_csv("/Users/17728/Desktop/Python/JupyterNotebook/Titanic/neural_network_15units_predictions2.csv", index=False)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"num of training data")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"score on training set")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"score on CV")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(modelL, u"learning curve", X, y)
plot_learning_curve(modelS, u"learning curve", X, y)
plot_learning_curve(modelN, u"learning curve", X, y)
