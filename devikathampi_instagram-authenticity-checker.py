import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_filepath = "../input/instagram-fake-spammer-genuine-accounts/train.csv"

test_filepath = "../input/instagram-fake-spammer-genuine-accounts/test.csv"

insta_train = pd.read_csv(train_filepath)

insta_test = pd.read_csv(test_filepath)
insta_train.head()
insta_train.describe()
insta_train.info()
print(insta_train.shape)

print(insta_test.shape)
print(insta_train.isna().values.any().sum())

print(insta_train.isna().values.any().sum())

corr= insta_train.corr()

sns.heatmap(corr)
sns.pairplot(insta_train)
train_Y = insta_train.fake

train_Y = pd.DataFrame(train_Y)

train_Y.tail(12)
train_X = insta_train.drop(columns="fake")

train_X.head()
test_Y = insta_test.fake

test_Y =pd.DataFrame(test_Y)

test_Y.tail(12)
test_X = insta_test.drop(columns="fake")

test_X.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

logreg = LogisticRegression()

model1 = logreg.fit(train_X,train_Y)

logreg_predict = model1.predict(test_X)
accuracy_score(logreg_predict,test_Y)
print(classification_report(logreg_predict,test_Y))
def plot_confusion_matrix(test_Y,predict_y):

    C = confusion_matrix(test_Y,predict_y)

    A = (((C.T)/(C.sum(axis=1))).T)

    B = (C/C.sum(axis=0))

    plt.figure(figsize=(20,4))

    labels = [1,2]

    cmap=sns.light_palette("seagreen")

    plt.subplot(1,3,1)

    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)

    plt.xlabel("Predicted Class")

    plt.ylabel("Original Class")

    plt.title("Confusion matrix")

    plt.subplot(1,3,2)

    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)

    plt.xlabel("Predicted Class")

    plt.ylabel("Original Class")

    plt.title("Precision matrix")

    plt.subplot(1,3,3)

    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels,yticklabels=labels)

    plt.xlabel("Predicted Class")

    plt.ylabel("Original Class")

    plt.title("Recall matrix")

    plt.show()

    

    

    
plot_confusion_matrix(test_Y,logreg_predict)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

model2 = knn.fit(train_X,train_Y)

knn_predict = model2.predict(test_X)
accuracy_score(knn_predict,test_Y)
print(classification_report(knn_predict,test_Y))
plot_confusion_matrix(test_Y,knn_predict)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

model3= rfc.fit(train_X,train_Y)

rfc_predict = model3.predict(test_X)

accuracy_score(rfc_predict,test_Y)
print(classification_report(rfc_predict,test_Y))
plot_confusion_matrix(test_Y,rfc_predict)
from xgboost import XGBClassifier

xgb= XGBClassifier()

model4 = xgb.fit(train_X,train_Y)

xgb_predict = model4.predict(test_X)
accuracy_score(xgb_predict,test_Y)
print(classification_report(xgb_predict,test_Y))
plot_confusion_matrix(test_Y,xgb_predict)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training Examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="b")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(model3,"RF learning curves",train_X,train_Y)

g = plot_learning_curve(model2,"KNN learning curves",train_X,train_Y)

g = plot_learning_curve(model1,"Logistic Rgeression learning curves",train_X,train_Y)

g = plot_learning_curve(model3,"XGB learning curves",train_X,train_Y)
print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))

print('KNN Accuracy:',accuracy_score(knn_predict,test_Y))

print('RFC Accuracy:',accuracy_score(rfc_predict,test_Y))

print('XGB Accuracy:',accuracy_score(xgb_predict,test_Y))