#imports



import pandas as pd

import numpy as np

from sklearn import metrics

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("fivethirtyeight")





import warnings

warnings.filterwarnings('ignore')
df= pd.read_csv('C:/Users/uknow/Desktop/bank_final2.csv')



df.head()
df.dtypes
df.rename(columns = {"['y']_yes":'y'}, inplace = True) 
X= pd.read_csv('C:/Users/uknow/Desktop/bank_final.csv')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df.y, test_size=0.2, random_state=2019)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from sklearn.model_selection import KFold

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train, y_train)
#ppnprep=y_pred are the class labels that we predicted

ppnpred = ppn.predict(X_test)
print(ppn.score(X_train, y_train), ppn.score(X_test, y_test))
# Calculate the classification accuracy of the perceptron



# Here y_test are the true class labels 



from sklearn import metrics



from sklearn.metrics import accuracy_score



print('Accuracy: %.2f' % accuracy_score(y_test, ppnpred))
from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_test, ppnpred))
from sklearn.model_selection import cross_val_score



PPNCV = (cross_val_score(ppn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

PPNCV
from sklearn.metrics import recall_score



print(round(metrics.recall_score(y_test, ppnpred),2))

from sklearn.metrics import precision_score



print(round(metrics.precision_score(y_test, ppnpred),2))
from sklearn.metrics import classification_report



print(' Performance Metrics Reports\n',classification_report(y_test, ppnpred))
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, ppnpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Perceptron model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))



z = np.arange(-7, 7, 0.1)

plt.plot(z, sigmoid(z))

plt.axvline(0.0, color='k')

plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')

plt.axhline(y=0.5, ls='dotted', color='k')

plt.yticks([0.0, 0.5, 1.0])

plt.ylim(-0.1, 1.1)

plt.xlabel('z= the net input ')

plt.title('$sigmoid (z)$')

plt.ylabel("y=responses")

plt.show()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear')

logreg.get_params()
logreg.fit(X_train, y_train)
# logistic regression is a linear model, so you have coefficients and intercepts:

logreg.coef_
logreg.intercept_
coeffs = pd.DataFrame({

    'features X': X_train.columns,

    'weights w'  : logreg.coef_[0]

})



coeffs


Y = X_train.dot(logreg.coef_.T) + logreg.intercept_

# and this gives us our predictions

sigmoid(Y)
# you can then use the predict method to predic out of sample data

logpred= logreg.predict(X_test)

logpred
print(logreg.score(X_train, y_train), logreg.score(X_test, y_test))
logreg.predict_proba(X_test)
print(confusion_matrix(y_test, logpred))



print(round(accuracy_score(y_test, logpred),2)*100)



print(round(metrics.recall_score(y_test, logpred),2))



print(round(metrics.precision_score(y_test, logpred),2))



LOGCV = (cross_val_score(logreg, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())



print(' Performance Metrics Reports\n',classification_report(y_test, logpred))
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, logpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for LogReg model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.model_selection import GridSearchCV



# set up the parameters of the model you'd like to fit

param_grid = {

    'penalty': ['l1', 'l2'],

    'C'      : [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000],

}
# load it into the grid

grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=10)



# fit it on your training data

grid.fit(X_train, y_train)



# get the version that gave you the best fit

grid.best_params_



# Note that tunning parameter is 1/C is small
params_lg = {'C': 10000, 'penalty': 'l1'}



logreg.set_params(**params_lg)
logreg.fit(X_train, y_train)
logreg.predict_proba(X_test)
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, logpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for LogReg model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini

dtree.fit(X_train, y_train)

dtreepred = dtree.predict(X_test)


print(confusion_matrix(y_test, dtreepred))

print(round(accuracy_score(y_test, dtreepred),2)*100)

DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(' Performance Metrics Reports\n',classification_report(y_test, logpred))
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, logpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Decision Tree model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini

rf.fit(X_train, y_train)

rfpred = rf.predict(X_test)

print(confusion_matrix(y_test, rfpred ))

print(round(accuracy_score(y_test, rfpred),2)*100)

RFCV = (cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(' Performance Metrics Reports\n',classification_report(y_test, rfpred))
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, logpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Random Forest model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)

gbpred = gb.predict(X_test)

print(confusion_matrix(y_test, gbpred ))

print(round(accuracy_score(y_test, gbpred),2)*100)

GBCV = (cross_val_score(gb, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, gbpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Gradient Boosting model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.svm import SVC

sv= SVC(kernel = 'sigmoid')

sv.fit(X_train, y_train)

svpred = sv.predict(X_test)

print(confusion_matrix(y_test, svpred))

print(round(accuracy_score(y_test, svpred),2)*100)

SVCV = (cross_val_score(sv, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
print(' Performance Metrics Reports\n',classification_report(y_test, svpred))
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, svpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for SVM model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=22)

knn.fit(X_train, y_train)

knnpred = knn.predict(X_test)
print(confusion_matrix(y_test, knnpred))

print(round(accuracy_score(y_test, knnpred),2)*100)

KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())
fprgbk, tprgbk, thresholdgbk = metrics.roc_curve(y_test, knnpred)



roc_aucgbk = metrics.auc(fprgbk, tprgbk)



plt.plot(fprgbk, tprgbk, 'b', label = 'AUC = %0.2f' % roc_aucgbk)

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for KNN model ',fontsize=10)

plt.ylabel('True Positive Rate',fontsize=20)

plt.xlabel('False Positive Rate',fontsize=15)

plt.legend(loc = 'lower right', prop={'size': 16})



plt.show()