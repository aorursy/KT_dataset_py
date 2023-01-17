import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/winequality-red.csv")

data.head()
data.info()
data.describe()
from sklearn.preprocessing import LabelEncoder



bins = (2, 6, 8)

group_names = ['bad', 'good']



data['quality'] = pd.cut(data["quality"], bins = bins, labels = group_names)



label_quality = LabelEncoder()



data['quality'] = label_quality.fit_transform(data['quality'].astype(str))

data['quality'].value_counts()
sns.countplot(data['quality'])

plt.show()
X = data.drop("quality", axis=1)

y = data["quality"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=10)
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")
# prepare configuration for cross validation test harness

seed = 7



# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(("RFC",RandomForestClassifier()))



# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



params_dict={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}

clf_lr=GridSearchCV(estimator=LogisticRegression(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_lr.fit(X_train,y_train)
clf_lr.best_params_
clf_lr.best_score_ 
pred=clf_lr.predict(X_test)

accuracy_score(pred,y_test)
from sklearn.model_selection import cross_val_score



scores = cross_val_score(clf_lr, X_train, y_train,n_jobs=-1, cv=10)

scores
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(scores)
y_test.value_counts()
from sklearn.metrics import confusion_matrix,  roc_auc_score

confusion_matrix(pred, y_test)
roc_auc_score(y_test, pred)
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)



plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve LR')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc = np.trapz(tpr,fpr)

print('AUC:', auc)
def evaluate_threshold(threshold):

    print('Sensitivity:', tpr[thresholds > threshold][-1])

    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
evaluate_threshold(0.5)
evaluate_threshold(.3)
# calculate cross-validated AUC

cross_val_score(clf_lr, X_train, y_train, cv=10, scoring='roc_auc').mean()
params_dict={'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10],'kernel':['linear','rbf']}

clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=5)

clf.fit(X_train,y_train)
clf.best_params_
clf.best_score_ 
pred_svm=clf.predict(X_test)

accuracy_score(pred_svm,y_test)
confusion_matrix(pred_svm, y_test)
roc_auc_score(y_test, pred_svm)
# calculate cross-validated AUC

cross_val_score(clf, X_train, y_train, cv=4, scoring='roc_auc').mean()
scores = cross_val_score(clf, X_train, y_train, cv=5)

display_scores(scores)
params_dict={'n_estimators':[500],'max_features':['auto','sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),param_grid=params_dict,scoring='accuracy',cv=5)

clf_rf.fit(X_train,y_train)
clf_rf.best_params_
clf_rf.best_score_ 
pred_rf=clf_rf.predict(X_test)

accuracy_score(pred_rf,y_test)
confusion_matrix(pred_rf, y_test)
roc_auc_score(y_test, pred_rf)
cross_val_score(clf_rf, X_train, y_train, cv=4, scoring='roc_auc').mean()
scores = cross_val_score(clf_rf, X_train, y_train, cv=5)

display_scores(scores)