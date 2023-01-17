import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import numpy as np

from collections import Counter

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

%matplotlib inline
### Read data



df = pd.read_csv('../input/creditcard.csv')

print(df.head())

### Time and Amount are not scaled , so scaling it

sc = StandardScaler()

df['Amount']=sc.fit_transform(df['Amount'].values.reshape(-1,1))

df['Time']=sc.fit_transform(df['Time'].values.reshape(-1,1))

##Separate features and o/p class

X = df.iloc[:,:-1]

y = df.iloc[:,-1]

sns.countplot(y)
##Separate data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=40, stratify=y)

print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))
##Feature selection excercise



model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')

plt.show()
##Correlation against each feature



corr = df.corr()

corr.round(2)

plt.figure(figsize=(28,28))

sns.heatmap(corr,annot=True,cmap='coolwarm')
from sklearn import discriminant_analysis

from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold



lda = discriminant_analysis.LinearDiscriminantAnalysis()#SVC(kernel="linear")



rfecv = RFECV(estimator=lda, step=1, cv=StratifiedKFold(3),scoring='accuracy')

rfecv.fit(X_train, y_train)



rfecv.n_features_

# Numbers of features selected

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

rfecv.grid_scores_



rfecv.support_



X_train.columns.values[rfecv.support_]
##Check for distribution of each feature on each class

##Used for feature selection



gs = gridspec.GridSpec(28,1)

plt.figure(figsize=(6,28*4))

for i,col in enumerate(df[df.iloc[:,0:28].columns]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[col][df.Class ==1],bins=50,color='r')

    sns.distplot(df[col][df.Class ==0],bins=50,color='g')

    ax.set_title('feature '+str(col))

plt.show()
### Import all models that will be needed



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.feature_selection import SelectFromModel , RFE

import statsmodels.api as sm

from sklearn.metrics import roc_curve, auc , confusion_matrix , classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn import svm
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Normal Logistic Regression with all columns as features")

print("F1 score is {}".format(f1_score(y_test, y_pred)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("AUC is {}".format(metrics.auc(fpr, tpr)))

print("Recall is {}".format(metrics.recall_score(y_test, y_pred)))

print("Precision is {}".format(metrics.precision_score(y_test, y_pred)))
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Random Forest with all columns as features")

print("F1 score is {}".format(f1_score(y_test, y_pred)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("AUC is {}".format(metrics.auc(fpr, tpr)))

print("Recall is {}".format(metrics.recall_score(y_test, y_pred)))

print("Precision is {}".format(metrics.precision_score(y_test, y_pred)))
## For over and under sampling

from imblearn.over_sampling import RandomOverSampler , SMOTE

from imblearn.under_sampling import RandomUnderSampler
#rus = RandomOverSampler(sampling_strategy=0.01)

rus = SMOTE(sampling_strategy=0.01)

X_res,y_res=rus.fit_resample(X_train, y_train)

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_res:', np.bincount(y_res))
model = LogisticRegression()

model.fit(X_res, y_res)

y_pred = model.predict(X_test)

print("Logisitic Regression with Random Sampling and sampling strategy and all columns as features")

print("F1 score is {}".format(f1_score(y_test, y_pred)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("AUC is {}".format(metrics.auc(fpr, tpr)))

print("Recall is {}".format(metrics.recall_score(y_test, y_pred)))

print("Precision is {}".format(metrics.precision_score(y_test, y_pred)))
clf = RandomForestClassifier()

clf.fit(X_res, y_res)

y_pred = clf.predict(X_test)

print("Random Forest with Random Sampling and sampling strategy and all columns as features")

print("F1 score is {}".format(f1_score(y_test, y_pred)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("AUC is {}".format(metrics.auc(fpr, tpr)))

print("Recall is {}".format(metrics.recall_score(y_test, y_pred)))

print("Precision is {}".format(metrics.precision_score(y_test, y_pred)))
### Hyper parameter tuning



clf.get_params()

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'log2']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
''' **** CAUTION - can take 10 hrs to run

Either reduce number of parameters or no of values you want the gridsearch to go through



rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_res, y_res)

'''
#rf_random.best_params_
clf = RandomForestClassifier(n_estimators= 1000,min_samples_split= 2,min_samples_leaf= 1,max_features= 'auto',max_depth= 50,bootstrap= True)

clf.fit(X_res, y_res)

#clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Random Forest with all columns as features")

print("F1 score is {}".format(f1_score(y_test, y_pred)))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

print("AUC is {}".format(metrics.auc(fpr, tpr)))

print("Recall is {}".format(metrics.recall_score(y_test, y_pred)))

print("Precision is {}".format(metrics.precision_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
y_prob = clf.fit(X_res, y_res).predict_proba(X_test)[::,1]

#print (y_prob)

fpr, tpr, _ = metrics.roc_curve(y_test,  y_prob)

auc = metrics.roc_auc_score(y_test, y_prob)

plt.plot(fpr,tpr,label="data 1 + auc "+str(auc))

plt.legend(loc=4)

plt.show()