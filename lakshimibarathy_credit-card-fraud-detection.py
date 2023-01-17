import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import fun_py as fp

%matplotlib inline



train= pd.read_csv("../input/creditc-fraud/credit_train.csv")

test= pd.read_csv("../input/creditc-fraud/credit_test.csv")
train["source"] = "train"

test["source"] = "test"

print("Train Data Shape aftr adding target col : ",train.shape)

print("Test Data Shape aftr adding target col : ",test.shape)

df = pd.concat([train,test])
my_colors = ['blue', 'red','yellow']  #red, green, blue, black, etc.



count_classes = pd.value_counts(df['Class'], sort = True).sort_index()



count_classes.plot(kind = 'pie')

#count_classes[1].set_color('r')



plt.title("LABEL : CLASS")

count_classes.plot(kind = 'bar',color=my_colors)

plt.xlabel("Class Labels")

plt.ylabel("Count")
print("Lables Counts of Column 'Class' \n")



print("Lable Counts")

print(fp.data_value_counts(df,'Class'))
fp.data_duplicates(df,0)

print("Dropping Dupicates")

df.drop_duplicates(inplace=True)
fp.data_duplicates(df,0)

fp.data_isna(df)
from sklearn.preprocessing import StandardScaler
df['sAmt'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))



df.head()
df.drop('Amount',axis=1,inplace=True)

df.head()
df['Class'].replace("'0'", "0",inplace=True)

df['Class'].replace("'1'", "1",inplace=True)

df.head()

df['Class']=df['Class'].apply(lambda x : int(x))
#data_corr_trg_col(df,'Class')



cols=abs(df.corr()['Class'].sort_values())

#cols

drop_cols=[]

drop_cols
#for i ,j in cols.items():

#    if j < .09:

[drop_cols.append(i) for i,j in cols.items() if j <.09]
drop_cols
data=df.copy()

data.drop(columns=drop_cols,axis=1)
pre_train=data[data['source']=='train']

pre_test=data[data['source']=='test']
#X.head()
X = df.loc[:, df.columns != 'Class']

X=X.loc[:, X.columns != 'source']

y = df.loc[:, df.columns == 'Class']
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
import warnings

warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(fp.data_value_counts(y_train,'Class').values[1]))

print("Before OverSampling, counts of label '0': {} \n".format(fp.data_value_counts(y_train,'Class').values[0]))
pd.options.display.max_rows = 40000
y_train["Class"]= y_train["Class"].replace("'0'", "0")

y_train["Class"]= y_train["Class"].replace("'1'", "1")
y_train['Class']=y_train['Class'].apply(lambda x : int(x))
sm = SMOTE(random_state=1)
X_train_samp, y_train_samp = sm.fit_sample(X_train, y_train.values.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_samp.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_samp.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_samp==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_samp==0)))
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}



classifier = {

    "LogisiticRegression": LogisticRegression(random_state=0)

}
import time



from sklearn.model_selection import cross_val_score



for key, classifier in classifier.items():

    t0 = time.time()

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train_samp, y_train_samp, cv=5)

    t1 = time.time()

    #print("Classifier :" ,classifier.__class__.__name__," Taken {:} s to Execute".format(abs(t1 - t0)))

    print("Classifier: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV



# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train_samp, y_train_samp)



# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_



print("Done with LR and the paramts are : ", log_reg)



# KNears Classifier

'''

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)



# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



print("Done with KNC and the paramts are : ", knears_neighbors)

'''

# Support Vector Classifier

'''svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_

'''

# DecisionTree Classifier

'''tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_



print("Done with DTC and the paramts are : ", tree_clf)

'''
log_reg_score = cross_val_score(log_reg, X_train_samp, y_train_samp, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_predict

# Create a DataFrame with all the scores and the classifiers names.



log_reg_pred = cross_val_predict(log_reg, X_train_samp, y_train_samp, cv=5)

                             
from sklearn.metrics import roc_auc_score



print('Logistic Regression: ', roc_auc_score(y_train_samp, log_reg_pred))
#X_train_samp, y_train_samp
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

mLoR=LogisticRegression(C=100)

mLoR.fit(X_train_samp, y_train_samp)

test_predict = mLoR.predict(X_test)

metrics.confusion_matrix(y_test,test_predict)
metrics.accuracy_score(y_test,test_predict)*100
from sklearn.metrics import classification_report

print(classification_report(y_test, test_predict))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, mLoR.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, mLoR.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()