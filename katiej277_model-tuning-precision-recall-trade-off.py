import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score,make_scorer



import matplotlib.pyplot as plt



heart = pd.read_csv('../input/heart.csv')



heart.shape
# create X (predictors) and y (target)

X = heart.iloc[:,:-1]

y = heart.iloc[:,-1]





# create train test splits - random state sets seed so can replicate, whilst stratify maintains class distribution when splitting

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,stratify = y)



# scale variables, applying scale of X_train to X_test to avoid data leakage

modelscaler = MinMaxScaler()

X_train = modelscaler.fit_transform(X_train)

X_test = modelscaler.transform(X_test)



# can see distribution of target value is v similar in both training and test sets

print(y_train.value_counts(normalize = True))

print(y_test.value_counts(normalize = True))
X
# run logisitc regression with 10 fold cross validation



log_model = LogisticRegressionCV(cv = 10,max_iter = 4000).fit(X_train,y_train)



log_model.score(X_test,y_test)
preds = log_model.predict(X_test)



pd.DataFrame(confusion_matrix(preds,y_test))
print('Recall score\n')

print(recall_score(y_test,preds))

print('\nPrecision score: \n')

print(precision_score(y_test,preds))

print('\nAccuracy score \n')

print(accuracy_score(y_test,preds))

precision,recall,thresholds = precision_recall_curve(y_test,preds)



plt.figure()

plt.plot(precision,recall)

plt.title('Precision-Recall Curve \nPrecision and Recall at each value of decision boundary')

plt.xlabel('Precision')

_ = plt.ylabel('Recall')
yprobs = log_model.predict_proba(X_test)

yprobs = pd.DataFrame(yprobs)

precision = []

recall = []

accuracy = []



# calculate precision and accuracy at each value of threshold - only to 0.99 because no true values are predicted at 100% confidence!

for t in np.arange(0,0.99,0.01):

    tmp = yprobs

    tmp['target'] = np.where(yprobs.iloc[:,1] >= t,1,0)

    

    precision.append(precision_score(y_test,tmp['target']))

    recall.append(recall_score(y_test,tmp['target']))

    accuracy.append(accuracy_score(y_test,tmp['target']))



plt.figure()

plt.plot(precision,recall)

plt.title('Precision-Recall Curve \nPrecision and Recall at each value of decision boundary')

plt.xlabel('Precision')

_ = plt.ylabel('Recall')
plt.figure()

plt.plot(np.arange(0,0.99,0.01),recall,label = 'Recall')

plt.plot(np.arange(0,0.99,0.01),precision,label = 'Precision')

plt.plot(np.arange(0,0.99,0.01),accuracy,label = 'Accuracy')

plt.title('Precision, Recall and Accuracy \n(For each value of decision boundary')

plt.xlabel('Decision Boundary')

plt.ylabel('Score')

_= plt.legend()
fpr,tpr,_ = roc_curve(y_test,preds,drop_intermediate = True)



plt.figure()

plt.plot(fpr,tpr)

roc_auc_score(y_test,preds)

fpr = []

tpr = recall



for t in np.arange(0,0.99,0.01):

    yprobs['target'] = np.where(yprobs.iloc[:,1] >=t,1,0)

    cm = confusion_matrix(yprobs['target'],y_test)

    tp = cm[1,1]

    fp = cm[1,0]

    tn = cm[0,0]

    fn = cm[0,1]

    

    fpr.append(fp/(tn+fp))
plt.figure()



plt.plot(fpr,tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

_ = plt.title('ROC curve')

print('AUC (area under curve) Score: {}'.format(roc_auc_score(y_test,preds)))
from sklearn.svm import SVC



# train model on training data selecting arbitrary gamma value = 0.1

svm_model = SVC(kernel = 'rbf',gamma =0.1).fit(X_train,y_train)



# predict on test data

preds = svm_model.predict(X_test)



print('Recall score:')

print(recall_score(y_test,preds))



print('\nPrecision score:')

print(precision_score(y_test,preds))



print('\nAccuracy score:')

print(accuracy_score(y_test,preds))
from sklearn.model_selection import GridSearchCV



gamma_values = {'gamma' : [0.01,0.1,0.5,1,5,10]}



svm = SVC(kernel = 'rbf')



gridsearch_recall = GridSearchCV(svm,param_grid = gamma_values,cv = 10,scoring = 'recall')

gridsearch_recall.fit(X_train,y_train)



gridsearch_precision = GridSearchCV(svm,param_grid = gamma_values,cv = 10,scoring = 'precision')

gridsearch_precision.fit(X_train,y_train)



gridsearch_accuracy = GridSearchCV(svm,param_grid = gamma_values, cv = 10,scoring = 'accuracy')

gridsearch_accuracy.fit(X_train,y_train)





print('Recall scoring:\n Best Gamma: {}\n Best Score: {}'.format(gridsearch_recall.best_params_,gridsearch_recall.best_score_))

print('Precision scoring:\n Best Gamma: {}\n Best Score: {}'.format(gridsearch_precision.best_params_,gridsearch_precision.best_score_))

print('Accuracy scoring:\n Best Gamma: {}\n Best Score: {}'.format(gridsearch_accuracy.best_params_,gridsearch_accuracy.best_score_))
preds = gridsearch_recall.predict(X_train)



print('Recall score: {}\nPrecision Score: {}\nAccuracy Score: {}'.format(recall_score(y_train,preds),precision_score(y_train,preds),accuracy_score(y_train,preds)))