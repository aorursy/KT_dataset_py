import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt, matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

N_IMAGES_PER_CLASS = 100 # Number of images to use: change this value for longer/shorter times higher/lower accuracy
SEED = 22
DIGIT_0 = 3
DIGIT_1 = 8
data = pd.read_csv('../input/train.csv')
sns.countplot(x = 'label', data = data)
# By grouping on the label and then sampling, we can keep sample equally among both labels. 
data01 = data[data.label.isin([DIGIT_0, DIGIT_1])] 
data01 = data01.groupby('label').apply(lambda x: x.sample(n=N_IMAGES_PER_CLASS, replace=False, random_state = SEED))
X = data01.iloc[:,1:]
y = data01.label
sns.countplot(x = data01.label, data = data01)
i=1
img = data01.iloc[i,1:].values.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(data01.iloc[i,0])
sns.distplot(X.values.reshape(1,-1))
# To make it work as binary for digits other than 1
y.update(pd.Series(0, index = y[y == DIGIT_0].index))
y.update(pd.Series(1, index = y[y == DIGIT_1].index))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=SEED, stratify = y)

data01_uneven = pd.concat([data01[data01.label == 0], data01[data01.label == 1].iloc[:round(N_IMAGES_PER_CLASS*0.2/0.8)]])
X_utrain, X_utest, y_utrain, y_utest = train_test_split(data01_uneven.iloc[:,1:], data01_uneven.label, 
                                                        test_size=0.8, stratify = data01_uneven.label ,random_state=SEED)
lr = LogisticRegression(solver = 'liblinear', random_state = SEED)
lr.fit(X_train, y_train)
lr_bal_score = lr.score(X_test, y_test)
print("Score on Train data:",lr.score(X_train, y_train), "Score on Test data =", lr_bal_score)
y_p = lr.predict(X_test)
y_pred = lr.predict_proba(X_test)
fpr, tpr, thresholds  = metrics.roc_curve(y_test, y_pred[:,1])
lr_roc_auc = metrics.roc_auc_score(y_test, y_pred[:,1])
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
cm_lr = metrics.confusion_matrix(y_test, y_p)
print(cm_lr)
print("TP:", cm_lr[1,1]/(cm_lr[0,1]+cm_lr[1,1]), "FP:", cm_lr[0,1]/(cm_lr[0,0]+cm_lr[0,1]))
sns.countplot(x = y_utrain)
lr.fit(X_utrain, y_utrain)
lr_unbal_score = lr.score(X_utest, y_utest)
print("Score on Train data:",lr.score(X_utrain, y_utrain), "Score on Test data =", lr_unbal_score)
y_p = lr.predict(X_utest)
y_pred = lr.predict_proba(X_utest)
fpr, tpr, thresholds  = metrics.roc_curve(y_utest, y_pred[:,1])
lr_unbal_roc_auc = metrics.roc_auc_score(y_utest, y_pred[:,1])
cm_lr = metrics.confusion_matrix(y_utest, y_p)
print(cm_lr)
print("TP:", cm_lr[1,1]/(cm_lr[0,1]+cm_lr[1,1]), "FP:", cm_lr[0,1]/(cm_lr[0,0]+cm_lr[0,1]), "AUC:", lr_unbal_roc_auc)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_bal_score = lr.score(X_test, y_test)
print("Score on Train data:",lda.score(X_train, y_train), "Score on Test data =", lda_bal_score)
y_p = lda.predict(X_test)
y_pred = lda.predict_proba(X_test)
fpr, tpr, thresholds  = metrics.roc_curve(y_test, y_pred[:,1])
lda_roc_auc = metrics.roc_auc_score(y_test, y_pred[:,1])
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
cm_lda = metrics.confusion_matrix(y_test, y_p)
print(cm_lda)
print("TP:", cm_lda[1,1]/(cm_lda[0,1]+cm_lda[1,1]), "FP:", cm_lda[0,1]/(cm_lda[0,0]+cm_lda[0,1]))
lda.fit(X_utrain, y_utrain)
lda_unbal_score = lr.score(X_utest, y_utest)
print("Score on Train data:",lda.score(X_train, y_train), "Score on Test data =", lda_unbal_score)
y_p = lda.predict(X_utest)
y_pred = lda.predict_proba(X_utest)
fpr, tpr, thresholds  = metrics.roc_curve(y_utest, y_pred[:,1])
lda_unbal_roc_auc = metrics.roc_auc_score(y_utest, y_pred[:,1])
cm_lda = metrics.confusion_matrix(y_utest, y_p)
print(cm_lda)
print("TP:", cm_lda[1,1]/(cm_lda[0,1]+cm_lda[1,1]), "FP:", cm_lda[0,1]/(cm_lda[0,0]+cm_lda[0,1]), "AUC:", lda_unbal_roc_auc)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
qda_bal_score = qda.score(X_test, y_test)
y_p = qda.predict(X_test)
y_pred = qda.predict_proba(X_test)
fpr, tpr, thresholds  = metrics.roc_curve(y_test, y_pred[:,1])
qda_roc_auc = metrics.roc_auc_score(y_test, y_pred[:,1])
print("Score on Train data:",qda.score(X_train, y_train), "Score on Test data =", qda_bal_score, "AUC:",qda_roc_auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
cm_qda = metrics.confusion_matrix(y_test, y_p)
print(cm_qda)
print("TP:", cm_qda[1,1]/(cm_qda[0,1]+cm_qda[1,1]), "FP:", cm_qda[0,1]/(cm_qda[0,0]+cm_qda[0,1]))
qda.fit(X_utrain, y_utrain)
qda_unbal_score = lr.score(X_utest, y_utest)
print("Score on Train data:",qda.score(X_train, y_train), "Score on Test data =", qda_unbal_score)
y_p = qda.predict(X_utest)
y_pred = qda.predict_proba(X_utest)
fpr, tpr, thresholds  = metrics.roc_curve(y_utest, y_pred[:,1])
qda_unbal_roc_auc = metrics.roc_auc_score(y_utest, y_pred[:,1])
cm_qda = metrics.confusion_matrix(y_utest, y_p)
print(cm_qda)
print("TP:", cm_qda[1,1]/(cm_qda[0,1]+cm_qda[1,1]), "FP:", cm_qda[0,1]/(cm_qda[0,0]+cm_qda[0,1]), "AUC:", qda_unbal_roc_auc)
# subsetting just the odd ones so that voting is always determining
neighbors = range(1,20,2)

# empty list that will hold cv scores
cv_scores = []
y_pred = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='roc_auc')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is",optimal_k, "with an AUC score of", scores[optimal_k])

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_bal_score = knn.score(X_test, y_test)
y_p = knn.predict(X_test)
y_pred = knn.predict_proba(X_test)
fpr, tpr, thresholds  = metrics.roc_curve(y_test, y_pred[:,1])
knn_roc_auc = metrics.roc_auc_score(y_test, y_pred[:,1])
print("Score on Train data:",knn.score(X_train, y_train), "Score on Test data =", knn_bal_score, "AUC:",knn_roc_auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
cm_knn = metrics.confusion_matrix(y_test, y_p)
print(cm_knn)
print("TP:", cm_knn[1,1]/(cm_knn[0,1]+cm_knn[1,1]), "FP:", cm_knn[0,1]/(cm_knn[0,0]+cm_knn[0,1]))
# subsetting just the odd ones so that voting is always determining
neighbors = range(1,6,2)

# empty list that will hold cv scores
cv_scores = []
y_pred = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_utrain, y_utrain, cv=5, scoring='roc_auc')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is",optimal_k, "with an AUC score of", scores[optimal_k])

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_utrain, y_utrain)
knn_unbal_score = knn.score(X_utest, y_utest)
y_p = knn.predict(X_utest)
y_pred = knn.predict_proba(X_utest)
fpr, tpr, thresholds  = metrics.roc_curve(y_utest, y_pred[:,1])
knn_unbal_roc_auc = metrics.roc_auc_score(y_utest, y_pred[:,1])
print("Score on Train data:",knn.score(X_utrain, y_utrain), "Score on Test data =", knn_unbal_score, "AUC:",knn_unbal_roc_auc)
results = pd.DataFrame(data = {'Balanced Accuracy':[], 'Balanced AUC':[], 'Unbalanced Acc':[], 'Unbalanced AUC':[]})
results = results.append(pd.Series(name = 'Logistic Regression', data = [lr_bal_score, lr_roc_auc, lr_unbal_score, lr_unbal_roc_auc], index = results.columns))
results = results.append(pd.Series(name = 'LDA', data = [lda_bal_score, lda_roc_auc, lda_unbal_score, lda_unbal_roc_auc], index = results.columns))
results = results.append(pd.Series(name = 'QDA', data = [qda_bal_score, qda_roc_auc, qda_unbal_score, qda_unbal_roc_auc], index = results.columns))
results = results.append(pd.Series(name = 'kNN', data = [knn_bal_score, knn_roc_auc, knn_unbal_score, knn_unbal_roc_auc], index = results.columns))
results