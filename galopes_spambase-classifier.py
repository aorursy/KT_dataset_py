import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.listdir('../input/spamdatabase')
train = pd.read_csv('../input/spamdatabase/train_data.csv', index_col='Id')
train.head()
test = pd.read_csv('../input/spamdatabase/test_features.csv', index_col='Id')
test.head()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
x_train = train.drop('ham', axis=1)
y_train = train.ham
from sklearn.model_selection import cross_val_score
nb_score = cross_val_score(gnb, x_train, y_train, cv=10)
nb_score.mean()
gnb.fit(x_train, y_train)
predictions_gnb = gnb.predict(test)
results = np.vstack((test.index, predictions_gnb)).T
cols = ['Id', 'ham']
df_pred_gnb = pd.DataFrame(columns=cols ,data=results)
df_pred_gnb.set_index('Id', inplace=True)
df_pred_gnb.to_csv('predictions_gnb.csv')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn_score = cross_val_score(knn, x_train, y_train, cv=10)
knn_score.mean()
knn.fit(x_train, y_train)
predictions_knn = knn.predict(test)
results = np.vstack((test.index, predictions_knn)).T
cols = ['Id', 'ham']
df_pred_knn = pd.DataFrame(columns=cols ,data=results)
df_pred_knn.set_index('Id', inplace=True)
df_pred_knn.to_csv('predictions_knn.csv')
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
from scipy import interp
cv = StratifiedKFold(n_splits=10, shuffle=False)
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
f_betas = []
i = 1
for train_cv,test_cv in cv.split(x_train,y_train):
    prediction = gnb.fit(x_train.iloc[train_cv], y_train.iloc[train_cv]).predict(x_train.iloc[test_cv])
    fpr, tpr, t = roc_curve(y_train.iloc[test_cv], prediction)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    f_betas.append(fbeta_score(y_train.iloc[test_cv], prediction, beta=3))
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
f_betas = np.array(f_betas)
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
f_betas.mean()
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb_score = cross_val_score(bnb, x_train, y_train, cv=10)
bnb_score.mean()
bnb.fit(x_train, y_train)
predictions_bnb = bnb.predict(test)
results = np.vstack((test.index, predictions_bnb)).T
cols = ['Id', 'ham']
df_pred_bnb = pd.DataFrame(columns=cols ,data=results)
df_pred_bnb.set_index('Id', inplace=True)
df_pred_bnb.to_csv('predictions_bnb_1.csv')
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb_score = cross_val_score(mnb, x_train, y_train, cv=10)
mnb_score.mean()
mnb.fit(x_train, y_train)
predictions_mnb = mnb.predict(test)
results = np.vstack((test.index, predictions_mnb)).T
cols = ['Id', 'ham']
df_pred_mnb = pd.DataFrame(columns=cols ,data=results)
df_pred_mnb.set_index('Id', inplace=True)
df_pred_mnb.to_csv('predictions_mnb.csv')
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
f_betas = []
i = 1
for train_cv,test_cv in cv.split(x_train,y_train):
    prediction = bnb.fit(x_train.iloc[train_cv], y_train.iloc[train_cv]).predict(x_train.iloc[test_cv])
    fpr, tpr, t = roc_curve(y_train.iloc[test_cv], prediction)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    f_betas.append(fbeta_score(y_train.iloc[test_cv], prediction, beta=3))
    i= i+1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
f_betas = np.array(f_betas)
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.text(0.32,0.7,'More accurate area',fontsize = 12)
plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()
f_betas.mean()