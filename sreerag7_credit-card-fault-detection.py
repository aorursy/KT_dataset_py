

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = data['Amount'].values
time_val = data['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))

data.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

data.head()
data.isnull().sum().sum()
data.describe()
plt.figure()
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=1,cmap="Reds")
plt.show()
var = data.columns.values

i = 0
t0 = data.loc[data['Class'] == 0]
t1 = data.loc[data['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
X=data.drop('Class',axis=1)
y=data['Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC # "Support Vector Classifier" 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
#gini_predictions = gini(y_test, y_pred)
#gini_max = gini(y_test, y_pred)
#ngini= gini_normalized(y_test, y_pred)

logreg = LogisticRegression(max_iter=150)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log=accuracy_score(y_test,y_pred)
prec_log,recall_log,f1_log,support_log=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_log=log_loss(y_test,y_pred)
kappa_log=cohen_kappa_score(y_test, y_pred)
ngini_log= gini_normalized(y_test, y_pred)

prob_log = logreg.predict_proba(X_test)
fpr_log, tpr_log, thresh_log = roc_curve(y_test, prob_log[:,1], pos_label=1)
random_probs_log = [0 for i in range(len(y_test))]
p_fpr_log, p_tpr_log, _ = roc_curve(y_test, random_probs_log, pos_label=1)
plt.title('ROC Logistic Regression')
plt.plot(fpr_log,tpr_log)
plt.show()
auc_log = roc_auc_score(y_test, prob_log[:,1])

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_nb=accuracy_score(y_test,y_pred)
prec_nb,recall_nb,f1_nb,support_nb=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_nb=log_loss(y_test,y_pred)
kappa_nb=cohen_kappa_score(y_test, y_pred)
ngini_nb= gini_normalized(y_test, y_pred)
prob_nb = gaussian.predict_proba(X_test)
fpr_nb, tpr_nb, thresh_nb = roc_curve(y_test, prob_nb[:,1], pos_label=1)
random_probs_nb = [0 for i in range(len(y_test))]
p_fpr_nb, p_tpr_nb, _ = roc_curve(y_test, random_probs_nb, pos_label=1)
plt.title('ROC Naive Bayes Classifier')
plt.plot(fpr_nb,tpr_nb)
plt.show()
auc_nb = roc_auc_score(y_test, prob_nb[:,1])
sgd = SGDClassifier(loss='log')
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd=accuracy_score(y_test,y_pred)
prec_sgd,recall_sgd,f1_sgd,support_sgd=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_sgd=log_loss(y_test,y_pred)
kappa_sgd=cohen_kappa_score(y_test, y_pred)
ngini_sgd= gini_normalized(y_test, y_pred)
prob_sgd = sgd.predict_proba(X_test)
fpr_sgd, tpr_sgd, thresh_sgd = roc_curve(y_test, prob_sgd[:,1], pos_label=1)
random_probs_sgd = [0 for i in range(len(y_test))]
p_fpr_sgd, p_tpr_sgd, _ = roc_curve(y_test, random_probs_sgd, pos_label=1)
plt.title('ROC Stochastic Gradient Descent')
plt.plot(fpr_sgd,tpr_sgd)
plt.show()
auc_sgd = roc_auc_score(y_test, prob_sgd[:,1])
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_dt=accuracy_score(y_test,y_pred)
prec_dt,recall_dt,f1_dt,support_dt=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_dt=log_loss(y_test,y_pred)
kappa_dt=cohen_kappa_score(y_test, y_pred)
ngini_dt= gini_normalized(y_test, y_pred)
prob_dt = decision_tree.predict_proba(X_test)
fpr_dt, tpr_dt, thresh_dt = roc_curve(y_test, prob_dt[:,1], pos_label=1)
random_probs_dt = [0 for i in range(len(y_test))]
p_fpr_dt, p_tpr_dt, _ = roc_curve(y_test, random_probs_dt, pos_label=1)
plt.title('ROC Decision Tree Classifier')
plt.plot(fpr_dt,tpr_dt)
plt.show()
auc_dt = roc_auc_score(y_test, prob_dt[:,1])
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
acc_rf=accuracy_score(y_test,y_pred)
prec_rf,recall_rf,f1_rf,support_rf=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_rf=log_loss(y_test,y_pred)
kappa_rf=cohen_kappa_score(y_test, y_pred)
ngini_rf= gini_normalized(y_test, y_pred)
prob_rf = random_forest.predict_proba(X_test)
fpr_rf ,tpr_rf, thresh_rf = roc_curve(y_test, prob_rf[:,1], pos_label=1)
random_probs_rf = [0 for i in range(len(y_test))]
p_fpr_rf, p_tpr_rf, _ = roc_curve(y_test, random_probs_rf, pos_label=1)
plt.title('ROC Random Forest Classifier')
plt.plot(fpr_rf,tpr_rf)
plt.show()
auc_rf = roc_auc_score(y_test, prob_rf[:,1])
svm=SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc_svm=accuracy_score(y_test,y_pred)
prec_svm,recall_svm,f1_svm,support_svm=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_svm=log_loss(y_test,y_pred)
kappa_svm=cohen_kappa_score(y_test, y_pred)
ngini_svm= gini_normalized(y_test, y_pred)
prob_svm = svm.predict_proba(X_test)
fpr_svm, tpr_svm, thresh_svm = roc_curve(y_test, prob_svm[:,1], pos_label=1)
random_probs_svm = [0 for i in range(len(y_test))]
p_fpr_svm, p_tpr_svm, _ = roc_curve(y_test, random_probs_svm, pos_label=1)
plt.title('ROC Support Vector Machines')
plt.plot(fpr_svm,tpr_svm)
plt.show()
auc_svm = roc_auc_score(y_test, prob_svm[:,1])
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn=accuracy_score(y_test,y_pred)
prec_knn,recall_knn,f1_knn,support_knn=precision_recall_fscore_support(y_test, y_pred, average='weighted')
log_knn=log_loss(y_test,y_pred)
kappa_knn=cohen_kappa_score(y_test, y_pred)
ngini_knn= gini_normalized(y_test, y_pred)
prob_knn = knn.predict_proba(X_test)
fpr_knn ,tpr_knn, thresh_knn = roc_curve(y_test, prob_knn[:,1], pos_label=1)
random_probs_knn = [0 for i in range(len(y_test))]
p_fpr_knn, p_tpr_knn, _ = roc_curve(y_test, random_probs_knn, pos_label=1)
plt.title('ROC K-Nearest Neighbours')
plt.plot(fpr_knn,tpr_knn)
plt.show()
auc_knn = roc_auc_score(y_test, prob_knn[:,1])
models = pd.DataFrame({
    'Model': ['Logistic Regression','Naive Bayes','Stochastic Gradient Decent','Support Vector Machines','Decision Tree', 'Random Forest','KNN'],
    'Accuracy': [acc_log,acc_nb,acc_sgd,acc_svm,acc_dt,acc_rf,acc_knn],
    'Precision': [prec_log,prec_nb,prec_sgd,prec_svm,prec_dt,prec_rf,prec_knn],
    'Recall': [recall_log,recall_nb,recall_sgd,recall_svm,recall_dt,recall_rf,recall_knn],
    'F1 Score':[f1_log,f1_nb,f1_sgd,f1_svm,f1_dt,f1_rf,f1_knn],
    'Log Loss': [log_log,log_nb,log_sgd,log_svm,log_dt,log_rf,log_knn],
    'Normalized GINI Score': [ngini_log,ngini_nb,ngini_sgd,ngini_svm,ngini_dt,ngini_rf,ngini_knn],
    'Kappa Score': [kappa_log,kappa_nb,kappa_sgd,kappa_svm,kappa_dt,kappa_rf,kappa_knn],
    'AUC Score': [auc_log,auc_nb,auc_sgd,auc_svm,auc_dt,auc_rf,auc_knn]
})
models