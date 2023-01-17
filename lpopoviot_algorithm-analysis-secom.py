import os

import time

import glob

import calendar

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats

%matplotlib inline
path = '../input/'



all_files = glob.glob(os.path.join(path, "*.data")) 



df_data = pd.read_csv('../input/secom.data', sep="\s+",header=None)

df_target = pd.read_csv('../input/secom_labels.data', sep="\s+",header=None)



df_data.head()
df_target.head()
df_data.shape, df_target.shape
df_target.iloc[:,1]



# Drop date cols

df_target.drop([1], axis=1,inplace=True)

df_target.rename(columns = {list(df_target)[0]:'Target'}, inplace=True)
df_target.head()
# Convert all Fails(1) to -1 

for i, val in enumerate(df_target.Target):

    if val == 1:

        df_target.iloc[i,0] = 0



for i, val in enumerate(df_target.Target):

    if val == -1:

        df_target.iloc[i,0] = 1

df_target.head() 
plt.figure()

sns.countplot(df_target.Target)

plt.title('Distribution of Pass/Fail')

plt.show()
df = pd.concat([df_data, df_target], axis=1)
correlation = df.corr()



f, ax = plt.subplots(figsize=(20,10))

plt.title('Correlations in dataset', size=20)

sns.heatmap(correlation)

plt.show()
missing_val_count_by_column = (df_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.impute import SimpleImputer



imputer = SimpleImputer()

df_data_imputed = imputer.fit_transform(df_data)
missing_val_count_by_column = (pd.DataFrame(df_data_imputed).isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Scale values

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df_data_imputed)
from sklearn.decomposition import PCA

pca = PCA()

X_scaled_pca = pca.fit(X_scaled)
plt.figure(figsize=(10, 4))

axes= plt.axes()

axes.grid()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.title('Cumulative variance of principle components')

plt.xticks(range(0,530,25))

plt.xlim(-1,525)

plt.yticks(np.arange(0,1.1,0.1))

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')

plt.tight_layout()
X = df_data

y = df_target



n_classes = y.shape[1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)





# Impute missing values

imputer = SimpleImputer()

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)

# Scale

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

y_test = y_test.values
X_train.shape,y_train.shape,y_test.shape
def plot_roc(model,y_test,y_pred,title,label_auc):

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    plt.figure()

    plt.plot(fpr, tpr, label= label_auc+' (area = %0.2f)' % logit_roc_auc)

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(title)

    plt.legend(loc="lower right")

    plt.savefig('Log_ROC')

    plt.show()
def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):

    auc_differences = []

    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())

    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())

    observed_difference = auc1 - auc2

    for _ in range(nsamples):

        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))

        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())

        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())

        auc1 = roc_auc_score(y_test.ravel(), p1)

        auc2 = roc_auc_score(y_test.ravel(), p2)

        auc_differences.append(auc1 - auc2)

    return print("difference in roc curves: {0:.4f} \nprobability to observe a larger difference on a shuffled data set: {1}".format(observed_difference, np.mean(auc_differences >= observed_difference)))

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from scipy import interp



start = time.time()



classifier = LogisticRegression(solver='sag', max_iter=4000, n_jobs=-1)

classifier.fit(X_train, np.ravel(y_train))

y_pred1 = classifier.predict(X_test)



end = time.time()

print("\nTime taken: {:.2f} seconds".format(end-start))
confusion_matrix = metrics.confusion_matrix(y_test,y_pred1)

confusion_matrix
from sklearn.metrics import classification_report

auc_roc = metrics.classification_report(y_test,y_pred1)

print('Logistic Regression Classification Report:\n {}'.format(auc_roc))
plot_roc(classifier,y_test, y_pred1, 'ROC Logistic Regression','Logistic Regression')
from sklearn.svm import SVC



start = time.time()



classifier = SVC(kernel="linear", probability=True, verbose=1)

classifier.fit(X_train, np.ravel(y_train))

y_pred2 = classifier.predict(X_test)



end = time.time()

print("\nTime taken: {:.2f} seconds".format(end-start))
confusion_matrix = metrics.confusion_matrix(y_test,y_pred2)

confusion_matrix
auc_roc = metrics.classification_report(y_test, y_pred2)

print('SVC Classification Report:\n {}'.format(auc_roc))
plot_roc(classifier,y_test, y_pred2, 'ROC SVC','SVC')


permutation_test_between_clfs(y_test, y_pred1, y_pred2, nsamples=1000)
from sklearn.feature_selection import RFE



start = time.time()



classifier = LogisticRegression(solver='sag', max_iter=4000, n_jobs=-1)

rfe = RFE(classifier,verbose=1,step=30)

rfe = rfe.fit(X_train, np.ravel(y_train))



end = time.time()

print("\nTime taken: {:.2f} seconds".format(end-start))
# Create new X train and test with est features from RFE

features = X.columns[rfe.support_]

print(features)

X_train_rfe = pd.DataFrame(X_train)[features]

X_test_rfe = pd.DataFrame(X_test)[features]
classifier = LogisticRegression(solver='sag', max_iter=4000, n_jobs=-1)

classifier.fit(X_train_rfe, np.ravel(y_train))

y_pred11 = classifier.predict(X_test_rfe)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred11)

confusion_matrix
auc_roc = metrics.classification_report(y_test,y_pred11)

print('Logistic Regression with RFE Classification Report:\n {}'.format(auc_roc))
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test_rfe))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_rfe)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression w/RFE (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Logistic Regression with RFE')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
permutation_test_between_clfs(y_test, y_pred1, y_pred11, nsamples=1000)
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import chi2 ,SelectKBest

norm = MinMaxScaler()



# Normalise training data 

X_train_norm = norm.fit_transform(X_train)
selector = SelectKBest(chi2, k=295)

selector.fit(X_train_norm, y_train)

X_train_kbest = selector.transform(X_train)

X_test_kbest = selector.transform(X_test)
classifier = LogisticRegression(solver='sag', max_iter=4000, n_jobs=-1)

classifier.fit(X_train_kbest, np.ravel(y_train))

y_pred12 = classifier.predict(X_test_kbest)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred12)

confusion_matrix
auc_roc = metrics.classification_report(y_test,y_pred12)

print('Logistic Regression with chi2 test Classification Report:\n {}'.format(auc_roc))
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test_kbest))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_kbest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression w/chi2 (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Logistic Regression with chi2 test')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
permutation_test_between_clfs(y_test, y_pred1, y_pred12, nsamples=1000)
start = time.time()



classifier = SVC(kernel="linear", probability=True, verbose=1)

rfe = RFE(classifier,verbose=1,step=30)

rfe = rfe.fit(X_train, np.ravel(y_train))



end = time.time()

print("\nTime taken: {:.2f} seconds".format(end-start))
# Create new X train and test with est features from RFE

features = X.columns[rfe.support_]

print(features)

X_train_rfe = pd.DataFrame(X_train)[features]

X_test_rfe = pd.DataFrame(X_test)[features]
classifier = SVC(kernel="linear", probability=True, verbose=1)

classifier.fit(X_train_rfe, np.ravel(y_train))

y_pred21 = classifier.predict(X_test_rfe)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred21)

confusion_matrix
auc_roc = metrics.classification_report(y_test,y_pred21)

print('SVC with RFE Classification Report:\n {}'.format(auc_roc))
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test_rfe))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_rfe)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='SVC w/RFE (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('SVC with RFE')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
permutation_test_between_clfs(y_test, y_pred2, y_pred21, nsamples=1000)
norm = MinMaxScaler()



# Normalise training data 

X_train_norm = norm.fit_transform(X_train)
selector = SelectKBest(chi2, k=295)

selector.fit(X_train_norm, y_train)

X_train_kbest = selector.transform(X_train)

X_test_kbest = selector.transform(X_test)
classifier = SVC(kernel="linear", probability=True, verbose=1)

classifier.fit(X_train_kbest, np.ravel(y_train))

y_pred22 = classifier.predict(X_test_kbest)
confusion_matrix = metrics.confusion_matrix(y_test,y_pred22)

confusion_matrix
auc_roc = metrics.classification_report(y_test,y_pred22)

print('SVC with chi2 test Classification Report:\n {}'.format(auc_roc))
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test_kbest))

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test_kbest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='SVC w/chi2 (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC SVC with chi2 test')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
permutation_test_between_clfs(y_test, y_pred2, y_pred22, nsamples=1000)