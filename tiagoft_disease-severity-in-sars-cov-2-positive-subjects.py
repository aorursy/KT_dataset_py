%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier
# Load dataset

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

df.head()
care_cols =  ['Patient addmited to regular ward (1=yes, 0=no)',\

                'Patient addmited to semi-intensive unit (1=yes, 0=no)',\

                 'Patient addmited to intensive care unit (1=yes, 0=no)']

default_hemogram = list(df.columns[6:20])

specialist_features = ['Lymphocytes', 'Neutrophils', 'Sodium', 'Potassium', 'Creatinine','Proteina C reativa mg/dL']



short_labels = ['No care', 'Regular', 'Semi-Intensive', 'Intensive']



cors = 'SARS-Cov-2 exam result'

# Specialist feature visualization

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

data = df[care_cols + specialist_features].dropna().to_numpy(copy=True)



y_ = data[:,0:3]

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]



# Lymphocytes vs. Neutrophils

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,0], X[idx,1], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Lymphocytes')

plt.ylabel('Neutrophils')

plt.show()



# Lymphocytes vs. Neutrophils

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,2], X[idx,3], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Sodium')

plt.ylabel('Potassium')

plt.show()



# Creatinine and PCR

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,4], X[idx,5], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Creatinine')

plt.ylabel('Proteina C reativa mg/dL')

plt.show()
# Specialist feature visualization only for SARS-Cov-2 positives

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

df_ = df[ df['SARS-Cov-2 exam result'] == 'positive']

data = df_[care_cols + specialist_features].dropna().to_numpy(copy=True)



y_ = data[:,0:3]

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]



# Lymphocytes vs. Neutrophils

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,0], X[idx,1], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Lymphocytes')

plt.ylabel('Neutrophils')

plt.show()



# Lymphocytes vs. Neutrophils

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,2], X[idx,3], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Sodium')

plt.ylabel('Potassium')

plt.show()



# Creatinine and PCR

plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,4], X[idx,5], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Creatinine')

plt.ylabel('Proteina C reativa mg/dL')

plt.show()

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]
plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,0], X[idx,5], alpha=0.6, label=short_labels[i])

plt.legend()

plt.xlabel('Lymphocites')

plt.ylabel('Proteina C reativa mg/dL')

plt.show()

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]
plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(X[idx,0], X[idx,5], alpha=0.6, label=short_labels[i])

    

plt.plot([-2, 1, 1, -2, -2], [-0.4, -0.4, 3.5, 3.5, -0.4], 'r')

plt.plot([1, 1], [-0.4, 1.5], 'r', label='Risk Boundaries')



plt.legend()

plt.xlabel('Lymphocites')

plt.ylabel('Proteina C reativa mg/dL')

plt.show()

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]
# Base classification experiment



skf = StratifiedKFold(n_splits=2)



c = np.array([ [0, 0],[0,0]])

y_gt = []

y_pr = []

for train_index, test_index in skf.split(X, y):

    X_train = X[train_index,:]

    y_train = y[train_index]

    X_test = X[test_index,:]

    y_test = y[test_index]

    

    

    #classifier = RandomForestClassifier(n_estimators=100, max_features=3, class_weight='balanced', criterion='entropy')

    #classifier = LinearSVC(C=10, class_weight='balanced')

    #classifier = SVC(C=10, class_weight='balanced', gamma='auto')

    classifier = LogisticRegression(penalty='l2', C=10, class_weight='balanced')#{'negative': .1, 'positive': 1}, l1_ratio=0.5)

    #classifier = GaussianNB()

    classifier.fit(X_train, y_train)

    y_ = classifier.predict(X_test)

    y_gt += list(y_test)

    y_pr += list(y_)



c = confusion_matrix(y_gt, y_pr, labels=range(4))

print(classification_report(y_gt, y_pr, labels=range(4)))

print(c)
# Two-label classification experiment



skf = StratifiedKFold(n_splits=2)



c = np.array([ [0, 0],[0,0]])

y_gt = []

y_pr = []

y0 = y

y0[y0>=1]=1

for train_index, test_index in skf.split(X, y0):

    X_train = X[train_index,:]

    y_train = y0[train_index]

    X_test = X[test_index,:]

    y_test = y0[test_index]

    

    

    #classifier = RandomForestClassifier(n_estimators=100, max_features=3, class_weight='balanced', criterion='entropy')

    #classifier = LinearSVC(C=10, class_weight='balanced')

    #classifier = SVC(C=10, class_weight='balanced', gamma='auto')

    classifier = LogisticRegression(penalty='l2', C=10, class_weight='balanced')#{'negative': .1, 'positive': 1}, l1_ratio=0.5)

    #classifier = GaussianNB()

    classifier.fit(X_train, y_train)

    y_ = classifier.predict(X_test)

    y_gt += list(y_test)

    y_pr += list(y_)





c = confusion_matrix(y_gt, y_pr, labels=range(2))

print(classification_report(y_gt, y_pr, labels=range(2)))

print(c)
# Hemogram data feature visualization

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

data = df[care_cols + specialist_features].dropna().to_numpy(copy=True)



y_ = data[:,0:3]

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]



scaler = StandardScaler()

scaler.fit(X)

X_ = scaler.transform(X)



# PCA visualization

pca = PCA(n_components=2)

Xpca = pca.fit_transform(X_)



plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(Xpca[idx,0], Xpca[idx,1], alpha=0.6, label=short_labels[i])



plt.title('PCA')

plt.legend()

plt.show()



# t-SNE visualization

tsne = TSNE(n_components=2)

Xtsne = tsne.fit_transform(X_)



plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=short_labels[i])



plt.title('t-SNE')

plt.legend()

plt.show()
# Hemogram data feature visualization for SARS-Cov-2-positives

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

df_ = df[ df['SARS-Cov-2 exam result'] == 'positive']

data = df_[care_cols + specialist_features].dropna().to_numpy(copy=True)



y_ = data[:,0:3]

y = y_[:,0]

y [y_[:,1]==1]=2

y [y_[:,2]==1]=3

X = data[:,3:]





scaler = StandardScaler()

scaler.fit(X)

X_ = scaler.transform(X)



# PCA visualization

pca = PCA(n_components=2)

Xpca = pca.fit_transform(X_)



plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(Xpca[idx,0], Xpca[idx,1], alpha=0.6, label=short_labels[i])



plt.title('PCA')

plt.legend()

plt.show()



# t-SNE visualization

tsne = TSNE(n_components=2, perplexity=3)

Xtsne = tsne.fit_transform(X_)



plt.figure()

for i in range(len(short_labels)):

    idx = np.where(y == i)[0]

    plt.scatter(Xtsne[idx,0], Xtsne[idx,1], alpha=0.6, label=short_labels[i])



plt.title('t-SNE')

plt.legend()

plt.show()
skf = StratifiedKFold(n_splits=2)



c = np.array([ [0, 0],[0,0]])

y_gt = []

y_pr = []

for train_index, test_index in skf.split(X, y):

    X_train = X[train_index,:]

    y_train = y[train_index]

    X_test = X[test_index,:]

    y_test = y[test_index]

    

    

    #classifier = RandomForestClassifier(n_estimators=100, max_features=3, class_weight='balanced', criterion='entropy')

    #classifier = LinearSVC(C=10, class_weight='balanced')

    #classifier = SVC(C=10, class_weight='balanced', gamma='auto')

    classifier = LogisticRegression(penalty='l2', C=10, class_weight='balanced')#{'negative': .1, 'positive': 1}, l1_ratio=0.5)

    #classifier = GaussianNB()

    classifier.fit(X_train, y_train)

    y_ = classifier.predict(X_test)

    y_gt += list(y_test)

    y_pr += list(y_)



c = confusion_matrix(y_gt, y_pr, labels=range(4))

print(classification_report(y_gt, y_pr, labels=range(4)))

print(c)
# Simplified classification experiment



skf = StratifiedKFold(n_splits=2)



c = np.array([ [0, 0],[0,0]])

y_gt = []

y_pr = []

y0 = y

y0[y0>=1]=1

for train_index, test_index in skf.split(X, y0):

    X_train = X[train_index,:]

    y_train = y0[train_index]

    X_test = X[test_index,:]

    y_test = y0[test_index]

    

    

    #classifier = RandomForestClassifier(n_estimators=100, max_features=3, class_weight='balanced', criterion='entropy')

    #classifier = LinearSVC(C=10, class_weight='balanced')

    #classifier = SVC(C=10, class_weight='balanced', gamma='auto')

    classifier = LogisticRegression(penalty='l2', C=10, class_weight='balanced')#{'negative': .1, 'positive': 1}, l1_ratio=0.5)

    #classifier = GaussianNB()

    classifier.fit(X_train, y_train)

    y_ = classifier.predict(X_test)

    y_gt += list(y_test)

    y_pr += list(y_)





c = confusion_matrix(y_gt, y_pr, labels=range(2))

print(classification_report(y_gt, y_pr, labels=range(2)))

print(c)