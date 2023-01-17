import numpy as np
from __future__ import print_function
import pandas as pd
from pandas import read_excel
from sklearn import decomposition, preprocessing, svm
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp
from matplotlib import pyplot as plt
%matplotlib inline
# Read in the input data
proteinD = pd.read_csv("../input/77_cancer_proteomes_CPTAC_itraq.csv",header = None, low_memory = False)
patientD = pd.read_csv("../input/clinical_data_breast_cancer.csv",header = None, low_memory = False)
# Convert the above to numpy arrays
bioData = proteinD.as_matrix()
patientData = patientD.as_matrix()
pRef = bioData[0,3:bioData.shape[1]];
# Instead of replacing each missing value through some sort of interpolation, let's see how removing each row will effect the outcome.

dOnly = bioData[1:bioData.shape[0],3:bioData.shape[1]].astype('float32');
dOnly = dOnly[~np.isnan(dOnly).any(axis=1)]
# First, classifying each patient by the 'PAM50 mRNA' label will be attempted.
h = patientData[0,:];
pos = np.where(h == 'PAM50 mRNA')
pathology = patientData[1:patientData.shape[0], pos]
pD = patientData[1:patientData.shape[0],0];
pD0 = [];
pD1 = [];
for i in range(0,pD.shape[0]-1):
    cur = pD[i];
    pD0.append(cur[5:12])
for i in range(0,pRef.shape[0]):
    cur = pRef[i]
    pD1.append(cur[0:7])
# Match up each group with the corresponding genomic data.
d = np.zeros((1,len(dOnly[:,1])+1))
for i in range(0, len(pD1)-1):
    for j in range(0,len(pD0)-1):
        if pD1[i] == pD0[j]:
            cur = np.hstack((dOnly[:,i].T,pathology[j][0]));
            d = np.vstack((d,cur));
d = np.delete(d, (0), axis=0)    
# Just a bit of pre-processing...
dR = d[:,0:d.shape[1]-1];
dN = preprocessing.minmax_scale(dR, feature_range=(-1, 1), axis=0, copy=True)
labels = d[:,d.shape[1]-1];
# Each label was transformed to numeric form and then LDA was used to reduce dimensionality.
le = preprocessing.LabelEncoder()
le.fit(labels)
nL = le.transform(labels);
# nL = np.reshape(nL,(len(nL),1))
lda = LinearDiscriminantAnalysis(n_components=3)
X = lda.fit(dN, nL).transform(dN)  
nL = np.reshape(nL,(len(nL),1))
# This line just allows for easier plotting via sort.
x = np.hstack((X,nL))
x = x[x[:,3].argsort()]
# How many of each class?
type0 = sum(np.isin(x[:,3], 0));
type1 = sum(np.isin(x[:,3], 1));
type2 = sum(np.isin(x[:,3], 2));
type3 = sum(np.isin(x[:,3], 3));

# Let's look at the first and second LDA loadings plot.
q=0;
r=1;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.scatter(x[0:type0,q],x[0:type0,r],s=25, c='blue', marker="s", label=le.classes_[0])
ax1.scatter(x[type0:type0+type1,q],x[type0:type0+type1,r],s=25, c='black', marker="x", label=le.classes_[1])
ax1.scatter(x[type0+type1:type0+type1+type2,q],x[type0+type1:type0+type1+type2,r],s=25, c='orange', marker="*", label=le.classes_[2])
ax1.scatter(x[type0+type1+type2:type0+type1+type2+type3,q],x[type0+type1+type2:type0+type1+type2+type3,r],s=25, c='purple', marker="o", label=le.classes_[3])
plt.xlabel('LDA Loading #1',fontsize=18)
plt.ylabel('LDA Loading #2',fontsize=18)
plt.title('2D LDA Scatter Plot: "PAM50 mRNA"',fontsize=18)
plt.legend(loc='upper left',prop={'size': 12});
plt.show()
# ... And then the second and third.
q=1;
r=2;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.scatter(x[0:type0,q],x[0:type0,r],s=25, c='blue', marker="s", label=le.classes_[0])
ax1.scatter(x[type0:type0+type1,q],x[type0:type0+type1,r],s=25, c='black', marker="x", label=le.classes_[1])
ax1.scatter(x[type0+type1:type0+type1+type2,q],x[type0+type1:type0+type1+type2,r],s=25, c='orange', marker="*", label=le.classes_[2])
ax1.scatter(x[type0+type1+type2:type0+type1+type2+type3,q],x[type0+type1+type2:type0+type1+type2+type3,r],s=25, c='purple', marker="o", label=le.classes_[3])
plt.xlabel('LDA Loading #2',fontsize=18)
plt.ylabel('LDA Loading #3',fontsize=18)
plt.title('2D LDA Scatter Plot: "PAM50 mRNA"',fontsize=18)
plt.legend(loc='lower left',prop={'size': 12});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
fiveF = cross_val_score(clf, x[:,0:3], x[:,3], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
# Time to check out the multi-class AUROC metric. This was obtained from the sklearn example at 'http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html'.
n_classes = 4;
y = label_binarize(x[:,3], classes=[0, 1, 2, 3])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(x[:,0:3], y, test_size=.5,
                                                    random_state=0)
                                                    
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
lw = 2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['blue','black','orange','purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(le.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.001])
plt.xlabel('False Positive Rate',fontsize = 18)
plt.ylabel('True Positive Rate',fontsize = 18)
plt.title('Multi-Class ROC',fontsize = 18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
pos = np.where(h == 'ER Status')
pathology = patientData[1:patientData.shape[0], pos]

d = np.zeros((1,len(dOnly[:,1])+1))
for i in range(0, len(pD1)-1):
    for j in range(0,len(pD0)-1):
        if pD1[i] == pD0[j]:
            cur = np.hstack((dOnly[:,i].T,pathology[j][0]));
            d = np.vstack((d,cur));
d = np.delete(d, (0), axis=0) 

dR = d[:,0:d.shape[1]-1];
dN = preprocessing.minmax_scale(dR, feature_range=(-1, 1), axis=0, copy=True)
labels = d[:,d.shape[1]-1];

le = preprocessing.LabelEncoder()
le.fit(labels)
nL = le.transform(labels);
# nL = np.reshape(nL,(len(nL),1))
lda = LinearDiscriminantAnalysis(n_components=1)
X = lda.fit(dN, nL).transform(dN)  
nL = np.reshape(nL,(len(nL),1))

x = np.hstack((X,nL))
x = x[x[:,1].argsort()]
# There is only one LDA loading this time around.
type0 = sum(np.isin(x[:,1], 0));
type1 = sum(np.isin(x[:,1], 1));

# Bar plot
q=0;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.bar(np.linspace(1,type0,type0), x[0:type0,q], align='center', label=le.classes_[0])
ax1.bar(np.linspace(type0+1,type0+type1,type1), x[type0:type0+type1,q], align='center', label=le.classes_[1])
plt.xlabel('Patient',fontsize=18)
plt.ylabel('LDA Loading',fontsize=18)
plt.title('1D LDA Bar Plot: "ER Status"',fontsize=18)
plt.legend(loc='upper left',prop={'size': 15});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
fiveF = cross_val_score(clf, x[:,0].reshape(x[:,0].shape[0],1), x[:,1], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
X = x[:,0].reshape(x[:,0].shape[0],1);
y = x[:,1];
n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC: "ER Status"',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
pos = np.where(h == 'PR Status')
pathology = patientData[1:patientData.shape[0], pos]

d = np.zeros((1,len(dOnly[:,1])+1))
for i in range(0, len(pD1)-1):
    for j in range(0,len(pD0)-1):
        if pD1[i] == pD0[j]:
            cur = np.hstack((dOnly[:,i].T,pathology[j][0]));
            d = np.vstack((d,cur));
d = np.delete(d, (0), axis=0) 

dR = d[:,0:d.shape[1]-1];
dN = preprocessing.minmax_scale(dR, feature_range=(-1, 1), axis=0, copy=True)
labels = d[:,d.shape[1]-1];

le = preprocessing.LabelEncoder()
le.fit(labels)
nL = le.transform(labels);
# nL = np.reshape(nL,(len(nL),1))
lda = LinearDiscriminantAnalysis(n_components=1)
X = lda.fit(dN, nL).transform(dN)  
nL = np.reshape(nL,(len(nL),1))

x = np.hstack((X,nL))
x = x[x[:,1].argsort()]

# There is only one LDA loading this time around.
type0 = sum(np.isin(x[:,1], 0));
type1 = sum(np.isin(x[:,1], 1));

# Bar plot
q=0;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.bar(np.linspace(1,type0,type0), x[0:type0,q], align='center', label=le.classes_[0])
ax1.bar(np.linspace(type0+1,type0+type1,type1), x[type0:type0+type1,q], align='center', label=le.classes_[1])
plt.xlabel('Patient',fontsize=18)
plt.ylabel('LDA Loading',fontsize=18)
plt.title('1D LDA Bar Plot: "PR Status"',fontsize=18)
plt.legend(loc='upper left',prop={'size': 15});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
fiveF = cross_val_score(clf, x[:,0].reshape(x[:,0].shape[0],1), x[:,1], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
X = x[:,0].reshape(x[:,0].shape[0],1);
y = x[:,1];
n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC: "PR Status"',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
pos = np.where(h == 'HER2 Final Status')
pathology = patientData[1:patientData.shape[0], pos]

d = np.zeros((1,len(dOnly[:,1])+1))
for i in range(0, len(pD1)-1):
    for j in range(0,len(pD0)-1):
        if pD1[i] == pD0[j]:
            cur = np.hstack((dOnly[:,i].T,pathology[j][0]));
            d = np.vstack((d,cur));
d = np.delete(d, (0), axis=0)

# This time, there was a patient who possessed a HER2 status of "equivocal". This needed to be taken care of.
k = [];
for i in range(0,len(d[:,d.shape[1]-2])):
    if d[i,d.shape[1]-1] != 'Positive' and d[i,d.shape[1]-1] != 'Negative':
        k.append(i)
d = np.delete(d, (k), axis=0)

dR = d[:,0:d.shape[1]-1];
dN = preprocessing.minmax_scale(dR, feature_range=(-1, 1), axis=0, copy=True)
labels = d[:,d.shape[1]-1];

le = preprocessing.LabelEncoder()
le.fit(labels)
nL = le.transform(labels);
# nL = np.reshape(nL,(len(nL),1))
lda = LinearDiscriminantAnalysis(n_components=1)
X = lda.fit(dN, nL).transform(dN)  
nL = np.reshape(nL,(len(nL),1))

x = np.hstack((X,nL))
x = x[x[:,1].argsort()]

# There is only one LDA loading this time around.
type0 = sum(np.isin(x[:,1], 0));
type1 = sum(np.isin(x[:,1], 1));

# Bar plot
q=0;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.bar(np.linspace(1,type0,type0), x[0:type0,q], align='center', label=le.classes_[0])
ax1.bar(np.linspace(type0+1,type0+type1,type1), x[type0:type0+type1,q], align='center', label=le.classes_[1])
plt.xlabel('Patient',fontsize=18)
plt.ylabel('LDA Loading',fontsize=18)
plt.title('1D LDA Bar Plot: "HER2 Final Status"',fontsize=18)
plt.legend(loc='upper left',prop={'size': 15});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
fiveF = cross_val_score(clf, x[:,0].reshape(x[:,0].shape[0],1), x[:,1], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
X = x[:,0].reshape(x[:,0].shape[0],1);
y = x[:,1];
n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC: "HER2 Final Status"',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
pos = np.where(h == 'RPPA Clusters')
pathology = patientData[1:patientData.shape[0], pos]

d = np.zeros((1,len(dOnly[:,1])+1))
for i in range(0, len(pD1)-1):
    for j in range(0,len(pD0)-1):
        if pD1[i] == pD0[j]:
            cur = np.hstack((dOnly[:,i].T,pathology[j][0]));
            d = np.vstack((d,cur));
d = np.delete(d, (0), axis=0)

k = [];
for i in range(0,len(d[:,d.shape[1]-2])):
    if d[i,d.shape[1]-1] == 'X':
        k.append(i)
d = np.delete(d, (k), axis=0)

dR = d[:,0:d.shape[1]-1];
dN = preprocessing.minmax_scale(dR, feature_range=(-1, 1), axis=0, copy=True)
labels = d[:,d.shape[1]-1];

le = preprocessing.LabelEncoder()
le.fit(labels)
nL = le.transform(labels);
lda = LinearDiscriminantAnalysis(n_components=3)
X = lda.fit(dN, nL).transform(dN)  
nL = np.reshape(nL,(len(nL),1))

x = np.hstack((X,nL))
x = x[x[:,3].argsort()]

# This time, six groups are present
type0 = sum(np.isin(x[:,3], 0));
type1 = sum(np.isin(x[:,3], 1));
type2 = sum(np.isin(x[:,3], 2));
type3 = sum(np.isin(x[:,3], 3));
type4 = sum(np.isin(x[:,3], 4));
type5 = sum(np.isin(x[:,3], 5));

q=0;
r=1;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.scatter(x[0:type0,q],x[0:type0,r],s=25, c='blue', marker="s", label=le.classes_[0])
ax1.scatter(x[type0:type0+type1,q],x[type0:type0+type1,r],s=25, c='black', marker="x", label=le.classes_[1])
ax1.scatter(x[type0+type1:type0+type1+type2,q],x[type0+type1:type0+type1+type2,r],s=25, c='orange', marker="*", label=le.classes_[2])
ax1.scatter(x[type0+type1+type2:type0+type1+type2+type3,q],x[type0+type1+type2:type0+type1+type2+type3,r],s=25, c='purple', marker="o", label=le.classes_[3])
ax1.scatter(x[type0+type1+type2+type3:type0+type1+type2+type3+type4,q],x[type0+type1+type2+type3:type0+type1+type2+type3+type4,r],s=25, c='red', marker=">", label=le.classes_[4])
ax1.scatter(x[type0+type1+type2+type3+type4:type0+type1+type2+type3+type4+type5,q],x[type0+type1+type2+type3+type4:type0+type1+type2+type3+type4+type5,r],s=25, c='green', marker="^", label=le.classes_[5])
plt.xlabel('LDA Loading #1',fontsize=18)
plt.ylabel('LDA Loading #2',fontsize=18)
plt.title('2D LDA Scatter Plot: "RPPA Clusters"',fontsize=18)
plt.legend(loc='lower left',prop={'size': 12});
plt.show()
q=1;
r=2;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.scatter(x[0:type0,q],x[0:type0,r],s=25, c='blue', marker="s", label=le.classes_[0])
ax1.scatter(x[type0:type0+type1,q],x[type0:type0+type1,r],s=25, c='black', marker="x", label=le.classes_[1])
ax1.scatter(x[type0+type1:type0+type1+type2,q],x[type0+type1:type0+type1+type2,r],s=25, c='orange', marker="*", label=le.classes_[2])
ax1.scatter(x[type0+type1+type2:type0+type1+type2+type3,q],x[type0+type1+type2:type0+type1+type2+type3,r],s=25, c='purple', marker="o", label=le.classes_[3])
ax1.scatter(x[type0+type1+type2+type3:type0+type1+type2+type3+type4,q],x[type0+type1+type2+type3:type0+type1+type2+type3+type4,r],s=25, c='red', marker=">", label=le.classes_[4])
ax1.scatter(x[type0+type1+type2+type3+type4:type0+type1+type2+type3+type4+type5,q],x[type0+type1+type2+type3+type4:type0+type1+type2+type3+type4+type5,r],s=25, c='green', marker="^", label=le.classes_[5])
plt.xlabel('LDA Loading #2',fontsize=18)
plt.ylabel('LDA Loading #3',fontsize=18)
plt.title('2D LDA Scatter Plot: "RPPA Clusters"',fontsize=18)
plt.legend(loc='upper right',prop={'size': 12});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
fiveF = cross_val_score(clf, x[:,0:3], x[:,3], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
n_classes = 6;
y = label_binarize(x[:,3], classes=[0, 1, 2, 3, 4, 5])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(x[:,0:3], y, test_size=.5,
                                                    random_state=0)
                                                    
random_state = np.random.RandomState(0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['blue','black','orange','purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(le.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.001])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Multi-Class ROC: "RPPA Clusters"',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()