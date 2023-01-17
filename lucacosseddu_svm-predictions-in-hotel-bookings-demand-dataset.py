import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def barplot(attribute,title,ylabel,xlabel):
    f = plt.figure(figsize=(8,5))
    ax = f.add_subplot(1,1,1)
    plt.bar(attribute.unique(),attribute.value_counts(dropna=False).values,color='#21618C',width=0.25)
    plt.ylabel(ylabel,fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.grid(color='w')
    ax.set_axisbelow(True)
    ax.set_facecolor('#E0E0E0')
    plt.title(title, fontsize=22)
    plt.savefig(title+'.png')
df=pd.read_csv("../input/filtrado.csv")
df.columns
df=df.drop('Unnamed: 0',axis=1)
barplot(df.is_repeated_guest,title='is_repeated_guest distribution',ylabel='Number of customers',xlabel='is_repeated_guest')
#Find Number of samples which are rep or not
no_rep = len(df[df['is_repeated_guest'] == 0])
rep = len(df[df['is_repeated_guest'] == 1])
#Get indices of non rep samples
no_rep_indices = df[df.is_repeated_guest == 0].index
#Random sample non rep indices
random_indices = np.random.choice(no_rep_indices, size=rep, replace=False)
#Find the indices of rep samples
rep_indices = df[df.is_repeated_guest == 1].index
#Concat rep indices with sample non-fraud ones
under_sample_indices = np.concatenate([random_indices,rep_indices])
#Get Balance Dataframe
under_sample = df.loc[under_sample_indices]
under_sample.reset_index().info()
Y=under_sample.is_repeated_guest.values
dff=under_sample
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_encoded=pd.get_dummies(dff,columns=list(dff.select_dtypes(exclude=numerics).columns))
X=df_encoded.values
X=np.nan_to_num(X)
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
PCA_df = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
explained_variance
# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 10)
LDA_df = lda.fit_transform(X,Y)
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA as KPCA
kpca = KPCA(n_components=10,kernel= 'sigmoid')
KPCA_df = kpca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
PCA_train, PCA_test, PCA_y_train, PCA_y_test = train_test_split(PCA_df,Y,test_size=0.3)
LDA_train, LDA_test, LDA_y_train, LDA_y_test = train_test_split(LDA_df,Y,test_size=0.3)
KPCA_train, KPCA_test, KPCA_y_train, KPCA_y_test = train_test_split(KPCA_df,Y,test_size=0.3)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
def report(results, n_top):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# Applying Random Search to find the best model and the best parameters
from sklearn.model_selection import RandomizedSearchCV
parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear'],
              'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9]}
             
rand_search = RandomizedSearchCV(estimator = classifier,
                           param_distributions = parameters,
                           cv = 10,
                           n_jobs = -1)
rand_search = rand_search.fit(X_train, y_train)
report(rand_search.cv_results_, n_top=3)
classifier = SVC(kernel = 'sigmoid', gamma = 0.1, C = 1000)
scores = cross_val_score(classifier, X, Y, cv=10)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

scores = cross_val_score(classifier, X, Y, cv=10, scoring='f1_macro')
print('F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
# Applying Random Search to find the best model and the best parameters
from sklearn.model_selection import RandomizedSearchCV
parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear'],
              'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9]}
             
rand_search = RandomizedSearchCV(estimator = classifier,
                           param_distributions = parameters,
                           #scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rand_search = rand_search.fit(PCA_train, PCA_y_train)
report(rand_search.cv_results_, n_top=3)
PCA_classifier = SVC(kernel = 'sigmoid', gamma = 0.1, C = 100)
scores = cross_val_score(classifier, PCA_df, Y, cv=10)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

scores = cross_val_score(classifier, PCA_df, Y, cv=10, scoring='f1_macro')
print('F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
classifier.fit(PCA_train, PCA_y_train)
PCA_y_pred = classifier.predict(PCA_test)
confusion_matrix(PCA_y_test, PCA_y_pred)
print('Accuracy %s' % accuracy_score(PCA_y_test, PCA_y_pred))
print('F1-score %s' % f1_score(PCA_y_test, PCA_y_pred, average=None))
# Applying Random Search to find the best model and the best parameters
from sklearn.model_selection import RandomizedSearchCV
parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear'],
              'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9]}
             
rand_search = RandomizedSearchCV(estimator = classifier,
                           param_distributions = parameters,
                           #scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rand_search = rand_search.fit(LDA_train, LDA_y_train)
report(rand_search.cv_results_, n_top=3)
classifier = SVC(kernel = 'sigmoid', gamma = 0.1, C = 10)
scores = cross_val_score(classifier, LDA_df, Y, cv=10)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

scores = cross_val_score(classifier, LDA_df, Y, cv=10, scoring='f1_macro')
print('F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
classifier.fit(LDA_train, LDA_y_train)
LDA_y_pred = classifier.predict(LDA_test)
confusion_matrix(LDA_y_test, LDA_y_pred)
print('Accuracy %s' % accuracy_score(LDA_y_test, LDA_y_pred))
print('F1-score %s' % f1_score(LDA_y_test, LDA_y_pred, average=None))
# Applying Random Search to find the best model and the best parameters
from sklearn.model_selection import RandomizedSearchCV
parameters = {'C': [1, 10, 100, 1000], 'kernel': ['linear'],
              'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9],
              'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.25, 0.5, 0.75, 0.9]}
             
rand_search = RandomizedSearchCV(estimator = classifier,
                           param_distributions = parameters,
                           #scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rand_search = rand_search.fit(KPCA_train, KPCA_y_train)
report(rand_search.cv_results_, n_top=3)
classifier = SVC(kernel = 'sigmoid', gamma = 0.5, C = 1000)
scores = cross_val_score(classifier, KPCA_df, Y, cv=10)
print('Accuracy: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

scores = cross_val_score(classifier, KPCA_df, Y, cv=10, scoring='f1_macro')
print('F1-score: %0.4f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
classifier.fit(KPCA_train, KPCA_y_train)
KPCA_y_pred = classifier.predict(KPCA_test)
confusion_matrix(KPCA_y_test, KPCA_y_pred)
print('Accuracy %s' % accuracy_score(KPCA_y_test,KPCA_y_pred))
print('F1-score %s' % f1_score(KPCA_y_test, KPCA_y_pred, average=None))