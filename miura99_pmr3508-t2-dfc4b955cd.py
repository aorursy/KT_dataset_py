import pandas as pd
raw = pd.read_csv('../input/dataset-tarefa-2/train_data.csv', engine = 'python')
raw.info()
raw_zeros = raw[raw==0].count()/3680*100
raw_corr = raw.corr().ham
import matplotlib.pyplot as plt
plt.scatter(raw_zeros,raw_corr)
plt.hist(raw['ham'], bins = 2)
ham0 = ((raw['ham'] == 0).sum()/3680)
print('proporções:')
print('ham = 0: %.2f' %ham0)
print('ham = 1: %.2f' %(1 - ham0) )
print('razão ham=0/ham=1: %.2f' %(ham0/(1-ham0) ) )
raw.describe()
plt.matshow(raw.iloc[:,:-2].corr())
anl0 = raw.corr().where(lambda x: abs (x) > 0.5)
anl0 = anl0.corr().where(lambda x: abs (x) < 1)
names = []
for column in anl0:
    series = anl0[column].dropna()
    if series.size != 0:
        names.append(column)
raw.corr().loc['ham',names]
anl1 = raw.iloc[:,:-5]
plt.style.use('seaborn-deep')
for column in anl1:
        plt.hist(anl1[column],alpha = 0.5, bins = 5)
anl2 = raw.loc[:,['word_freq_free','word_freq_money','ham']]
anl2true = anl2[anl2['ham'] == True]
anl2false =  anl2[anl2['ham'] == False]

plt.subplot(2, 1, 1)
plt.scatter(anl2true.iloc[:,0], anl2true.iloc[:,1], alpha = 0.1 ,c= 'blue')

plt.subplot(2, 1,2)
plt.scatter(anl2false.iloc[:,0], anl2false.iloc[:,1],alpha = 0.1 , c = 'red')
anl3 = raw.corr()['ham'].where(lambda x: abs(x) < 0.08).dropna()
anl3.index
anl3
index = [word for word in raw.columns if word not in anl3.index if word !='ham']
X_train = raw.loc[:,index]
Y_train = raw['ham']
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) 
X_train = X_train + 4
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
general_scores = []
clf_nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
alpha_values = np.array(range(1,333))
alpha_values = alpha_values*3/1000
param_grid_nb = dict(alpha = alpha_values)
clf_knn = KNeighborsClassifier(n_neighbors=5)
k_range = list(range(1,30))
weight_options = ['uniform', 'distance']
p_options = list(range(1,2))
param_grid_knn = dict(n_neighbors=k_range, p=p_options)#, p=p_options
#criarei um scorer para o gridsearch, a partir do fbetascore com beta = 3
from sklearn.metrics import fbeta_score, make_scorer
fbeta_scorer =  make_scorer(fbeta_score, beta=3)
grid = GridSearchCV(clf_knn, param_grid_knn, cv=10, scoring= fbeta_scorer, n_jobs = -2)  
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.best_score_)
grid2 = GridSearchCV(clf_nb, param_grid_nb, cv=10, scoring= fbeta_scorer, n_jobs = -2)  
grid2.fit(X_train, Y_train)
print(grid2.best_estimator_)
print(grid2.best_score_)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp
#Fazendo CV com 10 folds:
cv = StratifiedKFold(n_splits=10)
classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=29, p=1,
           weights='uniform')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X_train, Y_train):
    # Calculando probabilidades 
    probas_ = classifier.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
    # Computando a curva ROC e a AUC da ROC
    fpr, tpr, thresholds = roc_curve(Y_train[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
roc = plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
thresholds
from sklearn import preprocessing
fbs = []
mean_fbs = []
for threshold in thresholds:
    for train, test in cv.split(X_train, Y_train):
        probas_ = classifier.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
        y_test = preprocessing.binarize(probas_, threshold)[:,-1]
        fbs.append(fbeta_score(Y_train[test], y_test, beta = 3))
    fbs = np.array(fbs)
    mean_fbs.append(np.mean(fbs))
    fbs = []
plt.scatter(thresholds,mean_fbs)
maxes = [thresholds[i] + mean_fbs[i] for i in range(1,thresholds.shape[0])]
max_index = maxes.index(max(maxes) ) 
print(thresholds[max_index])
best_threshold = thresholds[max_index]
clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=29, p=1,
           weights='uniform')
clf.fit(X_train,Y_train)
test_raw = pd.read_csv('../input/dataset-tarefa-2/test_features.csv', engine = 'python')
X_test = test_raw.loc[:,index]
scaler = scaler.fit(X_test)
X_test = scaler.transform(X_test) 
Y_test = classifier.predict_proba(X_test)
Y_test = preprocessing.binarize(Y_test, best_threshold)[:,-1]
Y_test = pd.Series(Y_test)
Y_test = Y_test.replace(to_replace = [0,1], value = ['FALSE', 'TRUE'])
output = pd.DataFrame([test_raw['Id'],Y_test])
output = output.transpose()
output = output.rename(columns={'Id': 'Id', 'Unnamed 0': 'ham'})
output.to_csv("output.csv", index = False)
