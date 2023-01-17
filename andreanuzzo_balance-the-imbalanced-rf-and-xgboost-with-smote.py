import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/creditcard.csv')
df.head()
df.describe()
df.isnull().sum()
print('Fraud \n',df.Time[df.Class==1].describe(),'\n',
      '\n Non-Fraud \n',df.Time[df.Class==0].describe())
plt.figure(figsize=(12,30*4))
import matplotlib.gridspec as gridspec
features = df.iloc[:,0:30].columns
gs = gridspec.GridSpec(30, 1)
for i, feature in enumerate(df[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[feature][df.Class == 1], bins=50)
    sns.distplot(df[feature][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('Feature: ' + str(feature))
plt.show()
df2 = df.drop(['V15','V20','V22','V23','V25','V28', 'Time', 'Amount'], axis=1)
from sklearn.metrics import confusion_matrix
def plot_cm(classifier, predictions):
    cm = confusion_matrix(y_test, predictions)
    
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap='RdBu')
    classNames = ['Normal','Fraud']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                     horizontalalignment='center', color='White')
    
    plt.show()
        
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2*recall*precision/(recall+precision)

    print('Recall={0:0.3f}'.format(recall),'\nPrecision={0:0.3f}'.format(precision))
    print('F1={0:0.3f}'.format(F1))
from sklearn.metrics import average_precision_score, precision_recall_curve
def plot_aucprc(classifier, scores):
    precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=0)
    average_precision = average_precision_score(y_test, scores)

    print('Average precision-recall score: {0:0.3f}'.format(
          average_precision))

    plt.plot(recall, precision, label='area = %0.3f' % average_precision, color="green")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="best")
    plt.show()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
pre = RandomForestClassifier(n_jobs=-1, random_state = 42,
                             max_features= 'sqrt', 
                             criterion = 'entropy')
pre.fit(X_train, y_train)

#Make predictions
y_pred = pre.predict(X_test)
try:
    scores = pre.decision_function(X_test)
except:
    scores = pre.predict_proba(X_test)[:,1]

#Make plots
plot_cm(pre, y_pred)
plot_aucprc(pre, scores)
#from sklearn.model_selection import GridSearchCV
#param_grid = { 
#    'n_estimators': [10, 500],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'min_samples_leaf' : [len(X)//10000, len(X)//28000, 
#                          len(X)//50000, len(X)//100000]
#}

#CV_rfc = GridSearchCV(estimator=pre, 
#                      param_grid=param_grid, 
#                      scoring = 'f1',
#                      cv=10, 
#                      n_jobs=10,
#                      verbose=2,
#                      pre_dispatch='2*n_jobs',
#                      refit=False)
#CV_rfc.fit(X_train, y_train)

#CV_rfc.best_params_
#rfc = RandomForestClassifier(n_jobs=-1, random_state = 42,
#                             n_estimators=CV_rfc.best_params_['n_estimators'], 
#                             min_samples_leaf=CV_rfc.best_params_['min_samples_leaf'], 
#                             max_features= CV_rfc.best_params_['max_features'])

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=2,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=500, n_jobs=-1, oob_score=False, random_state=42,
#            verbose=0, warm_start=False)


rfc = RandomForestClassifier(n_jobs=-1, random_state = 42,
                             n_estimators=500, 
                             max_features='auto',
                             min_samples_leaf=2,
                             criterion = 'entropy')

rfc.fit(X_train, y_train)
#Make predictions
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 42, n_jobs = -1)
xgb.fit(X_train, y_train)
#Make predictions
y_pred = xgb.predict(X_test)
try:
    scores = xgb.decision_function(X_test)
except:
    scores = xgb.predict_proba(X_test)[:,1]
#Make plots
y_pred = xgb.predict(X_test)
plot_cm(xgb, y_pred)
plot_aucprc(xgb, scores)
fraud_ratio=y_train.value_counts()[1]/y_train.value_counts()[0]
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [1,3,5], 
             'min_child_weight': [1,3,5], 
             'n_estimators': [100,200,500,1000], 
             'scale_pos_weight': [1, 0.1, 0.01, fraud_ratio]}
#CV_GBM = GridSearchCV(estimator = xgb, 
#                      param_grid = param_grid,
#                      scoring = 'f1', 
#                      cv = 10, 
#                      n_jobs = -1,
#                      refit = True)

#CV_GBM.fit(X_train, y_train)

#CV_GBM.best_params_
#optimized_GBM = XGBClassifier(n_jobs=-1, random_state = 42,
#                             n_estimators=CV_GBM.best_params_['n_estimators'], 
#                             max_depth=CV_GBM.best_params_['max_depth'],
#                             min_child_weight=CV_GBM.best_params_['min_child_weight'],
#                             criterion = 'entropy')
optimized_GBM = XGBClassifier(n_jobs=-1, random_state = 42,
                             n_estimators=100, 
                             max_depth=1,
                             min_child_weight=1,
                             criterion = 'entropy',
                             scale_pos_weight=fraud_ratio)
optimized_GBM.fit(X_train, y_train)
#Make predictions
y_pred = optimized_GBM.predict(X_test)
try:
    scores = optimized_GBM.decision_function(X_test)
except:
    scores = optimized_GBM.predict_proba(X_test)[:,1]
    
#Make plots
plot_cm(optimized_GBM, y_pred)
plot_aucprc(optimized_GBM, scores)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import OneClassSVM
classifier = OneClassSVM(kernel="rbf", random_state = 42)
classifier.fit(X_train, y_train)
#Make predictions
y_pred = classifier.predict(X_test)
y_pred = np.array([y==-1 for y in y_pred])

try:
    scores = classifier.decision_function(X_test)
except:
    scores = classifier.predict_proba(X_test)[:,1]

#Make plots
plot_cm(classifier, y_pred)
plot_aucprc(classifier, scores)
df3 = df2#.sample(frac = 0.1, random_state=42)
train = df3[df3.Class==0].sample(frac=0.75, random_state = 42)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

X_test = df3.loc[~df3.index.isin(X_train.index)].iloc[:,:-1]#.sample(frac=.50, random_state = 42)
y_test = df3.loc[~df3.index.isin(y_train.index)].iloc[:,-1]#.sample(frac=.50, random_state = 42)

#X_cval = df3.loc[~df3.index.isin(X_test.index)& ~df3.index.isin(X_train.index)].iloc[:,:-1]
#y_cval = df3.loc[~df3.index.isin(y_test.index)& ~df3.index.isin(X_train.index)].iloc[:,-1]
print('df3', df3.shape,'\n',
      'train',train.shape,'\n',
      'X_train',X_train.shape,'\n',
      'y_train',y_train.shape,'\n',
      'X_test',X_test.shape,'\n',
      'y_test',y_test.shape,'\n', 
      #'X_val',X_cval.shape,'\n',
      #'y_val',y_cval.shape,'\n'
     )
df3.shape[0] == train.shape[0] + X_test.shape[0]
def covariance_matrix(X):
    X = X.values
    m, n = X.shape 
    tmp_mat = np.zeros((n, n))
    mu = X.mean(axis=0)
    for i in range(m):
        tmp_mat += np.outer(X[i] - mu, X[i] - mu)
    return tmp_mat / m
cov_mat = covariance_matrix(X_train)
cov_mat_inv = np.linalg.pinv(cov_mat)
cov_mat_det = np.linalg.det(cov_mat)
def multi_gauss(x):
    n = len(cov_mat)
    return (np.exp(-0.5 * np.dot(x, np.dot(cov_mat_inv, x.T))) 
            / (2. * np.pi)**(n/2.) 
            / np.sqrt(cov_mat_det))
eps = min([multi_gauss(x) for x in X_train.values])
predictions = np.array([multi_gauss(x) <= eps for x in X_test.values])
y_test = np.array(y_test, dtype=bool)
cm = confusion_matrix(y_test, predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap='RdBu')
classNames = ['Normal','Fraud']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                 horizontalalignment='center', color='White')

plt.show()

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)
print("F1=",F1)
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon
#fit the model
mu, sigma = estimateGaussian(X_train)
p = multivariateGaussian(X_train,mu,sigma)

p_cv = multivariateGaussian(X_test,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,y_test)
outliers = np.asarray(np.where(p < ep))
predictions = np.array([p_cv <= ep]).transpose()
y_test = np.array(y_test, dtype=bool)

cm = confusion_matrix(y_test, predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap='RdBu')
classNames = ['Normal','Fraud']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                 horizontalalignment='center', color='White')

plt.show()

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)
print("F1=",F1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from imblearn.metrics import classification_report_imbalanced

bagging = BaggingClassifier(random_state=0)
balanced_bagging = BalancedBaggingClassifier(random_state=0)

bagging.fit(X_train, y_train)
balanced_bagging.fit(X_train, y_train)

#Make predictions
print('Classification of original dataset with Bagging (scikit-learn)')
y_pred = bagging.predict(X_test)
try:
    scores = bagging.decision_function(X_test)
except:
    scores = bagging.predict_proba(X_test)[:,1]

#Make plots
plot_cm(bagging, y_pred)
plot_aucprc(bagging, scores)

#Make predictions
print('Classification of original dataset with BalancedBagging (imbalanced-learn)')
y_pred = balanced_bagging.predict(X_test)
try:
    scores = balanced_bagging.decision_function(X_test)
except:
    scores = balanced_bagging.predict_proba(X_test)[:,1]

#Make plots
plot_cm(balanced_bagging, y_pred)
plot_aucprc(balanced_bagging, scores)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25, random_state = 42)

#fit the best models so far
xgb.fit(X_train, y_train)
rfc.fit(X_train, y_train)

#Make predictions
print('Classification of SMOTE-resampled dataset with XGboost')
y_pred = xgb.predict(X_test)
try:
    scores = xgb.decision_function(X_test)
except:
    scores = xgb.predict_proba(X_test)[:,1]
#Make plots
y_pred = xgb.predict(X_test)
plot_cm(xgb, y_pred)
plot_aucprc(xgb, scores)

#Make predictions
print('Classification of SMOTE-resampled dataset with optimized RF')
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Make predictions
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)
