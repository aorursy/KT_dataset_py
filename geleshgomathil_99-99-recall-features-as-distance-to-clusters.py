# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
from IPython.display import Image
import os

import warnings
warnings.simplefilter('ignore')


# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# Read data into a panda dataframe

trainFile = data = pd.read_csv("../input/creditcard.csv") # pd.read_csv("creditcard.csv")
trainFile.info()
trainFile.head()
non_fraud = trainFile.loc[(trainFile["Class"] ==0)]
fraud = trainFile.loc[(trainFile["Class"] ==1)]
print ("Size of Fraud data:", fraud.shape)
print ("Size of non-fraud data: ", non_fraud.shape)
class_count = trainFile["Class"].value_counts()
class_count.plot(kind = 'bar')
plt.title("Transaction class histogram")
plt.xlabel("Class")
plt.ylabel("Count")
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_auc_score
trainFile["Class"].astype('float')
trainFile_corr = trainFile.corr()

sns.heatmap(trainFile_corr, cbar = True,  square = True, annot=False, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.show()
abs(trainFile_corr["Class"]).sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix

# Top three correlated inputs with housing_median_age
attributes = ["Class", "V17", "V14","V12","V10","V16","V3","V7"]

scatter_matrix(trainFile[attributes], figsize=(12, 8))
trainFile.plot(kind="scatter", x="V11", y="Class",
             alpha=0.1)
non_fraud_corr =  non_fraud.corr()
sns.heatmap(non_fraud_corr, cbar = True,  square = True, annot=False, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.show()
fraud_corr =  fraud.corr()
sns.heatmap(fraud_corr, cbar = True,  square = True, annot=False, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.show()
# Print Correlations above threshold of 0.15 for non fraud class
rows, cols = non_fraud.shape
flds = list(non_fraud.columns)

corr = non_fraud_corr.values

for i in range(cols):
    for j in range(i+1, cols):
        if abs(corr[i,j]) > 0.15:
            print (flds[i], ' ', flds[j], ' ', corr[i,j])
# Print Correlations above threshold of 0.8 for fraud class
rows, cols = fraud.shape
flds = list(fraud.columns)

corr = fraud_corr.values

for i in range(cols):
    for j in range(i+1, cols):
        if abs(corr[i,j]) > 0.8:
            print (flds[i], ' ', flds[j], ' ', corr[i,j])
%matplotlib inline
import matplotlib.pyplot as plt

fraud.hist( color='red', label='Fraud', bins=50, figsize=(20,15))
non_fraud.hist( color='blue', label='Non Fraud', bins=50, figsize=(20,15))

plt.show()
# random_state=42

fraud_count = len(fraud)
# fraud_count
smpl_non_fraud = non_fraud.sample(n=fraud_count, random_state=42)
# len(smpl_non_fraud)
train_data=smpl_non_fraud.append(fraud, ignore_index=True)

train_data = shuffle(train_data)
train_data.reset_index(drop=True)

len(train_data)
train_data.info()
%matplotlib inline
sns.countplot(x='Class', data=train_data)
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
# Select features to use for modeling.

cc_num_attribs = list(train_data)[1:-1] # To select all features except Time

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(cc_num_attribs)),
        ('std_scaler', StandardScaler()),
    ])

X = train_data.loc[:,train_data.columns != 'Class']
y = train_data.loc[:,train_data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
cc_prepared = num_pipeline.fit_transform(X_train)
cc_prepared
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(cc_prepared, y_train)
# let's try the full pipeline on the testing data set

test_prepared = num_pipeline.fit_transform(X_test)
y_pred = log_reg.predict(test_prepared)

# Let us calculate the False negative rate (FNR), Miss rate
def fnr(y_test, y_pred):
# from sklearn.metrics import confusion_matrix

    log_cm = confusion_matrix(y_test, y_pred)
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    tn, fp, fn, tp = log_cm.ravel()
    # (tn, fp, fn, tp)
    if isinstance(y_test, pd.DataFrame):
        true_pos = len(y_test.loc[(y_test["Class"] ==1)])
    if isinstance(y_test, np.ndarray):
        true_pos = np.count_nonzero(y_test == 1)
    
    #tp/true_pos # recall home_grown
    #tp/(tp+fp) # precision home_grown
    fnr = fn / true_pos
    return fnr
metrics = [precision_score, 
           recall_score,
           fnr,
           f1_score,
           lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=0.5),
           roc_auc_score]
metrics_names = ["Precision", 
                 "Recall", 
                 "False Negative",
                 "F1",
                 "F0.5",
                 "AUC"]
samples = [(test_prepared, y_test)]
models_names = ["Logistic, Ratio(1F:1NF)"]
def evaluate(models, metrics, samples, metrics_names, models_names):
    results = np.zeros((len(samples) * len(models), len(metrics)))
    samples_names = []
#     for m in models_names:
#         samples_names.extend([m + " Train", m + " Test"])
    for m_num, model in enumerate(models):
        for row, sample in enumerate(samples):
            for col, metric in enumerate(metrics):
                results[row + m_num * 2, col] = metric(sample[1], model.predict(sample[0]))
    results = pd.DataFrame(results, columns=metrics_names, index=models_names)
    return results
models = [log_reg]
res = evaluate(models, metrics, samples, metrics_names, models_names)
res
from sklearn.metrics import accuracy_score

log_acc = accuracy_score(y_test, y_pred)
log_acc
from sklearn.metrics import recall_score

log_recall = recall_score(y_test, y_pred)
log_recall
from sklearn.metrics import precision_score
log_pre = precision_score(y_test, y_pred)
log_pre
from sklearn.metrics import roc_auc_score
log_roc = roc_auc_score(y_test, y_pred)
log_roc
# Let us calculate the False negative rate (FNR), Miss rate
# def fnr(y_test, y_pred):
# from sklearn.metrics import confusion_matrix
log_cm = confusion_matrix(y_test, y_pred)
#tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
tn, fp, fn, tp = log_cm.ravel()
# (tn, fp, fn, tp)
true_pos = len(y_test.loc[(y_test["Class"] ==1)])
#tp/true_pos # recall home_grown
#tp/(tp+fp) # precision home_grown
fnr = fn / true_pos
fnr
import itertools

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(log_cm
                      , classes=class_names
                      , title='Confusion matrix for base model')
plt.show()
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.grid()
plt.show()

fraud_count = len(fraud)
# fraud_count
smpl_non_fraud = non_fraud.sample(n=fraud_count*10, random_state=42)
# len(smpl_non_fraud)
train_data=smpl_non_fraud.append(fraud, ignore_index=True)

train_data = shuffle(train_data)
train_data.reset_index(drop=True)

len(train_data)
train_data.info()
X = train_data.loc[:,train_data.columns != 'Class']
y = train_data.loc[:,train_data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
cc_prepared = num_pipeline.fit_transform(X_train)
test_prepared = num_pipeline.fit_transform(X_test)

log_reg.fit(cc_prepared, y_train)
y_pred = log_reg.predict(test_prepared)
log_pre = precision_score(y_test, y_pred)
log_pre
samples = [(test_prepared, y_test)]
models_names = ["Logistic, Ratio(1F:10NF)"]
models = [log_reg]
res_10 = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_10)
res

fraud_count = len(fraud)
# fraud_count
smpl_non_fraud = non_fraud.sample(n=fraud_count*20, random_state=42)
# len(smpl_non_fraud)
train_data=smpl_non_fraud.append(fraud, ignore_index=True)

train_data = shuffle(train_data)
train_data.reset_index(drop=True)

len(train_data)
X = train_data.loc[:,train_data.columns != 'Class']
y = train_data.loc[:,train_data.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
cc_prepared = num_pipeline.fit_transform(X_train)
test_prepared = num_pipeline.fit_transform(X_test)

log_reg.fit(cc_prepared, y_train)
y_pred = log_reg.predict(test_prepared)
log_pre = precision_score(y_test, y_pred)
log_pre
samples = [(test_prepared, y_test)]
models_names = ["Logistic, Ratio(1F:20NF)"]
res_20 = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_20)
res
from  sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(cc_prepared, y_train)
samples = [(test_prepared, y_test)]
models_names = ["RandomForest, Ratio(1F:1NF)"]
models = [rf_clf]
res_rf = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_rf)
res
from sklearn.model_selection import cross_val_score


score_randomforest = cross_val_score(rf_clf,test_prepared,y_test['Class'],scoring='recall',cv=10)
score_randomforest

validation_scores = pd.DataFrame(columns=['Model Name','mean','SD'])

validation_scores.loc[0] = ['RF',score_randomforest.mean(),score_randomforest.std()]

validation_scores
score_LR = cross_val_score(log_reg,test_prepared,y_test['Class'],scoring='recall',cv=10)
score_LR
validation_scores.loc[1] = ['LC',score_LR.mean(),score_LR.std()]
validation_scores
def stat_test(control, treatment):
    #paired t-test; two-tailed p-value      A   ,    B
    (t_score, p_value) = stats.ttest_rel(control, treatment)

    if p_value > 0.05/2:  #Two sided 
        print('There is no significant difference between the two machine learning pipelines (Accept H0)')
    else:
        print('The two machine learning pipelines are different (reject H0) \n(t_score, p_value) = (%.2f, %.5f)'%(t_score, p_value) )
        if t_score > 0.0: #in the case of regression lower RMSE is better; A is lower 
            print('Machine learning pipeline A is better than B')
        else:
            print('Machine learning pipeline B is better than A')
    return p_value
from sklearn.model_selection import cross_val_score
from scipy import stats
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# A sampling based bakeoff using *K-fold cross-validation*: 
# it randomly splits the training set into K distinct subsets (k=30)
# this bakeoff framework can be used for regression or classification
#Control system is a linear regression based pipeline

kFolds=10

y_test_ctrl = y_test
# Logistic Regression as base
control = cross_val_score(log_reg, test_prepared, y_test_ctrl['Class'],
                             scoring='recall', cv=kFolds)

# control_acc = control.mean()
# # control = control.mean()
# display_scores(control)

# display_scores(lin_rmse_scores)
#Treatment system is a random forest based pipeline

treatment = cross_val_score(rf_clf, test_prepared, y_test['Class'],
                         scoring='recall', cv=kFolds)

treatment_acc = treatment.mean()

pval = stat_test(control, treatment)

pval
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [5, 10, 20, 29]},
    # then try 8 (2×4) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [10, 20], 'max_features': [5, 10, 20,29]},
  ]

# train across 5 folds, that's a total of (12+8)*5=100 rounds of training 
grid_search = GridSearchCV(rf_clf, param_grid, cv=5,
                           scoring='recall')
grid_search.fit(cc_prepared, y_train['Class'])
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
pd.DataFrame(grid_search.cv_results_)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
attributes = list(X_train.columns.values)
attributes
sortedFeatures = sorted(zip(feature_importances,attributes), reverse=False)
sortedFeatures
np.array(sortedFeatures)[:, 0]
# Plot the feature importances of the forest
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.figure() 
plt.title("Feature importances")
sortedNames = np.array(sortedFeatures)[:, 1]
sortedImportances = np.array(sortedFeatures)[:, 0]

plt.title('Feature Importances')
plt.barh(range(len(sortedNames)), sortedImportances, color='b', align='center')
plt.yticks(range(len(sortedNames)), sortedNames)
plt.xlabel('Relative Importance')
plt.grid()
plt.show()
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class BestFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
k=5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices 
np.array(attributes)[top_k_feature_indices]
sorted(zip(feature_importances, attributes), reverse=True)[:k]
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', num_pipeline),
    ('feature_selection', BestFeatureSelector(feature_importances, k))
])
trainFile_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(X_train)
y_pred = preparation_and_feature_selection_pipeline.fit_transform(X_test)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

X=trainFile_prepared_top_k_features
Y=y_train



models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LR_L1', LogisticRegression(C=1,penalty='l1',max_iter=1000) ))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
models.append(('RF_10',RandomForestClassifier(n_estimators=10)))
models.append(('RF_100',RandomForestClassifier(n_estimators=100)))
#models.append(('RF_5.21',RandomForestClassifier(max_features=5,n_estimators=21)))

#models.append(('KNN_5', KNeighborsClassifier(n_neighbors=5,n_jobs=-1)))

#models.append(('CART', DecisionTreeClassifier()))
# evaluate each model in turn
results = []
names = []
scoring ='recall'#'roc_auc' #'recall' #'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure(figsize=(8, 6))
fig.suptitle('Classification Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.grid()
pyplot.show()
k=20
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', num_pipeline),
    ('feature_selection', BestFeatureSelector(feature_importances, k)),
    ('rf', RandomForestClassifier(bootstrap= False, n_estimators = 10, max_features=k))
   
])
prepare_select_and_predict_pipeline.fit(X_train, y_train)
prepare_select_and_predict_pipeline.predict(X_test)

samples = [(X_test, y_test)]
models_names = ["RandomForest, Feautres=20, Ratio(1F:1NF)"]

models = [prepare_select_and_predict_pipeline]
res_fe = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_fe)
res
# res.drop(res.tail(1).index,inplace=True)
# res
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

vot_clf = VotingClassifier(
    estimators=[
                ('rf', RandomForestClassifier(bootstrap= False, n_estimators = 10, max_features=20)),
#                 ('svc', SVC()),
                ('lr', LogisticRegression()),
                ('CART', DecisionTreeClassifier()),
#                 ('LDA', LinearDiscriminantAnalysis())
                ], 
    voting='hard')

k=20
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', num_pipeline),
    ('feature_selection', BestFeatureSelector(feature_importances, k)),
    ('vot_clf', vot_clf)
   
])
prepare_select_and_predict_pipeline.fit(X_train, y_train)
prepare_select_and_predict_pipeline.predict(X_test)

samples = [(X_test, y_test)]
models_names = ["Voting(LR,RF,DT), Feautres=20(1F:1NF)"]

models = [prepare_select_and_predict_pipeline]
res_vot = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_vot)
res
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
sns.pairplot(trainFile, hue="Class", size=2);
# Dont Run it Again
#Image(filename='.\pair_plot.jpg', width=500)
Image(filename='.\image2.jpeg', width=500)
Image(filename='.\image1.jpeg', width=500)
non_fraud = trainFile.loc[(trainFile["Class"] ==0)]
non_fraud = non_fraud.drop(["Class","Time"],axis=1)

fraud = trainFile.loc[(trainFile["Class"] ==1)]
fraud = fraud.drop(["Class","Time"],axis=1)

nfrdStd = num_pipeline.fit_transform(non_fraud)
frdStd = num_pipeline.fit_transform(fraud)
#non_fraud
from sklearn.cluster import KMeans
nfrdCentro = KMeans(n_clusters=6, random_state=0).fit(nfrdStd)
frdCentro = KMeans(n_clusters=4, random_state=0).fit(frdStd)

#kmeans = np.concatenate((nfrdCentro, frdCentro), axis=0)
kmeans = np.concatenate((nfrdCentro.cluster_centers_, frdCentro.cluster_centers_), axis=0)

#kmeans.labels_
#kmeans.cluster_centers_
import numpy as np
import scipy
#X=trainFile_0.values
dist_0=[]
dist_1=[]
dist_2=[]
dist_3=[]
dist_4=[]
dist_5=[]
dist_6=[]
dist_7=[]
dist_8=[]
dist_9=[]
isFraud=[]

for x in frdStd:
    dist_0.append(np.sqrt(np.sum((x-kmeans[0])**2,axis=0)))
    dist_1.append(np.sqrt(np.sum((x-kmeans[1])**2,axis=0)))
    dist_2.append(np.sqrt(np.sum((x-kmeans[2])**2,axis=0)))
    dist_3.append(np.sqrt(np.sum((x-kmeans[3])**2,axis=0)))
    dist_4.append(np.sqrt(np.sum((x-kmeans[4])**2,axis=0)))
    dist_5.append(np.sqrt(np.sum((x-kmeans[5])**2,axis=0)))
    dist_6.append(np.sqrt(np.sum((x-kmeans[6])**2,axis=0)))
    dist_7.append(np.sqrt(np.sum((x-kmeans[7])**2,axis=0)))
    dist_8.append(np.sqrt(np.sum((x-kmeans[8])**2,axis=0)))
    dist_9.append(np.sqrt(np.sum((x-kmeans[9])**2,axis=0)))
    isFraud.append(1)

distDf_frd = pd.DataFrame({
        "dist_0": dist_0,
        "dist_1": dist_1,
        "dist_2": dist_2,
        "dist_3": dist_3,
        "dist_4": dist_4,
        "dist_5": dist_5,
        "dist_6": dist_6,
        "dist_7": dist_7,
        "dist_8": dist_9,
        "dist_9": dist_9,
        "Class": isFraud})

ndist_0=[]
ndist_1=[]
ndist_2=[]
ndist_3=[]
ndist_4=[]
ndist_5=[]
ndist_6=[]
ndist_7=[]
ndist_8=[]
ndist_9=[]
nisFraud=[]

for x in nfrdStd:
    ndist_0.append(np.sqrt(np.sum((x-kmeans[0])**2,axis=0)))
    ndist_1.append(np.sqrt(np.sum((x-kmeans[1])**2,axis=0)))
    ndist_2.append(np.sqrt(np.sum((x-kmeans[2])**2,axis=0)))
    ndist_3.append(np.sqrt(np.sum((x-kmeans[3])**2,axis=0)))
    ndist_4.append(np.sqrt(np.sum((x-kmeans[4])**2,axis=0)))
    ndist_5.append(np.sqrt(np.sum((x-kmeans[5])**2,axis=0)))
    ndist_6.append(np.sqrt(np.sum((x-kmeans[6])**2,axis=0)))
    ndist_7.append(np.sqrt(np.sum((x-kmeans[7])**2,axis=0)))
    ndist_8.append(np.sqrt(np.sum((x-kmeans[8])**2,axis=0)))
    ndist_9.append(np.sqrt(np.sum((x-kmeans[9])**2,axis=0)))
    nisFraud.append(0)
    
distDf_nfrd = pd.DataFrame({
        "dist_0": ndist_0,
        "dist_1": ndist_1,
        "dist_2": ndist_2,
        "dist_3": ndist_3,
        "dist_4": ndist_4,
        "dist_5": ndist_5,
        "dist_6": ndist_6,
        "dist_7": ndist_7,
        "dist_8": ndist_8,
        "dist_9": ndist_9,
        "Class": nisFraud})
distDf_nfrd.describe()
distDf_frd.describe()
#dft_1=dft.loc[(dft["isFraud"] ==1)]
#print(len(dft_1))
#dft_0=dft.loc[(dft["isFraud"] ==0)]
distDf_nfrdSubSet=distDf_nfrd.sample(frac=0.03)
#print(len(dft_0))
trainFile=distDf_nfrdSubSet.append(distDf_frd, ignore_index=True)

#dft_1=dft_1.sample(frac=0.3)
#trainFile=dft_1.append(dft_0, ignore_index=True)
#trainFile=dft
trainFile= shuffle(trainFile)
#trainFile.reset_index(drop=True)
trainFile= shuffle(trainFile)
#trainFile.reset_index(drop=False)
trainFile= shuffle(trainFile)
dataY=trainFile[("Class")]
dataX=trainFile.drop(["Class"],axis=1)
#X,Y
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

%matplotlib inline

from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

#from sklearn import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

Y=dataY.values
X=dataX.values


#X_train=X
#y_train=Y
# prepare models
models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('LR_L1', LogisticRegression(C=1,penalty='l1',max_iter=1000) ))
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
models.append(('RF_10',RandomForestClassifier(n_estimators=10)))
models.append(('RF_100',RandomForestClassifier(n_estimators=100)))
#models.append(('RF_5.21',RandomForestClassifier(max_features=5,n_estimators=21)))

#models.append(('KNN_5', KNeighborsClassifier(n_neighbors=5,n_jobs=-1)))

#models.append(('CART', DecisionTreeClassifier()))
# evaluate each model in turn
results = []
names = []
scoring ='recall'#'roc_auc' #'recall' #'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = pyplot.figure(figsize=(8, 8))
fig.suptitle('Classification Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.grid()
pyplot.show()
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
param_distribs = {
        'penalty': ('l1','l2'),
        'C': (0.001,0.01,1,10,100),
    }

# scoring ='roc_auc' #'recall'
scoring ='recall'
log_reg = LogisticRegression(max_iter=10000)
rnd_search = RandomizedSearchCV(log_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring=scoring, random_state=42) #'neg_mean_squared_error'
X_train, X_test, y_train, y_test = train_test_split(dataX.values,dataY.values,test_size = 0.3, random_state = 42)

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3,10,30,100], 'max_features': [3,4,5,6,7,8,9]},
    # then try 8 (2×4) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators':  [3,10,30,100], 'max_features': [3,4,5,6,7,8,9]},
  ]

# train across 5 folds, that's a total of (12+8)*5=100 rounds of training 
grid_search = GridSearchCV(rf_clf, param_grid, cv=5,
                           scoring='recall')
#X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42)
grid_search.fit(X_train, y_train)

print("BEST PARAMS")
print(grid_search.best_params_)
grid_search.best_estimator_
cvres = grid_search.cv_results_
print("mean_score  params")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


pd.DataFrame(grid_search.cv_results_)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

attributes = list(dataX.columns.values)
attributes

sortedFeatures = sorted(zip(feature_importances,attributes), reverse=False)
sortedFeatures

# Plot the feature importances of the forest
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
plt.figure() 
plt.title("Feature importances")
sortedNames = np.array(sortedFeatures)[:, 1]
sortedImportances = np.array(sortedFeatures)[:, 0]

plt.title('Feature Importances')
plt.barh(range(len(sortedNames)), sortedImportances, color='b', align='center')
plt.yticks(range(len(sortedNames)), sortedNames)
plt.xlabel('Relative Importance')
plt.grid()
plt.show()

grid_search.best_estimator_.fit(X_train, y_train)


samples = [(X_test, y_test)]
models_names = ["RandomForest, Feature Transform"]
models = [grid_search.best_estimator_]
res_ft = evaluate(models, metrics, samples, metrics_names, models_names)
res = res.append(res_ft)
res
from sklearn.metrics import roc_curve

y_pred = grid_search.best_estimator_.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.grid()
plt.show()

# Plot confusion matrix
best_cm = confusion_matrix(y_test, y_pred)

class_names = [0,1]
plt.figure()
plot_confusion_matrix(best_cm
                      , classes=class_names
                      , title='Confusion matrix for best model')
plt.show()
from sklearn.model_selection import cross_val_score
from scipy import stats
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# A sampling based bakeoff using *K-fold cross-validation*: 
# it randomly splits the training set into K distinct subsets (k=30)
# this bakeoff framework can be used for regression or classification
#Control system is a linear regression based pipeline

kFolds=10
# Logistic Regression as base
control = cross_val_score(log_reg, test_prepared, y_test_ctrl['Class'],
                             scoring='recall', cv=kFolds)

# control_acc = control.mean()
# control = control.mean()
# display_scores(lin_rmse_scores)
# display_scores(control)


#Treatment system is a random forest based pipeline

treatment = cross_val_score(grid_search.best_estimator_, X_test, y_test,
                         scoring='recall', cv=kFolds)

# treatment_acc = treatment.mean()
# treatment = treatment.mean()
# display_scores(treatment)
# treatment = tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores)


pval = stat_test(control, treatment)

pval
    