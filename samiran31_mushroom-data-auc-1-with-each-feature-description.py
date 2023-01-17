import warnings

warnings.filterwarnings("ignore")

import shutil

import os

import pandas as pd

import matplotlib

matplotlib.use(u'nbAgg')

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pickle

from sklearn.manifold import TSNE

from sklearn import preprocessing

import pandas as pd

from multiprocessing import Process# this is used for multithreading

import multiprocessing

import codecs# this is used for file operations 

import random as r

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from collections import Counter
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print(data.shape)

data.head()
data['class'].unique()
data['class']=data['class'].map({'p':0,'e':1})
data['class'].unique()
data.info()
sns.countplot(data['class'])
sns.countplot(x='cap-shape',hue='class',data=data)
sns.catplot(x='cap-shape',y='class',data=data,kind='bar')
data['cap-shape'].value_counts()
data.loc[data['cap-shape']=='c']
data['cap-shape']=data['cap-shape'].map({'x':'x','f':'f','k':'KC','c':'KC','b':'BC','c':'BC'})
data['cap-shape'].value_counts()
sns.countplot(x='cap-surface',hue='class',data=data)
sns.catplot(x='cap-surface',y='class',data=data,kind='bar')
data['cap-surface'].value_counts()
data[data['cap-surface']=='g']
data['cap-surface']=data['cap-surface'].map({'y':'y','s':'SG','g':'SG','f':'f'})
data['cap-surface'].value_counts()
sns.countplot(x='cap-color',hue='class',data=data)
sns.catplot(x='cap-color',y='class',data=data,kind='bar')
data['cap-color'].value_counts()
data['cap-color']=data['cap-color'].map({'n':'n','g':'g','e':'e','y':'PBY','p':'PBY','b':'PBY','w':'CURW','c':'CURW','u':'CURW','r':'CURW'})
data['cap-color'].value_counts()
sns.countplot(x='bruises',hue='class',data=data)
sns.catplot(x='bruises',y='class',data=data,kind='bar')
data['bruises'].value_counts()
sns.countplot(x='odor',hue='class',data=data)
sns.catplot(x='odor',y='class',data=data,kind='bar')
data['odor'].value_counts()
data['odor']=data['odor'].map({'n':'n','p':'POI','f':'POI','c':'POI','y':'POI','s':'POI','m':'POI','a':'EDI','l':'EDI'})
data['odor'].value_counts()
sns.countplot(x='gill-attachment',hue='class',data=data)
sns.catplot(x='gill-attachment',y='class',data=data,kind='bar')
data['gill-attachment'].value_counts()
data[(data['gill-attachment']=='a')&(data['class']==1)]
data[(data['gill-attachment']=='a')&(data['class']==1)]['class'].count()
data[(data['gill-attachment']=='a')]['class'].count()
Edible_percent=data[(data['gill-attachment']=='a')&(data['class']==1)]['class'].sum()/data[(data['gill-attachment']=='a')]['class'].count()*100

print("The percentage of mushroom which are edible when gill is attached is:",Edible_percent)
sns.countplot(x='gill-spacing',hue='class',data=data)
sns.catplot(x='gill-spacing',y='class',data=data,kind='bar')
data['gill-spacing'].value_counts()
sns.countplot(x='gill-size',hue='class',data=data)
sns.catplot(x='gill-size',y='class',data=data,kind='bar')
data['gill-size'].value_counts()
sns.countplot(x='gill-color',hue='class',data=data)
sns.catplot(x='gill-color',y='class',data=data,kind='bar')
data['gill-color'].value_counts()
data['gill-color']=data['gill-color'].map({'b':'BR','p':'p','w':'WY','n':'OEKNU','g':'GH','h':'GH','u':'OEKNU','k':'OEKNU','e':'OEKNU','y':'WY','o':'OEKNU','r':"BR"})
data['gill-color'].value_counts()
sns.catplot(x='gill-color',y='class',data=data,kind='bar')
sns.countplot(x='stalk-shape',hue='class',data=data)
sns.catplot(x='stalk-shape',y='class',data=data,kind='bar')
data['stalk-shape'].value_counts()
sns.countplot(x='stalk-root',hue='class',data=data)
sns.catplot(x='stalk-root',y='class',data=data,kind='bar')
data['stalk-root'].value_counts()
data['stalk-root']=data['stalk-root'].map({'r':'RC','c':'RC','b':'b','?':'missing','e':'e'})
data['stalk-root'].value_counts()
sns.countplot(x='stalk-surface-above-ring',hue='class',data=data)
sns.catplot(x='stalk-surface-above-ring',y='class',data=data,kind='bar')
data['stalk-surface-above-ring'].value_counts()
data['stalk-surface-above-ring']=data['stalk-surface-above-ring'].map({'s':'SFY','f':'SFY','y':'SFY','k':'k'})
data['stalk-surface-above-ring'].value_counts()
sns.countplot(x='stalk-surface-below-ring',hue='class',data=data)
sns.catplot(x='stalk-surface-below-ring',y='class',data=data,kind='bar')
data['stalk-surface-below-ring'].value_counts()
data['stalk-surface-below-ring']=data['stalk-surface-below-ring'].map({'s':'SFY','f':'SFY','y':'SFY','k':'k'})
data['stalk-surface-below-ring'].value_counts()
sns.countplot(x='stalk-color-above-ring',hue='class',data=data)
sns.catplot(x='stalk-color-above-ring',y='class',data=data,kind='bar')
data['stalk-color-above-ring'].value_counts()
data['stalk-color-above-ring']=data['stalk-color-above-ring'].map({'e':'EOG','o':'EOG','g':'EOG','n':'NBCY','b':'NBCY','c':'NBCY','y':'NBCY','w':'w','p':'p'})
data['stalk-color-above-ring'].value_counts()
sns.countplot(x='stalk-color-below-ring',hue='class',data=data)
sns.catplot(x='stalk-color-below-ring',y='class',data=data,kind='bar')
data['stalk-color-below-ring'].value_counts()
data['stalk-color-below-ring']=data['stalk-color-below-ring'].map({'e':'EOG','o':'EOG','g':'EOG','n':'BYCN','b':'BYCN','y':'BYCN','c':'BYCN','w':'w','p':'p'})
data['stalk-color-below-ring'].value_counts()
sns.countplot(x='veil-type',hue='class',data=data)
sns.catplot(x='veil-type',y='class',data=data,kind='bar')
data=data.drop(['veil-type'],axis = 1)
sns.countplot(x='veil-color',hue='class',data=data)
sns.catplot(x='veil-color',y='class',data=data,kind='bar')
data['veil-color'].value_counts()
data=data.drop(['veil-color'],axis = 1)
sns.countplot(x='ring-number',hue='class',data=data)
sns.catplot(x='ring-number',y='class',data=data,kind='bar')
data['ring-number'].value_counts()
data['ring-number']=data['ring-number'].map({'o':'ON','n':'ON','t':'t'})
data['ring-number'].value_counts()
sns.countplot(x='ring-type',hue='class',data=data)
sns.catplot(x='ring-type',y='class',data=data,kind='bar')
data['ring-type'].value_counts()
data['ring-type']=data['ring-type'].map({'p':'PF','f':'PF','e':'e','l':'LN','n':'LN'})
data['ring-type'].value_counts()
sns.countplot(x='spore-print-color',hue='class',data=data)
sns.catplot(x='spore-print-color',y='class',data=data,kind='bar')
data['spore-print-color'].value_counts()
data['spore-print-color']=data['spore-print-color'].map({'o':'OYBU','y':'OYBU','b':'OYBU','u':'OYBU','h':'HR','r':'HR','w':'w','k':'KN','n':'KN'})
data['spore-print-color'].value_counts()
sns.countplot(x='population',hue='class',data=data)
sns.catplot(x='population',y='class',data=data,kind='bar')
data['population'].value_counts()
data['population']=data['population'].map({'n':'NAC','a':'NAC','c':'NAC','s':'SY','y':'SY','v':'v'})
data['population'].value_counts()
sns.countplot(x='habitat',hue='class',data=data)
sns.catplot(x='habitat',y='class',data=data,kind='bar')
data['habitat'].value_counts()
data['habitat']=data['habitat'].map({'g':'GD','d':'GD','u':'UL','l':'UL','m':'MW','w':'MW','p':'p'})
data['habitat'].value_counts()
data_dum=data.copy()

#data_dum=data_dum.drop(columns=['class'],axis=0)

data_dum
for i in data.columns:

    if (str(i)!=str('class')):

        data_dum=pd.get_dummies(data_dum,columns=[i],prefix=[i])

data_dum        
#saleprice correlation matrix

k = 50 #number of variables for heatmap

plt.figure(figsize=(16,8))

corrmat = data_dum.corr()

# picking the top 15 correlated features

cols = corrmat.nlargest(k, 'class')['class'].index

cm = np.corrcoef(data_dum[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
dataset=data_dum[cols]

dataset
Data_out=dataset['class']

input_data=dataset.drop(['class'],axis=1)
#data_y = result['Class']

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, X_test, y_train, y_test = train_test_split(input_data,Data_out,stratify=Data_out,test_size=0.10)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train,stratify=y_train,test_size=0.10)
print('Number of data points in train data:', X_train.shape[0])

print('Number of data points in test data:', X_test.shape[0])

print('Number of data points in cross validation data:', X_cv.shape[0])
print("-"*10, "Distribution of output variable in train data", "-"*10)

train_distr = Counter(y_train)#it will count how many 0 and how many 1 present.

train_len = len(y_train)

print("Class 0: ",int(train_distr[0])/train_len,"Class 1: ", int(train_distr[1])/train_len)

print("-"*10, "Distribution of output variable in test data", "-"*10)

test_distr = Counter(y_test)

test_len = len(y_test)

print("Class 0: ",int(test_distr[0])/test_len, "Class 1: ",int(test_distr[1])/test_len)

print(train_distr)
def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    #print("Number of misclassified points ",(len(test_y)-np.trace(C))/len(test_y)*100)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    plt.figure(figsize=(20,4))

    labels = [0,1]

    cmap=sns.light_palette("green")

    # representing A in heatmap format

    #print("-"*50, "Confusion matrix", "-"*50)

    #plt.figure(figsize=(10,5))

    plt.subplot(1, 3, 1)

    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Confusion matrix")

    #plt.show()



    #print("-"*50, "Precision matrix", "-"*50)

    #plt.figure(figsize=(10,5))

    plt.subplot(1, 3, 2)

    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Precision matrix")

    

    #plt.show()

    print("Sum of columns in precision matrix",B.sum(axis=0))

    

    # representing B in heatmap format

    #print("-"*50, "Recall matrix"    , "-"*50)

    #plt.figure(figsize=(10,5))

    plt.subplot(1, 3, 3)

    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Recall matrix")

    plt.show()

    print("Sum of rows in precision matrix",A.sum(axis=1))
from sklearn import metrics
test_data_len = X_test.shape[0]

cv_data_len = X_cv.shape[0]

#print(test_data_len)

#print(cv_data_len)

cv_predicted_y = np.zeros((cv_data_len,1))

#print(cv_predicted_y.shape)
rand_probs = np.random.rand(1,2)

#print(rand_probs)

#print(sum(sum(rand_probs)))

cv_predicted_y = ((rand_probs/sum(sum(rand_probs)))[0])

cv_predicted_y_12 = ((rand_probs/sum(sum(rand_probs))))

#print(cv_predicted_y)

#print(cv_predicted_y_12)



cv_predicted_y_1 = np.zeros((cv_data_len,1))

#print(cv_predicted_y_1)

#print(y_test.shape)

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
test_data_len = X_test.shape[0]

cv_data_len = X_cv.shape[0]



# we create a output array that has exactly same size as the CV data

cv_predicted_y = np.zeros((cv_data_len,2))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,2)

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))

#print(cv_predicted_y.shape)

#print(y_cv.shape)



# Test-Set error.

#we create a output array that has exactly same as the test data

test_predicted_y = np.zeros((test_data_len,2))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,2)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))





predicted_y =np.argmax(test_predicted_y, axis=1)

plot_confusion_matrix(y_test, predicted_y)

print("Scores using the AUC model",roc_auc_score(y_test,predicted_y))



#print(cv_predicted_y)

#print(test_predicted_y.shape)

#print(predicted_y)
from sklearn.linear_model import SGDClassifier

alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(X_train, y_train)

    predict_y = sig_clf.predict_proba(X_test)

    log_error_array.append(log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, log_error_array,c='g')

for i, txt in enumerate(np.round(log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(X_train, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(X_train, y_train)



predict_y = sig_clf.predict_proba(X_train)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_test)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y =np.argmax(predict_y,axis=1)

print("Total number of data points :", len(predicted_y))

print("Scores using the AUC model",roc_auc_score(y_test,predicted_y))





plot_confusion_matrix(y_test, predicted_y)
alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l1', loss='hinge', random_state=42)

    clf.fit(X_train, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(X_train, y_train)

    predict_y = sig_clf.predict_proba(X_test)

    log_error_array.append(log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, log_error_array,c='g')

for i, txt in enumerate(np.round(log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l1', loss='hinge', random_state=42)

clf.fit(X_train, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(X_train, y_train)



predict_y = sig_clf.predict_proba(X_train)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_test)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y =np.argmax(predict_y,axis=1)

print("Total number of data points :", len(predicted_y))

print("Scores using the AUC model",roc_auc_score(y_test,predicted_y))



plot_confusion_matrix(y_test, predicted_y)
import xgboost as xgb

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(X_train, label=y_train)

d_test = xgb.DMatrix(X_test, label=y_test)



watchlist = [(d_train, 'train'), (d_test, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=20, verbose_eval=10)



xgdmat = xgb.DMatrix(X_train,y_train)

predict_y = bst.predict(d_test)

print("The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



predicted_y =np.array(predict_y>0.5,dtype=int)

print("Total number of data points :", len(predicted_y))

print("Scores using the AUC model",roc_auc_score(y_test,predicted_y))



plot_confusion_matrix(y_test, predicted_y)
# --------------------------------

# default parameters 

# sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# class_weight=None)



# Some of methods of RandomForestClassifier()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# predict_proba (X)	Perform classification on samples in X.



# some of attributes of  RandomForestClassifier()

# feature_importances_ : array of shape = [n_features]

# The feature importances (the higher, the more important the feature).



# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# --------------------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------



alpha = [100,200,500,1000,2000]

max_depth = [5, 10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(X_train, y_train)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(X_train, y_train)

        sig_clf_probs = sig_clf.predict_proba(X_cv)

        cv_log_error_array.append(log_loss(y_cv, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(y_cv, sig_clf_probs)) 



'''fig, ax = plt.subplots()

features = np.dot(np.array(alpha)[:,None],np.array(max_depth)[None]).ravel()

ax.plot(features, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[int(i/2)],max_depth[int(i%2)],str(txt)), (features[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()

'''



best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(X_train, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(X_train, y_train)



predict_y = sig_clf.predict_proba(X_train)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_cv)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(X_test)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

predicted_y =np.argmax(predict_y,axis=1)

print("Scores using the AUC model",roc_auc_score(y_test,predicted_y))
