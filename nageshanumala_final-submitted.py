import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
transactions=pd.read_csv("/kaggle/input/ucrfinal/datasetForFinalAssignment.csv")
transactions.info()
test=pd.read_csv("/kaggle/input/ucrfinal/datasetForFinalTest.csv")
test.info()
transactions.rename(columns={'signup_time-purchase_time':'TimeBetween',

                          'N[device_id]':'TimesDeviceUsed',

                          }, 

                 inplace=True)
TimesIPaddress = transactions['ip_address'].value_counts()

TimesIPaddress.values
transactions["TimesIPaddress"] = transactions.groupby('ip_address')['user_id'].transform('count')
pd.pivot_table(transactions, index="class", values=["TimeBetween","TimesDeviceUsed","TimesIPaddress","purchase_value","age"], aggfunc="mean")
E = transactions[transactions['class'] == 1]

F= transactions[transactions['class'] == 0]

E["purchase_value"].describe()
F["purchase_value"].describe()
transactions['class'].value_counts().plot.bar()

transactions['class'].value_counts()
(7007/74691)*100
sns.countplot(x='source',hue="class",data=transactions) 
sns.countplot(x='browser', hue="class",data=transactions)

plt.xticks(rotation=-45)
sns.countplot(x='sex', hue="class",data=transactions) 

import seaborn as sns

sns.boxplot(x="class", y="TimesDeviceUsed", data=transactions) 
sns.boxplot(x="class", y="purchase_value", data=transactions)
sns.boxplot(x="class", y="age", data=transactions)
sns.boxplot(x="class", y="TimeBetween", data=transactions) 
sns.countplot(data=transactions, x="age",hue="class")
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

transactions['sex'] = lb_make.fit_transform(transactions['sex'])

sourcedummies=pd.get_dummies(transactions["source"], columns=["source"])

browserdummies=pd.get_dummies(transactions["browser"], columns=["browser"])



transactions = pd.concat([transactions, sourcedummies], axis=1)

transactions = pd.concat([transactions, browserdummies], axis=1)





transactions.info()
y = transactions["class"]



X = transactions[["age","sex","TimeBetween","purchase_value","TimesDeviceUsed","TimesIPaddress","Chrome","FireFox","IE","Opera","Safari","Ads","Direct","SEO"]]



I=X.copy(deep=True)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)

X.head()
plt.figure(figsize = (14,14))

plt.title(' Transactions features correlation plot (Pearson)')

corr = I.corr()

sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")

plt.show()
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test =   train_test_split(X,y,test_size = 0.1, stratify = y)
X_train.shape, X_test.shape
type(X_test)
indexA=X_test.index.values

indexA
A=I.iloc[indexA]

from imblearn.over_sampling import SVMSMOTE

#sm = SMOTE(sampling_strategy='str', random_state=None, k_neighbors=2, m_neighbors='deprecated', out_step='deprecated', kind='deprecated', svm_estimator='deprecated', n_jobs=1, ratio=1.0)

sm=SVMSMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=1, m_neighbors=10, svm_estimator=None, out_step=0.5)

from imblearn.over_sampling import SMOTE, ADASYN

#sm = SMOTE(random_state=1)

X_res, y_res = sm.fit_sample(X_train, y_train)

#ada = ADASYN(random_state=42)

#X_res, y_res = ada.fit_sample(X_train, y_train)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#X_res = scaler.fit_transform(X_res)

#X_test = scaler.fit_transform(X_test)  

X_res.shape, y_res.shape


                   

np.sum(y_res)/len(y_res)
from sklearn.tree import  DecisionTreeClassifier as dt

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier as gbm

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn import svm

from lightgbm import LGBMClassifier

import xgboost as xgb

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics

import time

pd.set_option('display.max_columns', 100)





RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier

NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier

NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier





from sklearn.tree import ExtraTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm.classes import OneClassSVM

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.neighbors.classification import RadiusNeighborsClassifier

from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.multioutput import ClassifierChain

from sklearn.multioutput import MultiOutputClassifier

from sklearn.multiclass import OutputCodeClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.linear_model.ridge import RidgeClassifierCV

from sklearn.linear_model.ridge import RidgeClassifier

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    

from sklearn.gaussian_process.gpc import GaussianProcessClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble.bagging import BaggingClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import GaussianNB

from sklearn.semi_supervised import LabelPropagation

from sklearn.semi_supervised import LabelSpreading

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.naive_bayes import MultinomialNB  

from sklearn.neighbors import NearestCentroid

from sklearn.svm import NuSVC

from sklearn.linear_model import Perceptron

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC

#from sklearn.mixture import DPGMM

#from sklearn.mixture import GMM 

from sklearn.mixture import GaussianMixture

#from sklearn.mixture import VBGMM

clfdt =dt(criterion='gini',

    splitter='best',

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features=None,

    random_state=None,

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    class_weight=None,

    presort=False,)
start = time.time()

clfdt = clfdt.fit(X_res,y_res)

end = time.time()

(end-start)/60                   

classes = clfdt.predict(X_test)

classes
f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

f

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

precision                    

recall = tp/(tp + fn)

recall 
accuracy=(tp+tn)/(tp+tn+fp+fn)

accuracy
from sklearn import metrics

clfdt.score(X_res,y_res)
from sklearn.metrics import r2_score

r2_score(y_test,classes )
from sklearn.metrics import f1_score

f1_score(y_test, classes, average='binary')


A['y_test'] = y_test.tolist()
A["classes"]=classes.tolist()
P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative
costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfrf =rf(n_estimators='warn',

    criterion='entropy',

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    bootstrap=True,

    oob_score=False,

    n_jobs=None,

    random_state=None,

    verbose=0,

    warm_start=False,

    class_weight=None,)
start = time.time()

clfrf = clfrf.fit(X_res,y_res)

end = time.time()

(end-start)/60 
classes = clfrf.predict(X_test)

classes
f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

f
tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

precision
recall = tp/(tp + fn)

recall
accuracy=(tp+tn)/(tp+tn+fp+fn)

accuracy
from sklearn.metrics import r2_score

r2_score(y_test,classes )
from sklearn import metrics

clfrf.score(X_res,y_res)
from sklearn.metrics import f1_score

f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative
costOfFaslePositive=8*fp

costOfFaslePositive
costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfetc=ExtraTreesClassifier( n_estimators='warn',

    criterion='gini',

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    bootstrap=False,

    oob_score=False,

    n_jobs=None,

    random_state=None,

    verbose=0,

    warm_start=False,

    class_weight=None,)
start = time.time()

clfetc = clfetc.fit(X_res,y_res)

end = time.time()

(end-start)/60 
classes = clfetc.predict(X_test)

classes
f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)

from sklearn import metrics

clfetc.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
from sklearn.linear_model.logistic import LogisticRegression

classifier=LogisticRegression()

classifier.fit(X_res,y_res)

classes = classifier.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

classifier.score(X_res,y_res)
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfkn=KNeighborsClassifier(n_neighbors=2)
start = time.time()

clfkn = clfkn.fit(X_res,y_res)

end = time.time()

(end-start)/60
classes = clfkn.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
clfkn.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfgbm = gbm(

    loss='exponential',

    learning_rate=0.1,

    n_estimators=100,

    subsample=1.0,

    criterion='friedman_mse',

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_depth=3,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    init=None,

    random_state=None,

    max_features=None,

    verbose=0,

    max_leaf_nodes=None,

    warm_start=False,

    presort='auto',

    validation_fraction=0.1,

    n_iter_no_change=None,

    tol=0.0001,

)
start = time.time()

clfgbm = clfgbm.fit(X_res,y_res)

end = time.time()

(end-start)/60
classes = clfgbm.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

clfgbm.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfxgb=XGBClassifier(criterion="gini", max_depth=4, max_features=14, min_weight_fraction_leaf=0.05997, n_estimators=499)
start = time.time()

clfxgb = clfxgb.fit(X_res,y_res)

end = time.time()

(end-start)/60  
classes = clfxgb.predict(X_test.values)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

clfxgb.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clflgb = lgb.LGBMClassifier(criterion="gini", max_depth=5, max_features=14, min_weight_fraction_leaf=0.01674, n_estimators=499)
start = time.time()

clflgb = clflgb.fit(X_res,y_res)

end = time.time()

(end-start)/60                   



classes = clflgb.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

clflgb.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfac = AdaBoostClassifier(base_estimator=None,

    n_estimators=50,

    learning_rate=1.0,

    algorithm='SAMME.R',

    random_state=None,)

start = time.time()

clfac = clfac.fit(X_res,y_res)

end = time.time()

(end-start)/60  
classes = clfac.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

clfac.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clfcb = CatBoostClassifier( )
start = time.time()

clfcb = clfcb.fit(X_res,y_res)

end = time.time()

(end-start)/60  
classes = clfcb.predict(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

clfcb.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_res, y_res)
classes = gnb.predict(X_test) 

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

gnb.score(X_res,y_res)
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

#create an object of type LinearSVC

svc_model = LinearSVC(random_state=0)

#train the algorithm on training data and predict using the testing data

classes = svc_model.fit(X_res, y_res).predict(X_test)

#print the accuracy score of the model

print("LinearSVC accuracy : ",accuracy_score(y_test, classes, normalize = True))
f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

svc_model.score(X_res,y_res)
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
y_res_onehot = pd.get_dummies(y_res)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)

from keras.models import Sequential



from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout,Flatten





model = Sequential()

model.add(Dense(32, input_shape=(14,)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(16))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(2))

model.add(Activation('softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

        

model.fit(X_res, y_res_onehot,class_weight=class_weights, epochs=6, batch_size=32)
classes = model.predict_classes(X_test)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

svc_model.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

import numpy as np

import warnings

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

warnings.simplefilter('ignore')
clf1 = XGBClassifier(criterion="gini", max_depth=4, max_features=4, min_weight_fraction_leaf=0.05997, n_estimators=499)

clf2=lgb.LGBMClassifier(criterion="gini", max_depth=5, max_features=53, min_weight_fraction_leaf=0.01674, n_estimators=499)

clf3 = LinearSVC(random_state=0)

clf4 = GaussianNB()

clf5=LogisticRegression() 

clf6=AdaBoostClassifier()

lr = gbm() 



sclf = StackingClassifier(classifiers=[clf1,clf2,clf3,clf4,clf5,clf6],

                          meta_classifier=lr)
start = time.time()

sclf = sclf.fit(X_test.values,y_test)

end = time.time()

(end-start)/60
classes = sclf.predict(X_test.values)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn import metrics

sclf.score(X_res,y_res)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
clf1 = XGBClassifier(criterion="gini", max_depth=4, max_features=4, min_weight_fraction_leaf=0.05997, n_estimators=499)

clf2=lgb.LGBMClassifier(criterion="gini", max_depth=5, max_features=53, min_weight_fraction_leaf=0.01674, n_estimators=499)

clf3 = LinearSVC(random_state=0)

clf4 = GaussianNB()

clf5=LogisticRegression() 

clf6=AdaBoostClassifier()

lr = gbm() 



sclf = StackingClassifier(classifiers=[clf1,clf2,clf3,clf4,clf5,clf6],

                          meta_classifier=lr)
start = time.time()

sclf = sclf.fit(X_test.values,y_test)

end = time.time()

(end-start)/60
classes = sclf.predict(X_test.values)

f  = confusion_matrix( y_test, classes )#confusion_matrix(y_true, y_pred)

tn,fp,fn,tp = f.ravel() #tn, fp, fn, tp = f.ravel()

(tn, fp, fn, tp)
precision = tp/(tp+fp)

recall = tp/(tp + fn)

accuracy=(tp+tn)/(tp+tn+fp+fn)

print( precision, recall, accuracy)
from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

r2_score(y_test,classes ),f1_score(y_test, classes, average='binary')
A['y_test'] = y_test.tolist()

A["classes"]=classes.tolist()

P=A.query(" classes==0")

Z=P.query("y_test!= classes")

CostOfFalseNegative=Z["purchase_value"].sum(axis = 0, skipna = True) 

CostOfFalseNegative

costOfFaslePositive=8*fp

costOfFaslePositive

costOfWorngPrediction=costOfFaslePositive+CostOfFalseNegative

costOfWorngPrediction
Specificity = tn / (tn + fp)

Specificity
Sensitivity = tp / (tp + fn)

Sensitivity
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test,classes)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, classes)
# calculate roc curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, classes)
from sklearn.metrics import average_precision_score

average_precision_score(y_test, classes)
from sklearn.metrics import log_loss

log_loss(y_test, classes, eps=1e-15, normalize=True, sample_weight=None, labels=None)
from matplotlib import pyplot

from sklearn.metrics import roc_curve, auc

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score

from matplotlib import pyplot

precision, recall, thresholds = precision_recall_curve(y_test, classes)

# calculate F1 score

f1=f1_score(y_test, classes, average='binary')

# calculate precision-recall AUC

auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y_test, classes)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the precision-recall curve for the model

pyplot.plot(recall, precision, marker='.')

# show the plot

pyplot.show()
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

from inspect import signature



precision, recall, _ = precision_recall_curve(y_test, classes)



step_kwargs = ({'step': 'post'}

               if 'step' in signature(plt.fill_between).parameters

               else {})

plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])



cm = pd.crosstab(y_test.values, classes, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cm, 

            xticklabels=['Not Fraud', 'Fraud'],

            yticklabels=['Not Fraud', 'Fraud'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkblue", cmap="Blues")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
test.rename(columns={'signup_time-purchase_time':'TimeBetween',

                          'N[device_id]':'TimesDeviceUsed',

                          }, 

                 inplace=True)
TimesIPaddress = test['ip_address'].value_counts()

test["TimesIPaddress"] = test.groupby('ip_address')['user_id'].transform('count')

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

test['sex'] = lb_make.fit_transform(test['sex'])

sourcedummies=pd.get_dummies(test["source"], columns=["source"])

browserdummies=pd.get_dummies(test["browser"], columns=["browser"])

test = pd.concat([test, sourcedummies], axis=1)

test = pd.concat([test, browserdummies], axis=1)

testdata = transactions[["sex","age","TimeBetween","purchase_value","TimesDeviceUsed","TimesIPaddress","Chrome","FireFox","IE","Opera","Safari","Ads","Direct","SEO"]]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

testdata = pd.DataFrame(ss.fit_transform(testdata),columns = testdata.columns)

testdata.head()
classes = sclf.predict(testdata.values)

classes
pd.DataFrame(classes).to_csv("CapstoneFinal_Nagesh_Anumala.csv",header=None,index=None)