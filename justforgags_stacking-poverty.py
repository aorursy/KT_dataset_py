# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#["rooms","r4h1","r4h2","tamviv","hhsize",

#for outside wall---
#brick-25.40--paredblolad
#zinc 4 ---- paredzocalo
#cement----4.25-paredpreb
#waste ----0.50---- pareddes
#wood----1.50----paredmad
#zinc-----12---paredzinc
#natural fiber-1.64--paredfibras
#paredother--0.75-other



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import seaborn as sns
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

path_train="../input/train.csv"
path_test="../input/test.csv"

train_data=pd.read_csv(path_train)
test_data=pd.read_csv(path_test)
train_data.head()
data_test_cpy=test_data.copy()
train_set, test_set = train_test_split (train_data, test_size = 0.1, random_state = 42)
train_set_y=train_set.Target
train_set_x=train_set.drop(['Target'],axis=1)
test_set_y=test_set.Target
test_set_x=test_set.drop(['Target'],axis=1)
train_data.info()
train_data.shape
train_data.describe()

print(train_data.isnull().sum().sum(),test_data.isnull().sum().sum())#We check number of NAN
y=['Target']#setting target variable
#to set X as all variables minustarget variables
X=[i for i in train_data.columns if i!="Target"]
train_data.isnull().sum().sum()#getting number of NAN data
feature_lis=[]#the list of features with less than 4500 NAN
feature_lis_no_target=[]#for test data above list - {'target'}
for cols in train_data:#getting all the columns with more than 7000NAN deleted
    #print(train_data[cols].isnull().sum().sum())
    if train_data[cols].isnull().sum().sum() <4500:
        feature_lis.append(cols)
        if cols != 'Target':
            feature_lis_no_target.append(cols)
train_data=train_data[feature_lis]
#doing same with test data
test_data=test_data[feature_lis_no_target]
print(train_data.isnull().sum().sum(),test_data.isnull().sum().sum())# now we see only 10 NAN values left out of soo many
train_data=train_data.fillna(method='ffill')
test_data=test_data.fillna(method='ffill')
print("after forward fill")
print(train_data.isnull().sum().sum(),test_data.isnull().sum().sum())

train_data.edjefe
all_strings=[i for i in train_data.columns if isinstance(train_data[i][0], str)]
print(all_strings)
#for edjefe and edjefa  we substitute yes with 1 and no with 0
substitute={"yes":1,"no":0}
train_data["edjefe"].replace(substitute,inplace=True)
train_data["edjefa"].replace(substitute,inplace=True)
test_data["edjefe"].replace(substitute,inplace=True)
test_data["edjefa"].replace(substitute,inplace=True)
train_data["dependency"].replace(substitute,inplace=True)
test_data["dependency"].replace(substitute,inplace=True)
train_data=train_data.drop(['Id', 'idhogar'],axis=1).select_dtypes(exclude=['object'])

test_data=test_data.drop(['Id', 'idhogar'],axis=1).select_dtypes(exclude=['object'])

print(train_data.isnull().sum().sum(),test_data.isnull().sum().sum())

dic_wall={"paredblolad":25.40,"paredzocalo":4,"paredpreb":4.25,"pareddes":0.50,"paredmad":1.50,"paredzinc":12,"paredfibras":1.64,"paredother":0.75}
dic_floor={"pisomoscer":9.07,"pisocemento":4.25,"pisonatur":15,"pisonotiene":0,"pisomadera":12,"pisoother":0.5,}#if pisonotiene=0
dic_roof={"techozinc":4,"techoentrepiso":9.07,"techocane":15,"techootro":0.5,}# if cielorazo=1 then else dont
feature_list_denoting_poor=["abastaguano","noelec","sanitario1","energcocinar1",]#abastaguano=1 if no water noelec=1 if no electricity sanitario1=1 if no toilet

train_data_test=train_data.copy()
mask=train_data_test["pisonotiene"]==0
mask2=test_data["pisonotiene"]==0
train_data_test.loc[mask,"pisonotiene"]=1
test_data.loc[mask2,"pisonotiene"]=1
#print(train_data_test["pisonotiene"])
#train_data_test["pisonotiene"]=train_data_test["pisonotiene"].apply(lambda x: if x ==0 1 else 0)
train_data_test["roof_cost_approx_on_material"]=train_data_test["cielorazo"]*(train_data_test["techozinc"]*dic_roof["techozinc"]+train_data_test["techoentrepiso"]*dic_roof["techoentrepiso"]+train_data_test["techocane"]*dic_roof["techocane"]+train_data_test["techootro"]*dic_roof["techootro"])
train_data_test["floor_cost_approx_on_material"]=train_data_test["pisonotiene"]*(train_data_test["pisomoscer"]*dic_floor["pisomoscer"]+train_data_test["pisocemento"]*dic_floor["pisocemento"]+train_data_test["pisonatur"]*dic_floor["pisonatur"]+train_data_test["pisomadera"]*dic_floor["pisomadera"]+train_data_test["pisoother"]*dic_floor["pisoother"])
test_data["roof_cost_approx_on_material"]=test_data["cielorazo"]*(test_data["techozinc"]*dic_roof["techozinc"]+test_data["techoentrepiso"]*dic_roof["techoentrepiso"]+test_data["techocane"]*dic_roof["techocane"]+test_data["techootro"]*dic_roof["techootro"])
test_data["floor_cost_approx_on_material"]=test_data["pisonotiene"]*(test_data["pisomoscer"]*dic_floor["pisomoscer"]+test_data["pisocemento"]*dic_floor["pisocemento"]+test_data["pisonatur"]*dic_floor["pisonatur"]+test_data["pisomadera"]*dic_floor["pisomadera"]+test_data["pisoother"]*dic_floor["pisoother"])
train_data_test["floor_cost_approx_on_material"]
#train_data_test["number_of_materials_used_in_roof"]==train_data_test["rooms"]#thus we se
df=train_data_test.groupby("Target").agg({"roof_cost_approx_on_material":['sum']}).plot.bar(figsize=(16,11),colormap='summer')
fig = plt.figure()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 50
fig_size[1] = 60
plt.rcParams["figure.figsize"] = fig_size
feature_list=["rooms","r4h1","r4h2","r4h3","r4m1","r4m2","r4m3","r4t3","escolari","hhsize","SQBovercrowding"]
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
sns.barplot("Target","rooms",data=train_data_test,hue="area1",ax=ax1)
sns.barplot("Target","rooms",data=train_data_test,hue="area2",ax=ax2)
sns.barplot("Target","meaneduc",data=train_data_test,ax=ax3)
sns.barplot("Target","floor_cost_approx_on_material",data=train_data_test,ax=ax4)

colormap = plt.cm.RdBu
all_strings=[i for i in train_data_test.columns if isinstance(train_data_test[i][0], str)]
print(all_strings)
plt.figure(figsize=(14,12))
#print([i for i in train_data_test.edjefe if i=="d6c086aa3"])
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data_test.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white')
#print(train_data.Target)
#g = sns.pairplot(train_data[[u'hacdor', u'hhsize', u'r4h1', u'sanitario1', u'SQBage',u'Target']], hue='Target', palette="husl",height=10 ,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))
g = sns.pairplot(train_data, vars=[u'hacdor', u'hhsize', u'r4h1', u'sanitario1', u'SQBage'],hue='Target', palette="husl",height=6,diag_kind = 'kde',plot_kws=dict(s=25))
g.set(xticklabels=[])
SEED = 0 # for reproducibility
NFOLDS = 10 # set folds for out-of-fold prediction
ntrain=train_data_test.shape[0]
kf = KFold(ntrain, n_folds= 10, random_state=SEED)

class SklearnHelper(BaseEstimator):
    def __init__(self, clf, params, seed=0):
        self.clf = clf(params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer
# confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Put in our parameters for said classifiers
# lgbm Forest parameters n_estimators,max_depth,learning_rate
lgbm_params = {
    'n_estimators': 5000,
    'max_depth': -1,
    'learning_rate': 0.1,
    'random_state':0
    #'objective':'multiclass',
    #'metric':'None',
    #'class_weight':'balanced',
    #'colsample_bytree':0.89,
    #'min_child_samples':30,
    #'num_leaves':32,
    #'subsample':0.96
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Stratified k fold parameters 
skf_params = {
    
    }
train_set, test_set = train_test_split (train_data_test, test_size = 0.1, random_state = 42)
train_set_y=train_set.Target
train_set_x=train_set.drop(['Target'],axis=1)
test_set_y=test_set.Target
test_set_x=test_set.drop(['Target'],axis=1)
lgbm=LGBMClassifier(random_state=0)

#lgbm = SklearnHelper(clf=LGBMClassifier,params=lgbm_params)
#et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
#ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
#gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
#skf = SklearnHelper(clf=StratifiedKFold, seed=SEED, params=skf_params)

ntrain=train_data_test.shape[0]
ntest=test_data.shape[0]
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]# x_te has both test X and test y
        y_te = y_train.iloc[test_index]
        #print(i,oof_train[test_index])
        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        
        #print(oof_train[test_index],test_index)
        print(i)
        oof_test_skf[i, :] = clf.predict(x_test)
        print(i,oof_test_skf[i,:])
        
    print(oof_test_skf)
    oof_test[:] = oof_test_skf.mean(axis=0)
    print(oof_test)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
'''
print("lgbm classifier")
print("current parameters for lgbm")
print(lgbm.get_params().keys())

lgbm_params_gs={
    'n_estimators': [10,100,1000],
    'max_depth':[5,10,15],
    'learning_rate':[0.1,0.2,0.3,0.4,0.5]


CV_rnd_cfl = GridSearchCV(estimator = lgbm, param_grid = lgbm_params_gs, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
CV_rnd_cfl.fit(train_set_x, train_set_y)

best_parameters = CV_rnd_cfl.best_params_
print("The best parameters for using this model is", best_parameters)

#lgbm_oof_train, lgbm_oof_test = get_oof(lgbm,X_train, y_train, test_data)
#plt.show()
#et_oof_train, et_oof_test = get_oof(et,X_train, y_train, test_data)
#ada_oof_train, ada_oof_test = get_oof(ada,X_train, y_train, test_data)
#gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, test_data)
'''
lgbm=LGBMClassifier(random_state=0,learning_rate=0.3,max_depth=10,n_estimators=1000)
lgbm.fit(train_set_x,train_set_y)
y_pred=lgbm.predict(test_set_x)
# Confusion maxtrix & metrics
cm = confusion_matrix(test_set_y, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'LGBM Confusion matrix')
plt.savefig('2.lgbm_confusion_matrix.png')
plt.show()
f1_score(test_set_y,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
et=ExtraTreesClassifier(random_state=0,n_jobs=-1,min_samples_leaf= 2)
print(et)
'''print("extra tree classifier")
print("current parameters for e-tree")
print(et.get_params().keys())

et_params_gs={
    'n_estimators': [10,100,1000,5000],
    'max_depth':[5,10],
    }

CV_rnd_cfl = GridSearchCV(estimator = et, param_grid = et_params_gs, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
CV_rnd_cfl.fit(train_set_x, train_set_y)

best_parameters = CV_rnd_cfl.best_params_
print("The best parameters for using this model is", best_parameters)
'''

et=ExtraTreesClassifier(random_state=0,n_jobs=-1,min_samples_leaf= 2,max_depth=10,n_estimators=10)
print(et)
et.fit(train_set_x,train_set_y)
y_pred=et.predict(test_set_x)
# Confusion maxtrix & metrics
cm = confusion_matrix(test_set_y, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Extra tree Confusion matrix')
plt.savefig('2.et_confusion_matrix.png')
plt.show()
f1_score(test_set_y,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
gbc=GradientBoostingClassifier(random_state=0)
print(gbc)
'''
print("gradient boost classifier")


gb_params_gs={
    'n_estimators':[500,700,1000,1500],
    'max_depth': [2,3,5]
}
print("start")
CV_rnd_cfl = GridSearchCV(estimator = gbc, param_grid = gb_params_gs, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
print("mid")
CV_rnd_cfl.fit(train_set_x, train_set_y)
print("end")
bes
t_parameters = CV_rnd_cfl.best_params_
print("The best parameters for using this model is", best_parameters)
'''
gbc=GradientBoostingClassifier(random_state=0,max_depth=5,n_estimators=1500)
print(gbc)

'''
gbc.fit(train_set_x,train_set_y)
y_pred=gbc.predict(test_set_x)
# Confusion maxtrix & metrics
cm = confusion_matrix(test_set_y, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'gbc boost Confusion matrix')
plt.savefig('2.gbc_confusion_matrix.png')
plt.show()
f1_score(test_set_y,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
'''
y_train=train_data_test.Target
X_train=train_data_test.drop(["Target"],axis=1)
lgbm_oof_train, lgbm_oof_test = get_oof(lgbm,X_train, y_train, test_data)
#plt.show()
gbc_oof_train, et_oof_test = get_oof(gbc,X_train, y_train, test_data)
#ada_oof_train, ada_oof_test = get_oof(ada,X_train, y_train, test_data)
#gb_oof_train, gb_oof_test = get_oof(gb,X_train, y_train, test_data)
base_models=pd.DataFrame({'lgbm':lgbm_oof_train.ravel(),
                          'Gradient_boost':gbc_oof_train.ravel()})
base_models.head()
plt.title('Pearson Correlation of Classifiers', y=1.05, size=15)
sns.heatmap(base_models.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True,annot=True, cmap=colormap, linecolor='white')
x_train = np.concatenate(( lgbm_oof_train, gbc_oof_train), axis=1)
x_test = np.concatenate(( lgbm_oof_test, et_oof_test), axis=1)
print(x_test)
gbmn = LGBMClassifier(n_estimators= 2000)
print(gbmn)
print("gradient boost classifier")


gbmn_params_gs={
    'n_estimators': [10,100,1000],
    'max_depth':[5,10,15],
    'learning_rate':[0.1,0.2,0.3,0.4,0.5]}
print("start")
CV_rnd_cfl = GridSearchCV(estimator = gbmn, param_grid = gbmn_params_gs, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
print("mid")
CV_rnd_cfl.fit(x_train, y_train)
print("end")

best_parameters = CV_rnd_cfl.best_params_
print("The best parameters for using this model is", best_parameters)
gbmn = LGBMClassifier(n_estimators= 100,max_depth=10,learning_rate=0.1)
print(gbmn)
gbmn.fit(x_train, y_train)
predictions = gbmn.predict(x_test)
print(predictions)
print(x_test)
df_ = pd.DataFrame(columns=['Id','Target'])
df_['Id']=data_test_cpy['Id']
df_['Target']=predictions
df_.to_csv('submission7.csv', index=False)
print(( lgbm_oof_train, gbc_oof_train))
print( lgbm_oof_test, et_oof_test)
print(predictions)
