# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import itertools
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier 
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve, f1_score
import warnings
from pprint import pprint
data=pd.read_csv("../input/train.csv")
data1=pd.read_csv("../input/test.csv")

df = data[['epared1','epared2','epared3']]
x = df.stack()
data['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data['epared'] = data['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))
df = data[['etecho1','etecho2','etecho3']]
x = df.stack()
data['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data['etecho'] = data['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))
df = data[['eviv1','eviv2','eviv3']]
x = df.stack()
data['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data['eviv'] = data['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))
data.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)
df = data1[['epared1','epared2','epared3']]
x = df.stack()
data1['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data1['epared'] = data1['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))
df = data1[['etecho1','etecho2','etecho3']]
x = df.stack()
data1['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data1['etecho'] = data1['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))
df = data1[['eviv1','eviv2','eviv3']]
x = df.stack()
data1['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
data1['eviv'] = data1['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))
data1.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)
#data['v2a1'] = data['v2a1'].fillna(lambda x : 84806.5000 if data['Target'] == 1 else (97015.166172 if data['Target'] == 2 else (102618.093333 if data['Target'] == 3 else 193589.258521)))
def data_cleaning(data):
    data['dependency']=np.sqrt(data['SQBdependency'])
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    
    conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
    ]
    choices = [0, 1, 1, data['edjefe'], data['edjefa']]
    data['edjefx']=np.select(conditions, choices)
    data['edjefx']=data['edjefx'].astype(int)
    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)
    
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
        
    return data

data_cleaning(data)
data_cleaning(data1)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
data1.loc[data1['rez_esc'] == 99.0 , 'rez_esc'] = 5
data['roof_waste_material'] = np.nan
data1['roof_waste_material'] = np.nan
data['electricity_other'] = np.nan
data1['electricity_other'] = np.nan

def fill_roof_exception(x):
    if (x['techozinc'] == 0) and (x['techoentrepiso'] == 0) and (x['techocane'] == 0) and (x['techootro'] == 0):
        return 1
    else:
        return 0
    
def fill_no_electricity(x):
    if (x['public'] == 0) and (x['planpri'] == 0) and (x['noelec'] == 0) and (x['coopele'] == 0):
        return 1
    else:
        return 0

data['roof_waste_material'] = data.apply(lambda x : fill_roof_exception(x),axis=1)
data1['roof_waste_material'] = data1.apply(lambda x : fill_roof_exception(x),axis=1)
data['electricity_other'] = data.apply(lambda x : fill_no_electricity(x),axis=1)
data1['electricity_other'] = data1.apply(lambda x : fill_no_electricity(x),axis=1)
def feature_engineering(train_set):
    train_set['adult'] = train_set['hogar_adul'] - train_set['hogar_mayor']
    train_set['dependency_count'] = train_set['hogar_nin'] + train_set['hogar_mayor']
    train_set['child_percent'] = train_set['hogar_nin']/train_set['hogar_total']*100
    train_set['elder_percent'] = train_set['hogar_mayor']/train_set['hogar_total']*100
    train_set['adult_percent'] = train_set['hogar_adul']/train_set['hogar_total']*100
    train_set['rent_per_person'] = train_set['v2a1']/train_set['hhsize']
    train_set['no_appliances'] = train_set['refrig'] + train_set['computer'] + train_set['television']
    train_set['rent_per_room'] = train_set['v2a1']/train_set['rooms']
    train_set['room_per_person_household'] = train_set['hhsize']/train_set['rooms']
    train_set['escolari_age'] = train_set['escolari']/train_set['age']
    train_set['rez_esc_escolari'] = train_set['rez_esc']/train_set['escolari']
    
    return train_set
    

feature_engineering(data)
feature_engineering(data1)
df_train = pd.DataFrame()
df_test = pd.DataFrame()
other_list = ['escolari', 'age', 'escolari_age']
for item in other_list:
    for function in ['mean','std','min','max','sum']:
        group_train = data[item].groupby(data['idhogar']).agg(function)
        group_test = data1[item].groupby(data1['idhogar']).agg(function)
        new_col = item + '_' + function
        df_train[new_col] = group_train
        df_test[new_col] = group_test
df_test = df_test.reset_index()
df_train = df_train.reset_index()

data = pd.merge(data, df_train, on='idhogar')
data1 = pd.merge(data1, df_test, on='idhogar')
data.head()
type(data)

'''
data1=pd.read_csv("../input/test.csv")

data1['v18q1'] = data1['v18q1'].fillna(0)
data1['rez_esc'] = data1['rez_esc'].fillna(0)
data1['v2a1'] = data1['v2a1'].fillna(0)
meaneduc_nan=data1[data1['meaneduc'].isnull()][['Id','idhogar','escolari']]
me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
for row in meaneduc_nan.iterrows():
    idx=row[0]
    idhogar=row[1]['idhogar']
    m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
    data1.at[idx, 'meaneduc']=m
    data1.at[idx, 'SQBmeaned']=m*m
    
data1['dependency']=np.sqrt(data1['SQBdependency'])
'''
data.fillna(0, inplace=True)
data1.fillna(0, inplace=True)
data = data.replace({np.inf: 0})
data1 = data1.replace({np.inf: 0})
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(f'new train set has {data.shape[0]} rows, and {data.shape[1]} features')

'''mapping = {"yes": 1, "no": 0}
for df in [data, data1]:
    # Fill in the values with the correct mapping
    #df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)
    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)
    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)'''
print([column for column in data1.columns if data1[column].isnull().sum()>0])

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split (data, test_size = 0.1, random_state = 42)
data = train_set
valid_data = test_set
valid_data=valid_data.drop(['Id','idhogar'],1)
y = valid_data.Target.tolist()
valid_data = valid_data.drop('Target', 1)
X = np.array(valid_data)
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
data=data.drop(['Id','idhogar'],1)
y = np.array(data.Target.tolist())
data = data.drop('Target', 1)
X = np.array(data.as_matrix())
skf = StratifiedKFold(n_splits=5 ,shuffle = True, random_state = 42)
for train_index, test_index in skf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
from sklearn.model_selection import GridSearchCV

'''xgb_cfl = xgb.XGBClassifier(n_jobs = -1)

xgb_cfl.fit(X_train, y_train)
y_pred = xgb_cfl.predict(X_test)
y_score = xgb_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title='XGB Confusion matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
#show_metrics()'''
'''print('Parameters currently in use:\n')
pprint(xgb_cfl.get_params())'''
'''param_grid = {
            'n_estimators': [700, 1000, 1200, 1300],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 2]
              }

CV_xgb_cfl = GridSearchCV(estimator = xgb_cfl, param_grid = param_grid, scoring= 'f1_macro', verbose = 2)
CV_xgb_cfl.fit(X_train, y_train)

best_parameters = CV_xgb_cfl.best_params_
print("The best parameters for using this model is", best_parameters)'''
xgb_cfl = xgb.XGBClassifier(n_jobs = -1, max_depth= 4,
                            n_estimators = 4000)

xgb_cfl.fit(X_train, y_train)
y_pred = xgb_cfl.predict(X_test)
y_score = xgb_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'XGB Confusion matrix')
plt.savefig('2.xgb_cfl_confusion_matrix.png')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
lgb_cfl = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.89, min_child_samples = 30, num_leaves = 32, subsample = 0.96)
lgb_cfl.fit(X_train, y_train)
y_pred = lgb_cfl.predict(X_test)
y_score = lgb_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix & metrics
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'LGB Confusion matrix')
plt.savefig('2.lgb_cfl_confusion_matrix.png')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)

'''rf_cfl = RandomForestClassifier(n_jobs = -1,
                                random_state = 42)

rf_cfl.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            early_stopping_rounds=400, verbose=100)
y_pred = rf_cfl.predict(X_test)
y_score = rf_cfl.predict_proba(X_test)[:,1]


# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'RF Confusion matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
'''print('Parameters currently in use:\n')
pprint(rf_cfl.get_params())'''
'''param_grid = {
            'n_estimators': [5,10,20,30,40,50],
            'criterion': ['gini','entropy'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 3, 5]
            }

CV_rnd_cfl = GridSearchCV(estimator = rf_cfl, param_grid = param_grid, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
CV_rnd_cfl.fit(X_train, y_train)

best_parameters = CV_rnd_cfl.best_params_
print("The best parameters for using this model is", best_parameters)'''
'''rf_cfl = RandomForestClassifier(random_state= 42, criterion= 'entropy', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 70)

rf_cfl.fit(X_train, y_train)
y_pred = rf_cfl.predict(X_test)
y_score = rf_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'RF Confusion matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
'''gb_cfl = GradientBoostingClassifier()

gb_cfl.fit(X_train, y_train)
y_pred = gb_cfl.predict(X_test)
y_score = gb_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Gradient Boosting matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
#print('Parameters currently in use:\n')
#pprint(dt_cfl.get_params())'''
'''print('Parameters currently in use:\n')
pprint(gb_cfl.get_params())'''
'''param_grid = {
            'n_estimators': [100, 300, 500, 700],
            'loss': ['deviance', 'exponential'],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
              }

CV_gb_cfl = GridSearchCV(estimator = gb_cfl, scoring= 'f1_macro', param_grid = param_grid, verbose = 0, n_jobs = -1)
CV_gb_cfl.fit(X_train, y_train)

best_parameters = CV_gb_cfl.best_params_
print("The best parameters for using this model is", best_parameters)'''
gb_cfl = GradientBoostingClassifier(loss= 'deviance', n_estimators=1000, max_depth=4 )
gb_cfl.fit(X_train, y_train)
y_pred = gb_cfl.predict(X_test)
y_score = gb_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'GradientBoostingClassifier matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
'''dt_cfl = DecisionTreeClassifier(random_state = 42)

dt_cfl.fit(X_train, y_train)
y_pred = dt_cfl.predict(X_test)
y_score = dt_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Decision Tree Matrix')
plt.show()

f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
'''print('Parameters currently in use:\n')
pprint(dt_cfl.get_params())'''
'''param_grid = {
            'criterion': ['gini','entropy'],
            'min_samples_leaf': [1, 2, 4, 5, 6],
            'min_samples_split': [2, 3, 5, 6]
            }

CV_ada_cfl = GridSearchCV(estimator = dt_cfl, param_grid = param_grid, scoring= 'f1_macro', verbose = 10, n_jobs = -1)
CV_ada_cfl.fit(X_train, y_train)

best_parameters = CV_ada_cfl.best_params_
print("The best parameters for using this model is", best_parameters)'''
'''dt_cfl = DecisionTreeClassifier( criterion= 'entropy', min_samples_leaf= 1, min_samples_split= 3)

dt_cfl.fit(X_train, y_train)
y_pred = dt_cfl.predict(X_test)
y_score = dt_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Decision Tree matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
'''et_cfl = ExtraTreesClassifier(random_state = 42)

et_cfl.fit(X_train, y_train)
y_pred = et_cfl.predict(X_test)
y_score = et_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Extra Trees Matrix')
plt.show()

f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
'''print('Parameters currently in use:\n')
pprint(et_cfl.get_params())'''
'''param_grid = {
            'n_estimators': [5,10,20,40,50,70],
            'criterion': ['gini','entropy'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 3, 5]
            }

CV_et_cfl = GridSearchCV(estimator = et_cfl, param_grid = param_grid, scoring= 'f1_macro', verbose = 0, n_jobs = -1)
CV_et_cfl.fit(X_train, y_train)

best_parameters = CV_et_cfl.best_params_
print("The best parameters for using this model is", best_parameters)'''
et_cfl = ExtraTreesClassifier( n_estimators = 100, min_samples_split= 3, criterion= 'entropy')

et_cfl.fit(X_train, y_train)
y_pred = et_cfl.predict(X_test)
y_score = et_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'Extra Trees Matrix')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
voting_cfl = VotingClassifier (
        estimators = [('xgb', xgb_cfl), ('gb', gb_cfl), ('lgb', lgb_cfl)],
                     voting='soft', weights = [1, 1, 1.2]) 
    
voting_cfl.fit(X_train,y_train)

y_pred = voting_cfl.predict(X_test)
#y_score = voting_cfl.predict_proba(X_test)[:,1]

# Confusion maxtrix
cm = confusion_matrix(y_test, y_pred)
class_names = [1,2,3,4]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'VOTING Confusion matrix')
plt.savefig('1.voting_confusion_matrix.png')
plt.show()
f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)
'''y_pred = voting_cfl.predict(X)
y_score = voting_cfl.predict_proba(X)[:,1]
cm = confusion_matrix(y, y_pred)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes = class_names, 
                      title = 'VOTING valid Confusion matrix')
plt.savefig('8.votingvf_cfl_confusion_matrix.png')
plt.show()
#f1_score(y_test,y_pred, labels=None, pos_label=1, average= 'macro', sample_weight=None)'''
Id= data1.Id
data1=data1.drop(['Id','idhogar'],1)
data1=data1.as_matrix()
#Target=voting_cfl.predict(data1)
#submission1=pd.merge(Id,Target)
#data1=data1.as_matrix()
Target=voting_cfl.predict(data1)

Id=pd.DataFrame(Id)
type(Target)
Target=pd.Series(Target)
#Target=Target.rename(index=str, columns={0:"Target"})


Id['Target'] = Target
submission=Id.copy()
submission
submission.to_csv('submission.csv',index=False)