import numpy as np
import pandas as pd


import scipy as sp
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
#from sklearn.metrics import mean_squared_error, r2_score

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time

%matplotlib inline
from bayes_opt import BayesianOptimization
df_train = pd.read_csv('../input/train.csv')
print(df_train.info())
df_train.head()
df_test = pd.read_csv('../input/test.csv')
print(df_test.info())
df_test.head()
df = pd.concat([df_train, df_test], sort=False)
df.head()
df[9550:9560]
columns = list(df.columns)
print(columns)
print(df[['Id', 'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q','v18q1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']].info())
df[['Id', 'v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q','v18q1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3']].head()

#v2a1, Monthly rent payment
#v18q1
df['v2a1'].fillna(df['v2a1'].mean(), inplace=True)
df['v18q1'].fillna(0, inplace=True)  # we may just end up removing v18q but maybe not (feature engineering already done?)
print(df[['tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']].info())
df[['tamhog', 'tamviv', 'escolari', 'rez_esc', 'hhsize', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']].head()
df['rez_esc'].unique()
#df['rez_esc'][df['rez_esc'] == 0.]
#df[df['rez_esc'].isnull()]

df['rez_esc'].fillna(0, inplace=True)
df[['techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele']].info()
df[['techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele']].head()
df[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']].info()
df[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']].head()
df[['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']].info()
df[['epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']].head()
df[['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12', 'idhogar', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq', 'Target']].info()
df[['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12', 'idhogar', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq', 'Target']].head()
#['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12', 'idhogar', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq', 'Target']

# Avg years of education
#df[['meaneduc','SQBmeaned']]
#df[['meaneduc','SQBmeaned']][df['meaneduc'].isnull()]
df['meaneduc'].fillna(df['meaneduc'].mean(), inplace=True)
df['SQBmeaned'].fillna(df['SQBmeaned'].mean(), inplace=True)


df[['idhogar', 'dependency', 'edjefe', 'edjefa']].info()
df[['idhogar', 'dependency', 'edjefe', 'edjefa']].head()
df['idhogar'].unique()

df['dependency'].unique()
df['dependency'].value_counts()
#edjefe, years of education of male head of household, 
#based on the interaction of escolari (years of education), 
#head of household and gender, yes=1 and no=0


df['edjefe'].value_counts()
#edjefa, years of education of female head of household, 
#based on the interaction of escolari (years of education), 
#head of household and gender, yes=1 and no=0

df['edjefa'].value_counts()

df[['dependency','edjefe', 'edjefa']] = df[['dependency','edjefe', 'edjefa']].replace('yes', 1)
df[['dependency','edjefe', 'edjefa']] = df[['dependency','edjefe', 'edjefa']].replace('no', 0)
#pd.to_numeric(df[['dependency','edjefe', 'edjefa']])
df['dependency'] = df['dependency'].astype(float)
df['edjefe'] = df['edjefe'].astype(float)
df['edjefa'] = df['edjefa'].astype(float)
#df[['dependency','edjefe', 'edjefa']].info()
#plt.subplot(121)
columns.remove('Id')
columns.remove('Target')
columns.remove('idhogar')
statistics = []
for name in columns:
    stuff = (name, sp.stats.skew(df[name]), sp.stats.kurtosis(df[name]))
    statistics.append( stuff)
from operator import itemgetter

sorted(statistics, key=itemgetter(1))
plt.hist(df['estadocivil2'])
columns.append('Target')
correlation_matrix = df[columns].corr()
variables = correlation_matrix.sort_values('Target', axis=0)['Target']
important_variables = variables[abs(variables)> 0.2].index
important_variables
plt.figure(figsize=(10,10))
sns.heatmap(df[important_variables].corr())


columns = list(df.columns)
columns.remove('Id')
columns.remove('Target')
columns.remove('idhogar')
X = df[columns][0:9557]
X_unknown = df[columns][9557:]
y = df['Target'][0:9557]
def PlotClassifierDiagnostics(y_test, y_pred, y_prob):
    accuracy = accuracy_score(y_test, y_pred)
    #roc_auc = roc_auc_score(y_test, y_prob, average='micro')
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ')
    print(confusion)
    print('Accuracy: ',  accuracy)
    print('F1 (micro/macro/weighted): ')
    
    f1 = f1_score(y_test, y_pred, average='micro')
    f2 = f1_score(y_test, y_pred, average='macro')
    f3 = f1_score(y_test, y_pred, average='weighted')
    
    print((f1, f2, f3))
    #print((accuracy, f1, roc_auc))
        
    #fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    #roc_auc2 = auc(fpr, tpr)
    #print('ROC curve')
    
    #plt.plot(fpr, tpr, alpha=0.3,label='(AUC = %0.2f)' % (roc_auc))
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')    
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#def xgb_crossval(learning_rate, n_estimators):
#    return cross_val_score(xgb.XGBClassifier(
#        learning_rate=learning_rate, 
#        n_estimators=int(n_estimators), 
#        silent=False,
#        objective='multi:softmax'),
#        X,y, scoring='f1_micro', cv=3, n_jobs=-1).mean()

#bayesian_optimizer = BayesianOptimization(xgb_crossval, 
#                                        {'learning_rate':(0.001, 0.2), 
#                                         'n_estimators': (100, 2000)} )
#best_optimizer = bayesian_optimizer.maximize(n_iter=10)
#bayesian_optimizer.res['max']
#xgb_model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=1000)

#https://github.com/fmfn/BayesianOptimization/commit/1ce5484c6fff6e4913d43fa41dbed29f2a95f187



#def xgb_crossval(learning_rate, n_estimators, max_depth, gamma, min_child_weight):
#    return cross_val_score(xgb.XGBClassifier(
#        learning_rate=learning_rate, 
#        n_estimators=int(n_estimators), 
#        max_depth = int(max_depth), 
#        gamma=int(gamma), 
#        min_child_weight = int(min_child_weight),
#        silent=False,
#        objective='multi:softmax'),
#        X,y, scoring='f1_micro', cv=3, n_jobs=-1).mean()

#bayesian_optimizer = BayesianOptimization(xgb_crossval, 
#                                        {'learning_rate':(0.001, 0.2), 
#                                         'n_estimators': (100, 2000), 
#                                         'max_depth':(1,10), 
#                                         'gamma': (0,1), 
#                                         'min_child_weight': (0,10)} )
#gp_params = {"alpha": 1e-5}
#best_optimizer = bayesian_optimizer.maximize(n_iter=10) #, **gp_params)                              
# Example from the people who made this library: 
# https://github.com/fmfn/BayesianOptimization/blob/master/examples/xgboost_example.py

#random_state= 0
#params = {
#        'eta': 0.1,
#        'silent': 1,
#        'eval_metric': 'mae',
#        'verbose_eval': True,
#        'seed': random_state
#    }
#    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
#                                                'colsample_bytree': (0.1, 1),
#                                                'max_depth': (5, 15),
#                                                'subsample': (0.5, 1),
#                                                'gamma': (0, 10),
#                                                'alpha': (0, 10),
#                                                })


#baysian_optimize
xgb_model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=6000, silent=False, objective='multi:softmax')
t = time.time()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

print('elapsed time: ', time.time()-t)

PlotClassifierDiagnostics(y_test, y_pred_xgb, y_prob_xgb)


# going up to 1500 yields benefits
###xgb_model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=1000, silent=False, objective='multi:softmax')
#Confusion Matrix: 
#[[ 101   14    0   17]
# [   8  225   11   79]
# [   9   15  133  105]
# [   5   13   10 1167]]
#Accuracy:  0.850418410042
#F1 (micro/macro/weighted): 
#(0.85041841004184104, 0.77623584606360785, 0.84031282560359011)

#xgb_model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=3000, silent=False, objective='multi:softmax')
#Confusion Matrix: 
#[[ 128   14    2   25]
# [   4  263    6   46]
# [   2   21  170   43]
# [   7    7    4 1170]]
#Accuracy:  0.905334728033
#F1 (micro/macro/weighted): 
#(0.90533472803347281, 0.85718856012394928, 0.90218972836421363)

#xgb_model = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=6000, silent=False, objective='multi:softmax')
#elapsed time:  142.39282512664795
#Confusion Matrix: 
#[[ 133   13    2   21]
# [   6  281    5   27]
# [   2   23  182   29]
# [   6    8    7 1167]]
#Accuracy:  0.922071129707
#F1 (micro/macro/weighted): 
#(0.92207112970711302, 0.87918487482829089, 0.92030419590693613)

lgb_model = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=6000)

t = time.time()

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)

print('elapsed time: ', time.time()-t)


PlotClassifierDiagnostics(y_test, y_pred_lgb, y_prob_lgb)
t = time.time()
xgb_model.fit(X,y)
y_pred_xgb = xgb_model.predict(X_unknown)
print('elapsed time: ', time.time()-t)



t = time.time()
lgb_model.fit(X,y)
y_pred_lgb =lgb_model.predict(X_unknown)
print('elapsed time: ', time.time()-t)

df_out = df[9557:]
df_out['Target'] = y_pred_xgb.astype(int)
df_out[['Id', 'Target']].to_csv('output_xgb.csv',index=False)

df_out['Target'] = y_pred_lgb.astype(int)
df_out[['Id', 'Target']].to_csv('output_lgb.csv',index=False)
