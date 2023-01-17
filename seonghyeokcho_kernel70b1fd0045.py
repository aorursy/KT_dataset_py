import pandas as pd
import numpy as np
import seaborn as sns
df1 = pd.read_csv('data/mem_data.csv').sort_values('MEM_ID')\
.set_index(['MEM_ID','GENDER','VISIT_CNT','SALES_AMT','RGST_DT','LAST_VST_DT']).reset_index()
df1.head(10)
df2 = pd.read_csv('data/mem_transaction.csv').sort_values(['MEM_ID','SELL_DT'])\
.set_index(['MEM_ID','STORE_ID','SELL_AMT','SELL_DT','MEMP_DT','MEMP_STY']).reset_index()
df2.head(10)
df3 = pd.read_csv('data/store_info.csv')
df3.head(10)
merged = df1.merge(df2,how='outer',on='MEM_ID')
merged.T
apply = np.where(merged['BIRTH_SL']=='S',1,0)
local = merged.columns.get_loc('BIRTH_SL')
merged.insert(loc=local, column='BIRTH_SL_CODE', value=apply)
merged.T
apply = np.where(merged['USED_PNT']==0,1,0)
local = merged.columns.get_loc('MEMP_TP')
merged.insert(loc=local, column='MEMP_TP_CODE', value=apply)
merged
apply = np.where(merged['SMS']=='Y',1,0)
local = merged.columns.get_loc('SMS')
merged.insert(loc=local, column='SMS_CODE', value=apply)
merged
apply = np.where(merged['MEMP_STY']=='O',1,0)
local = merged.columns.get_loc('MEMP_STY')
merged.insert(loc=local, column='MEMP_STY_CODE', value=apply)
merged
minus = merged['SALES_AMT'] - merged['SELL_AMT']
local = merged.columns.get_loc('SALES_AMT')+1
merged.insert(loc=local, column='SALES_INIT', value=minus)
merged
con = ['VISIT_CNT','SALES_AMT','USABLE_PNT','ACC_PNT','USED_PNT','USABLE_INIT','SELL_AMT']
merged[con] = merged[con].astype('int')
merged.info()
train = merged.GENDER!='UNKNOWN'
train_df = merged[train]
train_df.GENDER = (train_df.GENDER=='M').astype(int)
train_df
pred = merged.GENDER=='UNKNOWN'
pred_df = merged[pred].sort_values('MEM_ID')
prediction_df = pred_df[['MEM_ID','GENDER']]
prediction_df
train_df.isna().sum()
sns.heatmap(train_df[["GENDER","VISIT_CNT","SALES_AMT","BIRTH_SL_CODE","USABLE_PNT","USED_PNT","ACC_PNT",
                     "USABLE_INIT","SMS_CODE","STORE_ID","SELL_AMT"]]\
            .corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
train_df.info()
from sklearn.model_selection import train_test_split
feature_name = ['VISIT_CNT','SALES_AMT','USABLE_INIT','BIRTH_SL_CODE','ACC_PNT','USED_PNT','USABLE_PNT',
                'SMS_CODE','MEMP_TP_CODE']
label_name = ['GENDER']

X_train,X_test,y_train,y_test = train_test_split(train_df[feature_name],train_df[label_name],test_size=.25,random_state=0)

prediction_data = pred_df[feature_name]

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
29676+9893
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
%matplotlib inline
# bestModel
# input : model 객체
#   model    : 적용 알고리즘
#   nFolds   : 분할 개수
#  
# output : best parameter

def bestModel(model, nFolds, searchCV, X_train, y_train, X_test, y_test, isScaler, isPloy, isFeatureUnion, scoring, nJobs, nIter, verbose):
    # GridSearchCV을 위해 파라미터 값을 제한함.
    stepss = []
    grd_prams = {}

    if isPloy == True:
        stepss.append(('polynomialfeatures', PolynomialFeatures()))
        grd_prams.update({'polynomialfeatures__degree':[1, 2]})
        
        if isFeatureUnion == True:
            # create feature union
            features = []
            features.append(('pca', PCA(n_components=3)))
            features.append(('univ_select', SelectKBest(k=10)))
            stepss.append(('features', FeatureUnion(features)))       

    if isScaler == 'STANDARD':
        stepss.append(('standardscaler', StandardScaler()))
    else:
        stepss.append(('minmaxscaler', MinMaxScaler()))
            
    if model == 'SVC':
        stepss.append(('svc', SVC(random_state=0, C=100)))
        grd_prams.update({'svc__C':[0.1, 1, 10, 100], 'svc__gamma':[0.001, 0.01, 0.05, 0.1, 1, 10]})
    elif model == 'XGB': 
        stepss.append(('xgb', XGBClassifier(random_state=0, objective='binary:logistic')))
        grd_prams.update({'xgb__n_estimators': [300, 500],
            'xgb__learning_rate': [0.001, 0.01],
            'xgb__subsample': [0.5, 1],
            'xgb__max_depth': [5, 6],
            'xgb__colsample_bytree': [0.97, 1.24],
            'xgb__min_child_weight': [1, 2],
            'xgb__gamma': [0.001, 0.005],
            'xgb__nthread': [3, 4],
            'xgb__reg_lambda': [0.5, 1.0],
            'xgb__reg_alpha': [0.01, 0.1]
          })
        
    elif model == 'LGBM':
        # 그래디언트 부스팅 결정 트리(GBDT) : GOSS와 EFB 적용하여 GBDT를 새롭게 구현한 알고리즘
        stepss.append(('lgbm', LGBMClassifier(random_state=0, boosting_type='gbdt', objective='binary', metric='auc')))
        grd_prams.update({'lgbm__max_depth': [50, 100],
              'lgbm__learning_rate' : [0.01, 0.05],
              'lgbm__num_leaves': [150, 200],
              'lgbm__n_estimators': [300, 400],
              'lgbm__num_boost_round':[4000, 5000],
              'lgbm__subsample': [0.5, 1],
              'lgbm__reg_alpha': [0.01, 0.1],
              'lgbm__reg_lambda': [0.01, 0.1],
              'lgbm__min_data_in_leaf': [20, 30],
              'lgbm__lambda_l1': [0.01, 0.1],
              'lgbm__lambda_l2': [0.01, 0.1]
            })
        
    pipe = Pipeline(steps=stepss)
    
    cv = StratifiedShuffleSplit(n_splits=nFolds, test_size=0.2, random_state=0)
    grid = GridSearchCV(pipe, param_grid=grd_prams, n_jobs=nJobs, scoring=scoring, verbose=verbose, cv=cv)
    
    if searchCV == 'RANDOM':
        grid = RandomizedSearchCV(pipe, param_distributions=grd_prams, n_iter=nIter, scoring=scoring, error_score=3, verbose=verbose, n_jobs=nJobs, cv=cv)

    bestmodel = grid.fit(X_train, y_train).score(X_test,y_test)
    print("score_model ::: {}".format(bestmodel))
    print("-----------------------------------")
    print("{} best_parameters {}".format(model, grid.best_params_))
    print("{} best_estimator {}".format(model, grid.best_estimator_))
    
    report = grid.predict(X_test)
    print("-----------------------------------")
    print(classification_report(y_test, report))
    
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)
    print("-----------------------------------")
    print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
    print("Best model:"); print(confusion_matrix(y_test, report))
    
    return
bestModel('LGBM',5,'RANDOM',X_train,y_train,X_test,y_test,'MINMAX',True,True,'roc_auc',-1,15,3)
def bestGBDTNextModel(model, isKfold, nfold, searchCV, Xtrain, ytrain, Xtest, ytest, nIter, scoring, errScore, verbose, nJobs):
    grd_prams = {}
    classifier = LGBMClassifier(random_state=0, objective='binary:logistic')
    cv = KFold(n_splits=nfold, shuffle=True, random_state=0)
    if model == 'LGBM':  
        grd_prams.update({'max_depth': [50, 100],
              'learning_rate' : [0.01, 0.05],
              'num_leaves': [150, 200],
              'n_estimators': [300, 400],
              'num_boost_round':[4000, 5000],
              'subsample': [0.5, 1],
              'reg_alpha': [0.01, 0.1],
              'reg_lambda': [0.01, 0.1],
              'min_data_in_leaf': [20, 30],
              'lambda_l1': [0.01, 0.1],
              'lambda_l2': [0.01, 0.1]
            })
        classifier = LGBMClassifier(random_state=0, boosting_type='gbdt', objective='binary', metric='auc')
    if isKfold == False:
        cv = StratifiedShuffleSplit(n_splits=nfold, test_size=0.2, random_state=0)
    grid_ = RandomizedSearchCV(classifier, param_distributions=grd_prams,
                               n_iter=nIter, scoring=scoring, error_score=errScore, verbose=verbose, n_jobs=nJobs, cv=cv)
    grid_.fit(Xtrain, ytrain)
    score_ = grid_.score(Xtest, ytest)
    print("{} grid_.best_score {}".format(model, np.round(grid_.best_score_,3)))
    print("{} grid_.best_score {}".format(model, np.round(score_,3)))
    print("{} best_estimators {}".format(model, grid_.best_estimator_))
    print("{} best_parameters {}".format(model, grid_.best_params_))
    return
best_param = bestGBDTNextModel('LGBM', False, 5, 'RANDOM', X_train, y_train, X_test, y_test, 15, 'roc_auc', 0, 3, -1)
lgbm = LGBMClassifier(**best_param)
score_lgbm = lgbm.fit(X_train, y_train).score(X_test, y_test)
print("score_lgbm ::: {}".format(score_lgbm))
print("-----------------------------------")
y_lgbm = lgbm.predict(X_test)
print(classification_report(y_test, y_lgbm))
# 앞에서 학습한 베스트 모델 예측률 : 0.83
# 학습데이터에 알맞은 best estimators
'''super_model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', lambda_l1=0.1, lambda_l2=0.01,
               learning_rate=0.01, max_depth=100, metric='auc',
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=20, min_split_gain=0.0, n_estimators=300,
               n_jobs=-1, num_boost_round=5000, num_leaves=200,
               objective='binary', random_state=0, reg_alpha=0.01,
               reg_lambda=0.01, silent=True, subsample=0.5,
               subsample_for_bin=200000, subsample_freq=0)'''
super_model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', lambda_l1=0.1, lambda_l2=0.1,
               learning_rate=0.01, max_depth=50, metric='auc',
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=30, min_split_gain=0.0, n_estimators=400,
               n_jobs=-1, num_boost_round=4000, num_leaves=200,
               objective='binary', random_state=0, reg_alpha=0.01,
               reg_lambda=0.01, silent=True, subsample=1,
               subsample_for_bin=200000, subsample_freq=0)
best_score = super_model.fit(X_train, y_train).score(X_test, y_test)
y_super = super_model.predict(X_test)

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

print("super_model ---------------------------------------{}".format(np.round(best_score,3)))
print(classification_report(y_test, y_super))
print("-----------------------------------")
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("LightGBM:"); print(confusion_matrix(y_test, y_super))
fpr, tpr, _ = roc_curve(y_test, super_model.predict_proba(X_test)[:,1])
auc(fpr, tpr)
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
def plot_roc_curve(fpr, tpr, model, color=None) :
    model = model + ' (auc = %0.3f)' % auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model, color=color)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR (1 - specificity)')
    plt.ylabel('TPR (recall)')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, 
                                    dummy.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')
fpr_tree, tpr_tree, _ = roc_curve(y_test, 
                                  super_model.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'lightgbm', 'darkgreen')
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(precisions, recalls) :
    plt.plot(recalls, precisions, color='blue')
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
precisions, recalls, _ = precision_recall_curve(y_test, 
                                    super_model.predict_proba(X_test)[:,1])
plot_precision_recall_curve(precisions, recalls)
final = prediction_data.copy()
final['GENDER'] = super_model.predict(final)
prediction_df.GENDER = super_model.predict_proba(prediction_data.values)[:,1]
final['GENDER'].value_counts()
final[['GENDER']].head(40)
prediction_df.head(60)
prediction_data[['MEM_ID','GENDER']].to_csv('output_data.csv',index=False)
