import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
from collections import OrderedDict

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
colspecs=[[12,14],[78,79],[106,107],
          [119,120],[123,124],[148,150],[152,153],[162,163],[170,172],[173,174],
          [174,176],[181,182],[200,202],[216,218],[223,225],[241,243],
          [250,251],[260,261],[261,262],[262,263],[263,264],[268,269],[279,281],[286,287],[291,294],
          [312,313],[313,314],[314,315],[315,316],[316,317],[317,318],
          [324,325],[325,326],[326,327],[331,333],[336,337],[342,343],[343,344],[344,345],
          [345,346],[346,347],[352,353],[359,360],[360,361],[474,475],[491,493]]
names=['birth_month','mother_age9','mother_race6',
       'marital_status','mother_edu','father_age11','father_race6','father_edu','prior_births_living','prior_births_dead',
       'prior_terminations','birth_order_num','interval_last_birth11','interval_last_pregn11','month_prenatal_care','prenatal_visits',
       'WIC','cig_before','cig_1_trim','cig_2_trim','cig_3_trim','smoker','mother_height','mother_BMI','mother_pre_weight',
       'pre_diabetes','gest_diabetes','pre_hypertension','gest_hypertension','hypertasion_eclampsia','prev_preterm_births',
       'infertility_treat','fert_enhancing','asst_reproduct','num_prev_cesarean','no_risk_factors','gonorrhea','syphilis','chlamydia',
       'hepat_B','hepat_C','no_infections','successful_external_cephalic','failed_external_cephalic','infant_sex','gest_weeks10']
data2016 = pd.read_fwf('../input/us-birth-data-from-cdc/Nat2016PublicUS.c20170517.r20170913.txt',header=None,colspecs=colspecs,names=names)
data2016.head()
data2016 = data2016.replace({'N': 0, 'Y': 1, 'U': np.nan, 'X':np.nan})
data2016['mother_race6'] = data2016['mother_race6'].replace({6:np.nan})
data2016['father_race6'] = data2016['father_race6'].replace({6:np.nan,9:np.nan})
data2016['marital_status'] = data2016['marital_status'].replace({3:2, 9:np.nan})
data2016['father_age11'] = data2016['father_age11'].replace({11:np.nan})
data2016['mother_BMI'] = data2016['mother_BMI'].replace({99.9:np.nan})
data2016 = pd.get_dummies(data2016, columns=['infant_sex'])
data2016['target'] = np.where(data2016['gest_weeks10']<6, 1, 0)

cols9 = ['mother_edu','father_edu','birth_order_num','no_risk_factors','no_infections']
for i in cols9:
    data2016[i] = data2016[i].replace({9:np.nan})
cols99 = ['prior_births_living','prior_births_dead','prior_terminations','month_prenatal_care','prenatal_visits','cig_before'
         ,'cig_1_trim','cig_2_trim','cig_3_trim','mother_height','num_prev_cesarean','gest_weeks10']
for i in cols99:
    data2016[i] = data2016[i].replace({99:np.nan})
cols88 = ['interval_last_birth11','interval_last_pregn11']
for i in cols88:
    data2016[i] = data2016[i].replace({88:np.nan,99:np.nan})
cols999 = ['mother_pre_weight']
for i in cols999:
    data2016[i] = data2016[i].replace({999:np.nan})
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    
    
missing_values_table(data2016)
missing_df = missing_values_table(data2016);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 10].index)
data2016 = data2016.drop(columns = list(missing_columns))
data2016 = data2016.dropna()
data2016 = data2016.drop(columns = ['birth_month','no_risk_factors','no_infections','smoker','prenatal_visits'])
data2016 = pd.get_dummies(data2016, columns=['marital_status','WIC','pre_diabetes','gest_diabetes','pre_hypertension',
                                            'gest_hypertension','hypertasion_eclampsia','prev_preterm_births','infertility_treat',
                                            'gonorrhea','syphilis','chlamydia','hepat_B','successful_external_cephalic'
                                            ,'failed_external_cephalic'])
data2016.shape
c = ['r' if i <6 else 'b' for i in list(range(1,11))]
data2016.groupby(['gest_weeks10']).size().plot(kind='bar',color = c, title = 'Gestation weeks bins')
plt.show()
#corr = data2016.drop(columns = ['gest_weeks10']).corr()
#corr.style.background_gradient().set_precision(2)
t = data2016['target'].size
gr_tab = pd.DataFrame()
gr_tab['qnt'] = data2016.groupby(['target']).size() 
gr_tab['per'] = (data2016.groupby(['target']).size() /t)*100
gr_tab
train,test = split(data2016.drop(columns = ['gest_weeks10']),train_size = 0.8)
t = train['target'].size
gr_tab = pd.DataFrame()
gr_tab['qnt'] = train.groupby(['target']).size() 
gr_tab['per'] = (train.groupby(['target']).size() /t)*100
print ('train: \n ',gr_tab)
data2016all = train
train = data2016all[data2016all['target']==1]
d = data2016all[data2016all['target']==0].sample(n=314000)
train = train.append(d)

t = train['target'].size
gr_tab = pd.DataFrame()
gr_tab['qnt'] = train.groupby(['target']).size() 
gr_tab['per'] = (train.groupby(['target']).size() /t)*100
print ('train: \n ',gr_tab)

t = test['target'].size
gr_tab = pd.DataFrame()
gr_tab['qnt'] = test.groupby(['target']).size() 
gr_tab['per'] = (test.groupby(['target']).size() /t)*100
print ('test: \n ',gr_tab)
X_train = train.drop(columns = ['target'])
y_train = train.target
X_test = test.drop(columns = ['target'])
y_test = test.target
scaler = MaxAbsScaler()
sclr = scaler.fit(X_train)
X_train = sclr.transform(X_train)
X_test = sclr.transform(X_test)
rfc = RandomForestClassifier(max_depth=8,n_estimators=10).fit(X_train,y_train)
rfc_auc = roc_auc_score(y_test, rfc.predict(X_test))

#knn = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
#knn_auc = roc_auc_score(y_test, knn.predict(X_test))
logrg = LogisticRegression().fit(X_train,y_train)
log_auc = roc_auc_score(y_test, logrg.predict(X_test))
linrg = LinearRegression().fit(X_train,y_train)
lin_auc = roc_auc_score(y_test, linrg.predict(X_test).round())
#svc = SVC(kernel='linear', class_weight='balanced',probability=True).fit(X_train,y_train)
#svc_auc = roc_auc_score(y_test, svc.predict(X_test))
xgbm = xgb.XGBModel().fit(X_train,y_train)
xgb_auc = roc_auc_score(y_test, xgbm.predict(X_test).round())
print('RandomForest test AUC: ',rfc_auc)
#print('K nearest neighbors test AUC: ',knn_auc)
print('Logistic regression test AUC: ',log_auc)
print('Linear regression test AUC: ',lin_auc)
print('XGBoost test AUC: ',xgb_auc)
def report(clf, X, y):
    acc = accuracy_score(y_true=y, y_pred=clf.predict(X).round())
    auc = roc_auc_score(y, clf.predict(X).round())
    cm = confusion_matrix(y_true=y, y_pred=clf.predict(X).round())
    rep = classification_report(y_true=y,y_pred=clf.predict(X).round())
    return 'accuracy {:.3f}\nauc {:.3f}\n\n{}\n\n{}'.format(acc,auc, cm, rep)
print('train report: ',report(xgbm,X_train,y_train))
print('test report: ',report(xgbm,X_test,y_test))
plot_importance(xgbm)
plt.show()
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.3, 'n_estimators': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}

gbm = xgb.XGBClassifier(**ind_params).fit(X_train, y_train)

optimized_GBM = GridSearchCV(gbm, cv_params, scoring = 'roc_auc', cv = 5, n_jobs = 1) 

optimized_GBM.fit(X_train, y_train)

predictions_opt = optimized_GBM.predict(X_test)
#best_est = rs.best_estimator_
#print(best_est)
opt_xgb_auc = roc_auc_score(y_test, predictions_opt)
print ('Test auc: ',opt_xgb_auc)
best_est = optimized_GBM.best_estimator_
print(best_est)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 3, 'eta': 0.3, 'silent': 0, 'objective': 'binary:logistic'}
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 30
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
eval_auc = roc_auc_score(y_test, ypred)
print ('Test auc: ',eval_auc)
#confusion_matrix(y_test, ypred.round())
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 5, 'eta': 0.3, 'verbosity': 0, 'objective': 'binary:logistic','eval_metric':'auc'}
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 30
bst = xgb.train(param, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
eval_auc = roc_auc_score(y_test, ypred)
print ('Test auc: ',eval_auc)
