import os
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns  
from pandas import set_option
plt.style.use('ggplot')
from scipy.stats import randint
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV  
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.chdir("C:/D Drive Data/Max_Life/Data/")
data=pd.read_excel("case_study_data.xlsx")
data.head()
data.shape
data.columns
data.info()
data.isna().sum()
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_dtype

for i in data.columns:
    if bool(is_string_dtype(data[i]))==True:
        print(i,":",data[i].unique())
duplicate_rows_df = data[data.duplicated()]
print('number of duplicate rows:', duplicate_rows_df.shape)
set_option('display.width', 100)
set_option('precision', 2)

print("DESCRIPTIVE STATISTICS OF NUMERIC COLUMNS")
print()
print(data.describe().T)
# The frequency of default rate at the bank
data['status'].value_counts()

# Percentage
good_customer = round(data.loc[data['status']==1,'status'].sum()/len(data)*100, 1)
bad_customer = round(data.loc[data['status']==2,'status'].count()/len(data)*100, 1)
print("Percentage of good customer is :",good_customer,"\nPercentage of bad customer is :",bad_customer)
#Replacing 1 by 0 and 2 by 1
data.status.replace([1,2], [0,1], inplace=True)
import sys 
plt.figure(figsize=(7,4))
sns.set_context('notebook', font_scale=1.2)
sns.countplot('status',data=data, palette="Reds")
%config InlineBackend.figure_format ='retina'
# Calculate correlations
plt.figure(figsize=(20,10))
c= data.iloc[:,:-1].corr()
sns.heatmap(c,cmap='YlGnBu',annot=True)
## Barchart of categorical variables with credit default variable
cat=[]
for i in data.columns:
    if bool(is_string_dtype(data[i]))==True:
        cat.append(i)
for i in range(len(cat)):
    plt.figure()
    sns.countplot(cat[i], hue="status", data=data, palette="twilight_shifted_r")
#num=[]
#for i in data.columns:
 #   if bool(is_string_dtype(data[i]))==False:
  #      num.append(i)
#num=num[:-1]
#sns.set_style('whitegrid')

##Violin Plot
#for i in range(len(num)):
    #plt.figure()
 #   sns.violinplot(y='duration', hue='status', data=data,split = True)
  #  sns.swarmplot(x ='duration', y ='status', data = data, color ='black') 
   # plt.legend(loc='best', title= 'Default', facecolor='white')

##Pairwise Plot between numerical varaibles bifurcated by status columns,
var1=['duration','amount','inst_rate','residing_since','age','num_credits' ,'status' ]
#var1=['duration','amount','inst_rate' ,'status' ]

data_pair=data[var1]
data_pair['status'] = pd.Categorical(data_pair.status)
sns.pairplot(data_pair, hue ="status", palette ='coolwarm', diag_kws={'bw': 0.2}) 
##Jointplot between variables to assess the kernel density 
sns.jointplot(x ='duration', y ='status', data = data_pair, kind ='kde') 
sns.jointplot(x ='amount', y ='status', data = data_pair, kind ='kde') 
sns.jointplot(x ='inst_rate', y ='status', data = data_pair, kind ='kde') 
sns.jointplot(x ='residing_since', y ='status', data = data_pair, kind ='kde') 
sns.jointplot(x ='age', y ='status', data = data_pair, kind ='kde') 
sns.jointplot(x ='num_credits', y ='status', data = data_pair, kind ='kde') 
##Boxplot between numerical variables
num=[]
for i in data.columns:
    if bool(is_string_dtype(data[i]))==False:
        num.append(i)
num=num[:-1]
for i in range(len(num)):
    plt.figure()
    sns.boxplot(y=num[i],x='status', hue="status", data=data, palette='spring_r')
    plt.legend(loc='best', title= 'Default', facecolor='white')
ax = data[num].hist()
for axis in ax.flatten():
    axis.set_xticklabels([])
    axis.set_yticklabels([])
plt.show()
##subsetting data for calculating woe values:
data_woe=data
import numpy as np
def calculate_woe_iv(dataset, feature, target):
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        }) 
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    iv = dset['IV'].sum()
    dset = dset.sort_values(by='WoE')
    return dset, iv
lst1= []
IV_df = pd.DataFrame(columns=['Variable','IV'])
for col in data_woe.columns:
    if col == 'status': continue
    else:
          df, iv= calculate_woe_iv(data_woe, col, 'status')
    lst1.append(df)
    IV_df = IV_df.append({
                "Variable" :col ,
                "IV" : iv,
                },ignore_index=True)
IV_df

lst1[0]
data_checkin_acc= lst1[0]
data_checkin_acc = data_checkin_acc.rename(columns={"Value":"checkin_acc","WoE":"WoEcheckin_acc"})

data= pd.merge(data, data_checkin_acc[['checkin_acc','WoEcheckin_acc']], on='checkin_acc')
data_duration= lst1[1]
data_duration = data_duration.rename(columns={"Value":"duration","WoE":"WoEduration"})

data_credit_history= lst1[2]
data_credit_history = data_credit_history.rename(columns={"Value":"credit_history","WoE":"WoEcredit_history"})

data_age= lst1[12]
data_age = data_age.rename(columns={"Value":"age","WoE":"WoEage"})

data= pd.merge(data, data_duration[['duration','WoEduration']], on='duration')
data= pd.merge(data, data_credit_history[['credit_history','WoEcredit_history']], on='credit_history')
data= pd.merge(data, data_age[['age','WoEage']], on='age')
data.head()
data=data.drop(['checkin_acc','duration','credit_history','age'],axis=1)
#Capping outlier if any in the amount column
data['amount'] = np.where(data['amount'] <data['amount'].quantile(.05), data['amount'].quantile(.05),data['amount'])
data['amount']  = np.where(data['amount'] >data['amount'].quantile(.95), data['amount'].quantile(.95),data['amount']) 
sns.boxplot(x=data['amount'] )
data['amount'].describe()
# perform one hot encoding with k - 1, it automatically drop the first.
#not ordinal variables
one_hot_enc=pd.get_dummies(data[['purpose','svaing_acc','present_emp_since','personal_status','other_debtors',  'property', 'inst_plans', 'housing','job', 'telephone', 'foreign_worker']])
data=pd.concat([data,one_hot_enc],axis=1)
data.head()
#saving unstandardized data
data1=data
data=data.drop(['purpose','svaing_acc','present_emp_since','personal_status','other_debtors',  'property', 'inst_plans', 'housing','job', 'telephone', 'foreign_worker'],axis=1)
varlist=data.columns.difference(['status'])
data[varlist]=data[varlist].apply(lambda x: (x-min(x))/(max(x)-min(x)))
train,val=train_test_split(data,test_size=0.3,random_state =123)
train.shape
#predictors and dependent for train data
X=train[varlist]
Y=train.iloc[:,5]
#predictors and dependent for validation data
X_val=val[varlist]
Y_val=val.iloc[:,5]
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
1/300
1/700
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 6)]
#n_estimators=[25,50]
# Number of features to consider at every split
max_features = [15,'auto','sqrt']
#max_features = [20, 'sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
max_depth=[5,7,10]

# Minimum number of samples required to split a node
min_samples_split = [2,5,10]
#min_samples_split = [10]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
min_samples_leaf= [5,10,15]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
#n_jobs= -1, 
oob_score= [True]
class_weight=[{0:0.001,1:0.003},{0:1,1:1},{0:1,1:2},{0:1,1:10}]
6*3*3*3*3*4*5
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
             'oob_score':oob_score,
              'class_weight':class_weight}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X, Y)

best_RF = grid_search.best_estimator_
import pickle
filename = 'Model_best_RF.sav'
pickle.dump(best_RF, open(filename, 'wb'))
RF_model_final = pickle.load(open(filename, 'rb'))
score= RF_model_final.score(X, Y)
score_test= RF_model_final.score(X_val, Y_val)
# Predcited probability of each class.
y_pred_prob_RF = RF_model_final.predict_proba(X_val)
# Predicted value of each class
y_pred_class_RF = RF_model_final.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_RF))
prediction_class_RF=pd.DataFrame(y_pred_class_RF)
Y_val1= Y_val.reset_index()
final_test_pred_RF = pd.concat([Y_val1,pd.DataFrame(y_pred_prob_RF, columns=['Col_0', 'Col_1']),prediction_class_RF], axis=1)
final_test_pred_RF.to_csv("final_test_pred_RF1.csv")
best_RF
from sklearn.inspection import permutation_importance
r = permutation_importance(RF_model_final, X, Y,n_repeats=5,random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.columns[i]:<8}"f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
r = permutation_importance(RF_model_final, X_val, Y_val,n_repeats=5,random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.columns[i]:<8}"f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
#XGBOOST
import xgboost as xgb
# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
param_grid_xg = {'n_estimators': [640,800,1000],
               'max_depth': [7,15],
               'gamma': [0.01,.1],
                'reg_alpha' : [0.1,1],
                 'reg_lambda' : [0.1,1],
                 'colsample_bytree' : [.5,1],
                 'min_child_weight' : [5,10],
              'scale_pos_weight':[2,2.5,3],
                'eta':[0.2,.3,.1]}
3*2*2*2*2*2*2*3*3*5
xg_grid = GridSearchCV(estimator = XGBClassifier(objective= 'binary:logistic'), param_grid = param_grid_xg, scoring='roc_auc',n_jobs=-1, cv=5,verbose = 2)

# Fit the grid search to the data
xg_grid.fit(X, Y)
best_xg = xg_grid.best_estimator_
filename = 'Model_best_xgboost.sav'
pickle.dump(best_xg, open(filename, 'wb'))

xgboost_model_final = pickle.load(open(filename, 'rb'))
score= xgboost_model_final.score(X, Y)
print(score)
score_test= xgboost_model_final.score(X_val, Y_val)
print(score_test)
xgboost_model_final
param_grid_xg2 = {'n_estimators': [100,640],
               'max_depth': [5,6,7],
               'gamma': [0.001,.01],
                'reg_alpha' : [1,2],
                 'reg_lambda' : [0.01,.1,],
                 'colsample_bytree' : [.6,.5],
                 'min_child_weight' : [1,2],
              'scale_pos_weight':[2],
                'eta':[0.01,.001]}
xg_grid_iter2 = GridSearchCV(estimator = XGBClassifier(objective= 'binary:logistic'), param_grid = param_grid_xg2, scoring='roc_auc',n_jobs=-1, cv=5,verbose = 2)

xg_grid_iter2.fit(X, Y)
xg_iter2=xg_grid_iter2.best_estimator_
score= xg_iter2.score(X, Y)
print(score)
score= xg_iter2.score(X_val, Y_val)
print(score)
xg_iter2
filename = 'Model_best_xgboost_iter2.sav'
pickle.dump(xg_iter2, open(filename, 'wb'))

xgboost_model_final_iter2 = pickle.load(open(filename, 'rb'))
# Predcited probability of each class.
y_pred_prob_XG = xgboost_model_final_iter2.predict_proba(X_val)
# Predicted value of each class
y_pred_class_XG = xgboost_model_final_iter2.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_XG))
prediction_class_XG=pd.DataFrame(y_pred_prob_XG)
final_test_pred_XG = pd.concat([pd.DataFrame(y_pred_prob_XG, columns=['Col_0', 'Col_1']),prediction_class_XG], axis=1)
final_test_pred_XG.to_csv("final_test_pred_XG1.csv")
from sklearn.inspection import permutation_importance
r = permutation_importance(xgboost_model_final_iter2, X, Y,n_repeats=5,random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.columns[i]:<8}"f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
r = permutation_importance(xgboost_model_final_iter2, X_val, Y_val,n_repeats=5,random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.columns[i]:<8}"f"{r.importances_mean[i]:.3f}"f" +/- {r.importances_std[i]:.3f}")
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
param_grid_ada = {'n_estimators': [100,400,640,1000],
                'learning_rate':[0.01,.001,.0001]}
adb= AdaBoostClassifier()
adb_grid = GridSearchCV(estimator = adb, param_grid=param_grid_ada,scoring='roc_auc',n_jobs=-1, cv=10,verbose = 2)
adb_grid.fit(X, Y)
filename = 'Model_best_adaboost.sav'
pickle.dump(ada_best, open(filename, 'wb'))

Model_best_adaboost = pickle.load(open(filename, 'rb'))
score= Model_best_adaboost.score(X, Y)
print(score)
score_test= Model_best_adaboost.score(X_val, Y_val)
print(score_test)
# Predcited probability of each class.
y_pred_prob_ADA = Model_best_adaboost.predict_proba(X_val)
# Predicted value of each class
y_pred_class_ADA = Model_best_adaboost.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_ADA))
prediction_class_ADA=pd.DataFrame(y_pred_prob_ADA)
final_test_pred_ADA = pd.concat([val, pd.DataFrame(y_pred_prob_ADA, columns=['Col_0', 'Col_1']),prediction_class_ADA], axis=1)
final_test_pred_ADA.to_csv("final_test_pred_ADABOOST1.csv")
##Importing Logistic Regression Final Model
filename = 'Model_best_logistic.sav'
log_model_final = pickle.load(open(filename, 'rb'))

# Predcited probability of each class.
y_pred_prob_LOG = log_model_final.predict_proba(X_val)
# Predicted value of each class
y_pred_class_LOG = log_model_final.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_LOG))
prediction_class_LOG=pd.DataFrame(y_pred_prob_LOG)
final_test_pred_LOG = pd.concat([val, pd.DataFrame(y_pred_prob_LOG, columns=['Col_0', 'Col_1']),prediction_class_LOG], axis=1)
final_test_pred_LOG.to_csv("final_test_pred_LOG1.csv")
from sklearn.ensemble import VotingClassifier
final_ensemble = VotingClassifier(estimators=[('log_model_final', log_model_final), ('RF_model_final', RF_model_final),('xgboost_model_final_iter2',xgboost_model_final_iter2),('Model_best_adaboost',Model_best_adaboost)],voting='soft')
final_ensemble.fit(X, Y)
score= final_ensemble.score(X, Y)
print(score)
score_test= final_ensemble.score(X_val, Y_val)
print(score_test)
##weigheted classifier
final_ensemble_weights = VotingClassifier(estimators=[('lr', log_model_final), ('rf', RF_model_final),('xgb',xgboost_model_final_iter2),('adb',Model_best_adaboost)],voting='soft', weights=[1, 2, 2,1])
final_ensemble_weights.fit(X, Y)
score= final_ensemble_weights.score(X, Y)
print(score)
score_test= final_ensemble.score(X_val, Y_val)
print(score_test)
# Predcited probability of each class.
y_pred_prob_AVGENSEMB = final_ensemble.predict_proba(X_val)
# Predicted value of each class
y_pred_class_AVGENSEMB = final_ensemble.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_AVGENSEMB))
##EXPORTING The output of the test data
prediction = pd.DataFrame(y_pred_prob_AVGENSEMB)
prediction_class_AVGENSEMB=pd.DataFrame(y_pred_class_AVGENSEMB)
#final=pd.concat(prediction,prediction_class)
#prediction = pd.DataFrame(y_pred_prob, columns=['y_pred_prob']).to_csv('prediction.csv')
final_test_pred = pd.concat([pd.DataFrame(y_pred_prob, columns=['Col_0', 'Col_1']),prediction_class_AVGENSEMB], axis=1)
final_test_pred.to_csv("final_test_pred_avg_ensebmle.csv")
# Predcited probability of each class.
y_pred_prob_WAVGENSEMB = final_ensemble_weights.predict_proba(X_val)
# Predicted value of each class
y_pred_class_WAVGENSEMB = final_ensemble_weights.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_WAVGENSEMB))
#multilayer stacking
from sklearn.ensemble import StackingClassifier
final_layer = StackingClassifier(
estimators=[('adb',Model_best_adaboost),
('xgb',xgboost_model_final_iter2),('lr', log_model_final)],
final_estimator=RF_model_final)


final_layer.fit(X, Y)
#print('R2 score: {:.2f}'.format(multi_layer_regressor.score(X_test, y_test))
score= final_layer.score(X, Y)
print(score)
score_test= final_layer.score(X_val, Y_val)
print(score)
filename = 'Model_best_stacking.sav'
pickle.dump(final_layer, open(filename, 'wb'))
Model_best_stacking = pickle.load(open(filename, 'rb'))
# Predcited probability of each class.
y_pred_prob_STACK= final_layer.predict_proba(X_val)
# Predicted value of each class
y_pred_class_STACK = final_layer.predict(X_val)

print(metrics.classification_report(Y_val, y_pred_class_STACK))
##EXPORTING The output of the test data
prediction_STACK = pd.DataFrame(y_pred_prob_STACK)
prediction_class_STACK=pd.DataFrame(y_pred_class_STACK)
#final=pd.concat(prediction,prediction_class)
#prediction = pd.DataFrame(y_pred_prob, columns=['y_pred_prob']).to_csv('prediction.csv')
final_test_pred = pd.concat([pd.DataFrame(y_pred_prob_STACK, columns=['Col_0', 'Col_1']),prediction_class_STACK], axis=1)
final_test_pred.to_csv("final_test_pred_Stack.csv")
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# roc curve for ENSEMBLE
fpr1, tpr1, thresh1 = roc_curve(Y_val, y_pred_prob_AVGENSEMB[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(Y_val, y_pred_prob_STACK[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(Y_val, y_pred_prob_RF[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(Y_val, y_pred_prob_XG[:,1], pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(Y_val, y_pred_prob_LOG[:,1], pos_label=1)
fpr6, tpr6, thresh6 = roc_curve(Y_val, y_pred_prob_WAVGENSEMB[:,1], pos_label=1)
fpr7, tpr7, thresh7 = roc_curve(Y_val, y_pred_prob_ADA[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(Y_val))]
p_fpr, p_tpr, _ = roc_curve(Y_val, random_probs, pos_label=1)
# auc scores
auc_score1 = roc_auc_score(Y_val, y_pred_prob_AVGENSEMB[:,1])
auc_score2 = roc_auc_score(Y_val, y_pred_prob_STACK[:,1])
auc_score3 = roc_auc_score(Y_val, y_pred_prob_RF[:,1])
auc_score4 = roc_auc_score(Y_val, y_pred_prob_XG[:,1])
auc_score5 = roc_auc_score(Y_val, y_pred_prob_LOG[:,1])
auc_score6 = roc_auc_score(Y_val, y_pred_prob_WAVGENSEMB[:,1])
auc_score7 = roc_auc_score(Y_val, y_pred_prob_ADA[:,1])

plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Ensemble Learning Model (Avg)')
plt.plot(fpr2, tpr2, linestyle='--',color='blue', label='Stack Ensemble Learning Model')
plt.plot(fpr3, tpr3, linestyle='--',color='yellow', label='Random Forest Model')
plt.plot(fpr4, tpr4, linestyle='--',color='red', label='XGBOOST Model')
plt.plot(fpr5, tpr5, linestyle='--',color='pink', label='Logistic Regression Model')
plt.plot(fpr6, tpr6, linestyle='--',color='green', label='Ensemble Learning Model (Weighted Avg)')
plt.plot(fpr7, tpr7, linestyle='--',color='green', label='ADABoost Model')


plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig('ROC',dpi=800)
plt.show()
print("Ensemble Learning Model (Avg)",auc_score1,"\nStack Ensemble Learning Model",auc_score2,"\nRandom Forest Model",auc_score3,
     "\nXGBOOST Model",auc_score4,"\nLogistic Regression Model",auc_score5,"\nEnsemble Learning Model (Weighted Avg)",auc_score6,
     "\nADABoost Learning Model",auc_score7)
##Taking final Model to be Avg Ensembles and RF 
cm = metrics.confusion_matrix(Y_val, y_pred_class_RF)
cm
sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Non-Defaulter", "Defaulter"] , yticklabels = ["Non-Defaulter", "Defaulter"],)
plt.ylabel('True label',fontsize=12)
plt.xlabel('Predicted label',fontsize=12)
metrics.accuracy_score(Y_val,y_pred_class_RF)
fpr3, tpr3, thresh3 = roc_curve(Y_val, y_pred_prob_RF[:,1], pos_label=1)
roc_auc3 = metrics.auc(fpr3, tpr3)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr3, tpr3, 'b', label='ROC curve (area = %0.2f)' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#fpr3, tpr3, thresh3
#As we can notice, the minimum difference between the False Positive and True Positive is when our sensitivity value 
#is at 0.42 approx Now we will calculate the new cut off value based on this value of sensitivity and see how the accuracy of our model increases.

cutoff_prob3 = round(float(thresh3[(np.abs(tpr3 - 0.42)).argmin()]),2)
cutoff_prob3
prediction_class_RF=pd.DataFrame(y_pred_class_RF)
Y_val1= Y_val.reset_index()
final_test_pred_RF = pd.concat([Y_val1,pd.DataFrame(y_pred_prob_RF, columns=['Col_0', 'Col_1']),prediction_class_RF], axis=1)

final_test_pred_RF.rename(columns = {0: "Prediction"},  inplace = True) 
final_test_pred_RF.columns

final_test_pred_RF['decile'] = pd.qcut(final_test_pred_RF['Col_1'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
final_test_pred_RF.to_csv("final_test_pred_RF.csv")
final_test_pred_RF['Non-Defaulter'] = 1-final_test_pred_RF['Prediction']
df1 = pd.pivot_table(data=final_test_pred_RF,index=['decile'],values=['Prediction','Non-Defaulter','Col_1'],aggfunc={'Prediction':[np.sum],'Non-Defaulter':[np.sum],'Col_1' : [np.min,np.max]})


df1
df1.columns = ['max_score','min_score','Non-Defaulter_Count','Defaulter_Count']
df1['Total_Cust'] = df1['Defaulter_Count']+df1['Non-Defaulter_Count']

df2 = df1.sort_values(by='min_score',ascending=False)

df2['Default_Rate'] = (df2['Defaulter_Count'] / df2['Total_Cust']).apply('{0:.2%}'.format)
default_sum = df2['Defaulter_Count'].sum()
non_default_sum = df2['Non-Defaulter_Count'].sum()
df2['Default %'] = (df2['Defaulter_Count']/default_sum).apply('{0:.2%}'.format)
df2['Non_Default %'] = (df2['Non-Defaulter_Count']/non_default_sum).apply('{0:.2%}'.format)

df2['ks_stats'] = np.round(((df2['Defaulter_Count'] / df2['Defaulter_Count'].sum()).cumsum() -(df2['Non-Defaulter_Count'] / df2['Non-Defaulter_Count'].sum()).cumsum()), 4) * 100
flag = lambda x: '*****' if x == df2['ks_stats'].max() else ''
df2['max_ks'] = df2['ks_stats'].apply(flag)
df2
final_test_pred_RF['new_labels'] = final_test_pred_RF['Col_1'].map( lambda x: 1 if x >= 4.68e-01 else 0 )

cm1 = metrics.confusion_matrix( Y_val,final_test_pred_RF.new_labels )
sns.heatmap(cm1, annot=True,  fmt='.2f', xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],)
plt.ylabel('True label',fontsize=12)
plt.xlabel('Predicted label',fontsize=12)
metrics.accuracy_score(Y_val,final_test_pred_RF.new_labels )
##Exporting the final labels 
final_test_pred_RF.to_csv("final_test_pred_RF.csv")
# Predcited probability of each class.
prediction_prob_RF_train= RF_model_final.predict_proba(X)
# Predicted value of each class
prediction_class_RF_train = RF_model_final.predict(X)

prediction_class_RF_train=pd.DataFrame(prediction_class_RF_train)
Y1= Y.reset_index()
final_train_pred_RF = pd.concat([Y1,pd.DataFrame(prediction_prob_RF_train, columns=['Col_0', 'Col_1']),prediction_class_RF_train], axis=1)
final_train_pred_RF.rename(columns = {0: "Prediction"},  inplace = True) 
final_train_pred_RF['decile'] = pd.qcut(final_train_pred_RF['Col_1'],10,labels=['1','2','3','4','5','6','7','8','9','10'])
final_train_pred_RF.to_csv("final_test_pred_RF.csv")
final_train_pred_RF['Non-Defaulter'] = 1-final_train_pred_RF['Prediction']
df1train = pd.pivot_table(data=final_train_pred_RF,index=['decile'],values=['Prediction','Non-Defaulter','Col_1'],aggfunc={'Prediction':[np.sum],'Non-Defaulter':[np.sum],'Col_1' : [np.min,np.max]})
df1train.columns = ['max_score','min_score','Non-Defaulter_Count','Defaulter_Count']
df1train['Total_Cust'] = df1train['Defaulter_Count']+df1train['Non-Defaulter_Count']
df2train = df1train.sort_values(by='min_score',ascending=False)

df2train['Default_Rate'] = (df2train['Defaulter_Count'] / df2train['Total_Cust']).apply('{0:.2%}'.format)
default_sumtrain = df2train['Defaulter_Count'].sum()
non_default_sumtrain = df2train['Non-Defaulter_Count'].sum()
df2train['Default %'] = (df2train['Defaulter_Count']/default_sum).apply('{0:.2%}'.format)
df2train['Non_Default %'] = (df2train['Non-Defaulter_Count']/non_default_sumtrain).apply('{0:.2%}'.format)

df2train['ks_stats'] = np.round(((df2train['Defaulter_Count'] / df2train['Defaulter_Count'].sum()).cumsum() -(df2train['Non-Defaulter_Count'] / df2train['Non-Defaulter_Count'].sum()).cumsum()), 4) * 100
flagtrain = lambda x: '*****' if x == df2train['ks_stats'].max() else ''
df2train['max_ks'] = df2train['ks_stats'].apply(flagtrain)
df2train

df2['default_cum_test%'] = np.round(((df2['Defaulter_Count'] / df2['Defaulter_Count'].sum()).cumsum()), 4) * 100
df2
df_gains = df2[['default_cum_test%']]

df2train['default_cum_train%'] = np.round(((df2train['Defaulter_Count'] / df2train['Defaulter_Count'].sum()).cumsum()), 4) * 100
df2train
df_gainstrain = df2train[['default_cum_train%']]
df_gainstrain.reset_index()
df_gainstrain['Base %'] = [10,20,30,40,50,60,70,80,90,100]


final_g = pd.concat([df_gainstrain,df_gains],axis=1)
final_g

gains_chart = final_g.plot(kind='line',use_index=False)
gains_chart.set_ylabel("Proportion of Defaulters",fontsize=12)
gains_chart.set_xlabel("Decile",fontsize=12)
gains_chart.set_title("Gains Chart")
cm1
FP = cm1.sum(axis=0) - np.diag(cm1) 
FN = cm1.sum(axis=1) - np.diag(cm1)
TP = np.diag(cm1)
TN = cm1.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
TPR = 100*TP/(TP+FN)
TNR = 100*TN/(TN+FP) 
FPR = 100*FP/(FP+TN)
FNR = 100*FN/(TP+FN)
ACC = 100*((TP+TN)/(TP+FP+FN+TN))
print("Sensitivity:",TPR[0],"\nSpecificity:",TNR[0],"\nPrecison :","\nFalse Postive Rate:",FPR[0],"\nFalse Negative Rate",FNR[0],"\nOverall Acuracy",ACC[0])