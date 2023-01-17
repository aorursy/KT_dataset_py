#Importing the neccessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load the input file

my_local_path='../input/'

device_data=pd.read_csv(my_local_path+'device_failure.csv')
device_data.info()
device_data.head()
device_data.describe()
device_data['failure'].value_counts()
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler().fit(device_data[['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',

                                            'attribute7','attribute8','attribute9']])

device_data_scaled=scale.fit_transform(device_data[['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',

                                            'attribute7','attribute8','attribute9']])
device_df_scaled=pd.DataFrame(device_data_scaled,columns=['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',

                                            'attribute7','attribute8','attribute9'])
device_df_scaled.head()
device_df_scaled['failure']=device_data['failure']
device_df_scaled.head()
corr=device_df_scaled.corr()
corr
sns.heatmap(corr,annot=True,fmt=".1f")
device_df_scaled.drop('attribute8',axis=1,inplace=True)
device_df_scaled.head()
sns.boxplot(x='failure',y='attribute1',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute2',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute3',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute5',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute6',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute7',data=device_df_scaled)
sns.boxplot(x='failure',y='attribute9',data=device_df_scaled)
from datetime import datetime

device_df_scaled['month']=pd.to_datetime(device_data['date']).dt.month
month_dummies=pd.get_dummies(device_df_scaled.month,prefix='month',drop_first=False)

device_df_scaled=pd.concat([device_df_scaled,month_dummies],axis=1)
device_df_scaled.pivot_table(index='month',columns='failure',aggfunc='size')
features=['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute9']
#Spliting the features and labels

X=device_df_scaled[features]

Y=device_df_scaled['failure']
X.head()
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=333)
x_train.shape
x_test.shape
np.bincount(y_train)
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=555)

x_res,y_res=sm.fit_resample(x_train,y_train)
print("resample data set class distrbibution :", np.bincount(y_res))
x_res.shape
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(criterion='entropy',max_depth =4, n_estimators = 150,max_leaf_nodes=10, random_state = 1,min_samples_leaf=5,

                            min_samples_split=10)
# Train the model with the resampled data 

my_forest=model.fit(x_res,y_res)
# Training accuracy

my_forest.score(x_res,y_res)
y_train_pred=my_forest.predict(x_res)
y_pred=my_forest.predict(x_test)
from sklearn import metrics

from sklearn.metrics import confusion_matrix,accuracy_score
# Test accuracy

accuracy_score(y_pred,y_test)
# Testing Confusion Matrix

print(confusion_matrix(y_test,y_pred))
#Training CF

print(confusion_matrix(y_res,y_train_pred))
cr=metrics.classification_report(y_test,y_pred)

print(cr)
dt_parameters={"criterion":['gini','entropy'],"max_depth":[3,7],"max_leaf_nodes": [20,30],"n_estimators":[100,200,300]}
from sklearn.model_selection  import GridSearchCV

grid_rf=GridSearchCV(RandomForestClassifier(),dt_parameters)
grid_rf_model=grid_rf.fit(x_res,y_res)
grid_rf_model.best_params_
grid_predictor=grid_rf_model.predict(x_test)
print(confusion_matrix(y_test,grid_predictor))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=0.2)

logreg.fit(x_res,y_res)
y_logi_pred=logreg.predict(x_test)
print(confusion_matrix(y_test,y_logi_pred))
from sklearn.utils import resample
train_data=pd.concat([x_train,y_train],axis=1)
train_data.shape
df_majority = train_data[train_data.failure==0]

df_minority = train_data[train_data.failure==1]
df_majority.shape
df_minority.shape
df_minorty_upsample=resample(df_minority, 

                             replace=True,     # sample with replacement

                             n_samples=30000,    # to some good number

                             random_state=123) # reproducible results

 
df_majoirty_downsample=resample(df_majority, 

                             replace=False,     # sample without replacement

                             n_samples=50000,    # to some good number

                             random_state=321) # reproducible results

 
df_majoirty_downsample.shape
df_minorty_upsample.shape
final_sample_merged=pd.concat([df_minorty_upsample,df_majoirty_downsample],axis=0)
final_sample_merged.head()
final_sample_merged['failure'].value_counts()
x_resample_train=final_sample_merged[features]

y_resample_train=final_sample_merged['failure']
y_resample_train.shape
model_rf=RandomForestClassifier(criterion='entropy', max_depth = 6, n_estimators = 200, max_leaf_nodes=10,

                                    min_samples_leaf=10,min_samples_split=40,random_state = 1)
model_rf.fit(x_resample_train,y_resample_train)
y_pred_train=model_rf.predict(x_resample_train)
y_pred=model_rf.predict(x_test)
confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
from sklearn.model_selection import cross_val_score
model_kfold_rf=RandomForestClassifier(criterion='entropy', max_depth = 6, n_estimators = 200, max_leaf_nodes=10,

                                    min_samples_leaf=10,min_samples_split=40,random_state = 1,)
scores = cross_val_score(model_kfold_rf, x_resample_train, y_resample_train, scoring='recall', cv=10)
scores
model_kfold_rf.fit(x_resample_train,y_resample_train)
y_pred_prob=model_kfold_rf.predict_proba(x_test)
y_cv_pred=model_kfold_rf.predict(x_test)
confusion_matrix(y_test,y_cv_pred)
print(metrics.classification_report(y_test,y_cv_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:,1])
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Device Failure classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
roc_df=pd.DataFrame(columns={'fpr','tpr','threshold'})
roc_df['fpr']=fpr

roc_df['tpr']=tpr

roc_df['threshold']=thresholds
roc_df.loc[(roc_df['fpr']<0.16) & (roc_df['tpr']>0.77) ]
def adjusted_classes(y_scores, t):

    """

    This function adjusts class predictions based on the prediction threshold (t).

    """

    return [1 if y >= t else 0 for y in y_scores]
y_pred=adjusted_classes(y_pred_prob[:,1],0.22)
confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
# We will also see the Area under curve

print(metrics.roc_auc_score(y_test, y_pred_prob[:,1]))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=0.6)
logreg.fit(x_resample_train,y_resample_train)
y_log_pred=logreg.predict(x_test)
print(confusion_matrix(y_test,y_log_pred))
y_pred_log_prob=logreg.predict_proba(x_test)
scores_logreg = cross_val_score(logreg, x_resample_train, y_resample_train, scoring='recall', cv=5)
scores_logreg
# ROC Curve for the logistic regression

fpr_log, tpr_log, thresholds_log = metrics.roc_curve(y_test, y_pred_log_prob[:,1])



plt.plot(fpr_log, tpr_log)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Device Failure classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
# We will also see the Area under curve

print(metrics.roc_auc_score(y_test, y_pred_log_prob[:,1]))
from sklearn.ensemble import AdaBoostClassifier
adb_model=AdaBoostClassifier(n_estimators=300,

                             learning_rate=0.2)
adb_model.fit(x_resample_train,y_resample_train)
y_adb_predict=adb_model.predict(x_test)
print(confusion_matrix(y_test,y_adb_predict))
# GridSearch on Adaboost

from sklearn.model_selection import  GridSearchCV

dt_parameters={"n_estimators":[100,200],"learning_rate":[0.1,0.2,0.4]}



grid_adaboost=GridSearchCV(AdaBoostClassifier(),dt_parameters)

grid_adaboost.fit(x_resample_train,y_resample_train)
grid_adaboost.best_params_
y_pred_adb_prob=adb_model.predict_proba(x_test)
# ROC AUC Curve

fpr_adb, tpr_adb, thresholds_adb = metrics.roc_curve(y_test, y_pred_adb_prob[:,1])



plt.plot(fpr_adb, tpr_adb)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Device Failure classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
print(metrics.roc_auc_score(y_test, y_pred_adb_prob[:,1]))
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(learning_rate_init=0.00008)

mlp_clf.fit(x_resample_train,y_resample_train)
y_mlp_pred=mlp_clf.predict(x_test)

confusion_matrix(y_test,y_mlp_pred)
y_pred_mlp_prob=mlp_clf.predict_proba(x_test)
print(metrics.classification_report(y_test,y_mlp_pred))
fpr_mlp, tpr_mlp, thresholds_mlp = metrics.roc_curve(y_test, y_pred_mlp_prob[:,1])



plt.plot(fpr_mlp, tpr_mlp)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Device Failure classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
print(metrics.roc_auc_score(y_test, y_pred_mlp_prob[:,1]))
#Build Stacking model

from sklearn.model_selection import KFold



base1_clf = model_rf

base2_clf = logreg

base3_clf = adb_model

base4_clf = mlp_clf

final_clf =logreg



# Defining the K Fold

n_folds = 5

n_class = 2

kf = KFold(n_splits= n_folds, shuffle=True, random_state=42)

def get_oof(clf, x_train, y_train, x_test):

    ntest = x_test.shape[0]

    oof_train = np.zeros((x_train.shape[0],n_class))

    oof_test  = np.zeros((x_test.shape[0],n_class))

    oof_test_temp = np.empty((n_folds, ntest))

   

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]

  

        

        clf.fit(x_tr, y_tr)



        pred_te = clf.predict_proba(x_te)

        oof_train[test_index,:] = pred_te

        

        pred_test = clf.predict_proba(x_test)

        oof_test += pred_test



    return oof_train, oof_test/n_folds
base1_oof_train, base1_oof_test = get_oof(base1_clf, x_resample_train.values,y_resample_train.values, x_test.values)

base2_oof_train, base2_oof_test = get_oof(base2_clf, x_resample_train.values,y_resample_train.values, x_test.values)

base3_oof_train, base3_oof_test = get_oof(base3_clf, x_resample_train.values,y_resample_train.values, x_test.values)

base4_oof_train, base4_oof_test = get_oof(base4_clf, x_resample_train.values,y_resample_train.values, x_test.values)
base1_oof_train
base1_oof_test
x_train_stack = np.concatenate((base1_oof_train, 

                          base2_oof_train,

                          base3_oof_train,

                          base4_oof_train), axis=1)

x_test_stack = np.concatenate((base1_oof_test,

                         base2_oof_test,

                         base3_oof_test,

                         base4_oof_test),axis=1)
x_train_stack.shape
y_resample_train.shape
x_test_stack.shape
final_clf.fit(x_train_stack,y_resample_train)
y_stacked_predict=final_clf.predict(x_test_stack)
confusion_matrix(y_test,y_stacked_predict)
print(metrics.classification_report(y_test,y_stacked_predict))