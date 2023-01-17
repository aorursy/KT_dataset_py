import numpy as np

import pandas as pd

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

import scikitplot as skplt

import warnings

import shap



shap.initjs()



from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

import lightgbm as lgbm

import xgboost as xgb



from sklearn.model_selection import KFold, StratifiedKFold

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score, classification_report

from imblearn.pipeline import make_pipeline



%matplotlib inline

np.random.seed(42)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.3f}'.format

sns.set_style("whitegrid", {'axes.grid' : False})
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)
df.head()
df.describe()
df.info()
print('Verificando o total')

print('Total missing:', df.isnull().sum().max())

print('-------------------------------')

print('if (missing > 0): Verificar a porcentagem de missing em cada variável')

percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,

                                 'percent_missing': percent_missing})

print(missing_value_df.to_string(index=False))

sns.countplot('Class',data=df).set_title('Target Distribution\n[0 - No Fraud]  [1 - Fraud]', size=15)

plt.show()
print('No Fraud:', df[df['Class'] == 0].shape)

print('Fraud:', df[df['Class'] == 1].shape)
print('Time - Not Fraud')

print(df.Time[df['Class'] == 0].describe())

print('--------------------------')

print('Time - Fraud')

print(df.Time[df['Class'] == 1].describe())



f, axes = plt.subplots(1, 2, figsize=(25, 8))

sns.boxplot(x="Class", y="Time", ax=axes[0], data=df).set_title('Time BoxPlot\n[0 - No Fraud]  [1 - Fraud]', size=15)

sns.distplot(df.Time[df['Class'] == 1], ax=axes[1], bins=50, label='Fraud', color='r').set_title('Transactions on the variable Time', size=15)

sns.distplot(df.Time[df['Class'] == 0], ax=axes[1], bins=50, label='Not Fraud', color='b')

plt.legend()

plt.show()
df['Hour'] = np.ceil(df['Time']/3600).mod(24)



plt.figure(figsize=(17, 6))

sns.distplot(df.Hour[df['Class'] == 1], bins=50, label='Fraud', color='r').set_title('Transactions on the variable Hour', size=15)

sns.distplot(df.Hour[df['Class'] == 0], bins=50, label='Not Fraud', color='b')

plt.xticks(range(0,24))

plt.legend()

plt.show()
print('Amount - Not Fraud')

print(df.Amount[df['Class'] == 0].describe())

print('----------------------------')

print('Amount - Fraud')

print(df.Amount[df['Class'] == 1].describe())





f, axes = plt.subplots(1, 2, figsize=(25, 8))

sns.boxplot(x="Class", y="Amount", ax=axes[0], data=df).set_title('Amount BoxPlot\n[0 - No Fraud]  [1 - Fraud]', size=15)

sns.distplot(df.Amount[df['Class'] == 1], ax=axes[1], bins=50, label='Fraud', color='r').set_title('Transactions on the variable Amount', size=15)

sns.distplot(df.Amount[df['Class'] == 0], ax=axes[1], bins=50, label='Not Fraud', color='b')

plt.legend()

plt.show()
f, axes = plt.subplots(2, 2, figsize=(20,10))



sns.scatterplot(df.Time[df['Class'] == 1], df.Amount[df['Class'] == 1], label='Fraud', ax=axes[0,0], color ='r').set_title('Transactions on the variable Amount and Time')

sns.scatterplot(df.Time[df['Class'] == 0], df.Amount[df['Class'] == 0],  label='Not Fraud', ax=axes[1,0], color = 'b')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')



sns.scatterplot(df.Hour[df['Class'] == 1], df.Amount[df['Class'] == 1], label='Fraud', ax=axes[0,1], color ='r').set_title('Transactions on the variable Amount and Hour')

sns.scatterplot(df.Hour[df['Class'] == 0], df.Amount[df['Class'] == 0],  label='Not Fraud', ax=axes[1, 1], color = 'b')

plt.xlabel('Hour')

plt.ylabel('Amount')

plt.show()
plt.figure()

fig, ax = plt.subplots(7,4,figsize=(20,30))



i = 0

for c in df.columns[1:-3]:

    i += 1

    plt.subplot(7,4,i)

    sns.kdeplot(df.loc[df['Class'] == 0][c],label="Not Fraud")

    sns.kdeplot(df.loc[df['Class'] == 1][c],label="Fraud")

    plt.xlabel(c, fontsize=11)

    locs, labels = plt.xticks()

plt.show();
cols = list(df.columns.values)

cols.pop(cols.index('Time'))

cols.pop(cols.index('Hour'))

cols.pop(cols.index('Amount'))

df = df[['Time','Hour', 'Amount'] + cols]
corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.figure(figsize=(16,11))

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values).set_title('Correlation between all features', size=22)

plt.show()
corr1 = df[df['Class'] == 1].corr()

corr0 = df[df['Class'] == 0].corr()



mask = np.zeros_like(corr1, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)



f, axes = plt.subplots(1, 2, figsize=(30,11))

sns.heatmap(corr1, mask=mask, cmap=cmap, ax=axes[0], vmin=-1, vmax=1,

            xticklabels=corr1.columns.values,

            yticklabels=corr1.columns.values).set_title('Fraud Correlation', size=22)

sns.heatmap(corr0, mask=mask, cmap=cmap, ax=axes[1], vmin=-1, vmax=1,

            xticklabels=corr0.columns.values,

            yticklabels=corr0.columns.values).set_title('Not Fraud Correlation', size=22)

plt.show()
X = df.drop('Class', axis=1)

y = df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify = y)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42, stratify = y_train)



print('Train:', X_train.shape)

print('Test:', X_test.shape)

print('Valid:', X_valid.shape)
sm = SMOTE(sampling_strategy='minority', random_state=42)

X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)



print('X_smote shape:', X_train_smote.shape)

print('Distribuição da variável alvo após aplicação do método SMOTE:', np.unique(y_train_smote, return_counts=True))
adasyn = ADASYN(sampling_strategy='minority', random_state=42)

X_train_adasyn, y_train_adasyn = adasyn.fit_sample(X_train, y_train)



print('X_adasyn shape:', X_train_adasyn.shape)

print('Distribuição da variável alvo após aplicação do método ADASYN:', np.unique(y_train_adasyn, return_counts=True))
random_grid = {

    'bootstrap': [True, False],

    'max_depth': [1, 2, 3, 5, 10, None],

    'max_features': ['auto', 'sqrt'],

    'n_estimators': [50, 100, 200],

    "criterion": ["entropy", "gini"],

    "class_weight": [None, 'balanced']

}



clr_rf = RandomForestClassifier(random_state=42)

gs = RandomizedSearchCV(clr_rf, param_distributions=random_grid, n_iter=2, verbose=0, cv=3, scoring='roc_auc', random_state=42)

gs.fit(X_train, y_train)



print('RANDOM FOREST')

print("Best Score: {}".format(gs.best_score_))

print("Best Parameters: {}\n".format(gs.best_params_))
# Unbalanced

clr_best_rf = RandomForestClassifier(**gs.best_params_)

clr_best_rf.fit(X_train, y_train)

predictions_rf = clr_best_rf.predict(X_test)

roc_auc_rf = np.round(roc_auc_score(y_test, predictions_rf), 2)



print('Random Forest Classifier with Unbalanced Data')

print('ROC AUC score on test data:', roc_auc_rf)

print('Classification Report:')

print(classification_report(y_test, predictions_rf))

print('------------------------------------------------------')

    

# SMOTE 

if gs.best_params_['class_weight'] == 'balanced':

    gs.best_params_['class_weight'] = None



clr_best_rf_smote = RandomForestClassifier(**gs.best_params_)

clr_best_rf_smote.fit(X_train_smote, y_train_smote)

predictions_rf_smote = clr_best_rf_smote.predict(X_test)

roc_auc_rf_smote = np.round(roc_auc_score(y_test, predictions_rf_smote), 2)



print('Random Forest Classifier with SMOTE')

print('ROC AUC score on test data:', roc_auc_rf_smote)

print('Classification Report:')

print(classification_report(y_test, predictions_rf_smote))

print('------------------------------------------------------')



# ADASYN

if gs.best_params_['class_weight'] == 'balanced':

    gs.best_params_['class_weight'] = None



clr_best_rf_adasyn = RandomForestClassifier(**gs.best_params_)

clr_best_rf_adasyn.fit(X_train_adasyn, y_train_adasyn)

predictions_rf_adasyn = clr_best_rf_adasyn.predict(X_test)

roc_auc_rf_adasyn = np.round(roc_auc_score(y_test, predictions_rf_adasyn), 2)



print('Random Forest Classifier with ADASYN')

print('ROC AUC score on test data:', roc_auc_rf_adasyn)

print('Classification Report:')

print(classification_report(y_test, predictions_rf_adasyn))
plt.figure(figsize=(30,25))



ax1 = plt.subplot(331)

skplt.metrics.plot_confusion_matrix(y_test, predictions_rf, normalize=False, ax=ax1)

ax1.set_yticklabels(['Not Fraud', 'Fraud'])

ax1.set_xticklabels(['Not Fraud', 'Fraud']) 

ax1.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_rf, 1 - roc_auc_rf))

ax1.set_title('Confusion Matrix (Unbalanced)')



y_probas = clr_best_rf.predict_proba(X_test)

ax2 = plt.subplot(332)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax2)

ax2.set_title('ROC Curves (Unbalanced)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_rf.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax3 = plt.subplot(333)

plt.title('Features importance (Unbalanced)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax4 = plt.subplot(334)

skplt.metrics.plot_confusion_matrix(y_test, predictions_rf_smote, normalize=False, ax=ax4)

ax4.set_yticklabels(['Not Fraud', 'Fraud'])

ax4.set_xticklabels(['Not Fraud', 'Fraud']) 

ax4.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_rf_smote, 1 - roc_auc_rf_smote))

ax4.set_title('Confusion Matrix (SMOTE)')



y_probas = clr_best_rf_smote.predict_proba(X_test)

ax5 = plt.subplot(335)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax5)

ax5.set_title('ROC Curves (SMOTE)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_rf_smote.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax6 = plt.subplot(336)

plt.title('Features importance (SMOTE)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax7 = plt.subplot(337)

skplt.metrics.plot_confusion_matrix(y_test, predictions_rf_adasyn, normalize=False, ax=ax7)

ax7.set_yticklabels(['Not Fraud', 'Fraud'])

ax7.set_xticklabels(['Not Fraud', 'Fraud']) 

ax7.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_rf_adasyn, 1 - roc_auc_rf_adasyn))

ax7.set_title('Confusion Matrix (ADASYN)')



y_probas = clr_best_rf_adasyn.predict_proba(X_test)

ax8 = plt.subplot(338)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax8)

ax8.set_title('ROC Curves (ADASYN)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_rf_adasyn.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax9 = plt.subplot(339)

plt.title('Features importance (ADASYN)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



plt.show() 
balanced_weight = np.unique(y_train, return_counts=True)[1][0] / np.unique(y_train, return_counts=True)[1][1]



fit_params = {"early_stopping_rounds" : 50, 

             "eval_metric" : 'auc', 

             "eval_set" : [(X_valid, y_valid)],

             "verbose" :  0}



random_grid = {

        'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

        'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

        'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': np.linspace(0.6, 1, 10),

        'max_depth': [1, 2, 3, 4, 5, 6, 7],

        'scale_pos_weight' : [1, balanced_weight]

        }



clr_xgb = XGBClassifier(objective='binary:logistic', tree_method='gpu_hist', random_state=42)

gs = RandomizedSearchCV(clr_xgb, param_distributions=random_grid, n_iter=30, verbose=0, cv=5, scoring='roc_auc', random_state=42)

gs.fit(X_train, y_train, **fit_params)



print('XGBOOST')

print("Best Score: {}".format(gs.best_score_))

print("Best Parameters: {}\n".format(gs.best_params_))
opt_parameters =  gs.best_params_



# Unbalanced

clr_best_xgb = XGBClassifier(**clr_xgb.get_params())

clr_best_xgb.set_params(**opt_parameters)

clr_best_xgb.fit(X_train, y_train)

predictions_xgb = clr_best_xgb.predict(X_test)

roc_auc_xgb = np.round(roc_auc_score(y_test, predictions_xgb), 2)



print('XGBoost with Unbalanced Data:')

print('ROC AUC score on test data:', roc_auc_xgb)

print('Classification Report:')

print(classification_report(y_test, predictions_xgb))

print('------------------------------------------------------')





# SMOTE

if opt_parameters['scale_pos_weight'] != 1:

    opt_parameters['scale_pos_weight'] = 1



clr_best_xgb_smote = XGBClassifier(**clr_xgb.get_params())

clr_best_xgb_smote.set_params(**opt_parameters)

clr_best_xgb_smote.fit(X_train_smote, y_train_smote)

predictions_xgb_smote = clr_best_xgb_smote.predict(X_test.values)

roc_auc_xgb_smote = np.round(roc_auc_score(y_test, predictions_xgb_smote), 2)



print('XGBoost with SMOTE:')

print('ROC AUC score on test data:', roc_auc_xgb_smote)

print('Classification Report:')

print(classification_report(y_test, predictions_xgb_smote))

print('------------------------------------------------------')





# ADASYN

if opt_parameters['scale_pos_weight'] != 1:

    opt_parameters['scale_pos_weight'] = 1



clr_best_xgb_adasyn = XGBClassifier(**clr_xgb.get_params())

clr_best_xgb_adasyn.set_params(**opt_parameters)

clr_best_xgb_adasyn.fit(X_train_adasyn, y_train_adasyn)

predictions_xgb_adasyn = clr_best_xgb_adasyn.predict(X_test.values)

roc_auc_xgb_adasyn = np.round(roc_auc_score(y_test, predictions_xgb_adasyn), 2)



print('XGBoost with ADASYN:')

print('ROC AUC score on test data:', roc_auc_xgb_adasyn)

print('Classification Report:')

print(classification_report(y_test, predictions_xgb_adasyn))
plt.figure(figsize=(25,20))



ax1 = plt.subplot(331)

skplt.metrics.plot_confusion_matrix(y_test, predictions_xgb, normalize=False, ax=ax1)

ax1.set_yticklabels(['Not Fraud', 'Fraud'])

ax1.set_xticklabels(['Not Fraud', 'Fraud']) 

ax1.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_xgb, 1 - roc_auc_xgb))

ax1.set_title('Confusion Matrix (Unbalanced)')



y_probas = clr_best_xgb.predict_proba(X_test)

ax2 = plt.subplot(332)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax2)

ax2.set_title('ROC Curves (Unbalanced)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_xgb.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax3 = plt.subplot(333)

plt.title('Features importance (Unbalanced)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax4 = plt.subplot(334)

skplt.metrics.plot_confusion_matrix(y_test, predictions_xgb_smote, normalize=False, ax=ax4)

ax4.set_yticklabels(['Not Fraud', 'Fraud'])

ax4.set_xticklabels(['Not Fraud', 'Fraud']) 

ax4.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_xgb_smote, 1 - roc_auc_xgb_smote))

ax4.set_title('Confusion Matrix (SMOTE)')



y_probas = clr_best_xgb_smote.predict_proba(X_test.values)

ax5 = plt.subplot(335)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax5)

ax5.set_title('ROC Curves (SMOTE)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_xgb_smote.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax6 = plt.subplot(336)

plt.title('Features importance (SMOTE)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax7 = plt.subplot(337)

skplt.metrics.plot_confusion_matrix(y_test, predictions_xgb_adasyn, normalize=False, ax=ax7)

ax7.set_yticklabels(['Not Fraud', 'Fraud'])

ax7.set_xticklabels(['Not Fraud', 'Fraud']) 

ax7.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_xgb_adasyn, 1 - roc_auc_xgb_adasyn))

ax7.set_title('Confusion Matrix (ADASYN)')



y_probas = clr_best_xgb_adasyn.predict_proba(X_test.values)

ax8 = plt.subplot(338)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax8)

ax8.set_title('ROC Curves (ADASYN)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_xgb_adasyn.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax9 = plt.subplot(339)

plt.title('Features importance (ADASYN)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



plt.show() 
fit_params = {"early_stopping_rounds" : 50, 

             "eval_metric" : 'binary', 

             "eval_set" : [(X_valid,y_valid)],

             "eval_names": ['valid'],

             "verbose": 0}



random_test = {

    'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

    'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000],

    'num_leaves': np.random.randint(6, 50), 

    'min_child_samples': np.random.randint(100, 500), 

    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

    'subsample': np.linspace(0.5, 1, 100), 

    'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],

    'colsample_bytree': np.linspace(0.6, 1, 10),

    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],

    'scale_pos_weight' : [1, balanced_weight]

    }



clr_gbm = LGBMClassifier(objective='binary', metric='auc', tree_method='gpu_hist', random_state=42)

gs = RandomizedSearchCV(clr_gbm, param_distributions=random_grid, n_iter=30, verbose=0, cv=5, scoring='roc_auc', random_state=42)

gs.fit(X_train, y_train, **fit_params)



print('LIGHTGBM')

print("Best Score: {}".format(gs.best_score_))

print("Best Parameters: {}\n".format(gs.best_params_))

opt_parameters =  gs.best_params_



# Unbalanced

clr_best_gbm = LGBMClassifier(**clr_gbm.get_params())

clr_best_gbm.set_params(**opt_parameters)

clr_best_gbm.fit(X_train, y_train)

predictions_gbm = clr_best_gbm.predict(X_test)

roc_auc_gbm = np.round(roc_auc_score(y_test, predictions_gbm), 2)



print('LightGBM with Unbalanced Data:')

print('ROC AUC score on test data:', roc_auc_gbm)

print('Classification Report:')

print(classification_report(y_test, predictions_gbm))

print('------------------------------------------------------')





# SMOTE

if gs.best_params_['scale_pos_weight'] != 1:

    gs.best_params_['scale_pos_weight'] = 1



clr_best_gbm_smote = LGBMClassifier(**clr_gbm.get_params())

clr_best_gbm_smote.set_params(**opt_parameters)

clr_best_gbm_smote.fit(X_train_smote, y_train_smote)

predictions_gbm_smote = clr_best_gbm_smote.predict(X_test.values)

roc_auc_gbm_smote = np.round(roc_auc_score(y_test, predictions_gbm_smote), 2)



print('LightGBM with SMOTE:')

print('ROC AUC score on test data:', roc_auc_gbm_smote)

print('Classification Report:')

print(classification_report(y_test, predictions_gbm_smote))

print('------------------------------------------------------')





# ADASYN

if gs.best_params_['scale_pos_weight'] != 1:

    gs.best_params_['scale_pos_weight'] = 1



clr_best_gbm_adasyn = LGBMClassifier(**clr_gbm.get_params())

clr_best_gbm_adasyn.set_params(**opt_parameters)

clr_best_gbm_adasyn.fit(X_train_adasyn, y_train_adasyn)

predictions_gbm_adasyn = clr_best_gbm_adasyn.predict(X_test.values)

roc_auc_gbm_adasyn = np.round(roc_auc_score(y_test, predictions_gbm_adasyn), 2)



print('LightGBM with ADASYN:')

print('ROC AUC score on test data:', roc_auc_gbm_adasyn)

print('Classification Report:')

print(classification_report(y_test, predictions_gbm_adasyn))
plt.figure(figsize=(25,20))



ax1 = plt.subplot(331)

skplt.metrics.plot_confusion_matrix(y_test, predictions_gbm, normalize=False, ax=ax1)

ax1.set_yticklabels(['Not Fraud', 'Fraud'])

ax1.set_xticklabels(['Not Fraud', 'Fraud']) 

ax1.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_gbm, 1 - roc_auc_gbm))

ax1.set_title('Confusion Matrix (Unbalanced)')



y_probas = clr_best_gbm.predict_proba(X_test)

ax2 = plt.subplot(332)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax2)

ax2.set_title('ROC Curves (Unbalanced)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_gbm.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax3 = plt.subplot(333)

plt.title('Features importance (Unbalanced)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax4 = plt.subplot(334)

skplt.metrics.plot_confusion_matrix(y_test, predictions_gbm_smote, normalize=False, ax=ax4)

ax4.set_yticklabels(['Not Fraud', 'Fraud'])

ax4.set_xticklabels(['Not Fraud', 'Fraud']) 

ax4.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_gbm_smote, 1 - roc_auc_gbm_smote))

ax4.set_title('Confusion Matrix (SMOTE)')



y_probas = clr_best_gbm_smote.predict_proba(X_test.values)

ax5 = plt.subplot(335)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax5)

ax5.set_title('ROC Curves (SMOTE)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_gbm_smote.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax6 = plt.subplot(336)

plt.title('Features importance (SMOTE)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



ax7 = plt.subplot(337)

skplt.metrics.plot_confusion_matrix(y_test, predictions_gbm_adasyn, normalize=False, ax=ax7)

ax7.set_yticklabels(['Not Fraud', 'Fraud'])

ax7.set_xticklabels(['Not Fraud', 'Fraud']) 

ax7.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_gbm_adasyn, 1 - roc_auc_gbm_adasyn))

ax7.set_title('Confusion Matrix (ADASYN)')



y_probas = clr_best_gbm_adasyn.predict_proba(X_test.values)

ax8 = plt.subplot(338)

skplt.metrics.plot_roc(y_test, y_probas, figsize=(10,8), ax=ax8)

ax8.set_title('ROC Curves (ADASYN)')



rf_fi = pd.DataFrame({'Feature': X.columns, 'Feature importance': clr_best_gbm_adasyn.feature_importances_})

rf_fi = rf_fi.sort_values(by='Feature importance', ascending=False)

ax9 = plt.subplot(339)

plt.title('Features importance (ADASYN)',fontsize=14)

sns.barplot(x='Feature importance',y='Feature',data=rf_fi)



plt.show() 
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(clr_best_xgb_smote, scoring='roc_auc', random_state=42).fit(X_test.values, y_test.values)

eli5.show_weights(perm, feature_names = X.columns.tolist())
booster = clr_best_xgb_smote.get_booster()

shap_values = booster.predict(xgb.DMatrix(X_test.values), pred_contribs=True)
shap.summary_plot(shap_values[:,:-1], X_test, feature_names=X.columns)
shap.dependence_plot("V4", shap_values[:,:-1], X_test)
shap.dependence_plot("V14", shap_values[:,:-1], X_test)
import tensorflow.keras.backend as K



def preprocessing_fnn(data):

    norm = (data - data.mean())/data.std()

    return norm



X_train_fnn = preprocessing_fnn(X_train)

X_test_fnn = preprocessing_fnn(X_test)

X_valid_fnn = preprocessing_fnn(X_valid)



def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed



n_inputs = X_train.shape[1]



def model_fnn(loss_f):

    model = tf.keras.models.Sequential()

    

    model.add(tf.keras.layers.Dense(512, input_shape=(X_train.shape[1],)))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Activation("relu"))

    #model.add(tf.keras.layers.Dropout(0.2))

    

    model.add(tf.keras.layers.Dense(256, kernel_initializer="he_normal", use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Activation("relu"))

    #model.add(tf.keras.layers.Dropout(0.2))

    

    model.add(tf.keras.layers.Dense(128, kernel_initializer="he_normal", use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Activation("relu"))

    #model.add(tf.keras.layers.Dropout(0.2))

    

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    

    model.compile(loss=loss_f, optimizer=tf.keras.optimizers.Nadam())

    

    return model



model_fnn_focal_loss_low_alpha = model_fnn(focal_loss())

model_fnn_focal_loss_high_alpha = model_fnn(focal_loss(alpha=4.))

model_fnn_binary_ce = model_fnn('binary_crossentropy')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=10)



model_fnn_focal_loss_low_alpha.fit(X_train_fnn, y_train, batch_size=1024, epochs=100, validation_data=(X_valid_fnn, y_valid),

                                   callbacks=[early_stopping])

print("----------------------------------------------")

model_fnn_focal_loss_high_alpha.fit(X_train_fnn, y_train, batch_size=1024, epochs=100, validation_data=(X_valid_fnn, y_valid),

                                   callbacks=[early_stopping])

print("----------------------------------------------")

model_fnn_binary_ce.fit(X_train, y_train, batch_size=1024, epochs=100, validation_data=(X_valid, y_valid),

                                   callbacks=[early_stopping])



preds_focal_loss_low_alpha = model_fnn_focal_loss_low_alpha.predict_classes(X_test, verbose=0)

roc_auc_focal_loss_low_alpha = np.round(roc_auc_score(y_test, preds_focal_loss_low_alpha), 2)



preds_focal_loss_high_alpha = model_fnn_focal_loss_high_alpha.predict_classes(X_test, verbose=0)

roc_auc_focal_loss_high_alpha = np.round(roc_auc_score(y_test, preds_focal_loss_high_alpha), 2)



preds_binary_ce = model_fnn_binary_ce.predict_classes(X_test, verbose=0)

roc_auc_binary_ce = np.round(roc_auc_score(y_test, preds_binary_ce), 2)





plt.figure(figsize=(25,20))

ax1 = plt.subplot(331)

skplt.metrics.plot_confusion_matrix(y_test, preds_focal_loss_low_alpha, normalize=False, ax=ax1)

ax1.set_yticklabels(['Not Fraud', 'Fraud'])

ax1.set_xticklabels(['Not Fraud', 'Fraud']) 

ax1.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_focal_loss_low_alpha, 1 - roc_auc_focal_loss_low_alpha))

ax1.set_title('Confusion Matrix (Focal Loss with LOW Alpha)')



ax2 = plt.subplot(332)

skplt.metrics.plot_confusion_matrix(y_test, preds_focal_loss_high_alpha, normalize=False, ax=ax2)

ax2.set_yticklabels(['Not Fraud', 'Fraud'])

ax2.set_xticklabels(['Not Fraud', 'Fraud']) 

ax2.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_focal_loss_high_alpha, 1 - roc_auc_focal_loss_high_alpha))

ax2.set_title('Confusion Matrix (Focal Loss with HIGH Alpha)')



ax3 = plt.subplot(333)

skplt.metrics.plot_confusion_matrix(y_test, preds_binary_ce, normalize=False, ax=ax3)

ax3.set_yticklabels(['Not Fraud', 'Fraud'])

ax3.set_xticklabels(['Not Fraud', 'Fraud']) 

ax3.set_xlabel('Predicted label\nroc auc={:0.2f}; missclass={:0.2f}'.format(roc_auc_binary_ce, 1 - roc_auc_binary_ce))

ax3.set_title('Confusion Matrix (Binary CrossEntropy)')



plt.show()