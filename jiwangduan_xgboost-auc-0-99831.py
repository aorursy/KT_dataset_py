import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score

from sklearn import metrics





import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import style

style.use('ggplot')

import seaborn as sns

%matplotlib inline





from scipy.stats import skew, boxcox

import warnings

warnings.filterwarnings("ignore")



plt.rcParams['font.sans-serif'] = ['SimHei']  

plt.rcParams['axes.unicode_minus'] = False
# Checking data

csvfile_path = '../input/PS_20174392719_1491204439457_log.csv'

raw_data = pd.read_csv(csvfile_path)

print ("raw data shape:", raw_data.shape)

print(raw_data.head())

print(raw_data.describe())

print(raw_data.info())
# check the missing values

raw_data.isnull().any()
# exploratory the colum of 'type'

print(raw_data['type'].value_counts())

raw_data['type'].value_counts().plot.pie(autopct='%.2f',figsize=(5, 5))

plt.title('type')

plt.tight_layout()
pd.crosstab(raw_data['type'], raw_data['isFraud'])
pd.crosstab(raw_data['type'], raw_data['isFlaggedFraud'])
raw_data[['nameOrig', 'nameDest']].describe()
data_used = raw_data.loc[(raw_data['type'].isin(['TRANSFER', 'CASH_OUT'])),:]

data_used.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)



data_used = data_used.reset_index(drop=True)



type_encoder = preprocessing.LabelEncoder()

type_category = type_encoder.fit_transform(data_used['type'].values)

data_used['type_code'] = type_category



print ("data_used shape:", data_used.shape)

print(data_used.head())

print(data_used.info())
sns.heatmap(data_used.corr())
# check the balance of the data_used

print(data_used['isFraud'].value_counts())



data_used.isFraud.value_counts().plot.pie(autopct='%.2f',figsize=(5, 5))

plt.title('Ratio of class')

plt.tight_layout()
# under sample

feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 

                 'oldbalanceDest', 'newbalanceDest', 'type_code']

number_records_fraud = len(data_used[data_used['isFraud'] == 1])



#　indices of fraud_indices

fraud_indices = data_used[data_used['isFraud'] == 1].index.values



#indices of the normal records

nonfraud_indices = data_used[data_used['isFraud'] == 0].index



random_nonfraud_indices = np.random.choice(nonfraud_indices, number_records_fraud, replace=False)

random_nonfraud_indices = np.array(random_nonfraud_indices)



under_sample_indices = np.concatenate([fraud_indices, random_nonfraud_indices])

under_sample_data = data_used.iloc[under_sample_indices, :]

                          

print(under_sample_data[feature_names].head())

X_undersample = under_sample_data[feature_names].values

y_undersample = under_sample_data['isFraud'].values

print("--------------------------------------------------------------------------")



print("Ratio of nomal: ", len(under_sample_data[under_sample_data['isFraud'] == 0]) / len(under_sample_data))

print("Ratio of fraud: ", len(under_sample_data[under_sample_data['isFraud'] == 1]) / len(under_sample_data))

print("Number of data for model: ", len(under_sample_data))

X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegressionCV

alpha = np.logspace(-2, 2, 20)

lr_model_cv = LogisticRegressionCV(Cs=alpha, penalty='l1', solver='liblinear', cv=5)

lr_model_cv.fit(X_train, y_train)



y_pred_score_cv = lr_model_cv.predict_proba(X_test)

# print('y_pred_score_cv:')

# print(y_pred_score_cv)



fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_test, y_pred_score_cv[:, 1])

roc_auc_cv = auc(fpr_cv,tpr_cv)

print('auc:', roc_auc_cv)

# 绘制ROC曲线

plt.title('Receiver Operating Characteristic')

plt.plot(fpr_cv, tpr_cv, 'b',label='AUC = %0.4f'% roc_auc_cv)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



print('coef_:', lr_model_cv.coef_)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV



param_test0b = {    

    'n_estimators':[80, 100, 120,160,200],

    'max_depth':range(2,16,2),

#     'min_child_weight':range(1,8,2),

#     'gamma':[0,0.1,0.2,0.3,0.4,0.5,0.6]，

#     'subsample':[i/100.0 for i in range(75,90,5)],

#     'colsample_bytree':[i/100.0 for i in range(75,90,5)],

    

#     'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}



xgb_cv0b = GridSearchCV(estimator=xgb.XGBClassifier(learning_rate =0.1,

                                                   n_estimators=100, 

                                                   max_depth=6,

                                                   min_child_weight=1,  

                                                   gamma=0, 

                                                   subsample=0.8,

                                                   colsample_bytree=0.8,

                                                   objective= 'binary:logistic', 

                                                   nthread=4,

                                                   scale_pos_weight=1, 

                                                   seed=27),

                       param_grid = param_test0b,

                       scoring='roc_auc',

                       n_jobs=4,

                       iid=False, 

                       cv=5

                       )



xgb_cv0b.fit(X_train, y_train)

test_est_xgb_cv = xgb_cv0b.predict(X_test)

test_est_p_xgb_cv = xgb_cv0b.predict_proba(X_test)[:,1]



train_est_xgb_cv = xgb_cv0b.predict(X_train)

train_est_p_xgb_cv = xgb_cv0b.predict_proba(X_train)[:,1]



test_est_xgb_cv = xgb_cv0b.predict(X_test)

test_est_p_xgb_cv = xgb_cv0b.predict_proba(X_test)[:,1]



train_est_xgb_cv = xgb_cv0b.predict(X_train)

train_est_p_xgb_cv = xgb_cv0b.predict_proba(X_train)[:,1]



fpr_test_xgb_cv, tpr_test_xgb_cv, th_test_xgb_cv = metrics.roc_curve(y_test, test_est_p_xgb_cv)

test_roc_auc_xgb_cv = auc(fpr_test_xgb_cv,tpr_test_xgb_cv)



fpr_train_xgb_cv, tpr_train_xgb_cv, th_train_xgb_cv = metrics.roc_curve(y_train, train_est_p_xgb_cv)

train_roc_auc_xgb_cv = auc(fpr_train_xgb_cv,tpr_train_xgb_cv)

print('auc_test:', test_roc_auc_xgb_cv)

print('auc_train:', train_roc_auc_xgb_cv)

print('best_params',xgb_cv0b.best_params_)

print('\n')

print('test classification report: \n',metrics.classification_report(y_test, test_est_xgb_cv))

print('train classification report: \n',metrics.classification_report(y_train, train_est_xgb_cv))



# 绘制ROC曲线

plt.title('Receiver Operating Characteristic')

plt.plot(fpr_test_xgb_cv, tpr_test_xgb_cv, color="k",label='AUC_test = %0.5f'% test_roc_auc_xgb_cv)

plt.plot(fpr_train_xgb_cv, tpr_train_xgb_cv, color="y",label='AUC_test = %0.5f'% train_roc_auc_xgb_cv)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

print(xgb_cv0b)