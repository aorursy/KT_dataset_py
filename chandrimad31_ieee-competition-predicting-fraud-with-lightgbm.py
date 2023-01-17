import pandas as pd

train_transact = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_transact = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')
train = train_transact.merge(train_id,how='left',left_index=True,right_index=True)

test = test_transact.merge(test_id,how='left',left_index=True,right_index=True)
train.info()

test.info()
# Freeing up some space

del train_transact, train_id, test_transact, test_id
# Taken from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

import numpy as np

def reduce_mem_usage(df):

   

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
reduce_mem_usage(train)
reduce_mem_usage(test)
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,5))

train.isFraud.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','red'], alpha = 0.8)

plt.title('Fraud and Non-Fraud (Imbalanced Dataset)')

plt.show()
# Undersampling majority to remove Class Imbalance

import pandas as pds

from sklearn.utils import resample



fraud = train[train.isFraud==1]

not_fraud = train[train.isFraud==0]



not_fraud_undsamp = resample(not_fraud,replace = False,n_samples = len(fraud),random_state = 42)

undsamp_train = pds.concat([not_fraud_undsamp, fraud])



fig = plt.figure(figsize = (8,5))

undsamp_train.isFraud.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','red'], alpha = 0.8)

plt.title('Fraud and Non-Fraud after Undersampling (Balanced Dataset)')

plt.show()
# Drop features having >40% missing values in Train Dataset



pct_null = undsamp_train.isnull().sum() / len(undsamp_train)

missing_features = pct_null[pct_null > 0.40].index

undsamp_train.drop(missing_features, axis=1, inplace=True)

undsamp_train.info()
del train, fraud, not_fraud #free space
undsamp_train.select_dtypes(include=['category']).columns
# Impute categorical var with Mode

undsamp_train['ProductCD'] = undsamp_train['ProductCD'].fillna(undsamp_train['ProductCD'].mode()[0])

undsamp_train['card4'] = undsamp_train['card4'].fillna(undsamp_train['card4'].mode()[0])

undsamp_train['card6'] = undsamp_train['card6'].fillna(undsamp_train['card6'].mode()[0])

undsamp_train['P_emaildomain'] = undsamp_train['P_emaildomain'].fillna(undsamp_train['P_emaildomain'].mode()[0])

undsamp_train['M4'] = undsamp_train['M4'].fillna(undsamp_train['M4'].mode()[0])
# Convert categorical features to continuous features with Label Encoding

from sklearn.preprocessing import LabelEncoder

lencoders = {}

for col in undsamp_train.select_dtypes(include=['category']).columns:

    lencoders[col] = LabelEncoder()

    undsamp_train[col] = lencoders[col].fit_transform(undsamp_train[col])
# Multiple Imputation by Chained Equations

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

train_MiceImputed = undsamp_train.copy(deep=True) 

mice_imputer = IterativeImputer()

train_MiceImputed.iloc[:, :] = mice_imputer.fit_transform(undsamp_train)
train_MiceImputed.info()
# Feature Selection using Extra Trees Classifier (max_features='log2')



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



x_t = train_MiceImputed.drop('isFraud', axis=1)

y_t = train_MiceImputed['isFraud']

clf_1 = SelectFromModel(ExtraTreesClassifier(max_features='log2'))

clf_1.fit(x_t, y_t)

select_feats_1 = x_t.columns[(clf_1.get_support())]

print(select_feats_1)



# Feature Selection using Extra Trees Classifier (max_features = 'auto')



clf_2 = SelectFromModel(ExtraTreesClassifier(max_features='auto'))

clf_2.fit(x_t, y_t)

select_feats_2 = x_t.columns[(clf_2.get_support())]

print(select_feats_2)
# Feature Selection using Random Forest (max_features='log2')



from sklearn.ensemble import RandomForestClassifier

clf_3 = SelectFromModel(RandomForestClassifier(max_features='log2'))

clf_3.fit(x_t, y_t)

select_feats_3 = x_t.columns[(clf_3.get_support())]

print(select_feats_3)



# Feature Selection using Random Forest (max_features 'auto') 



clf_4 = SelectFromModel(RandomForestClassifier(max_features='auto'))

clf_4.fit(x_t, y_t)

select_feats_4 = x_t.columns[(clf_4.get_support())]

print(select_feats_4)
# Combine all selected features

x_train = train_MiceImputed[['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2','card3', 'card4', 'card5', 

                             'card6','addr1', 'addr2', 'P_emaildomain', 'C1', 'C2', 'C6', 'C7','C8', 'C9','C10','C11',

                             'C12', 'C13', 'C14','D1', 'D4', 'D10', 'D15','M4','V12', 'V13', 'V15', 'V22','V29', 'V30',

                             'V34', 'V35','V36', 'V37','V38','V42', 'V43', 'V44','V45', 'V47', 'V48','V49','V50', 'V51', 

                             'V52', 'V53', 'V54','V57','V69', 'V70','V71', 'V74','V75', 'V76', 'V78', 'V81','V82','V83', 

                             'V84', 'V85', 'V86','V87', 'V90', 'V91', 'V92','V93', 'V94', 'V102', 'V127', 'V128', 'V133', 

                             'V280','V282','V283', 'V285', 'V294','V295', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313',

                             'V314','V315', 'V317', 'V318']]
y_train = train_MiceImputed['isFraud']
skewness_of_features=[]

for col in x_train:

        skewness_of_features.append(x_train[col].skew())

print(skewness_of_features)
skewed_vars = x_train[['TransactionAmt','C2','C6','C7','C8','C9','C10','C11','C12','C13','C14','V29','V37','V38','V44',

                       'V45','V47','V52','V53','V69','V75','V78','V81','V82','V86','V87','V91','V102','V127','V128','V133',

                       'V280','V283','V285','V294','V295','V306','V307','V308','V310','V312','V313','V314','V315','V317',

                       'V318']]
import numpy

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from pyearth import Earth



# Fit an Earth model (Multi Variate Adaptive Regression Splines)

# Residual Sum of Squares (RSS), Generalized Cross Validation (GCV), No. of Subsets of MARS Model Terms 

criteria = ('rss', 'gcv', 'nb_subsets')  

model = Earth(max_degree=3,

              max_terms=10,

              minspan_alpha=.5,

              feature_importance_type=criteria,

              verbose=True)

model.fit(skewed_vars, y_train)

rf = RandomForestRegressor()

rf.fit(skewed_vars, y_train)

print(model.trace())

print(model.summary())

print(model.summary_feature_importances(sort_by='gcv'))



# Plot the feature importances

importances = model.feature_importances_

importances['random_forest'] = rf.feature_importances_

criteria = criteria + ('random_forest',)

idx = 1



fig = plt.figure(figsize=(30, 12))

labels = [i for i in range(len(skewed_vars.columns))]

for crit in criteria:

    plt.subplot(2, 2, idx)

    plt.bar(numpy.arange(len(labels)),

            importances[crit],

            align='center',

            color='blue')

    plt.xticks(numpy.arange(len(labels)), labels)

    plt.title(crit)

    plt.ylabel('importances')

    idx += 1

title = 'Feature Importance Plots using Earth Package'

fig.suptitle(title, fontsize="18")

plt.show()
x_train['TransactionAmt']=np.log(x_train['TransactionAmt'])
# Check skewness again after log-transormation

print(x_train['TransactionAmt'].skew())
# Keeping only the important features

x_train_work = x_train[['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2','card3', 'card4', 'card5', 

                        'card6','addr1', 'addr2', 'P_emaildomain', 'D1','D4', 'D10', 'D15','M4','C2','C6','C7','C8',

                        'C9','C10','C11','C12','C13','C14','V12', 'V13', 'V15', 'V22','V29','V30','V34', 'V35','V36',

                        'V37','V38','V42', 'V43','V44','V45','V47','V48','V49', 'V50', 'V51', 'V52','V53','V54','V57',

                        'V69','V70','V71', 'V74','V75','V76','V78','V83', 'V84', 'V85', 'V87','V90','V91','V92','V93',

                        'V94', 'V282','V294','V308','V315','V318']]
# Check correlation matrix for the selected important features

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

mask = np.zeros_like(x_train_work.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(20,20))

sns.heatmap(x_train_work.corr(), mask = mask, vmin = -1, annot = False, cmap = 'RdYlGn')
import lightgbm as lgb

from hyperopt import hp, tpe, fmin

from sklearn.model_selection import cross_val_score



valgrid = {'n_estimators':hp.quniform('n_estimators', 1000, 5000, 50),

         'subsample_for_bin':hp.uniform('subsample_for_bin',10,300000),

         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),

         'max_depth':hp.quniform('max_depth', 3,8,1),

         'num_leaves':hp.quniform('num_leaves', 7,256,1),

         'subsample':hp.uniform('subsample', 0.60, 0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.95),

         'min_child_samples':hp.quniform('min_child_samples', 1, 500,1),

         'min_child_weight':hp.uniform('min_child_weight', 0.60, 0.95),

         'min_split_gain':hp.uniform('min_split_gain', 0.60, 0.95),  

         'reg_lambda':hp.uniform('reg_lambda', 1, 25)

         #'reg_alpha':hp.uniform('reg_alpha', 1, 25)  

        }



def objective(params):

    params = {'n_estimators': int(params['n_estimators']),

              'subsample_for_bin': int(params['subsample_for_bin']),

              'learning_rate': params['learning_rate'],

              'max_depth': int(params['max_depth']),

              'num_leaves': int(params['num_leaves']),

              'subsample': params['subsample'],

              'colsample_bytree': params['colsample_bytree'],

              'min_child_samples': int(params['min_child_samples']),

              'min_child_weight': params['min_child_weight'],

              'min_split_gain': params['min_split_gain'],

              'reg_lambda': params['reg_lambda']}

              #'reg_alpha': params['reg_alpha']}

    

    lgb_a = lgb.LGBMClassifier(**params)

    score = cross_val_score(lgb_a, x_train_work, y_train, cv=5, n_jobs=-1).mean()

    return score



bestP = fmin(fn= objective, space= valgrid, max_evals=20, rstate=np.random.RandomState(123), algo=tpe.suggest)
print(bestP)
# Missing value analysis for the selected feature set in Test Data

total = test[['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2','card3', 'card4', 'card5', 

                        'card6','addr1', 'addr2', 'P_emaildomain', 'D1','D4', 'D10', 'D15','M4','C2','C6','C7','C8',

                        'C9','C10','C11','C12','C13','C14','V12', 'V13', 'V15', 'V22','V29','V30','V34', 'V35','V36',

                        'V37','V38','V42', 'V43','V44','V45','V47','V48','V49', 'V50', 'V51', 'V52','V53','V54','V57',

                        'V69','V70','V71', 'V74','V75','V76','V78','V83', 'V84', 'V85', 'V87','V90','V91','V92','V93',

                        'V94', 'V282','V294','V308','V315','V318']].isnull().sum().sort_values(ascending=False)



percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_test.head(25)
# Impute categorical vars with Mode

test['ProductCD']=test['ProductCD'].fillna(test['ProductCD'].mode()[0])

test['card4']=test['card4'].fillna(test['card4'].mode()[0])

test['card6']=test['card6'].fillna(test['card6'].mode()[0])

test['P_emaildomain']=test['P_emaildomain'].fillna(test['P_emaildomain'].mode()[0])

test['M4']=test['M4'].fillna(test['M4'].mode()[0])
# Convert categorical features to continuous features with Label Encoding

from sklearn.preprocessing import LabelEncoder

lencoders_te = {}

for col in test[['ProductCD','card4','card6','P_emaildomain','M4']]:

    lencoders_te[col] = LabelEncoder()

    test[col] = lencoders_te[col].fit_transform(test[col])  
test_working = test[['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2','card3', 'card4', 'card5', 

                        'card6','addr1', 'addr2', 'P_emaildomain', 'D1','D4', 'D10', 'D15','M4','C2','C6','C7','C8',

                        'C9','C10','C11','C12','C13','C14','V12', 'V13', 'V15', 'V22','V29','V30','V34', 'V35','V36',

                        'V37','V38','V42', 'V43','V44','V45','V47','V48','V49', 'V50', 'V51', 'V52','V53','V54','V57',

                        'V69','V70','V71', 'V74','V75','V76','V78','V83', 'V84', 'V85', 'V87','V90','V91','V92','V93',

                        'V94', 'V282','V294','V308','V315','V318']]

del test #free space
# MICE Imputation for test data set

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



test_MiceImputed = test_working.copy(deep=True) 

mice_imputer_te = IterativeImputer()

test_MiceImputed.iloc[:, :] = mice_imputer_te.fit_transform(test_working)
test_MiceImputed.head()
test_MiceImputed['TransactionAmt']=np.log(test_MiceImputed['TransactionAmt'])
test_working_final = test_MiceImputed[['TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2','card3', 'card4', 'card5', 

                        'card6','addr1', 'addr2', 'P_emaildomain', 'D1','D4', 'D10', 'D15','M4','C2','C6','C7','C8',

                        'C9','C10','C11','C12','C13','C14','V12', 'V13', 'V15', 'V22','V29','V30','V34', 'V35','V36',

                        'V37','V38','V42', 'V43','V44','V45','V47','V48','V49', 'V50', 'V51', 'V52','V53','V54','V57',

                        'V69','V70','V71', 'V74','V75','V76','V78','V83', 'V84', 'V85', 'V87','V90','V91','V92','V93',

                        'V94', 'V282','V294','V308','V315','V318']]

del test_MiceImputed #free space
# Prediction using Light GBM with Best Hyperparameters 

import pandas as pd_out

import lightgbm 

from sklearn.model_selection import train_test_split

import sklearn.metrics 

import numpy as np



clf_final = lightgbm.LGBMClassifier(n_estimators = int(bestP['n_estimators']),

                                    subsample_for_bin = int(bestP['subsample_for_bin']),

                                    learning_rate = bestP['learning_rate'],

                                    max_depth = int(bestP['max_depth']),

                                    num_leaves = int(bestP['num_leaves']),

                                    subsample = bestP['subsample'],

                                    colsample_bytree = bestP['colsample_bytree'],

                                    min_child_samples = int(bestP['min_child_samples']),

                                    min_child_weight = bestP['min_child_weight'],

                                    min_split_gain = bestP['min_split_gain'],

                                    reg_lambda = bestP['reg_lambda'],

                                    #reg_alpha = bestP['reg_alpha'], 

                                    random_state = 123)



X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=123, stratify=y_train)

clf_final.fit(x_train_work,y_train)

y_pred = clf_final.predict(test_working_final)

#print(accuracy_score(Y_test,y_pred))
y_prob = clf_final.predict_proba(x_train_work)[:,-1] # Positive class prediction probabilities  

y_thresh = np.where(y_prob > 0.5, 1, 0) # Threshold the probabilities to give class predictions

clf_final.score(x_train_work, y_thresh)
predicted_proba = clf_final.predict_proba(x_train_work)
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, predicted_proba[:,-1]) 

roc_auc = auc(false_positive_rate, true_positive_rate)

print(roc_auc) # Checking area under ROC Curve
# Plotting ROC Curve

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
# Detailed Classification Report

from sklearn.metrics import classification_report

print(classification_report(clf_final.predict(x_train_work),y_train,digits=4))
from sklearn import metrics

confusion_matrix=metrics.confusion_matrix(clf_final.predict(x_train_work),y_train)

print(confusion_matrix)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf_final, x_train_work, y_train,cmap=plt.cm.Blues, normalize = 'all')
submission = pd_out.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

submission['isFraud'] = y_pred

submission.head()

submission.to_csv('IEEEfraud_submission.csv', index=False)

print("Submission successful")