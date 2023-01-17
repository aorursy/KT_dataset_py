import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from datetime import date,datetime
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

#data prep
train = pd.read_csv('../input/y_train.csv')
demographic = pd.read_csv('../input/demo.csv')
saving = pd.read_csv('../input/sa.csv')
credit = pd.read_csv('../input/cc_txn.csv')
# join demographic + address + train\

#onehot gnd_cd mar_st_cd ctf_tp_cd ocp_cd from demo and prov from address
groomed = pd.get_dummies(demographic,dummy_na = True,columns = ['gnd_cd','mar_st_cd','ctf_tp_cd','ocp_cd'])

#drop account start date and calculate the age of each account instead
groomed['act_age_days'] = (pd.Timestamp.today() - pd.to_datetime(groomed['act_strt_dt'])).dt.days
groomed = groomed.drop(['act_strt_dt'],axis=1)

#analyze saving & credit
saving['dt'] = pd.to_datetime(saving['dt'])
txn_sv_grp = saving.groupby(['ip_id','tp',saving['dt'].dt.strftime('%Y')])['amt'].sum()
#extract savings to features
# CR + 2017 / DR + 2017 / CR + 2018 / DR + 2018 = 4 features
tx_sv = txn_sv_grp.reset_index()
tx_sv['new'] = tx_sv['tp'] + tx_sv['dt']
tx_sv = tx_sv.reset_index().set_index(['ip_id', 'new'])['amt'].unstack().rename_axis(None, axis=1).reset_index()
groomed = groomed.merge(tx_sv,left_on='ip_id', right_on='ip_id', how='left')


#extract credit to features
#group by category, sum over date, make it a column like saving. Should I count # of unique card hash?
credit['dt'] = pd.to_datetime(credit['dt'])
txn_cc_grp = credit.groupby(['ip_id',credit['dt'].dt.strftime('%Y')])['txn_amt'].sum()

txn_cc = txn_cc_grp.reset_index()

#txn_cc = txn_cc.reset_index().set_index(['ip_id'])['txn_amt'].unstack().rename_axis(None, axis=1).reset_index()

txn_cc = txn_cc.reset_index().set_index(['ip_id', 'dt'])['txn_amt'].unstack().rename_axis(None, axis=1).reset_index()

groomed = groomed.merge(txn_cc, left_on='ip_id', right_on='ip_id', how='left')

groomed["brth_yr"] = 2018-groomed["brth_yr"];

#imputed missing value
g_cloned = groomed.copy()
#find missing
col_with_missings = (col for col in g_cloned.columns if g_cloned[col].isnull().any())

for col in col_with_missings:
     g_cloned[col + '_was_missing'] = g_cloned[col].isnull()

#no more object, can now impute?
my_imputer = SimpleImputer()
imputed_groomed = my_imputer.fit_transform(g_cloned)
complete_df = pd.DataFrame(imputed_groomed)
complete_df.columns = g_cloned.columns



complete_df = complete_df.merge(train, left_on='ip_id', right_on='ip_id', how='left')

complete_df = complete_df.loc[:, ~complete_df.columns.str.endswith('_was_missing')]

from sklearn import preprocessing

x = complete_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=complete_df.columns)

df['ip_id'] = complete_df['ip_id']
df['label'] = complete_df['label']
complete_df = df

complete_df.to_csv('complete_dataframe.csv',index=False)

test = pd.read_csv('../input/y_test_index.csv')
actual_test = complete_df[complete_df.ip_id.isin(test.ip_id)]

test_frame = complete_df[complete_df['label'].isnull()]
train_frame = complete_df[complete_df['label'].notnull()]

default = train_frame.loc[train_frame['label'] == 1]
not_default = train_frame.loc[train_frame['label'] == 0]

default = train_frame.loc[train_frame['label'] == 1]
not_default = train_frame.loc[train_frame['label'] == 0]


#under sampling
count_not_default, count_default = train_frame.label.value_counts()

df_not_default_under = not_default.sample(count_default)
df_test_under = pd.concat([df_not_default_under, default], axis=0)

print('Random under-sampling:')
print(df_test_under.label.value_counts())

df_test_under.label.value_counts().plot(kind='bar', title='Count (label)');



from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(df_test_under.drop(columns=['label'],axis=1), df_test_under['label'], verbose=False)


actual_test_features = actual_test.loc[:, actual_test.columns != 'label']
prd_prob = grid.predict_proba(actual_test_features)
print(prd_prob[:,1])
output_df = pd.DataFrame(prd_prob[:,1])
#output_df.to_csv('TJ2018-AUDITION-10423.csv',index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import calibration_curve

cols = [c for c in complete_df.columns if c.lower()[:5] != 'ip_id']

complete_df=complete_df[cols]

labels = complete_df['label']
features = complete_df

def classification_setup(df_data):
    '''Returns X_data, y_data, ls_features'''
    X_data, y_data = df_data, df_data['label']
    ls_features = list(X_data.keys())
    class_index = ls_features.index('label')
    ls_features.pop(class_index)
    return X_data, y_data, ls_features


ls_features = list(features.keys())
class_index = ls_features.index('label') 
ls_features.pop(class_index)

X_data, y_data, xss = classification_setup(complete_df)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def under_sample_kfold(X_data, y_data, n_folds = 10):
    '''Returns list of 10 (default) folds of
    X_train, X_test, y_train, y_test data'''
    
    pos_events = X_data[X_data['label'] == 1]
    neg_events = X_data[X_data['label'] == 0]
    
    #Randomize and pick same n number of events
    number_pos_events = len(pos_events)  
    undersampled_folds = []

    for fold in range(0, n_folds):
        pos_events = pos_events.reindex(np.random.permutation(pos_events.index))
        neg_events = neg_events.reindex(np.random.permutation(neg_events.index))
        undersampled_events = pd.concat([neg_events.head(number_pos_events), pos_events])
        X_data_u, y_data_u = undersampled_events, undersampled_events['label']
        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_data_u, y_data_u, test_size=0.3)
        undersampled_folds.append([X_train_u, X_test_u, y_train_u, y_test_u])
    return undersampled_folds



def adaboost_undersample(folds, full_test_set = False,test_set=None):
    
    for fold in folds:
        X_train, X_test, y_train, y_test = fold[0], fold[1], fold[2], fold[3]
    
        dt_clf = DecisionTreeClassifier(max_depth = 1)
        ada_real = AdaBoostClassifier(base_estimator = dt_clf, 
                                      learning_rate = 0.1, 
                                      n_estimators = 100)
        ada_real.fit(X_train[ls_features], y_train)

        if full_test_set == False:
            y_pred = ada_real.predict(X_test[ls_features])
            test_conf = confusion_matrix(y_test, y_pred)
            print (test_conf)
    
        else:
            X_train_and_test = pd.concat([X_train, X_data])
            X_test_full = (X_train_and_test.reset_index()
                                           .drop_duplicates(subset= 'index', keep= False)
                                           .set_index('index'))
            y_test_full = X_test_full['label']
        
            #Eval
            y_pred = ada_real.predict(X_test_full[ls_features])
            test_conf = confusion_matrix(y_test_full, y_pred)
            print (test_conf)
    #ada_real.predict_proba()





folds = under_sample_kfold(features, labels)
adaboost_undersample(folds,full_test_set = True)
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

labels = train_frame['label']
features = train_frame.loc[:, train_frame.columns != 'label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=42)


cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)
rus = make_pipeline(RandomUnderSampler(),tree.DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5))
forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)

cart.fit(X_train, y_train)
rus.fit(X_train, y_train)
forest.fit(X_train, y_train)
n_splits = 10

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt

ub = BaggingClassifier(warm_start=True, n_estimators=0)

for split in range(n_splits):
    X_res, y_res = RandomUnderSampler(random_state=split).fit_sample(X_train,y_train) 
    ub.n_estimators += 1
    ub.fit(X_res, y_res)
    
def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

f, ax = plt.subplots(figsize=(6,6))

roc_auc_plot(y_test,ub.predict_proba(X_test),label='UB ',l='-')
roc_auc_plot(y_test,forest.predict_proba(X_test),label='FOREST ',l='--')
roc_auc_plot(y_test,cart.predict_proba(X_test),label='CART', l='-.')
roc_auc_plot(y_test,rus.predict_proba(X_test),label='RUS',l=':')

ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        label='Random Classifier')    
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic curves')
sns.despine()

from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

x_train, x_val, y_train, y_val = train_test_split(features, labels,
                                                  test_size = .1,
                                                  random_state=12)
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)

print('Validation Results')
print(clf_rf.score(x_val, y_val))
print(recall_score(y_val, clf_rf.predict(x_val)))
#visualize imbalance
target_count = complete_df.label.value_counts()
print('label 0:', target_count[0])
print('label 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

labels = train_frame['label']
features = train_frame.loc[:, train_frame.columns != 'label']


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



#one feature training, reducing accuracy
feature2 = train_frame[['cis_income']]

X_train, X_test, y_train, y_test = train_test_split(feature2, labels, test_size=0.2, random_state=1)
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



default = train_frame.loc[train_frame['label'] == 1]
not_default = train_frame.loc[train_frame['label'] == 0]


#under sampling
count_not_default, count_default = train_frame.label.value_counts()

df_not_default_under = not_default.sample(count_default)
df_test_under = pd.concat([df_not_default_under, default], axis=0)

print('Random under-sampling:')
print(df_test_under.label.value_counts())

df_test_under.label.value_counts().plot(kind='bar', title='Count (label)');


#use imblearn for random resampling
import imblearn
from sklearn.decomposition import PCA

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    

labels = train_frame['label']
features = train_frame.loc[:, train_frame.columns != 'label']

pca = PCA(n_components=2)
pca_features = pca.fit_transform(features)

plot_2d_space(pca_features, labels, 'Imbalanced dataset (2 PCA components)')

#over sampling
df_default_over = default.sample(count_not_default, replace=True)
df_test_over = pd.concat([not_default, df_default_over], axis=0)

print('Random over-sampling:')
print(df_test_over.label.value_counts())

df_test_over.label.value_counts().plot(kind='bar', title='Count (target)');


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(features, labels)

#print('Removed indexes:', id_rus)
plot_2d_space(X_rus, y_rus, 'Random under-sampling')


ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(features, labels)
print(X_ros.shape[0] - features.shape[0], 'new random picked points')
plot_2d_space(X_ros, y_ros, 'Random over-sampling')
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(features, labels)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
from imblearn.over_sampling import SMOTE  # or: import RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     StratifiedKFold)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.33, random_state=444, stratify=labels)

# This doesn't work with sklearn.pipeline.Pipeline because
# RandomOverSampler doesn't have a .tranform() method.
# (It has .fit_sample() or .sample().)
pipe = imbPipeline([
    ('oversample', SMOTE(random_state=444)),
    ('clf', RandomForestClassifier(random_state=444, n_jobs=-1))
    ])

skf = StratifiedKFold()
param_grid = {'clf__max_depth': [25, 40],
              'clf__max_features': ['sqrt', 'log2']}
grid = GridSearchCV(pipe, param_grid, return_train_score=False,
                    n_jobs=-1, scoring='roc_auc', cv=skf)
grid.fit(X_train, y_train)

test_0, test_1 = y_test.value_counts()
print(test_0)
print(test_1)


train_0, train_1 = y_train.value_counts()
print(train_0)
print(train_1)

grid.score(X_test, y_test)


# 1.0 ?????


np.set_printoptions(precision=3)
test = pd.read_csv('../input/y_test_index.csv')
actual_test = complete_df[complete_df.ip_id.isin(test.ip_id)]
actual_test_features = actual_test.loc[:, actual_test.columns != 'label']
prd_prob = grid.predict_proba(actual_test_features)
#print(prd_prob[:,1])
output_df = pd.DataFrame(prd_prob[:,1])
#output_df.to_csv('TJ2018-AUDITION-10423.csv',index=False)
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report


X_train, X_val, y_train, y_val = train_test_split(train_frame.loc[:,train_frame.columns != 'label'],train_frame['label'],
                                                  test_size = .1,
                                                  random_state=12)


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


#parameters = {'C': np.linspace(1, 50, 50)}
#lr = LogisticRegression()
#clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
#clf.fit(X_train_res, y_train_res.ravel())
##best = 1
#clf.best_params_

lr1 = LogisticRegression(C=1,penalty='l1', verbose=5)
lr1.fit(X_train_res, y_train_res.ravel())


y_train_pre = lr1.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()
from sklearn.utils import resample


# Separate majority and minority classes
df_majority = train_frame[train_frame.label==0]
df_minority = train_frame[train_frame.label==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=6978,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.label.value_counts()



X_train, X_val, y_train, y_val = train_test_split(df_upsampled.loc[:,df_upsampled.columns != 'label'],df_upsampled.loc[:,df_upsampled.columns == 'label'],
                                                  test_size = .1,
                                                  random_state=12)

#parameters = {'C': np.linspace(1, 50, 50)}
#lr = LogisticRegression()
#clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
#clf.fit(X_train_res, y_train_res.ravel())
##best = 1
#clf.best_params_

lr1 = LogisticRegression(C=1,penalty='l1', verbose=5)
lr1.fit(X_train_res, y_train_res.ravel())


y_train_pre = lr1.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()

print(complete_df.columns)
complete_df.plot.scatter(x='cis_income',y='act_age_days',c='label',colormap='viridis')
complete_df.plot.scatter(x='cis_income',y='crn_bal',c='label',colormap='viridis')
complete_df.plot.scatter(x='act_age_days',y='crn_bal',c='label',colormap='viridis')
complete_df.plot.scatter(x='brth_yr',y='no_of_dpnd_chl',c='label',colormap='viridis')