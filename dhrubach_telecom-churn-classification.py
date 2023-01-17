# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import math

import itertools



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from sklearn.metrics import classification_report, make_scorer



from imblearn.over_sampling import SMOTE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
# load telecom churn dataset into a dataframe

df_telecom = pd.read_csv('../input/telecom_churn_data.csv')
df_telecom.shape
df_telecom.head()
# create a new dataframe with only mobile number and recharge amount for the first 2 months

df_cust_rech_6_7 = df_telecom[['mobile_number', 'total_rech_amt_6', 'total_rech_amt_7']]



# create a new column to capture average recharge per customer for the first 2 months

df_cust_rech_6_7['avg_rech_6_7'] = np.mean(

    [df_cust_rech_6_7['total_rech_amt_6'], df_cust_rech_6_7['total_rech_amt_7']],

    axis=0

)



# calculate the 70th percentile of calculated average amount

df_cust_rech_6_7['avg_rech_6_7'].describe(percentiles=[0.7])
# mobile numbers of high value customers

lst_high_value_cust = list(df_cust_rech_6_7[df_cust_rech_6_7['avg_rech_6_7'] >= 368.50]['mobile_number'])



# filter original dataframe for only high value customers

df_high_value_cust = df_telecom[df_telecom['mobile_number'].isin(lst_high_value_cust)]



# of the total 99,999 customers, 30,011 are high-value customers

# i.e. 30% of the total population are high-value customers

print(df_high_value_cust.shape)

round((len(df_high_value_cust.index) / len(df_telecom.index)) * 100, 2)
df_high_value_cust['churn'] = (

    (df_high_value_cust['total_ic_mou_9'] == 0) &

    (df_high_value_cust['total_og_mou_9'] == 0) &

    (df_high_value_cust['vol_2g_mb_9'] == 0) &

    (df_high_value_cust['vol_3g_mb_9'] == 0)).astype(int)



# total number of churned customers

df_high_value_cust['churn'].sum()
round((df_high_value_cust['churn'].sum() / len(df_high_value_cust.index)) * 100, 2)
arr_all_columns = list(df_high_value_cust.columns)



arr_churned_month_cols = filter(lambda item: '_9' in item, arr_all_columns)



df_high_value_cust.drop(list(arr_churned_month_cols), axis=1, inplace=True)

df_high_value_cust.drop('sep_vbc_3g', axis=1, inplace=True)



df_high_value_cust.shape
# drop the following columns as these show no variance

# mobile_number --- can't be used for either prediction or interpretation

# circle_id --- single value

# 'date' columns --- single value

# remaining columns --- single value '0'

arr_cols_to_drop = [

    'mobile_number', 'circle_id', 'loc_og_t2o_mou', 'std_og_t2o_mou',

    'loc_ic_t2o_mou', 'last_date_of_month_6', 'last_date_of_month_7',

    'last_date_of_month_8', 'std_og_t2c_mou_6', 'std_og_t2c_mou_7',

    'std_og_t2c_mou_8', 'std_ic_t2o_mou_6', 'std_ic_t2o_mou_7',

    'std_ic_t2o_mou_8'

]



df_high_value_cust.drop(columns=arr_cols_to_drop, inplace=True)



df_high_value_cust.shape
# calculate the percentage of NaN values present in each column

df_high_value_cust_col_NaN = pd.DataFrame({

    'NaN%': round((df_high_value_cust.isnull().sum() / len(df_high_value_cust.index)) *100, 2)

})



df_high_value_cust_col_NaN['NaN%'].describe(percentiles=[0, .25, .50, .75, .80, .90, .95])
# check those columns which has 61% or more of NaN values present

arr_cols_NaN = df_high_value_cust_col_NaN[

    df_high_value_cust_col_NaN['NaN%'] > 61].index.values



list(arr_cols_NaN)
# take the numerical columns and impute missing column values with '0'

arr_cols_NaN_filterObj = filter(lambda col: 'date' not in col,

                                df_high_value_cust.columns)

arr_cols_NaN_continuous = list(arr_cols_NaN_filterObj)



# total number of non - date columns

# includes columns with both categorical and continuous data elements

len(arr_cols_NaN_continuous)
# remove 'churn' column from the list as we don't want to impute that column

arr_cols_NaN_continuous = arr_cols_NaN_continuous[:-1]
list(

    map(lambda col: df_high_value_cust[col].fillna(0, inplace=True),

        arr_cols_NaN_continuous))



# re-calculate the percentage of NaN values present in each column

df_high_value_cust_col_NaN = pd.DataFrame({

    'NaN%':

    round((df_high_value_cust.isnull().sum() / len(df_high_value_cust.index)) *

          100, 2)

})

df_high_value_cust_col_NaN.sort_values('NaN%', ascending=False).head(n=10)
arr_date_cols = [

    'date_of_last_rech_data_6', 'date_of_last_rech_data_7',

    'date_of_last_rech_data_8', 'date_of_last_rech_6', 'date_of_last_rech_7',

    'date_of_last_rech_8'

]



list(

    map(lambda col: df_high_value_cust[col].fillna('01/01/1900', inplace=True),

        arr_date_cols))



# re-calculate the percentage of NaN values present in each column

df_high_value_cust_col_NaN = pd.DataFrame({

    'NaN%':

    round((df_high_value_cust.isnull().sum() / len(df_high_value_cust.index)) * 100, 2)

})



df_high_value_cust_col_NaN.describe()
def create_recharge_col(suffix):

    columnName = 'recharge_done_' + suffix

    column1 = 'date_of_last_rech_' + suffix

    column2 = 'date_of_last_rech_data_' + suffix



    df_high_value_cust[columnName] = df_high_value_cust.apply(

        lambda x: 0 if (x[column1] == '01/01/1900' and x[column2] == '01/01/1900') else 1,axis=1)
for suffix in ['6', '7', '8']:

    create_recharge_col(suffix)

    

df_high_value_cust[['recharge_done_6', 'recharge_done_7', 'recharge_done_8']].sum()
# format all date values to a uniform format : MM/DD/YYYY

for col in arr_date_cols:

    df_high_value_cust[col] = pd.to_datetime(df_high_value_cust[col])

    df_high_value_cust[col] = df_high_value_cust[col].dt.day
# check the distribution of days across date columns

df_high_value_cust[arr_date_cols].describe()
# plot scatter plots of 'date' columns with customer churn



df_churn_cust = df_high_value_cust[df_high_value_cust['churn'] == 1] 



plt.figure(figsize=(10, 10))



plt.subplot(2,2,1)

sns.scatterplot(x = df_churn_cust['date_of_last_rech_data_6'], y = df_churn_cust['date_of_last_rech_data_7'])

plt.subplot(2,2,2)

sns.scatterplot(x = df_churn_cust['date_of_last_rech_data_6'], y = df_churn_cust['date_of_last_rech_data_8'])

plt.subplot(2,2,3)

sns.scatterplot(x = df_churn_cust['date_of_last_rech_6'], y = df_churn_cust['date_of_last_rech_7'])

plt.subplot(2,2,4)

sns.scatterplot(x = df_churn_cust['date_of_last_rech_6'], y = df_churn_cust['date_of_last_rech_8'])



plt.show()
bins = [0, 10, 20, 31]

labels = [1, 2, 3]



df_high_value_cust['bin_date_of_last_rech_6'] = pd.cut(

    df_high_value_cust['date_of_last_rech_6'], bins=bins, labels=labels)



df_high_value_cust['bin_date_of_last_rech_7'] = pd.cut(

    df_high_value_cust['date_of_last_rech_7'], bins=bins, labels=labels)



df_high_value_cust['bin_date_of_last_rech_8'] = pd.cut(

    df_high_value_cust['date_of_last_rech_8'], bins=bins, labels=labels)
df_high_value_cust.drop(arr_date_cols, inplace = True, axis = 1)
# calculating sum of monthly onnet_mou and offnet_mou



df_high_value_cust['total_mou'] = df_high_value_cust.apply(

    lambda x: x['onnet_mou_6'] + x['onnet_mou_7'] + x['onnet_mou_8'] + 

    x['offnet_mou_6'] + x['offnet_mou_7'] + x['offnet_mou_8'], axis=1)
len(df_high_value_cust[df_high_value_cust['total_mou'] == 0].index)
df_high_value_cust['ratio_mou_6'] = df_high_value_cust.apply(

    lambda x: 0 if x['total_mou'] == 0 else (round((x['offnet_mou_6'] + x['onnet_mou_6']) / x['total_mou'] * 100, 2)), 

    axis=1)



df_high_value_cust['ratio_mou_7'] = df_high_value_cust.apply(

    lambda x: 0 if x['total_mou'] == 0 else (round((x['offnet_mou_7'] + x['onnet_mou_7']) / x['total_mou'] * 100, 2)), 

    axis=1)



df_high_value_cust['ratio_mou_8'] = df_high_value_cust.apply(

    lambda x: 0 if x['total_mou'] == 0 else (round((x['offnet_mou_8'] + x['onnet_mou_8']) / x['total_mou'] * 100, 2)), 

    axis=1)
df_high_value_cust[['ratio_mou_6', 'ratio_mou_7', 'ratio_mou_8', 'total_mou']].head(10)
df_high_value_cust[['ratio_mou_6', 'ratio_mou_7', 'ratio_mou_8', 'total_mou']].isnull().sum()
# drop 'offnet' and 'onnet' variables

columns_to_deleted = ['onnet_mou_6','onnet_mou_7','onnet_mou_8','offnet_mou_6','offnet_mou_7','offnet_mou_8']



df_high_value_cust.drop(columns_to_deleted, axis=1, inplace=True)
df_categorical = df_high_value_cust.select_dtypes(include = ['object'])

df_categorical.columns
X = df_high_value_cust.drop(['churn'], axis = 1)

y = df_high_value_cust['churn']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)



print("Number customers X_train dataset: ", X_train.shape)

print("Number customers y_train dataset: ", y_train.shape)

print("Number customers X_test dataset: ", X_test.shape)

print("Number customers y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))



smt = SMOTE(random_state=2)

X_train_resampled, y_train_resampled = smt.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_resampled.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_resampled.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_resampled == 1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_resampled == 0)))
def create_grid_search_object(param, refit_score = 'precision_score'):

    

    skf = StratifiedKFold(n_splits=5, random_state=100)



    ran_for = RandomForestClassifier(criterion='gini', max_features = 'sqrt', random_state=100)



    scorers = {

        'precision_score': make_scorer(precision_score),

        'recall_score': make_scorer(recall_score),

        'accuracy_score': make_scorer(accuracy_score)

    }

    

    tree = GridSearchCV(ran_for,

                        param,

                        cv=skf,

                        scoring=scorers,

                        return_train_score=True,

                        refit = refit_score,

                        n_jobs=-1)



    return tree
%%time



parameters = {'max_depth': range(1, 40, 5), 'n_estimators': [100, 300]}



tree = create_grid_search_object(parameters, 'precision_score')



tree.fit(X_train, y_train)
scores = tree.cv_results_



df_scores = pd.DataFrame(scores)



df_score_max_depth = df_scores[[

    'mean_test_precision_score', 

    'rank_test_precision_score', 

    'param_max_depth', 

    'params', 

    'mean_test_recall_score', 

    'mean_test_accuracy_score']]



df_score_max_depth.sort_values('mean_test_precision_score', ascending = False).head()
plt.figure(figsize = (10, 10))

plt.plot(scores["param_max_depth"], 

         scores["mean_train_precision_score"], 

         label="training precision")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_precision_score"], 

         label="test precision")

plt.plot(scores["param_max_depth"], 

         scores["mean_train_recall_score"], 

         'g--',

         label="training recall")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_recall_score"], 

         'r--',

         label="test recall")

plt.xlabel("Max Depth")

plt.ylabel("Score")

plt.legend()

plt.show()
%%time



parameters = {'min_samples_leaf': range(10, 2000, 200), 'n_estimators': [100, 300]}



tree = create_grid_search_object(parameters, 'precision_score')



tree.fit(X_train, y_train)
scores = tree.cv_results_



df_score = pd.DataFrame(scores)



df_score_min_samples_leaf = df_score[[

    'rank_test_precision_score', 'mean_test_precision_score',

    'param_min_samples_leaf', 'param_n_estimators',

    'rank_test_recall_score', 'mean_test_recall_score',

    'rank_test_accuracy_score', 'mean_test_accuracy_score',

]]



df_score_min_samples_leaf.sort_values('rank_test_precision_score').head(n=10)
tree.best_params_
tree.best_estimator_
plt.figure(figsize = (10, 10))

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_precision_score"], 

         label="training precision")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_precision_score"], 

         label="test precision")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_recall_score"], 

         'g--',

         label="training recall")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_recall_score"], 

         'r--',

         label="test recall")

plt.xlabel("Min Samples Leaf")

plt.ylabel("Score")

plt.legend()

plt.show()
%%time



parameters = {

    'min_samples_split': range(5, 2000, 200),

    'n_estimators': [100, 300]

}



tree = create_grid_search_object(parameters, 'precision_score')



tree.fit(X_train, y_train)
scores = tree.cv_results_



df_score = pd.DataFrame(scores)



df_score_min_samples_split = df_score[[

    'rank_test_precision_score', 'mean_test_precision_score',

    'param_min_samples_split', 'param_n_estimators',

    'rank_test_recall_score', 'mean_test_recall_score',

    'rank_test_accuracy_score', 'mean_test_accuracy_score',

]]



df_score_min_samples_split.sort_values('rank_test_precision_score').head(n=10)
tree.best_params_
tree.best_estimator_
plt.figure(figsize = (10, 10))

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_precision_score"], 

         label="training precision")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_precision_score"], 

         label="test precision")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_recall_score"], 

         'g--',

         label="training recall")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_recall_score"], 

         'r--',

         label="test recall")

plt.xlabel("Min Samples Split")

plt.ylabel("Score")

plt.legend()

plt.show()
%%time



optimal_tree = RandomForestClassifier(n_estimators=300,

                                      criterion='entropy',

                                      max_depth=36,

                                      min_samples_leaf=810,

                                      min_samples_split=1800,

                                      random_state=100)



optimal_tree.fit(X_train, y_train)
y_pred = optimal_tree.predict(X_train)
print(classification_report(y_train, y_pred))
y_test_pred = optimal_tree.predict(X_test)
print(classification_report(y_test, y_test_pred))
print(accuracy_score(y_test, y_test_pred))
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion Matrix',

                          cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    else:

        1



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix_tra = confusion_matrix(y_test,

                                  y_test_pred)



class_names = [0, 1]

plt.figure(figsize = (5, 5))

plot_confusion_matrix(cnf_matrix_tra,

                      classes=class_names,

                      title='Confusion matrix')

plt.show()
precision, recall, thresholds = precision_recall_curve(y_test_pred, y_test)



precision_recall_auc = auc(recall, precision)



fig = plt.figure(figsize = (8,8))

plt.title('Precision Recall Curve')

plt.plot(precision, recall, 'b',label='AUC = %0.3f'% precision_recall_auc)

plt.legend(loc='upper right')

plt.plot([0,1],[0.1,0.1],'r--')

plt.xlim([-0.1,1.05])

plt.ylim([-0.1,1.01])

plt.ylabel('Precision Score')

plt.xlabel('Recall Score')

plt.show()