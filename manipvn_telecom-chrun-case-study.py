# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.read_excel('https://cdn.upgrad.com/UpGrad/temp/a625d1ee-b8d7-4edb-bdde-b1d82beaf5b4/Data+Dictionary-+Telecom+Churn+Case+Study.xlsx')
telecom_data = pd.read_csv('https://upgradstorage.s3.ap-south-1.amazonaws.com/Telecom_churn_caseStudy/telecom_churn_data.csv')

telecom_data.head()
telecom_data.info()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print(telecom_data.describe())
telecom_data.shape
def get_null_percentage(_data):

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        return (_data.isnull().sum(axis=0)/_data.shape[0]).sort_values(ascending= False)
null_values_per = get_null_percentage(telecom_data)

null_values_per
telecom_data['good_phase_recharge'] = telecom_data['total_rech_amt_6'] + telecom_data['total_rech_amt_7']
seventy_percentile = int(telecom_data['good_phase_recharge'].quantile(.70))

telecom_data = telecom_data[(telecom_data.good_phase_recharge > seventy_percentile)]

telecom_data.shape
null_values_per = get_null_percentage(telecom_data)

null_values_per
filtered_columns = list(null_values_per[null_values_per < 0.6].index) 
filtered_data = telecom_data[filtered_columns]

filtered_data.head()
get_null_percentage(filtered_data)
# filtered_data.apply(pd.value_counts)



filtered_data.describe()
# std_og_t2c_mou_6

# std_ic_t2o_mou_6

# std_ic_t2o_mou_8

# std_og_t2c_mou_8

# std_og_t2c_mou_9

# std_ic_t2o_mou_9



# ===> Drop these columns as there is no change in the data overall

dropping_columns = [

    'std_og_t2c_mou_6', 'std_ic_t2o_mou_6', 'std_ic_t2o_mou_8',

    'std_og_t2c_mou_8', 'std_og_t2c_mou_9', 'std_ic_t2o_mou_9',

    'std_og_t2c_mou_7', 'std_ic_t2o_mou_7', 'loc_og_t2o_mou', 'std_og_t2o_mou',

    'loc_ic_t2o_mou', 'circle_id', 'mobile_number'

]

filtered_data = filtered_data.drop(columns=dropping_columns, axis=1)
filtered_data.describe()
filtered_data.roam_og_mou_9.fillna(0, inplace=True)
def is_churned(_x):

    if ((_x.total_ic_mou_9 == 0) & (_x.total_og_mou_9 == 0) & (_x.vol_2g_mb_9 == 0) & (_x.vol_3g_mb_9 == 0)):

        return 1

    else:

        return 0

filtered_data['churn'] = filtered_data.apply(is_churned, axis=1)
filtered_data.churn.value_counts()
filtered_data.info(verbose=True)
# ===> Drop columns 'last_date_of_month_6', 'last_date_of_month_7' as it's same across all columns

filtered_data.drop(columns=['last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8', 'last_date_of_month_9'], axis=1, inplace=True)
def convert_to_datetime(_x, _columns):

    _x[_columns] = _x[_columns].apply(pd.to_datetime, format='%m/%d/%Y')

    return _x





_columns = [

     'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_9', 'date_of_last_rech_8'

]

filtered_data[_columns] = convert_to_datetime(filtered_data[_columns], _columns)
without9_columns = filtered_data.columns.drop(list(filtered_data.filter(regex='_9')))

filtered_data = filtered_data[without9_columns] 

filtered_columns = filtered_data.columns
filtered_data[(filtered_data.arpu_7 <= 0) & (filtered_data.arpu_8 <= 0)].churn.value_counts() 
for _column in filtered_columns:

    plt.figure()

    sns.boxplot(y=_column, x='churn', data=filtered_data, orient='v')

    plt.show()
filtered_data.info(verbose=True)
#### Feature engineering: 
filtered_data.date_of_last_rech_7.dtype
def filter_date_and_day(_data):

    for _column in _data.select_dtypes(include=['datetime64']).columns:

        _data[_column + '_year'] = _data[_column].dt.year 

        _data[_column + '_month'] = _data[_column].dt.month 

        _data[_column + '_day'] = _data[_column].dt.day

        _data.drop(columns=[_column], axis=1, inplace=True)

    return _data



filtered_data = filter_date_and_day(filtered_data)

filtered_data.info(verbose=True)
filtered_data.select_dtypes(include=['datetime64']).columns
# remove rows whose rows has null values more than 80

filtered_data = filtered_data[~(filtered_data.apply(lambda x: sum(x.isnull().values), axis = 1)> 80)]
filtered_data.shape
filtered_data.dropna(inplace=True)
filtered_data.shape
qar = filtered_data['loc_og_t2m_mou_6'].quantile(1.0)

filtered_data[ filtered_data['onnet_mou_8'] < qar].shape
filtered_data.shape
def quantile_percentage(data):  

    quantile = pd.DataFrame(columns=['col', '10','50','85','90','95','99','100','max'])

    for col in data.columns:

        _tmp = data[col].quantile([0.1,0.5,0.85,0.9,0.95,0.99,1.0])

        quantile = quantile.append({'col': col, 

                                    '10': str(round(_tmp[0.1],2)), 

                                    '50': str(round(_tmp[0.5],2)),

                                    '85': str(round(_tmp[0.85],2)),

                                    '90': str(round(_tmp[0.9],2)),

                                    '95': str(round(_tmp[0.95],2)),

                                    '99': str(round(_tmp[0.99],2)),

                                    '100': str(round(_tmp[1.0],2)),

                                   'max':max(data[col])}, ignore_index=True)

    return quantile



pd.set_option('display.max_rows', 500)

quantile_percentage(filtered_data.select_dtypes([np.number]))
Q1 = filtered_data.quantile(0.05)

Q3 = filtered_data.quantile(0.99)

IQR = Q3 - Q1



filtered_data = filtered_data[~((filtered_data < (Q1 - 1.5 * IQR)) |(filtered_data > (Q3 + 1.5 * IQR))).any(axis=1)]

filtered_data.shape
X = filtered_data.drop(columns=['churn'], axis=1)

Y = filtered_data[['churn']]



scaler = StandardScaler()

_columns = X.columns

X[_columns] = scaler.fit_transform(X)

X.head()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
x_train.head()
pca = PCA(random_state=100, svd_solver='randomized')

pca.fit_transform(x_train)
pca.components_.round(4)
colnames = list(x_train.columns)

pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':colnames})

pcs_df.head(10)
pca.explained_variance_ratio_.round(4)
fig = plt.figure(figsize=(12, 8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
pca_final = IncrementalPCA(n_components=85)

df_train_pca = pca_final.fit_transform(x_train)

df_train_pca.shape
corrmat = np.corrcoef(df_train_pca.transpose())

plt.figure(figsize=(40, 40))

sns.heatmap(corrmat, annot=True)
pca_test_data = pca_final.transform(x_test)
pca_test_data.shape
logistic_regression = LogisticRegression()

logistic_model = logistic_regression.fit(df_train_pca, y_train)
predicted_proba = logistic_model.predict_proba(pca_test_data)[:, 1]
"{:2.2}".format(metrics.roc_auc_score(y_test, predicted_proba))
plt.figure(figsize=(8, 8))

plt.scatter(df_train_pca[:, 0], df_train_pca[:, 1], c= y_train['churn'].map({0: 'red', 1: 'green'}))

plt.xlabel('Principle component 1')

plt.ylabel('Principle component 2')

plt.show()
fig = plt.figure(figsize=(8,8))

ax = Axes3D(fig)

# ax = plt.axes(projection='3d')

ax.scatter(df_train_pca[:,2], df_train_pca[:,0], df_train_pca[:,1], c=y_train['churn'].map({0:'green',1:'red'}))
len(logistic_model.coef_[0])
len(colnames)
pca_column_frame = pcs_df.head(85)

pca_column_frame.head()
# result =  pd.DataFrame({'columns': colnames})

pca_column_frame['coeff'] = logistic_model.coef_[0]

pca_column_frame.sort_values(by=['coeff'])
rfc = RandomForestClassifier(n_jobs=-1, bootstrap=True,

                             max_depth=4,

                             min_samples_leaf=50, 

                             min_samples_split=50,

                             max_features=8,

                             n_estimators=60)

rfc.fit(df_train_pca,y_train)
predictions = rfc.predict(pca_test_data)
print(classification_report(y_test, predictions))
accuracy_score(y_true=y_test, y_pred=predictions).round(2)
print(confusion_matrix(y_test, predictions))
predictions[:20]
predicted_proba[:10].round(3)
print(confusion_matrix(y_test, predictions))
metrics.roc_auc_score(y_test, predicted_proba)
import imblearn
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True)

x_rus, y_run, ind = rus.fit_sample(x_train, y_train)
x_rus.shape
y_run.sum()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(return_indices=True)



# x_ros, y_ros, ind = ros.fit_sample(x_train, y_train)

# x_test_ros, y_test_ros, ind = ros.fit_sample(x_test, y_test)



x_ros, y_ros, ind = ros.fit_sample(df_train_pca, y_train)

x_test_ros, y_test_ros, ind = ros.fit_sample(pca_test_data, y_test)
x_ros.shape
y_ros.sum()
def fit_random_forest(x_train_data, x_test_data, y_train_data, y_test_data):

    rf = RandomForestClassifier(n_jobs=-1,

                                bootstrap=True,

                                max_depth=4,

                                min_samples_leaf=50,

                                min_samples_split=50,

                                n_estimators=60)

    rf.fit(x_train_data, y_train_data)

    predictions = rf.predict(x_test_data)

    print(classification_report(y_test_data, predictions))

    print(accuracy_score(y_true=y_test_data, y_pred=predictions))

    print(confusion_matrix(y_test_data, predictions))

    return rf
rfe_algo = fit_random_forest(x_ros, x_test_ros, y_ros, y_test_ros)
from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')

x_smt_train, y_smt_train = smt.fit_sample(df_train_pca, y_train)

x_smt_test, y_smt_test = smt.fit_sample(pca_test_data, y_test)
rfe_algo = fit_random_forest(x_smt_train, x_smt_test, y_smt_train, y_smt_test)
rfe_algo.feature_importances_.round(3)
def fit_logistic_regression(x_train_data, x_test_data, y_train_data, y_test_data):

    log = LogisticRegression(random_state=True)

    log.fit(x_train_data, y_train_data)

    predicted_proba = log.predict_proba(x_test_data)[:,1]

    print(metrics.roc_auc_score(y_test_data, predicted_proba).round(3)*100)

    return log
log_alg = fit_logistic_regression(x_smt_train, x_smt_test, y_smt_train, y_smt_test)
_percentage = log_alg.predict_proba(x_smt_train)[:,1]
churn_predicted = pd.DataFrame({})

churn_predicted['ChurnProbability'] = _percentage

churn_predicted['y_train'] = y_smt_train

churn_predicted.head(10)
log_alg.classes_
probabilities = [i/10 for i in range(10) ]

cutoff = pd.DataFrame(columns=['prob', 'accuracy', 'sensitivity', 'specificity'])

for _prob in probabilities:

    churn_predicted[_prob] = churn_predicted.ChurnProbability.map(lambda x: 1 if x > _prob else 0)

    cm = metrics.confusion_matrix(churn_predicted.y_train, churn_predicted[_prob])

    print(cm)

    total = sum(sum(cm))

    _accuracy = round((cm[0, 0] + cm[1, 1])/total, 3)

    _sensitivity = round(cm[1,1] / (cm[1,0] + cm[1,1]), 3)

    _specificity = round(cm[0,0] / (cm[0,0] + cm[0,1]), 3)

    cutoff.loc[_prob] = [ _prob, _accuracy, _sensitivity,  _specificity ]

    

cutoff
cutoff.plot.line(x='prob', y=['accuracy', 'sensitivity', 'specificity'])
_percentage
_test_percentage = log_alg.predict_proba(x_smt_test)[:,1]

test_churn_predicted = pd.DataFrame({'ChurnProbability': log_alg.predict_proba(x_smt_test)[:,1]})

test_churn_predicted['y_test'] = y_smt_test

test_churn_predicted['0.55_predict'] = test_churn_predicted.ChurnProbability.map(lambda x: 1 if x > 0.55 else 0)

test_churn_predicted['0.65_predict'] = test_churn_predicted.ChurnProbability.map(lambda x: 1 if x > 0.65 else 0)

test_churn_predicted['0.60_predict'] = test_churn_predicted.ChurnProbability.map(lambda x: 1 if x > 0.6 else 0)

test_churn_predicted.head(10)
cm = metrics.confusion_matrix(test_churn_predicted.y_test, test_churn_predicted['0.65_predict'])

cm
cm = metrics.confusion_matrix(test_churn_predicted.y_test, test_churn_predicted['0.60_predict'])

cm
cm = metrics.confusion_matrix(test_churn_predicted.y_test, test_churn_predicted['0.55_predict'])

cm
_percentage = log_alg.predict_proba(x_smt_test)[:,1]
def draw_roc(actual, probability):

    fpr, tpr, thershoulds = metrics.roc_curve(actual, probability, drop_intermediate=False)

    auc = metrics.roc_auc_score(actual, probability)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)

    plt.plot([0,1], [0,1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    return None



draw_roc(y_smt_test, _percentage)
log_alg.coef_.round(2)
pca_column_frame['sampling_coeff'] = log_alg.coef_[0]

pca_column_frame.iloc[(-np.abs(pca_column_frame['sampling_coeff'].values)).argsort()]
# x_smt_train, x_smt_test, y_smt_train, y_smt_test



model_rf = RandomForestClassifier(bootstrap=True,

                                  max_depth=10,

                                  min_samples_leaf=100, 

                                  min_samples_split=200,

                                  n_estimators=1000,

                                  oob_score = True, n_jobs = -1,

                                  random_state =50,

                                  max_features = 15,

                                  max_leaf_nodes = 30)

model_rf.fit(x_smt_train, y_smt_train)



# Make predictions

prediction_test = model_rf.predict(x_smt_test)



print(classification_report(y_smt_test,prediction_test))

print(confusion_matrix(y_smt_test,prediction_test))

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
r_model = RandomForestClassifier()

params = {

    'max_features': range(30, 40, 5),

    'n_estimators': [40],

    'min_samples_leaf': range(100, 150, 25),

    'min_samples_split': range(100, 150, 25),

    'max_depth': [7, 8, 9]

}



folds = KFold(n_splits=2, shuffle=True, random_state=101)

grid_cv1 = GridSearchCV(r_model,

                        param_grid=params,

                        scoring='accuracy',

                        n_jobs=-1,

                        cv=folds,

                        verbose=1,

                        return_train_score=True)

grid_cv1.fit(x_smt_train, y_smt_train)
results = pd.DataFrame(grid_cv1.cv_results_)

results
grid_cv1.best_params_
best_estimator = grid_cv1.best_estimator_
best_estimator.fit(x_smt_train, y_smt_train)



# Make predictions

prediction_test = best_estimator.predict(x_smt_test)



print(classification_report(y_smt_test,prediction_test))

print(confusion_matrix(y_smt_test,prediction_test))

best_estimator.feature_importances_.round(3)
pca_column_frame



plt.figure(figsize=(20, 10))

top_10_features = pca_column_frame.iloc[(-np.abs(pca_column_frame['sampling_coeff'].values)).argsort()].head(10)

sns.barplot(x='Feature', y='sampling_coeff', data=top_10_features)

plt.xlabel('Features', size=20)

plt.ylabel('Coefficient', size=20)

plt.xticks(size = 14, rotation='vertical')

plt.show()