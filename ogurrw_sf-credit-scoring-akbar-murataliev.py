# Importing modules.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from pandas import Series
from sklearn import metrics 
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score 

%matplotlib inline
#Setting the conditions for experiments.
random_seed = 42
current_date = pd.to_datetime('21/10/2020')
pd.set_option('display.max_columns', None)
data_directory = '/kaggle/input/sf-dst-scoring/'
!pip freeze > requirements.txt
# Defining a function for detecting outliers.
def outlier_detect(data, column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    lower_number = len(data[column<lower_range])
    upper_number = len(data[column>upper_range])
    print('Lower Range:', lower_range,
          'Upper Range:', upper_range,
          'Lower Outliers:', lower_number,
          'Upper Outliers:', upper_number, 
          sep='\n')
# Defining a function for visualization of confusion matrix.
def show_confusion_matrix(y_true, y_pred):
    color_text = plt.get_cmap('PuBu')(0.95)
    class_names = ['Default', 'Non-Default']
    cm = confusion_matrix(y_true, y_pred)
    cm[0,0], cm[1,1] = cm[1,1], cm[0,0]
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), title="Confusion Matrix")
    ax.title.set_fontsize(15)
    sns.heatmap(df, square=True, annot=True, fmt="d", linewidths=1, cmap="PuBu")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor", fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=14, color = color_text)
    ax.set_xlabel('Real Values', fontsize=14, color = color_text)
    b, t = plt.ylim()
    plt.ylim(b+0.5, t-0.5)
    fig.tight_layout()
    plt.show()
# Defining a function for visualization of metrics for logistic regression.
def all_metrics(y_true, y_pred, y_pred_prob):
    dict_metric = {}
    P = np.sum(y_true==1)
    N = np.sum(y_true==0)
    TP = np.sum((y_true==1)&(y_pred==1))
    TN = np.sum((y_true==0)&(y_pred==0))
    FP = np.sum((y_true==0)&(y_pred==1))
    FN = np.sum((y_true==1)&(y_pred==0))
    
    dict_metric['Positive, P'] = [P,'default']
    dict_metric['Negative, N'] = [N,'non-default']
    dict_metric['True Positive, TP'] = [TP,'correctly identified default']
    dict_metric['True Negative, TN'] = [TN,'correctly identified non-default']
    dict_metric['False Positive, FP'] = [FP,'incorrectly identified default']
    dict_metric['False Negative, FN'] = [FN,'incorrectly identified non-default']
    dict_metric['Accuracy'] = [accuracy_score(y_true, y_pred),'Accuracy=(TP+TN)/(P+N)']
    dict_metric['Precision'] = [precision_score(y_true, y_pred),'Precision = TP/(TP+FP)'] 
    dict_metric['Recall'] = [recall_score(y_true, y_pred),'Recall = TP/P']
    dict_metric['F1-score'] = [f1_score(y_true, y_pred),'Harmonical mean of Precision и Recall']
    dict_metric['ROC_AUC'] = [roc_auc_score(y_true, y_pred_prob),'ROC AUC Score']    

    temp_df = pd.DataFrame.from_dict(dict_metric, orient='index', columns=['Value', 'Description'])
    display(temp_df)   
# Importing datasets.
data_train = pd.read_csv(data_directory+'train.csv')
data_test = pd.read_csv(data_directory+'test.csv')
sample_submission = pd.read_csv(data_directory+'/sample_submission.csv')
# Checking the data.
data_train.info()
data_train.head()
# Checking the data.
data_test.info()
data_test.head()
# Merging the datasets.
data_train['sample'] = 1
data_test['sample'] = 0
data = data_train.append(data_test, sort=False).reset_index(drop=True)
# Checking the data.
data.info()
data.head()
# Checking for missing values.
data.isna().sum()
# Checking the number of unique values.
data.nunique()
# Grouping column names by data type.
time_cols = ['app_date']
cat_cols = ['education', 'region_rating', 'home_address', 'work_address', 'sna', 'first_time']
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
num_cols = ['age','decline_app_cnt','score_bki','bki_request_cnt','income']
# Checking the missing values.
data['education'].value_counts(dropna = False)
# Creating a new feature.
data['education_nan'] = pd.isna(data['education']).astype('uint8')
# Filling the missing values with the most frequent value ('SCH').
data['education'] = data['education'].fillna('SCH')
# Encoding binary variables.
label_encoder = LabelEncoder()
for column in bin_cols:
    data[column] = label_encoder.fit_transform(data[column])
# Checking the data.
data.head()
# Converting the data to datetime.
data['app_date'] = pd.to_datetime(data['app_date'], format='%d%b%Y')
data.head()
# Finding the minimum.
data_min = min(data['app_date'])
data_min
# Creating a new feature.
data['app_date_timedelta'] = (data['app_date'] - data_min).dt.days.astype('int')
data.head()
# Adding a feature to the list.
num_cols.append('app_date_timedelta')
# Checking the frequency distribution.
data['app_date_timedelta'].hist(bins=50)
# Checking the frequency distribution.
data.boxplot(column=['app_date_timedelta'])
# Detection of outliers.
outlier_detect(data,data['app_date_timedelta'])
# Checking the frequency distribution.
data['education'].hist()
# Encoding a categorical variable.
education_dict = {
    'SCH': 1,
    'GRD': 2,
    'UGR': 3,
    'PGR': 4,
    'ACD': 5,
}

data['education'] = data['education'].map(education_dict)
# Checking the frequency distribution.
data['sex'].hist(bins=2)
# Checking the frequency distribution.
data['age'].hist()
# Checking the frequency distribution.
data.boxplot(column=['age'])
# Detection of outliers.
outlier_detect(data,data['age'])
# Taking the logarithm.
np.log(data['age'] + 1).hist()
# Taking the logarithm.
data['age'] = np.log(data['age'] + 1)
data.head()
# Checking the frequency distribution.
data.boxplot(column=['age'])
# Detection of outliers.
outlier_detect(data,data['age'])
# Checking the frequency distribution.
data['car'].hist(bins=2)
# Checking the frequency distribution.
data['car_type'].hist(bins=2)
# Checking the frequency distribution.
data['decline_app_cnt'].hist()
# Checking the frequency distribution.
data.boxplot(column=['decline_app_cnt'])
# Detection of outliers.
outlier_detect(data,data['decline_app_cnt'])
# Taking the logarithm.
np.log(data['decline_app_cnt'] + 1).hist()
# Taking the logarithm.
data['decline_app_cnt'] = np.log(data['decline_app_cnt'] + 1)
# Checking the frequency distribution.
data.boxplot(column=['decline_app_cnt'])
# Detection of outliers.
outlier_detect(data,data['decline_app_cnt'])
# Checking the frequency distribution.
data['good_work'].hist(bins=2)
# Checking the frequency distribution.
data['score_bki'].hist()
# Checking the frequency distribution.
data.boxplot(column=['score_bki'])
# Detection of outliers.
outlier_detect(data,data['score_bki'])
# Checking the frequency distribution.
data['bki_request_cnt'].hist()
# Checking the frequency distribution.
data.boxplot(column=['bki_request_cnt'])
# Detection of outliers.
outlier_detect(data,data['bki_request_cnt'])
# Taking the logarithm.
np.log(data['bki_request_cnt'] + 1).hist()
# Taking the logarithm.
data['bki_request_cnt'] = np.log(data['bki_request_cnt'] + 1)
# Checking the frequency distribution.
data.boxplot(column=['bki_request_cnt'])
# Detection of outliers.
outlier_detect(data,data['bki_request_cnt'])
# Checking the frequency distribution.
data['region_rating'].hist()
# Checking the frequency distribution.
data['home_address'].hist(bins=3)
# Checking the frequency distribution.
data['work_address'].hist(bins=3)
# Checking the frequency distribution.
data['income'].hist(bins=100)
# Checking the frequency distribution.
data.boxplot(column=['income'])
# Checking the frequency distribution.
outlier_detect(data,data['income'])
# Taking the logarithm.
np.log(data['income'] + 1).hist(bins=100)
# Taking the logarithm.
data['income'] = np.log(data['income'] + 1)
# Checking the frequency distribution.
data.boxplot(column=['income'])
# Detection of outliers.
outlier_detect(data,data['income'])
# Checking the frequency distribution.
data['sna'].hist()
# Checking the frequency distribution.
data['first_time'].hist()
# Checking the frequency distribution.
data['foreign_passport'].hist(bins=2)
# Checking the frequency distribution.
data['default'].hist(bins=2)
# Checking the correlation matrix
data_train_temp = data[data['sample']==1]
sns.heatmap(data_train_temp[num_cols].corr().abs(), vmin=0, vmax=1)
# Checking the correlation matrix
data_train_temp[num_cols].corr().abs().sort_values(by='decline_app_cnt', ascending=False)
# Checking the frequency distribution.
fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()
for i in range(len(num_cols)):
    sns.boxplot(x="default", y=num_cols[i], data=data_train_temp, ax=axes[i])
# Checking the importance of features.
imp_num = Series(f_classif(data_train_temp[num_cols], 
                           data_train_temp['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')
# Checking the importance of features.
imp_cat = Series(mutual_info_classif(
    data_train_temp[bin_cols + cat_cols], data_train_temp['default'], 
    discrete_features =True
), index = bin_cols + cat_cols)

imp_cat.sort_values(inplace = True)
imp_cat.plot(kind = 'barh')
# Standardization of data.
ss = StandardScaler()
data[num_cols] = pd.DataFrame(ss.fit_transform(data[num_cols]),columns = data[num_cols].columns)
# Checking the data.
data.info()
data.head(5)
# Data processing and model training.
data_temp = data.drop(['sample', 'client_id', 'app_date', 'default'], axis=1)
data_education_nan = data_temp[data_temp['education_nan']==1]
data_no_nan = data_temp[data_temp['education_nan']==0]
y = data_no_nan['education'].values
X = data_no_nan.drop(['education'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_seed)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state = random_seed)
model.fit(X_train, y_train)
y_pred = np.round(model.predict(X_test))
# Predicting the values.
predict = np.round(model.predict(data_education_nan.drop(['education'], axis=1)))
# Adding predicted values to the dataset.
index_education_nan = data[data['education_nan']==1].index
data.loc[index_education_nan,'education'] = predict
# Encoding categorical variables.
data = pd.get_dummies(data, prefix=cat_cols, columns=cat_cols)
data.info()
# Checking the data.
data.head()
# Splitting the dataset.
data_train = data.query('sample == 1').drop(['sample', 'client_id', 'app_date'], axis=1)
data_test = data.query('sample == 0').drop(['sample', 'client_id', 'app_date'], axis=1)
# Training and predicting.
X = data_train.drop(['default'], axis=1)
y = data_train['default'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
# Plotting the ROC curve
probs = model.predict_proba(X_test)
probs = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
# Checking the confusion matrix.
show_confusion_matrix(y_test, y_pred)
# Checking the metrics.
all_metrics(y_test, y_pred, y_pred_prob)
model = LogisticRegression(random_state=random_seed)

iter_ = 50
epsilon_stop = 1e-3

param_grid = [
    {'penalty': ['l1'], 
     'solver': ['liblinear', 'lbfgs'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['l2'], 
     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
    {'penalty': ['none'], 
     'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 
     'class_weight':['none', 'balanced'], 
     'multi_class': ['auto','ovr'], 
     'max_iter':[iter_],
     'tol':[epsilon_stop]},
]

gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=5)
gridsearch.fit(X_train, y_train)
model = gridsearch.best_estimator_

best_parameters = model.get_params()
for param_name in sorted(best_parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

preds = model.predict(X_test)
print('Accuracy: %.4f' % accuracy_score(y_test, preds))
print('Precision: %.4f' % precision_score(y_test, preds))
print('Recall: %.4f' % recall_score(y_test, preds))
print('F1: %.4f' % f1_score(y_test, preds))
# Training and predicting.
model = LogisticRegression(random_state=random_seed, 
                           C=1, 
                           class_weight='balanced', 
                           dual=False, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           l1_ratio=None, 
                           multi_class='auto', 
                           n_jobs=None, 
                           penalty='l1', 
                           solver='liblinear', 
                           verbose=0, 
                           warm_start=False)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)
# Plotting the ROC curve
probs = model.predict_proba(X_test)
probs = probs[:,1]

fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
# Checking the confusion matrix.
show_confusion_matrix(y_test, y_pred)
# Checking the metrics.
all_metrics(y_test, y_pred, y_pred_prob)
data_train = data.query('sample == 1').drop(['sample', 'client_id', 'app_date'], axis=1)
data_test = data.query('sample == 0').drop(['sample', 'client_id', 'app_date'], axis=1)
X_train = data_train.drop(['default'], axis=1)
y_train = data_train['default'].values
X_test = data_test.drop(['default'], axis=1)
model = LogisticRegression(random_state=random_seed, 
                           C=1, 
                           class_weight='balanced', 
                           dual=False, 
                           fit_intercept=True, 
                           intercept_scaling=1, 
                           l1_ratio=None, 
                           multi_class='auto', 
                           n_jobs=None, 
                           penalty='l1', 
                           solver='liblinear', 
                           verbose=0, 
                           warm_start=False,
                           max_iter=1000)

model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:,1]
submit = pd.DataFrame(data.query('sample == 0')['client_id'])
submit['default'] = y_pred_prob
submit.to_csv('submission.csv', index=False)