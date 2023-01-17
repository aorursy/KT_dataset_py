import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.describe()
data.info()
data.shape
data.drop('salary', axis=1, inplace=True)
grid = sns.PairGrid(data= data, hue='status')
grid = grid.map_upper(plt.scatter)
grid = grid.map_diag(sns.kdeplot, shade=True)
grid = grid.map_lower(sns.kdeplot)
plt.title('Distribution of the features')
cat_feats = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']

fig, axes = plt.subplots(2, 4, figsize=(20, 15))

sns.countplot(data.gender, hue=data.status, ax=axes[0][0])
sns.countplot(data.ssc_b, hue=data.status, ax=axes[0][1])
sns.countplot(data.hsc_b, hue=data.status, ax=axes[0][2])
sns.countplot(data.hsc_s, hue=data.status, ax=axes[0][3])
sns.countplot(data.degree_t, hue=data.status, ax=axes[1][0])
sns.countplot(data.workex, hue=data.status, ax=axes[1][1])
sns.countplot(data.specialisation, hue=data.status, ax=axes[1][2])
num_feats = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

sns.boxplot(data.status, data.ssc_p, ax=axes1[0][0])
sns.boxplot(data.status, data.hsc_p, ax=axes1[0][1])
sns.boxplot(data.status, data.degree_p, ax=axes1[0][2])
sns.boxplot(data.status, data.etest_p, ax=axes1[1][0])
sns.boxplot(data.status, data.mba_p, ax=axes1[1][1])
data.gender.value_counts()
data.ssc_b.value_counts()
data.hsc_b.value_counts()
data.hsc_s.value_counts()
data.degree_t.value_counts()
data.workex.value_counts()
data.specialisation.value_counts()
encoder = LabelEncoder()

data['gender'] = encoder.fit_transform(data['gender'])
data['ssc_b'] = encoder.fit_transform(data['ssc_b'])
data['hsc_b'] = encoder.fit_transform(data['hsc_b'])
data['hsc_s'] = encoder.fit_transform(data['hsc_s'])
data['degree_t'] = encoder.fit_transform(data['degree_t'])
data['workex'] = encoder.fit_transform(data['workex'])
data['specialisation'] = encoder.fit_transform(data['specialisation'])

data.head()
#encode target labels.
def encode(col):
    if col[0] == 'Placed':
        return 1
    else:
        return 0
    
data['status'] = data[['status']].apply(encode, axis=1)
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data.status)

print('train size : ' + str(train_data.shape[0]))
print('test size : ' + str(test_data.shape[0]))
sns.countplot(train_data.status)
train_data.status.value_counts()
test_data.head()
train_labels = pd.DataFrame(train_data.status, columns=['status'])
train_data.drop('status', axis=1, inplace=True)

test_labels = pd.DataFrame(test_data.status, columns=['status'])
test_data.drop('status', axis=1, inplace=True)

sampler = TomekLinks()
train_res, labels_res = sampler.fit_resample(train_data, train_labels)
labels_res.status.value_counts()
corr = pd.concat([train_res, labels_res], axis=1).corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, cmap='YlGnBu', annot=True)
extra_tree_forest = ExtraTreesClassifier() 
  
extra_tree_forest.fit(train_res, labels_res) 

feature_importance = extra_tree_forest.feature_importances_ 

plt.figure(figsize=(15, 15))
plt.bar(train_res.columns, feature_importance) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()
def print_metrics(predicts, true_labels):
    print('Accuracy : ' + str(round(accuracy_score(predicts, true_labels), 2)))
    print('Precision : ' + str(round(precision_score(predicts, true_labels), 2)))
    print('Recall : ' + str(round(recall_score(predicts, true_labels), 2)))
    print('f1score : ' + str(round(f1_score(predicts, true_labels), 2)))
test_data.reset_index(inplace=True, drop=True)
test_labels.reset_index(inplace=True, drop=True)

train_res.reset_index(inplace=True, drop=True)
labels_res.reset_index(inplace=True, drop=True)

train_res.drop('index', axis=1, inplace=True)
labels_res.drop('index', axis=1, inplace=True)
test_data.drop('index', axis=1, inplace=True)
test_labels.drop('index', axis=1, inplace=True)
def scale_data(train, test, num_cols):
    scaler = MinMaxScaler(feature_range=(0, 3))

    temp_data = train.copy()
    temp_test = test.copy()

    scaled_data = pd.DataFrame(scaler.fit_transform(temp_data[num_cols]), columns = num_cols)
    temp_data.drop(num_cols, axis=1, inplace=True)
    final_data = pd.concat([temp_data, scaled_data], axis=1)

    scaled_test = pd.DataFrame(scaler.fit_transform(temp_test[num_cols]), columns = num_cols)
    temp_test.drop(num_cols, axis=1, inplace=True)
    final_test = pd.concat([temp_test, scaled_test], axis=1)
    
    return final_data, final_test
logistic = LogisticRegression()
log_data, log_test = scale_data(train_res, test_data, num_feats + ['sl_no'])
logistic.fit(log_data, labels_res)
preds = logistic.predict(log_test)
print_metrics(preds, test_labels)

log_matrix = confusion_matrix(preds, test_labels)
dtc = DecisionTreeClassifier()
dtc.fit(train_res, labels_res)
preds = dtc.predict(test_data)
print_metrics(preds, test_labels)

dtc_matrix = confusion_matrix(preds, test_labels)
rfc = RandomForestClassifier()
rfc.fit(train_res, labels_res)
preds = rfc.predict(test_data)
print_metrics(preds, test_labels)

rfc_matrix = confusion_matrix(preds, test_labels)
fig, axes2 = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(log_matrix, cmap='YlGnBu', annot=True, ax=axes2[0])
sns.heatmap(dtc_matrix, cmap='YlGnBu', annot=True, ax=axes2[1])
sns.heatmap(rfc_matrix, cmap='YlGnBu', annot=True, ax=axes2[2])

axes2[0].set_title("Logistic Regression")
axes2[1].set_title("Decision Tree")
axes2[2].set_title("Random Forest")
logistic_tune = LogisticRegression()
log_data, log_test = scale_data(train_res, test_data, num_feats + ['sl_no'])

params = {
    'penalty' : ['l1', 'l2'],
    'max_iter' : [80, 90, 100, 110, 120]
}

search = RandomizedSearchCV(logistic_tune, params, n_iter=20, cv=6, random_state=21)

best_model = search.fit(log_data, labels_res)
best_logistic = LogisticRegression(**best_model.best_estimator_.get_params())
best_logistic.fit(log_data, labels_res)
preds = best_logistic.predict(log_test)
print_metrics(preds, test_labels)

log_matrix = confusion_matrix(preds, test_labels)
dtc_tune = DecisionTreeClassifier()

params = {
    'max_depth' : [6, 7, 8],
    'max_features' : [7, 8, 9]    
}

search = RandomizedSearchCV(dtc_tune, params, n_iter=50, cv=8, random_state=21)

best_model = search.fit(train_res, labels_res)
best_dtc = DecisionTreeClassifier(**best_model.best_estimator_.get_params())
best_dtc.fit(train_res, labels_res)
preds = best_dtc.predict(test_data)
print_metrics(preds, test_labels)

dtc_matrix = confusion_matrix(preds, test_labels)
rfc_tune = RandomForestClassifier()

params = {
    'n_estimators' : [160, 170, 180],
    'max_depth' : [6, 7, 8],
    'max_features' : [5, 6, 7],
    'bootstrap' : [True],
    'min_samples_leaf' : [2, 3]    
}

search = RandomizedSearchCV(rfc_tune, params, n_iter=40, cv=8, random_state=21)

best_model = search.fit(train_res, labels_res)
best_rfc = RandomForestClassifier(**best_model.best_estimator_.get_params())
best_rfc.fit(train_res, labels_res)
preds = best_rfc.predict(test_data)
print_metrics(preds, test_labels)

rfc_matrix = confusion_matrix(preds, test_labels)
fig, axes3 = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(log_matrix, cmap='YlGnBu', annot=True, ax=axes3[0])
sns.heatmap(dtc_matrix, cmap='YlGnBu', annot=True, ax=axes3[1])
sns.heatmap(rfc_matrix, cmap='YlGnBu', annot=True, ax=axes3[2])

axes3[0].set_title("Logistic Regression Tuned")
axes3[1].set_title("Decision Tree Tuned")
axes3[2].set_title("Random Forest Tuned")