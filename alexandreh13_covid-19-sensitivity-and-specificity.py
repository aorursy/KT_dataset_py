import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
data.shape
data.describe
data.tail()
data.columns = [x.lower().strip().replace(',','').replace(' ','_') for x in data.columns]
data.columns
data.isnull().sum()
total_cells = np.product(data.shape)
total_missing = data.isnull().sum()


percentage_missing = (total_missing/total_cells)*100
percentage_missing
missing_values_count = data.isnull().sum()
total_missing = missing_values_count.sum()

# total of % data missing
(total_missing/total_cells)*100
#50 is just a random value
percentage_missing.sort_values(ascending=True).head(50)
correlations = data.corr(method='pearson')
correlations
import seaborn as sb
plt.figure(figsize = (20, 8))
sns.heatmap(correlations, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = False, vmax = 0.8)
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
object_cols
for col in object_cols:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes
data.tail()
correlations = data.corr(method='pearson')
correlations
cor_target = abs(correlations['sars-cov-2_exam_result'])
cor_target.sort_values(ascending=False).head(50)
data = data.fillna(data.median())
data.tail()
#22 features. Just a random number
data_model = data[['sars-cov-2_exam_result', 'patient_age_quantile', 'patient_addmited_to_regular_ward_(1=yes_0=no)', 'patient_addmited_to_semi-intensive_unit_(1=yes_0=no)', 'patient_addmited_to_intensive_care_unit_(1=yes_0=no)', 'influenza_b', 'respiratory_syncytial_virus', 'influenza_a', 'rhinovirus/enterovirus', 'inf_a_h1n1_2009', 'coronavirusoc43', 'coronavirus229e', 'parainfluenza_4', 'adenovirus', 'chlamydophila_pneumoniae', 'parainfluenza_3', 'coronavirus_hku1', 'coronavirusnl63', 'parainfluenza_1', 'bordetella_pertussis', 'parainfluenza_2', 'metapneumovirus']]
data_model.isnull().sum()
data_model.shape
data_model['sars-cov-2_exam_result'].value_counts()
data_model['sars-cov-2_exam_result'].value_counts().plot(kind='bar', title='Count (target)')
# Class count
count_class_0, count_class_1 = data_model['sars-cov-2_exam_result'].value_counts()

# Divide by class
class_0 = data_model[data_model['sars-cov-2_exam_result'] == 0]
class_1 = data_model[data_model['sars-cov-2_exam_result'] == 1]
class_0_under = class_0.sample(count_class_1)
new_data_under = pd.concat([class_0_under, class_1], axis=0)

print('Under-sampling:')
print(new_data_under['sars-cov-2_exam_result'].value_counts())

new_data_under['sars-cov-2_exam_result'].value_counts().plot(kind='bar', title='Count (target)')
data_model = new_data_under
data_model['sars-cov-2_exam_result'].value_counts()
from sklearn.model_selection import cross_val_score, KFold, train_test_split
X = data_model.drop(['sars-cov-2_exam_result'], axis=1)
y = data_model['sars-cov-2_exam_result'].copy()
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train.shape
X_test.shape
results_dict = {}
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
kf = KFold(n_splits=10, random_state=0, shuffle=True)
lr = LogisticRegression(C=0.5, random_state=1)
mean_auc_lr = cross_val_score(lr, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
results_dict['Logistic Regression'] = mean_auc_lr
results_dict
knn = KNeighborsClassifier(n_neighbors=5)
mean_auc_knn = cross_val_score(knn, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
results_dict['KNN'] = mean_auc_knn
results_dict
svm = svm.SVC()
mean_auc_svm = cross_val_score(svm, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
results_dict['SVM'] = mean_auc_svm
results_dict
nb = GaussianNB()
mean_auc_nb = cross_val_score(nb, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
results_dict['NB'] = mean_auc_nb
results_dict
x = ['Logistic Regression', 'KNN', 'SVM', 'NB']
y = [results_dict['Logistic Regression'], results_dict['KNN'], results_dict['SVM'], results_dict['NB']]
plt.title("AUC comparison")
plt.ylabel("AUC")
plt.bar(x,y)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
nb.fit(X_train, y_train)
predicted = nb.predict(X_test)
roc_auc = roc_auc_score(y_test, predicted)
mae = mean_absolute_error(y_test, predicted)

print("Mean Absolute Error: {} | ROC AUC: {}".format(mae, roc_auc))
from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(nb, X_test, y_test,
                                 display_labels=data_model['sars-cov-2_exam_result'],
                                 cmap=plt.cm.Blues)

disp.ax_.set_title("Confusion Matrix")
disp.confusion_matrix
plt.show()
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, predicted)
confusion
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)
#17 features. Just a random number
new_data_model = data[['sars-cov-2_exam_result', 'pco2_(arterial_blood_gas_analysis)', 'ph_(arterial_blood_gas_analysis)', 'po2_(arterial_blood_gas_analysis)', 'arteiral_fio2', 'ionized_calcium', 'leukocytes', 'platelets', 'cto2_(arterial_blood_gas_analysis)', 'total_co2_(arterial_blood_gas_analysis)', 'hco3_(arterial_blood_gas_analysis)', 'monocytes', 'eosinophils', 'lipase_dosage', 'segmented', 'ferritin', 'urine_-_density']]
new_data_model.shape
correlations = new_data_model.corr(method='pearson')
correlations
plt.figure(figsize = (20, 8))
sns.heatmap(correlations, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.8)
X = new_data_model.drop(['sars-cov-2_exam_result'], axis=1)
y = new_data_model['sars-cov-2_exam_result'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
new_result_dict = {}
mean_auc_lr = cross_val_score(lr, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
new_result_dict['Logistic Regression'] = mean_auc_lr
new_result_dict
mean_auc_knn = cross_val_score(knn, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
new_result_dict['KNN'] = mean_auc_knn
new_result_dict
mean_auc_svm = cross_val_score(svm, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
new_result_dict['SVM'] = mean_auc_svm
new_result_dict
mean_auc_nb = cross_val_score(nb, X_train, y_train, n_jobs=-1, cv=kf, scoring='roc_auc').mean()
new_result_dict['NB'] = mean_auc_nb
new_result_dict
x = ['Logistic Regression', 'KNN', 'SVM', 'NB']
y = [new_result_dict['Logistic Regression'], new_result_dict['KNN'], new_result_dict['SVM'], new_result_dict['NB']]
plt.title("AUC comparison")
plt.ylabel("AUC")
plt.bar(x,y)
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
roc_auc = roc_auc_score(y_test, predicted)
mae = mean_absolute_error(y_test, predicted)

print("Mean Absolute Error: {} | ROC AUC: {}".format(mae, roc_auc))
disp = plot_confusion_matrix(lr, X_test, y_test,
                                 display_labels=new_data_model['sars-cov-2_exam_result'],
                                 cmap=plt.cm.Blues)

disp.ax_.set_title("Confusion Matrix")
disp.confusion_matrix
plt.show()
confusion = confusion_matrix(y_test, predicted)
confusion
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)