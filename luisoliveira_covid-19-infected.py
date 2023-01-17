# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook

import os
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import numpy as np


# sklearn libs compatilhadas
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

## Tratando dados desbalanceados
from imblearn.over_sampling import SMOTE




### Clustenização 1
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 



## Classificacao 1
from sklearn.linear_model import LogisticRegression

## Classificação 2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from lightgbm import LGBMClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    df = pd.DataFrame(cm, index=["no", "yes"], columns=["no", "yes"])
    ax = sns.heatmap(df, annot=True)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return ax

def plot_roc(y_true, y_score, figsize=(8, 8)):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (AUC = {100*roc_auc:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc


def plot_precision_recall(precisions, recalls, thresholds):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(thresholds, precisions[:-1], "r--", label="Precisions")
    ax.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
    ax.set_title("Precision and Recall \n Tradeoff", fontsize=18)
    ax.set_ylabel("Level of Precision and Recall", fontsize=16)
    ax.set_xlabel("Thresholds", fontsize=16)
    ax.legend(loc="best", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return ax
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.columns = ['Patient ID',
 'Patient_age_quantile',
 'SARS-Cov-2_result',
 'Patient_addmited_regular_ward_care_bool',
 'Patient_addmited_semi-intensive_care_bool',
 'Patient_addmited_intensive_care_bool',
 'Hematocrit',
 'Hemoglobin',
 'Platelets',
 'Mean_platelet_volume',
 'Red_blood_Cells',
 'Lymphocytes',
 'MCHC',
 'Leukocytes',
 'Basophils',
 'MCH',
 'Eosinophils',
 'MCV',
 'Monocytes',
 'RDW',
 'Serum Glucose',
 'Respiratory_Syncytial Virus',
 'Influenza_A',
 'Influenza_B',
 'Parainfluenza_1',
 'CoronavirusNL63',
 'Rhinovirus_Enterovirus',
 'Mycoplasma_pneumoniae',
 'Coronavirus_HKU1',
 'Parainfluenza_3',
 'Chlamydophila_pneumoniae',
 'Adenovirus',
 'Parainfluenza 4',
 'Coronavirus229E',
 'CoronavirusOC43',
 'Inf_A_H1N1_2009',
 'Bordetella_ertussis',
 'Metapneumovirus',
 'Parainfluenza_2',
 'Neutrophils',
 'Urea',
 'Proteina_C',
 'Creatinine',
 'Potassium',
 'Sodium',
 'Influenza_B_test',
 'Influenza_A_test',
 'Alanine_ransaminase',
 'Aspartate_transaminase',
 'Gamma-glutamyltransferase',
 'Total_Bilirubin',
 'Direct_Bilirubin',
 'Indirect_Bilirubin',
 'Alkaline_phosphatase',
 'Ionized_calcium',
 'Strepto_A',
 'Magnesium_',
 'pCO2_',
 'Hb_saturation_',
 'Bas_excess_',
 'pO2_',
 'Fio2_',
 'Total_CO2_',
 'pH_',
 'HCO3_venon',
 'Rods',
 'Segmented',
 'Promyelocytes',
 'Metamyelocytes',
 'Myelocytes',
 'Myeloblasts',
 'Urine_Esterase',
 'Urine_Aspect',
 'Urine_pH',
 'Urine_Hemoglobin',
 'Urine_Bile pigments',
 'Urine_Ketone Bodies',
 'Urine_Nitrite',
 'Urine_Density',
 'Urine_Urobilinogen',
 'Urine_Protein',
 'Urine_Sugar',
 'Urine_Leukocytes',
 'Urine_Crystals',
 'Urine_Red blood cells',
 'Urine_Hyaline cylinders',
 'Urine_Granular cylinders',
 'Urine_Yeasts',
 'Urine_Color',
 'Partial_thromboplastin_time',
 'Relationship',
 'INR',
 'Lactic_Dehydrogenase',
 'Prothrombin_time (PT)',
 'Vitamin_B12',
 'Creatine_phosphokinase',
 'Ferritin',
 'Lactic_Acid',
 'Lipase_dosage',
 'D-Dimer',
 'Albumin',
 'Hb_saturation',
 'pCO2',
 'Base_excess',
 'pH',
 'Total_CO2',
 'HCO3_artery',
 'pO2',
 'Arteiral_Fio2',
 'Phosphor',
 'ctO2']
# sorted([[df.shape[0]-j,i] for i,j in df.isna().sum().items() if j > 0])


# [0, 'D-Dimer'],
#  [0, 'Mycoplasma_pneumoniae'],
#  [0, 'Partial_thromboplastin_time'],
#  [0, 'Prothrombin_time (PT)'],
#  [0, 'Urine_Sugar'],
#  [1, 'Fio2_'],
#  [1, 'Urine_Nitrite'],
#  [3, 'Vitamin_B12'],
#  [8, 'Lipase_dosage'],
## 
df.drop(['Prothrombin_time (PT)', 'D-Dimer', 'Mycoplasma_pneumoniae', 'Urine_Sugar', 'Partial_thromboplastin_time', 'Fio2_', 'Urine_Nitrite', 'Vitamin_B12'], axis = 1, inplace = True)
## Preenchendo as colunas  dos testes que não foram feitos.
columns_to_fill = pd.DataFrame(df.isna().sum()/df.shape[0], columns=['Missing'])
columns_to_fill = columns_to_fill[(columns_to_fill.Missing < 0.87)].index

for col in columns_to_fill:
    df[col] = df[col].fillna('not_done')

# df['null_cols'] = df.isna().sum(axis=1)
null_col = df.isna().sum(axis=1)
## Map CATEGORICO
fullMapper={'negative': 0, 'positive': 1,
           'not_detected': 0, 'detected': 1,
            'not_done': -1, 'absent': -1,
            'Não Realizado': -1,
               ## Urine Aspects
              'clear': 0,
              'cloudy': 1,
              'lightly_cloudy': 2,
              'altered_coloring': 3,
               #Urine_Leukocytes
               '<1000': 1000,
                #Urine_urobilinogen
                'normal':0,
            #'Urine_Crystals': {
              'Ausentes': 0,
              'Urato Amorfo --+': 1,
              'Urato Amorfo +++': 2,
              'Oxalato de Cálcio +++': 3,
              'Oxalato de Cálcio -++': 4,
            # Urine_Color
              'yellow': 0,
              'light_yellow': 1,
              'orange': 2,
              'citrus_yellow': 3,
            # Urine_Hemoglobin
            'present': 1,
               }

df.replace(fullMapper, inplace=True)
## Preenchendo os exames de sangue com base na média por idade
columns_to_fill = pd.DataFrame(df.isna().sum()/df.shape[0], columns=['Missing'])
columns_to_fill = columns_to_fill[columns_to_fill.Missing > 0.87].index

#Prenche quando tiver o dado da idade
for value in df.Patient_age_quantile.unique():
    df_aux = df[df.Patient_age_quantile == value].copy()
    
    for col in columns_to_fill:
        df_aux[col] = df_aux[col].fillna(df_aux[col].median())
        
    df.loc[df_aux.index] = df_aux

#Preenche com a media geral
for col in columns_to_fill:
    df[col] = df[col].fillna(df[col].median())
try:
    df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals','null_cols']]  = df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals', 'null_cols']].astype('category')
except:
    df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals']]  = df[['Urine_Aspect', 'Urine_Color', 'Urine_Crystals']].astype('category')
df.isna().sum().sum()
use_sample = False
if use_sample:
    df_samp = df.sample(frac=0.8,random_state=76)
else:
    df_samp = df

X = df_samp.drop(['Patient ID','SARS-Cov-2_result', 'Patient_addmited_regular_ward_care_bool',
       'Patient_addmited_semi-intensive_care_bool',
       'Patient_addmited_intensive_care_bool'], axis=1)
y = df_samp['SARS-Cov-2_result']
X_scaled = StandardScaler().fit_transform(X)

best_cluster = 0
best_total = 0

for r in tqdm_notebook(range(50)):
    kmodel = KMeans(n_clusters=20).fit(X_scaled)
    X['cluster'] = kmodel.predict(X_scaled)

# for num in X.cluster.unique():
#     print(str(num) + '   |   ' + str(y[X.cluster == num].sum())  + '    |     ' + str(y[X.cluster == num].sum()/X[X.cluster == num].shape[0]))
    
    for num in X.cluster.unique():
#         print(str(num) + '   |   ' + str(y[X.cluster == num].sum())  + '    |     ' + str(y[X.cluster == num].sum()/X[X.cluster == num].shape[0]))
        if (y[X.cluster == num].sum() <= 2) & (X[X.cluster == num].shape[0] > 10):
#             print('--Replace')
            X['cluster'].replace(num, -1, inplace = True)
    
    if (y[X.cluster == -1].sum() >= best_cluster) & (X[X.cluster == -1].shape[0] > best_total):
        best_kmodel = kmodel
        best_cluster = y[X.cluster == -1].sum()
        best_total = X[X.cluster == -1].shape[0]

# X_scaled = StandardScaler().fit_transform(X)


# X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
df['null_cols'] = null_col
    
X = df.drop(['Patient ID','SARS-Cov-2_result', 'Patient_addmited_regular_ward_care_bool',
       'Patient_addmited_semi-intensive_care_bool',
       'Patient_addmited_intensive_care_bool'], axis=1)
y = df['SARS-Cov-2_result']
try:
    X_scaled = StandardScaler().fit_transform(X)
    X['cluster'] = best_kmodel.predict(X_scaled)
except:
    X_scaled = StandardScaler().fit_transform(X.drop(['null_cols'], axis=1))
    X['cluster'] = best_kmodel.predict(X_scaled)
    
for num in X.cluster.unique():
    print(str(num) + '   |   ' + str(y[X.cluster == num].sum())  + '/' + str(X[X.cluster == num].shape[0]) + '    |     ' + str(y[X.cluster == num].sum()/X[X.cluster == num].shape[0]))
    if (y[X.cluster == num].sum() <= 2) & (X[X.cluster == num].shape[0] > 15):
        print('--Replace')
        X['cluster'].replace(num, -1, inplace = True)

X_scaled = StandardScaler().fit_transform(X)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_scaled[X.cluster != -1], y[X.cluster != -1], test_size=0.3, random_state=0)

use_smote = False

if use_smote:
    os = SMOTE(random_state=131)
    columns = X.columns

    os_data_X, os_data_y = os.fit_sample(X_train_all, y_train_all)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y)

    X_train_all = os_data_X
    y_train_all = os_data_y['SARS-Cov-2_result']
    
    X_test_all = pd.DataFrame(data=X_test_all,columns=columns)

else:
    columns = X.columns
    X_train_all = pd.DataFrame(data=X_train_all,columns=columns)
    X_test_all = pd.DataFrame(data=X_test_all,columns=columns)


print(f"""
tamanho do dataset de treino:
X_train:{X_train_all.shape}
y_train:{y_train_all.shape}
~~~~~~~
tamanho do dataset de teste:
X_test:{X_test_all.shape}
y_test:{y_test_all.shape}
""")

X_train = X_train_all.copy()
y_train = y_train_all.copy()
X_test = X_test_all.copy()
y_test = y_test_all.copy()

# params = dict(
#     n_estimators=[150,500,1000],
#     max_depth=[3, 5, 10],
#     min_samples_split=[2,50],
#     min_samples_leaf=[1,5,10],
# )
# model = RandomForestClassifier(n_jobs=-1, random_state=42)
# grid = GridSearchCV(model, param_grid = params,verbose=True, n_jobs=-1, return_train_score= True)
# grid.fit(X_train, y_train)

best_params = {'max_depth': 10,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 500}

model = RandomForestClassifier(**best_params,n_jobs=-1,verbose=0, random_state=42).fit(X_train, y_train)
y_pred_rf = model.predict_proba(X_test)

y_test = y_test.append(y[X.cluster == -1])
y_pred_rf = np.append(y_pred_rf, np.repeat([[1,0]], y[X.cluster == -1].shape[0], axis=0), axis=0)
pred_train = model.predict(X_train)
scores = sklearn.metrics.accuracy_score(y_train, pred_train)
print('Accuracy on training data: {:.2f}%'.format(scores))   
 
pred_test = y_pred_rf[:,1] > 0.5
scores2 = sklearn.metrics.accuracy_score(y_test, pred_test)
print('Accuracy on test data: {:.2f}%'.format(scores2))    
_ =plot_roc(y_test, y_pred_rf[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_rf[:,1])
_ =plot_precision_recall(precisions, recalls, thresholds)
for i in range(0,5000):
    thr = i/5000
    y_pred_ = y_pred_rf[:,1] > thr
    
    if confusion_matrix(y_test, y_pred_)[1,0] > 9:
        thr -= 1/5000
        break

print(thr)

threshold = thr

y_pred_ = y_pred_rf[:,1] > threshold

_ =plot_confusion_matrix(y_test, y_pred_)
confusion_matrix(y_test, y_pred_)
shap.initjs()
plt.figure(figsize=(20,10))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.title("Top 20 important features")
plt.show()
explainer = shap.TreeExplainer(model, X_train.sample(100), model_output='probability', feature_dependence='independent')
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
def force_plot(explainer, patient):
    shap_values = explainer.shap_values(patient)
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
y_pred_[0:50]
# explainer = shap.TreeExplainer(model, X_train.sample(100), model_output='probability', feature_dependence='independent')
force_plot(explainer, patient=X_test.iloc[[5], :])
force_plot(explainer, patient=X_test.iloc[[7], :])
from sklearn.feature_selection import RFECV
from scipy.stats import randint, uniform

X_train = X_train_all.copy()
y_train = y_train_all.copy()
X_test = X_test_all.copy()
y_test = y_test_all.copy()


clf1 = LGBMClassifier(n_estimators=100, min_data=1, random_state=0, is_unbalance=True)
selector = RFECV(clf1, min_features_to_select=5, cv=5, scoring='roc_auc')

param_test ={
    'num_leaves': randint(2, 50), 
    'min_child_samples': randint(50, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
    'learning_rate': uniform(0.01, 0.2) }
    
#     'num_leaves': randint(3, 10), 
#     'min_child_samples': randint(50, 150),
#     'learning_rate': uniform(0.01, 0.1)

# clf2 = LGBMClassifier(n_estimators=100, random_state=77, min_data=1, silent=True, is_unbalance=True)
# model = RandomizedSearchCV(
#     estimator=clf2,
#     param_distributions=param_test, 
#     n_iter=200,
#     scoring='roc_auc',
#     cv=4,
#     refit=True,
#     random_state=0,
#     verbose=2,
#     n_jobs=-1
# )

# pipeline = make_pipeline(selector, model)

# pipeline.fit(X_train, y_train)

# model.best_estimator_
model = LGBMClassifier(boosting_type='gbdt', class_weight=None,
               colsample_bytree=0.5386420305589861, importance_type='split',
               is_unbalance=True, learning_rate=0.1177737006790299,
               max_depth=-1, min_child_samples=258, min_child_weight=1e-05,
               min_data=1, min_split_gain=0.0, n_estimators=100, n_jobs=4,
               num_leaves=4, objective=None, random_state=77, reg_alpha=0.1,
               reg_lambda=10, silent=True, subsample=0.3552182525036206,
               subsample_for_bin=200000, subsample_freq=0)

pipeline = make_pipeline(selector, model)

pipeline.fit(X_train, y_train)
y_pred_boost = pipeline.predict_proba(X_test)

y_test = y_test.append(y[X.cluster == -1])
y_pred_boost = np.append(y_pred_boost, np.repeat([[1,0]], y[X.cluster == -1].shape[0], axis=0), axis=0)
_ =plot_roc(y_test, y_pred_boost[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_boost[:,1])
_ =plot_precision_recall(precisions, recalls, thresholds)
for i in range(0,5000):
    thr = i/5000
    y_pred_ = y_pred_boost[:,1] > thr
    
    if confusion_matrix(y_test, y_pred_)[1,0] > 9:
        thr -= 1/5000
        break

print(thr)

threshold = thr

y_pred_ = y_pred_boost[:,1] > threshold

_ =plot_confusion_matrix(y_test, y_pred_)
confusion_matrix(y_test, y_pred_)
X_train = X_train_all.copy()
y_train = y_train_all.copy()
X_test = X_test_all.copy()
y_test = y_test_all.copy()

# Create smote object
smt = SMOTE(k_neighbors=5, random_state=1206)

# Do the process
X_train, y_train = smt.fit_sample(X_train, y_train)

# Defining parameter range to grid search
# param_gridSVM = {'C': [0.1, 1, 10, 100, 1000],
#                  'shrinking':[True, False],
#                  'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 
#                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  

# Best result found after grid search! This trick is to improve commit speed
# param_gridSVM = {'C': [10], 'gamma': [0.1], 'kernel': ['rbf'], 'shrinking': [True]}

# gridSVM = GridSearchCV(cv=3, estimator=SVC(class_weight='balanced', random_state=101, probability=True), param_grid=param_gridSVM, refit = True, verbose = 2, scoring='roc_auc', n_jobs=3)

# Define grid instance
gridSVM = SVC(class_weight='balanced', random_state=101, probability=True, C=10, gamma=0.1, kernel = 'rbf', shrinking = True) 

# Initialize grid search, fitting the best model
gridSVM.fit(X_train, y_train);

# print(gridSVM.best_params_)
y_pred_svm = gridSVM.predict_proba(X_test)

y_test = y_test.append(y[X.cluster == -1])
y_pred_svm = np.append(y_pred_svm, np.repeat([[1,0]], y[X.cluster == -1].shape[0], axis=0), axis=0)
_ =plot_roc(y_test, y_pred_svm[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_svm[:,1])
_ =plot_precision_recall(precisions, recalls, thresholds)
for i in range(0,5000):
    thr = i/5000
    y_pred_ = y_pred_svm[:,1] > thr
    
    if confusion_matrix(y_test, y_pred_)[1,0] > 9:
        thr -= 1/5000
        break

print(thr)

threshold = thr

y_pred_ = y_pred_svm[:,1] > threshold

_ =plot_confusion_matrix(y_test, y_pred_)
confusion_matrix(y_test, y_pred_)
best_alpha = 0
best_thr = 0
num_false_pos = 100000

for alpha in tqdm_notebook(range(100)):
    
    y_pred_combined = (alpha/100)*(y_pred_rf) + (1-alpha/100)*y_pred_boost
    
    for i in range(0,500):
        thr = i/500
        y_pred_ = y_pred_combined[:,1] > thr

        if confusion_matrix(y_test, y_pred_)[1,0] > 9:
            thr -= 1/500
            break

    threshold = thr

    y_pred_ = y_pred_combined[:,1] > threshold

    if confusion_matrix(y_test, y_pred_)[0,1] < num_false_pos:
        best_alpha = alpha/100
        best_thr = thr
        num_false_pos = confusion_matrix(y_test, y_pred_)[0,1]
    

print(best_alpha)
print(best_thr)
    
y_pred_combined = best_alpha*(y_pred_rf) + (1-best_alpha)*y_pred_boost
y_pred_ = y_pred_combined[:,1] > best_thr

_ =plot_confusion_matrix(y_test, y_pred_)
confusion_matrix(y_test, y_pred_)
X_scaled = StandardScaler().fit_transform(X)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

os = SMOTE(random_state=131)
columns = X.columns

os_data_X, os_data_y = os.fit_sample(X_train_all, y_train_all)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y = pd.DataFrame(data=os_data_y)

X_train_all = os_data_X
y_train_all = os_data_y['SARS-Cov-2_result']

X_test_all = pd.DataFrame(data=X_test_all,columns=columns)

X_train = X_train_all.copy()
y_train = y_train_all.copy()
X_test = X_test_all.copy()
y_test_ =y_test_all.copy()

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test_)

count_classes = y_test.shape[1]
print(count_classes)
# build the model
model_rn = Sequential()
model_rn.add(Dense(250, activation='relu', input_dim=(X_train.shape[1])))
model_rn.add(Dropout(.2))
model_rn.add(Dense(200, activation='relu'))
model_rn.add(Dropout(.2))
model_rn.add(Dense(200, activation='tanh'))
model_rn.add(Dropout(.2))
model_rn.add(Dense(100, activation='relu'))
model_rn.add(Dropout(.3))
model_rn.add(Dense(50, activation='relu'))
model_rn.add(Dense(2, activation='softmax'))

# Compile the model
model_rn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting The Mdoel

# model_rn.fit(X_train, y_train, sample_weight= y_train[:,1]*5 + 1, validation_data = (X_test, y_test, y_test[:,1]*5 + 1), batch_size=None, epochs=300, verbose = 1, workers=-1, use_multiprocessing=True)
model_rn.fit(X_train, y_train, batch_size=None, epochs=300, verbose = 0, workers=-1, use_multiprocessing=True)
pred_train = model_rn.predict(X_train)
scores = model_rn.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {:.2f}% \n Error on training data: {:.2f}'.format(scores[1], 1 - scores[1]))   
 
pred_test = model_rn.predict(X_test)
scores2 = model_rn.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {:.2f}% \n Error on test data: {:.2f}'.format(scores2[1], 1 - scores2[1]))    
y_pred_rn =  model_rn.predict(X_test)

_ =plot_roc(y_test[:,1], y_pred_rn[:,1])
precisions, recalls, thresholds = precision_recall_curve(y_test_, y_pred_rn[:,1])
_ =plot_precision_recall(precisions, recalls, thresholds)
for i in range(0,5000):
    thr = i/5000
    y_pred_ = y_pred_rn[:,1] > thr
    
    if confusion_matrix(y_test_, y_pred_)[1,0] > 17:
        thr -= 1/5000
        break

print(thr)

threshold = thr

y_pred_ = y_pred_rn[:,1] > threshold

_ =plot_confusion_matrix(y_test_, y_pred_)
confusion_matrix(y_test_, y_pred_)
