# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import randint, uniform 

pd.options.display.max_rows = 120

pd.options.display.max_columns = 120
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()
df.shape
new_cols = ['Patient_ID','Patient_age_quantile','SARS_Cov_2_exam_result','Patient addmited to regular ward',

 'Patient addmited to semi-intensive unit','Patient addmited to intensive care unit','Hematocrit','Hemoglobin','Platelets',

 'Mean platelet volume ','Red blood Cells','Lymphocytes','Mean corpuscular hemoglobin concentration','Leukocytes',

 'Basophils','Mean corpuscular hemoglobin','Eosinophils','Mean corpuscular volume','Monocytes','Red blood cell distribution width',

 'Serum Glucose','Respiratory Syncytial Virus','Influenza A','Influenza B','Parainfluenza 1','CoronavirusNL63','Rhinovirus_Enterovirus',

 'Mycoplasma pneumoniae','Coronavirus HKU1','Parainfluenza 3','Chlamydophila pneumoniae','Adenovirus','Parainfluenza 4','Coronavirus229E',

 'CoronavirusOC43','Inf A H1N1 2009','Bordetella pertussis','Metapneumovirus','Parainfluenza 2','Neutrophils','Urea',

 'Proteina C reativa mg/dL','Creatinine','Potassium','Sodium','Influenza B, rapid test','Influenza A, rapid test','Alanine transaminase',

 'Aspartate transaminase','Gamma-glutamyltransferase','Total Bilirubin','Direct Bilirubin','Indirect Bilirubin',

 'Alkaline phosphatase','Ionized calcium','Strepto A','Magnesium','pCO2 (venous blood gas analysis)','Hb saturation (venous blood gas analysis)',

 'Base excess (venous blood gas analysis)','pO2_venous blood gas analysis_','Fio2 (venous blood gas analysis)',

 'Total CO2_venous blood gas analysis','pH_venous blood gas analysis','HCO3 (venous blood gas analysis)',

 'Rods','Segmented','Promyelocytes','Metamyelocytes','Myelocytes','Myeloblasts','Urine - Esterase','Urine - Aspect',

 'Urine - pH','Urine - Hemoglobin','Urine - Bile pigments','Urine - Ketone Bodies','Urine - Nitrite','Urine - Density',

 'Urine - Urobilinogen','Urine - Protein','Urine - Sugar','Urine - Leukocytes','Urine - Crystals','Urine - Red blood cells',

 'Urine - Hyaline cylinders','Urine - Granular cylinders','Urine - Yeasts','Urine - Color','Partial thromboplastin time',

 'Relationship (Patient/Normal)','International normalized ratio (INR)','Lactic Dehydrogenase','Prothrombin time (PT), Activity',

 'Vitamin B12','Creatine phosphokinase','Ferritin','Arterial Lactic Acid','Lipase dosage','D-Dimer', 'Albumin',

 'Hb saturation (arterial blood gases)','pCO2 (arterial blood gas analysis)','Base excess (arterial blood gas analysis)',

 'pH (arterial blood gas analysis)','Total CO2 (arterial blood gas analysis)','HCO3 (arterial blood gas analysis)',

 'pO2 (arterial blood gas analysis)','Arteiral Fio2','Phosphor','ctO2 (arterial blood gas analysis)']
df.columns = new_cols
df.columns = df.columns.str.replace(" ", "_")

df.columns = df.columns.str.replace("(", "_")

df.columns = df.columns.str.replace(")", "_")

df.columns = df.columns.str.replace("-", "")

df.columns = df.columns.str.replace("__", "_")
df.head()
df.dtypes
object_columns = df.select_dtypes('object').columns
for col in object_columns:

    print(col, df[col].unique())
df = df.replace("negative", 0).replace("positive", 1)

df = df.replace("not_detected", 0).replace("detected", 1)
object_columns = df.select_dtypes('object').columns

for col in object_columns:

    print(col, df[col].unique())
df.corr()
plt.figure(figsize=(20, 10))

matrix = np.triu(df.corr())

sns.heatmap(df.corr(), mask=matrix)

# SARS-Cov-2 exam result
df_notnull = df.dropna(subset=['SARS_Cov_2_exam_result']).drop(columns=['Patient_ID'])

df_notnull.head()
df_notnull.SARS_Cov_2_exam_result.value_counts()
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, plot_confusion_matrix

from sklearn.inspection import plot_partial_dependence

from lightgbm import LGBMClassifier, plot_metric, plot_tree, create_tree_digraph

from sklearn.feature_selection import RFECV, SelectFromModel

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import LabelEncoder, FunctionTransformer

from tqdm import tqdm
def plot_roc_curve(y_true, y_pred):

    fpr, tpr, threshold = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10,7))

    ax.set_title('Receiver Operating Characteristic')

    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    ax.legend(loc = 'lower right')

    ax.plot([0, 1], [0, 1],'r--')

    ax.set_xlim([0, 1])

    ax.set_ylim([0, 1])

    ax.set_ylabel('True Positive Rate')

    ax.set_xlabel('False Positive Rate')

    return ax



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





def plot_confusion_matrix_2(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    df = pd.DataFrame(cm.T)

    ax = sns.heatmap(df, annot=True)

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    return ax
object_columns = df_notnull.select_dtypes('object').columns
df_notnull[object_columns]=df[object_columns].astype('category')
X = df_notnull.drop(columns='SARS_Cov_2_exam_result')

y = df_notnull.SARS_Cov_2_exam_result


cols = ['V{}'.format(i) for i in range(len(X.columns))]

  
X.columns = cols
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
param_test ={'num_leaves': randint(6, 50), 

             'min_child_samples': randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


clf = LGBMClassifier(random_state=42,  min_data=1, silent=True, is_unbalance=True)

model = RandomizedSearchCV(

    estimator=clf,

    param_distributions=param_test, 

    n_iter=100,

    scoring='roc_auc',

    cv=5,

    refit=True,

    random_state=42,

    verbose=True)
answer = model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]


plot_roc_curve(y_test, y_pred_proba)

roc_auc_score(y_test, y_pred_proba)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

plot_precision_recall(precisions, recalls, thresholds)
threshold = 0.528

# i tried a lot of values, this is the best balance until now

y_pred = y_pred_proba > threshold



r = confusion_matrix(y_test, y_pred)

s = [

        [r[0][0]/y_test.value_counts()[0], r[0][1]/y_test.value_counts()[0]], 

        [r[1][0]/y_test.value_counts()[1], r[1][1]/y_test.value_counts()[1]]

    ]

sns.heatmap(s, annot=True,cmap=plt.cm.Blues)