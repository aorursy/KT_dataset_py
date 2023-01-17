# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import time
import datetime
from datetime import datetime
import calendar

from sklearn import model_selection # model assesment and model selection strategies
from sklearn import metrics # model evaluation metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix


sns.set_style('white')

pd.options.display.float_format = '{:,.2f}'.format
full_df=pd.read_pickle('/kaggle/input/easymoney/fulldf_base_nandropped_FEOk-solo0y1.pkl',compression='zip')
full_df.info(verbose=True)
# split the data into train, validation and test dataset
train_index = sorted(list(full_df["pk_partition"].unique()))[5:-3]

valida_index = [sorted(list(full_df["pk_partition"].unique()))[-3]]

test_index = [sorted(list(full_df["pk_partition"].unique()))[-2]]
train_index
variables_borrar=['pk_cid', 
"pk_partition",
'segment',
'gender', 
'deceased',
'entry_date',
'mesesAlta',
'entry_channel',
'isNewClient',
'isActive',
'active_customer',
'totalAssets',
'Provincia',
'SalaryQtil',
'totalCuentas',
'totalAhorro',
'totalFinanciacion',  
'totalIngresos',
'totalBeneficio',
'diasDesdeUltimaAltaInt',
'diasDesdeUltimaAlta',
'hayAlta', 
'diasDesdeUltimaBajaInt',
'diasDesdeUltimaBaja',
'hayBaja',           
'country_id',                 
 'dif_loans',
 'dif_mortgage',
 'dif_funds',
 'dif_securities',
 'dif_long_term_deposit',
 'dif_short_term_deposit',
 'dif_em_account_pp',
 'dif_credit_card',
 'dif_payroll',
 'dif_pension_plan',
 'dif_payroll_account',
 'dif_emc_account',
 'dif_debit_card',
 'dif_em_acount',
 'dif_em_account_p',
 'loans',
 'mortgage',
 'funds',
 'securities',
 'long_term_deposit',
 'short_term_deposit',
 'em_account_pp',
 'credit_card',
 'payroll',
 'pension_plan',
 'payroll_account',
 'emc_account',
 'debit_card',
 'em_acount',
 'em_account_p']
#del X_train,Y_train, X_valida, Y_valida,X_test,Y_test
gc.collect()
X_train = full_df[full_df["pk_partition"].isin(train_index)].drop(variables_borrar, axis=1)
Y_train = full_df[full_df["pk_partition"].isin(train_index)]['dif_em_acount']

X_valida = full_df[full_df["pk_partition"].isin(valida_index)].drop(variables_borrar, axis=1)
Y_valida = full_df[full_df["pk_partition"].isin(valida_index)]['dif_em_acount']

# No lo creamos para ahorrar memoria.
#X_test = full_df[full_df["pk_partition"].isin(test_index)].drop(variables_borrar, axis = 1)
#Y_test = full_df[full_df["pk_partition"].isin(test_index)]['dif_em_acount']


del full_df
gc.collect()
dt = DecisionTreeClassifier(max_depth=6,random_state=42)
dt.fit(X_train,Y_train)
score_train=dt.score(X_train, Y_train)
score_test=dt.score(X_valida, Y_valida)
print('Resultados para: Train: {} - Valida: {}'.format(score_train,score_test))

X_test=X_valida
Y_test=Y_valida
y_test_pred = pd.DataFrame(dt.predict(X_test), index=Y_test.index, columns=['altaPrediction'])
results_df = Y_test.to_frame().join(y_test_pred)
results_df['Success']=(results_df['dif_em_acount']==results_df['altaPrediction']).astype(int)
results_df[results_df['dif_em_acount']!=0].sample(20)
results_df[results_df['dif_em_acount']==1]['Success'].hist()
results_df[results_df['dif_em_acount']==1]['Success'].value_counts()
results_df[results_df['dif_em_acount']==0]['Success'].hist()
results_df[results_df['dif_em_acount']==0]['Success'].value_counts()
# La podemos calcular directamente con la funcion confusion_matrix de skitlearn
conf_matrix=confusion_matrix(Y_test, dt.predict(X_test))
conf_matrix
# Tambien la podemos definir asi:
conf_matrix = pd.crosstab(results_df['dif_em_acount'], results_df['altaPrediction'])
conf_matrix
TP = conf_matrix.iloc[1,1]
TN = conf_matrix.iloc[0,0]
FP = conf_matrix.iloc[0,1]
FN = conf_matrix.iloc[1,0]
accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
f1_score = 2 / ( 1/Precision + 1/Recall )
Precision, Recall
f1_score
metrics.f1_score(Y_test, y_test_pred)
dt.predict_proba(X_test)
dt.predict_proba(X_test)[:,1]
y_score = pd.DataFrame(dt.predict_proba(X_test)[:,1], index=Y_test.index, columns=['AltaScore'])
results_df = results_df.join(y_score)
results_df
results_df[results_df['dif_em_acount']!=0].sample(20)
results_by_score = results_df.pivot_table(index='AltaScore', values='Success', aggfunc=[len, sum, np.mean])
results_by_score.columns = ['Count', 'Sum', 'Mean']
fig, ax = plt.subplots(figsize = (15, 6))
results_by_score['Mean'].plot(kind='bar')
print(metrics.roc_auc_score(results_df['dif_em_acount'], results_df['AltaScore']))
fpr, tpr, _ = metrics.roc_curve(results_df['dif_em_acount'], results_df['AltaScore'])
fig, ax = plt.subplots(figsize = (15, 6))
plt.clf()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
top_features = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(30)
top_features
