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
import pandas as pd

# Importing the dataset
bcell = pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')

bcell.head()
sars = pd.read_csv('/kaggle/input/epitope-prediction/input_sars.csv')

sars.head()
bs = pd.concat([bcell, sars], ignore_index=True)

bs
from sklearn.model_selection import train_test_split
train, test = train_test_split(bs, test_size=0.1)
!pip install pycaret
from pycaret.classification import *

experiment = setup(
    data = train 
    ,target = 'target'
    ,ignore_features = ['parent_protein_id', 'protein_seq', 'peptide_seq']
    ,normalize = True
)
compare_models()
from sklearn.metrics import roc_auc_score

best_models = ['et','catboost','xgboost','lightgbm','rf']

df_Results = pd.DataFrame(columns=['Classification', 'Dataset', 'Model', 'AUC'])

for m in best_models:
    
    print('-----------------------------------------------------')
    print('[START] - Processing model: ', m)
    print('-----------------------------------------------------')
    
    mo = create_model(m)
    
    print('-----------------------------------------------------')
    print('[START] - Tunning model: ', m)
    print('-----------------------------------------------------')
    
    tu = tune_model(mo)
    
    print('-----------------------------------------------------')
    print('[START] - Ensemble model: ', m)
    print('-----------------------------------------------------')
    
    en = ensemble_model(mo)
    
    mo_pred = predict_model(mo, test)
    mo_pred.dropna(inplace=True)
    
    tu_pred = predict_model(tu, test)
    tu_pred.dropna(inplace=True)
    
    en_pred = predict_model(en, test)
    en_pred.dropna(inplace=True)
    
    try:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Model', roc_auc_score(mo_pred['target'], mo_pred['Label'])]
    except:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Model', 'NA']
    try:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Tunned', roc_auc_score(tu_pred['target'], tu_pred['Label'])]
    except:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Tunned', 'NA']
    try:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Ensembled', roc_auc_score(en_pred['target'], en_pred['Label'])]
    except:
        df_Results.loc[len(df_Results)] = [m, 'Valid', 'Ensembled', 'NA']
    
    print('-----------------------------------------------------')
    print('[FINISHED] - Model: ', m)
    print('-----------------------------------------------------')
print(df_Results.sort_values(by=['Dataset', 'AUC'], ascending=False))
rf = create_model('rf')
rf_tunned = tune_model(rf)
plot_model(rf_tunned)
plot_model(rf_tunned, 'confusion_matrix')
plot_model(rf_tunned, 'threshold')
plot_model(rf_tunned, 'error')
plot_model(rf_tunned, 'class_report')
plot_model(rf_tunned, 'feature')
covid = pd.read_csv('/kaggle/input/epitope-prediction/input_covid.csv')

covid.head()
pred_covid = predict_model(rf_tunned, covid)

pred_covid
pred_covid.to_csv('submit.csv', index=False)