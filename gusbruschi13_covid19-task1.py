import pandas as pd
import io
import requests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import gc
import random
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
# Import dataset

dataset = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
dataset.shape
dataset.head()
dataset.columns
## Convert ID

dataset["id"] = dataset["Patient ID"].values
dataset = dataset.drop(columns=["Patient ID"])
## Convert Target

dataset['target'] = np.where(dataset['SARS-Cov-2 exam result']=='negative',0,1)
dataset['target'] = dataset['target'].astype(np.float64)
dataset = dataset.drop(columns=["SARS-Cov-2 exam result"])
#### Generate Metadata Function

def GenerateMetadata(train,var_id,targetname): 
    print('Running metadata...')
    
    for ids in var_id:
        print('Renaming ---> ', ids,'to ---> ', 'ID_'+ids)
        train = train.rename(columns={ids: 'ID_'+ids})
   
    train = train.rename(columns={targetname: 'target'})
    # Verifying type of columns
    t = []
    for i in train.columns:
            t.append(train[i].dtype)

    n = []
    for i in train.columns:
            n.append(i)

    aux_t = pd.DataFrame(data=t,columns=["Type"])
    aux_n = pd.DataFrame(data=n,columns=["Features"])
    df_tipovars = pd.merge(aux_n, aux_t, left_index=True, right_index=True) 

    data = []
    for f in train.columns:
        # Defining variable roles:
        if f == 'target':
            role = 'target'
        elif f[0:3] == 'ID_':
            role = 'id'
        else:
            role = 'input'

        # Defining variable types: nominal, ordinal, binary ou interval
        if f == 'target':
            level = 'binary'
        if train[f].dtype == 'object' or f == 'id': 
            level = 'nominal'
        elif train[f].dtype in ['float','float64'] :
            level = 'interval'
        elif train[f].dtype in ['int','int64','int32'] :
            level = 'ordinal'
        else:
            level = 'NA'

        # Remove IDs
        keep = True
        if f[0:3] == 'ID_':
            keep = False

        #  Defining the type of input table variables
        dtype = train[f].dtype

        # Metadata list
        f_dict = {
            'Features': f,
            'Role': role,
            'Level': level,
            'Keep': keep,
            'Type': dtype
        }
        data.append(f_dict)

    meta = pd.DataFrame(data, columns=['Features', 'Role', 'Level', 'Keep', 'Type'])

    # Cardinality of columns
    card = []

    v = train.columns
    for f in v:
        dist_values = train[f].value_counts().shape[0]
        f_dict = {
                'Features': f,
                'Cardinality': dist_values
            }
        card.append(f_dict)

    card = pd.DataFrame(card, columns=['Features', 'Cardinality'])

    metadados_train = pd.merge(meta, card, on='Features')
    print('Metadada successfully completed')
    return metadados_train,train 
lista_ids = ['id']
targetname = 'target'
metadados,abt_desenv_01 = GenerateMetadata(dataset,lista_ids,targetname)
metadados
metadados['Type'].unique()
### Convert numbers to "float64" and categorical to "str"

numeric_list = metadados[((metadados.Level  == 'ordinal')|(metadados.Level == 'interval')) & (metadados.Role == 'input')]
category_list = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input')]

numeric_list = list(numeric_list['Features'].values)
category_list = list(category_list['Features'].values)
abt_desenv_02 = abt_desenv_01[numeric_list].astype(np.float64)
abt_desenv_03 = pd.merge(abt_desenv_02, abt_desenv_01[category_list].astype(np.str), left_index=True, right_index=True)
abt_desenv_03.shape
abt_desenv_03['ID_id'] = abt_desenv_01['ID_id'].values
abt_desenv_03['target'] = abt_desenv_01['target'].values
abt_desenv_03.shape
abt_desenv_03.head()
def DataPrep(metadados,input_df,var_id,targetname):
    
    print('Starting data preparation ...')
    
    #-------------- Handling missing of numeric columns -----------------
    input_df.rename(columns={var_id: 'id', targetname: 'target'}, inplace=True)
    df_00 = input_df
    targetname = 'target'
    print('Executing')
    
    #--------- Numeric Features --------------------
    vars_numericas_df = metadados[((metadados.Level  == 'ordinal')|(metadados.Level == 'interval')) & (metadados.Role == 'input')]
    lista_vars_numericas = list(vars_numericas_df['Features'])
    df01 = df_00[lista_vars_numericas]
    df01 = df01.fillna(0)
    df01 = df01.round(4)
    
    print('Missings done')

    #--------- Nominal Features - Low Cardinality --------------------
    vars_char_baix_cardin_df = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input') & (metadados.Cardinality <= 50)]
    lista_char_baix_cardin_df = list(vars_char_baix_cardin_df['Features'])
    
    df_00[lista_char_baix_cardin_df].apply(lambda x: x.fillna(x.mode, inplace=True))
    df02 = df_00[lista_char_baix_cardin_df]
    
    df03 = pd.get_dummies(df02,columns=lista_char_baix_cardin_df,drop_first=True,
                          prefix=lista_char_baix_cardin_df,prefix_sep='_')
    print('Dummifications done')    
    
    #--------- Nominal Features - High Cardinality --------------------
    vars_char_alta_cardin_df = metadados[(metadados.Level  == 'nominal') & (metadados.Role == 'input') & (metadados.Cardinality > 50)]
    lista_char_alta_cardin_df = list(vars_char_alta_cardin_df['Features'])
    
    df_00[lista_char_alta_cardin_df].apply(lambda x: x.fillna(x.mode, inplace=True)) 
    df04 = df_00[lista_char_alta_cardin_df]

    def MultiLabelEncoder(columnlist,dataframe):
        for i in columnlist:
            labelencoder_X=LabelEncoder()
            dataframe[i]=labelencoder_X.fit_transform(dataframe[i])

    MultiLabelEncoder(lista_char_alta_cardin_df,df04)
    print('Label Encodings done')
    
    #---------- Checking IDs -----------------------
    vars_ids_df = metadados[(metadados.Role  == 'id')]
    lista_ids = list(vars_ids_df['Features'])

    df1_3 = pd.merge(df01, df03, left_index=True, right_index=True)
    df1_3_4 = pd.merge(df1_3, df04, left_index=True, right_index=True)
    
    lista_vars_keep = lista_ids + [targetname]
    
    df_out = pd.merge(input_df[lista_vars_keep], df1_3_4, left_index=True, right_index=True)    
    
    print('Data Preparation Sucess')
    
    return df_out
abt_desenv_04 = DataPrep(metadados, abt_desenv_03,'id','target')
abt_desenv_04.shape
abt_desenv_04.head()
abt_desenv_04.isnull().sum()
df_to_select = abt_desenv_04.drop(columns=['ID_id'])
Y = df_to_select['target']
X = df_to_select.drop(columns='target', axis=1)
correlated_features = set()
correlation_matrix = X.corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.875:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
correlated_features
X = X.drop(columns=correlated_features)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

rfc = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=10, random_state=12345)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(15), scoring='accuracy')
rfecv.fit(X, Y)
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation - Random Forest', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show(), print('Optimal number of features: {}'.format(rfecv.n_features_))
rfecvcv_cols = X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

df_rfecv = pd.DataFrame()
df_rfecv['attr'] = X.columns
df_rfecv['importance'] = rfecv.estimator_.feature_importances_
df_rfecv = df_rfecv.sort_values(by='importance', ascending=False)

plt.figure(figsize=(16, 14))
plt.barh(y=df_rfecv['attr'], width=df_rfecv['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()
rfecvcv_cols
X.columns
# Split Train and Test

from sklearn.model_selection import train_test_split

explicativas = X
resposta = Y

x_train, x_test, y_train, y_test = train_test_split(explicativas,
                                                    resposta,
                                                    test_size = 0.2,
                                                    random_state = 666)
# Gradient Boosting 

from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(123456)

gbc = GradientBoostingClassifier(loss='exponential', 
                                 learning_rate=0.05,
                                 n_estimators=500, 
                                 subsample=1.0,
                                 random_state=123456)

gbc.fit(x_train, y_train)

# Train
y_pred_gbc_train = gbc.predict(x_train)
y_score_gbc_train = gbc.predict_proba(x_train)[:,1]

# Test
y_pred_gbc_test = gbc.predict(x_test)
y_score_gbc_test = gbc.predict_proba(x_test)[:,1]
# 1) Accuracy
from sklearn.metrics import accuracy_score

#Train
acc_gbc_train = round(accuracy_score(y_pred_gbc_train, y_train) * 100, 2)

#Test
acc_gbc_test = round(accuracy_score(y_pred_gbc_test, y_test) * 100, 2)

# 2) AUC ROC and Gini
from sklearn.metrics import roc_curve, auc

# Train
fpr_gbc_train, tpr_gbc_train, thresholds = roc_curve(y_train, y_score_gbc_train)
roc_auc_gbc_train = 100*round(auc(fpr_gbc_train, tpr_gbc_train), 2)
gini_gbc_train = 100*round((2*roc_auc_gbc_train/100 - 1), 2)

# Test
fpr_gbc_test, tpr_gbc_test, thresholds = roc_curve(y_test, y_score_gbc_test)
roc_auc_gbc_test = 100*round(auc(fpr_gbc_test, tpr_gbc_test), 2)
gini_gbc_test = 100*round((2*roc_auc_gbc_test/100 - 1), 2)


# 3) ROC Curve graph
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,6))

plt.plot(fpr_gbc_train, tpr_gbc_train, color='blue',lw=2, label='ROC (Train = %0.0f)' % roc_auc_gbc_train)
plt.plot(fpr_gbc_test, tpr_gbc_test, color='green',lw=2, label='ROC (Test = %0.0f)' % roc_auc_gbc_test)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive', fontsize=14)
plt.ylabel('True Positive', fontsize=14)
plt.legend(loc="lower right")
plt.legend(fontsize=20) 
plt.title('ROC Curve - Gradient Boosting - Covid19', fontsize=22)
plt.show()

print('Accuracy, Gini and ROC Curve Area (Train): ',acc_gbc_train, gini_gbc_train, roc_auc_gbc_train)
print('Accuracy, Gini and ROC Curve Area (Test): ',acc_gbc_test, gini_gbc_test, roc_auc_gbc_test)
# XG Boost

import xgboost as xgb

np.random.seed(123456)

xgb = xgb.XGBClassifier(loss='exponential',
                        learning_rate=0.05,
                        n_estimators=500,
                        subsample=1.0,
                        random_state=123456)

xgb.fit(x_train, y_train)

# Train
y_pred_xgb_train = xgb.predict(x_train)
y_score_xgb_train = xgb.predict_proba(x_train)[:,1]

# Test
y_pred_xgb_test = xgb.predict(x_test)
y_score_xgb_test = xgb.predict_proba(x_test)[:,1]
# 1) Accuracy
from sklearn.metrics import accuracy_score

#Train
acc_xgb_train = round(accuracy_score(y_pred_xgb_train, y_train) * 100, 2)

#Test
acc_xgb_test = round(accuracy_score(y_pred_xgb_test, y_test) * 100, 2)

# 2) AUC ROC and Gini
from sklearn.metrics import roc_curve, auc

# Train
fpr_xgb_train, tpr_xgb_train, thresholds = roc_curve(y_train, y_score_xgb_train)
roc_auc_xgb_train = 100*round(auc(fpr_xgb_train, tpr_xgb_train), 2)
gini_xgb_train = 100*round((2*roc_auc_xgb_train/100 - 1), 2)

# Test
fpr_xgb_test, tpr_xgb_test, thresholds = roc_curve(y_test, y_score_xgb_test)
roc_auc_xgb_test = 100*round(auc(fpr_xgb_test, tpr_xgb_test), 2)
gini_xgb_test = 100*round((2*roc_auc_xgb_test/100 - 1), 2)


# 3) ROC Curve graph
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,6))

plt.plot(fpr_xgb_train, tpr_xgb_train, color='blue',lw=2, label='ROC (Train = %0.0f)' % roc_auc_xgb_train)
plt.plot(fpr_xgb_test, tpr_xgb_test, color='green',lw=2, label='ROC (Test = %0.0f)' % roc_auc_xgb_test)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive', fontsize=14)
plt.ylabel('True Positive', fontsize=14)
plt.legend(loc="lower right")
plt.legend(fontsize=20) 
plt.title('ROC Curve - XG Boost - Covid19', fontsize=22)
plt.show()

print('Accuracy, Gini and ROC Curve Area (Train): ',acc_xgb_train, gini_xgb_train, roc_auc_xgb_train)
print('Accuracy, Gini and ROC Curve Area (Test): ',acc_xgb_test, gini_xgb_test, roc_auc_xgb_test)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

np.random.seed(123456)

rndforest = RandomForestClassifier(n_estimators=30,
                                   criterion='gini',
                                   max_depth=10,
                                   min_samples_split=2, 
                                   min_samples_leaf=1,
                                   n_jobs=100,
                                   random_state=123456)

rndforest.fit(x_train, y_train)

# Train
y_pred_rndforest_train = rndforest.predict(x_train)
y_score_rndforest_train = rndforest.predict_proba(x_train)[:,1]

# Test
y_pred_rndforest_test = rndforest.predict(x_test)
y_score_rndforest_test = rndforest.predict_proba(x_test)[:,1]
# 1) Accuracy
from sklearn.metrics import accuracy_score

#Train
acc_rndforest_train = round(accuracy_score(y_pred_rndforest_train, y_train) * 100, 2)

#Test
acc_rndforest_test = round(accuracy_score(y_pred_rndforest_test, y_test) * 100, 2)

# 2) AUC ROC and Gini
from sklearn.metrics import roc_curve, auc

# Train
fpr_rndforest_train, tpr_rndforest_train, thresholds = roc_curve(y_train, y_score_rndforest_train)
roc_auc_rndforest_train = 100*round(auc(fpr_rndforest_train, tpr_rndforest_train), 2)
gini_rndforest_train = 100*round((2*roc_auc_rndforest_train/100 - 1), 2)

# Test
fpr_rndforest_test, tpr_rndforest_test, thresholds = roc_curve(y_test, y_score_rndforest_test)
roc_auc_rndforest_test = 100*round(auc(fpr_rndforest_test, tpr_rndforest_test), 2)
gini_rndforest_test = 100*round((2*roc_auc_rndforest_test/100 - 1), 2)


# 3) ROC Curve graph
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,6))

plt.plot(fpr_rndforest_train, tpr_rndforest_train, color='blue',lw=2, label='ROC (Train = %0.0f)' % roc_auc_rndforest_train)
plt.plot(fpr_rndforest_test, tpr_rndforest_test, color='green',lw=2, label='ROC (Test = %0.0f)' % roc_auc_rndforest_test)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive', fontsize=14)
plt.ylabel('True Positive', fontsize=14)
plt.legend(loc="lower right")
plt.legend(fontsize=20) 
plt.title('ROC Curve - Random Forest - Covid19', fontsize=22)
plt.show()

print('Accuracy, Gini and ROC Curve Area (Train): ',acc_rndforest_train, gini_rndforest_train, roc_auc_rndforest_train)
print('Accuracy, Gini and ROC Curve Area (Test): ',acc_rndforest_test, gini_rndforest_test, roc_auc_rndforest_test)
# Neural Networks

from sklearn.neural_network import MLPClassifier

clf1 = MLPClassifier(solver='adam', activation='logistic', alpha=1e-3, hidden_layer_sizes=(40, 4), random_state=10)
clf1.fit(x_train, y_train)
# Train
y_pred_nn1_train = clf1.predict(x_train)
y_score_nn1_train = clf1.predict_proba(x_train)[:,1]
# Test
y_pred_nn1_test = clf1.predict(x_test)
y_score_nn1_test = clf1.predict_proba(x_test)[:,1]

clf2 = MLPClassifier(solver='adam', activation='tanh', alpha=1e-3, hidden_layer_sizes=(40, 4), random_state=10)
clf2.fit(x_train, y_train)
# Train
y_pred_nn2_train = clf2.predict(x_train)
y_score_nn2_train = clf2.predict_proba(x_train)[:,1]
# Test
y_pred_nn2_test = clf2.predict(x_test)
y_score_nn2_test = clf2.predict_proba(x_test)[:,1]
# 1) Accuracy
from sklearn.metrics import accuracy_score

#Train
acc_nn1_train = round(accuracy_score(y_pred_nn1_train, y_train) * 100, 2)

#Test
acc_nn1_test = round(accuracy_score(y_pred_nn1_test, y_test) * 100, 2)

# 2) AUC ROC and Gini
from sklearn.metrics import roc_curve, auc

# Train
fpr_nn1_train, tpr_nn1_train, thresholds = roc_curve(y_train, y_score_nn1_train)
roc_auc_nn1_train = 100*round(auc(fpr_nn1_train, tpr_nn1_train), 2)
gini_nn1_train = 100*round((2*roc_auc_nn1_train/100 - 1), 2)

# Test
fpr_nn1_test, tpr_nn1_test, thresholds = roc_curve(y_test, y_score_nn1_test)
roc_auc_nn1_test = 100*round(auc(fpr_nn1_test, tpr_nn1_test), 2)
gini_nn1_test = 100*round((2*roc_auc_nn1_test/100 - 1), 2)


# 3) ROC Curve graph
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,6))

plt.plot(fpr_nn1_train, tpr_nn1_train, color='blue',lw=2, label='ROC (Train = %0.0f)' % roc_auc_nn1_train)
plt.plot(fpr_nn1_test, tpr_nn1_test, color='green',lw=2, label='ROC (Test = %0.0f)' % roc_auc_nn1_test)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive', fontsize=14)
plt.ylabel('True Positive', fontsize=14)
plt.legend(loc="lower right")
plt.legend(fontsize=20) 
plt.title('ROC Curve - Neural Network 1 - Covid19', fontsize=22)
plt.show()

print('Accuracy, Gini and ROC Curve Area (Train): ',acc_nn1_train, gini_nn1_train, roc_auc_nn1_train)
print('Accuracy, Gini and ROC Curve Area (Test): ',acc_nn1_test, gini_nn1_test, roc_auc_nn1_test)
# 1) Accuracy
from sklearn.metrics import accuracy_score

#Train
acc_nn2_train = round(accuracy_score(y_pred_nn2_train, y_train) * 100, 2)

#Test
acc_nn2_test = round(accuracy_score(y_pred_nn2_test, y_test) * 100, 2)

# 2) AUC ROC and Gini
from sklearn.metrics import roc_curve, auc

# Train
fpr_nn2_train, tpr_nn2_train, thresholds = roc_curve(y_train, y_score_nn2_train)
roc_auc_nn2_train = 100*round(auc(fpr_nn2_train, tpr_nn2_train), 2)
gini_nn2_train = 100*round((2*roc_auc_nn2_train/100 - 1), 2)

# Test
fpr_nn2_test, tpr_nn2_test, thresholds = roc_curve(y_test, y_score_nn2_test)
roc_auc_nn2_test = 100*round(auc(fpr_nn2_test, tpr_nn2_test), 2)
gini_nn2_test = 100*round((2*roc_auc_nn2_test/100 - 1), 2)


# 3) ROC Curve graph
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,6))

plt.plot(fpr_nn2_train, tpr_nn2_train, color='blue',lw=2, label='ROC (Train = %0.0f)' % roc_auc_nn2_train)
plt.plot(fpr_nn2_test, tpr_nn2_test, color='green',lw=2, label='ROC (Test = %0.0f)' % roc_auc_nn2_test)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive', fontsize=14)
plt.ylabel('True Positive', fontsize=14)
plt.legend(loc="lower right")
plt.legend(fontsize=20) 
plt.title('ROC Curve - Neural Network 2 - Covid19', fontsize=22)
plt.show()

print('Accuracy, Gini and ROC Curve Area (Train): ',acc_nn2_train, gini_nn2_train, roc_auc_nn2_train)
print('Accuracy, Gini and ROC Curve Area (Test): ',acc_nn2_test, gini_nn2_test, roc_auc_nn2_test)
