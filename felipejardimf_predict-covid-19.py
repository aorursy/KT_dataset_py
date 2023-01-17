# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(style="whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
#iremos assumir o resultado positivo como 1 e negativo como 0

df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].replace('positive',1)

df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].replace('negative',0)

df = df.rename(columns = {'SARS-Cov-2 exam result':'target'})
v = df.columns

card=[]

for f in v:

    dist_values = df[f].value_counts().shape[0]

    f_dict = {

                'Features': f,

                'Cardinality': dist_values

            }

    card.append(f_dict)

card = pd.DataFrame(card, columns=['Features', 'Cardinality'])   
#removendo colunas com 100% de missing

card[card['Cardinality'] == 0 ]



df.drop(card['Features'][card['Cardinality'] == 0 ], axis=1, inplace=True)
#colunas removidas

card[card['Cardinality'] == 0 ]
df_types = df.dtypes.to_frame()

df_types.reset_index(level=0, inplace=True)

df_types = df_types.rename(columns={'index':'coluna',0:'tipo'})
#procurando por variáveis binárias

card['Features'][card['Cardinality'] <= 2]
#

df['Respiratory Syncytial Virus'] = df['Respiratory Syncytial Virus'].replace('not_detected',0)

df['Respiratory Syncytial Virus'] = df['Respiratory Syncytial Virus'].replace('detected',1)



df['Influenza A'] = df['Influenza A'].replace('not_detected',0)

df['Influenza A'] = df['Influenza A'].replace('detected',1)

                      

df['Influenza B'] = df['Influenza B'].replace('not_detected',0)

df['Influenza B'] = df['Influenza B'].replace('detected',1) 



df['Parainfluenza 1'] = df['Parainfluenza 1'].replace('not_detected',0)

df['Parainfluenza 1'] = df['Parainfluenza 1'].replace('detected',1) 



df['CoronavirusNL63'] = df['CoronavirusNL63'].replace('not_detected',0)

df['CoronavirusNL63'] = df['CoronavirusNL63'].replace('detected',1)



df['Rhinovirus/Enterovirus'] = df['Rhinovirus/Enterovirus'].replace('not_detected',0)

df['Rhinovirus/Enterovirus'] = df['Rhinovirus/Enterovirus'].replace('detected',1)



df['Coronavirus HKU1'] = df['Coronavirus HKU1'].replace('not_detected',0)

df['Coronavirus HKU1'] = df['Coronavirus HKU1'].replace('detected',1)



df['Parainfluenza 3'] = df['Parainfluenza 3'].replace('not_detected',0)

df['Parainfluenza 3'] = df['Parainfluenza 3'].replace('detected',1) 



df['Chlamydophila pneumoniae'] = df['Chlamydophila pneumoniae'].replace('not_detected',0)

df['Chlamydophila pneumoniae'] = df['Chlamydophila pneumoniae'].replace('detected',1)



df['Adenovirus'] = df['Adenovirus'].replace('not_detected',0)

df['Adenovirus'] = df['Adenovirus'].replace('detected',1)



df['Parainfluenza 4'] = df['Parainfluenza 4'].replace('not_detected',0)

df['Parainfluenza 4'] = df['Parainfluenza 4'].replace('detected',1) 



df['Coronavirus229E'] = df['Coronavirus229E'].replace('not_detected',0)

df['Coronavirus229E'] = df['Coronavirus229E'].replace('detected',1)



df['CoronavirusOC43'] = df['CoronavirusOC43'].replace('not_detected',0)

df['CoronavirusOC43'] = df['CoronavirusOC43'].replace('detected',1)



df['Inf A H1N1 2009'] = df['Inf A H1N1 2009'].replace('not_detected',0)

df['Inf A H1N1 2009'] = df['Inf A H1N1 2009'].replace('detected',1)



df['Bordetella pertussis'] =df['Bordetella pertussis'].replace('not_detected',0)

df['Bordetella pertussis'] =df['Bordetella pertussis'].replace('detected',1)



df['Metapneumovirus'] =df['Metapneumovirus'].replace('not_detected',0)

df['Metapneumovirus'] =df['Metapneumovirus'].replace('detected',1)



df['Parainfluenza 2'] = df['Parainfluenza 2'].replace('not_detected',0)



df['Influenza B, rapid test'] = df['Influenza B, rapid test'].replace('negative',0)

df['Influenza B, rapid test'] = df['Influenza B, rapid test'].replace('positive',1)



df['Influenza A, rapid test'] = df['Influenza A, rapid test'].replace('negative',0)

df['Influenza A, rapid test'] = df['Influenza A, rapid test'].replace('positive',1)





df['Fio2 (venous blood gas analysis)'] = df['Fio2 (venous blood gas analysis)'].replace('0.',0)

df['Myeloblasts'] = df['Myeloblasts'].replace('0.',0)





df['Urine - Esterase'] = df['Urine - Esterase'].replace('absent', 0)

df['Urine - Esterase'] = df['Urine - Esterase'].replace('not_done', np.nan)



df['Urine - Bile pigments'] = df['Urine - Bile pigments'].replace('absent', 0)

df['Urine - Bile pigments'] = df['Urine - Bile pigments'].replace('not_done', np.nan)



df['Urine - Ketone Bodies'] = df['Urine - Ketone Bodies'].replace('absent', 0)

df['Urine - Ketone Bodies'] = df['Urine - Ketone Bodies'].replace('not_done', np.nan)



#só exames com status 'not_done', iremos tirar de nossa base de treino.

del df['Urine - Nitrite']



#assumimos 'normal' como 0

df['Urine - Urobilinogen'] = df['Urine - Urobilinogen'].replace('normal',0)

df['Urine - Urobilinogen'] = df['Urine - Urobilinogen'].replace('not_done',np.nan)



df['Urine - Protein'] = df['Urine - Protein'].replace('absent',0)

df['Urine - Protein'] = df['Urine - Protein'].replace('not_done',np.nan)



df['Urine - Hyaline cylinders'] = df['Urine - Hyaline cylinders'].replace('absent',0)



df['Urine - Granular cylinders'] = df['Urine - Granular cylinders'].replace('absent',0)



df['Urine - Yeasts'] = df['Urine - Yeasts'].replace('absent',0)













v = df.columns

card2=[]

for f in v:

    dist_values = df[f].value_counts().shape[0]

    f_dict = {

                'Features': f,

                'Cardinality': dist_values

            }

    card2.append(f_dict)

card2 = pd.DataFrame(card2, columns=['Features', 'Cardinality'])   
#procurando mais binárias

card2['Features'][card2['Cardinality'] == 3].tolist()
#essa não é binária

df['Vitamin B12'].unique()
df['Strepto A'] = df['Strepto A'].replace('positive',1)

df['Strepto A'] = df['Strepto A'].replace('negative',0)

df['Strepto A'] = df['Strepto A'].replace('not_done',np.nan)



df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace('absent',0)

df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace('present',1)

df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace('not_done',np.nan)

card2['Features'][card2['Cardinality'] <= 3].tolist()
dataset_bin_nom = df[[

 'Patient addmited to regular ward (1=yes, 0=no)',

 'Patient addmited to semi-intensive unit (1=yes, 0=no)',

 'Patient addmited to intensive care unit (1=yes, 0=no)',

 'Respiratory Syncytial Virus',

 'Influenza A',

 'Influenza B',

 'Parainfluenza 1',

 'CoronavirusNL63',

 'Rhinovirus/Enterovirus',

 'Coronavirus HKU1',

 'Parainfluenza 3',

 'Chlamydophila pneumoniae',

 'Adenovirus',

 'Parainfluenza 4',

 'Coronavirus229E',

 'CoronavirusOC43',

 'Inf A H1N1 2009',

 'Bordetella pertussis',

 'Metapneumovirus',

 'Parainfluenza 2',

 'Influenza B, rapid test',

 'Influenza A, rapid test',

 'Strepto A',

 'Fio2 (venous blood gas analysis)',

 'Myeloblasts',

 'Urine - Esterase',

 'Urine - Hemoglobin',

 'Urine - Bile pigments',

 'Urine - Ketone Bodies',

 'Urine - Urobilinogen',

 'Urine - Protein',

 'Urine - Hyaline cylinders',

 'Urine - Granular cylinders',

 'Urine - Yeasts',

 'Urine - Aspect',

 'Urine - Crystals',

 'Urine - Color']]
df_nom = pd.get_dummies(dataset_bin_nom, 

                      columns=dataset_bin_nom.columns,

                      drop_first=False, 

                      prefix = dataset_bin_nom.columns,

                      dummy_na=True,

                      prefix_sep='_')
df['Urine - Leukocytes'] = df['Urine - Leukocytes'].replace('<1000','500')

df['Urine - pH'] = df['Urine - pH'].replace('Não Realizado',np.nan)

df['Urine - pH'] = df['Urine - pH'].astype('float')

df['Urine - Leukocytes'] = df['Urine - Leukocytes'].astype('float')
dataset_num = df[[

'Patient age quantile',

'Hematocrit',

'Hemoglobin',

'Platelets',

'Mean platelet volume ',

'Red blood Cells',

'Lymphocytes',

'Mean corpuscular hemoglobin concentration\xa0(MCHC)',

'Leukocytes',

'Basophils',

'Mean corpuscular hemoglobin (MCH)',

'Eosinophils',

'Mean corpuscular volume (MCV)',

'Monocytes',

'Red blood cell distribution width (RDW)',

'Serum Glucose',

'Neutrophils',

'Urea',

'Proteina C reativa mg/dL',

'Creatinine',

'Potassium',

'Sodium',

'Alanine transaminase',

'Aspartate transaminase',

'Gamma-glutamyltransferase\xa0',

'Total Bilirubin',

'Direct Bilirubin',

'Indirect Bilirubin',

'Alkaline phosphatase',

'Ionized calcium\xa0',

'Magnesium',

'pCO2 (venous blood gas analysis)',

'Urine - Leukocytes',

'Urine - pH',

'Hb saturation (venous blood gas analysis)',

'Base excess (venous blood gas analysis)',

'pO2 (venous blood gas analysis)',

'Total CO2 (venous blood gas analysis)',

'pH (venous blood gas analysis)',

'HCO3 (venous blood gas analysis)',

'Rods #',

'Segmented',

'Metamyelocytes',

'Myelocytes',

'Vitamin B12',

'Arterial Lactic Acid',

'Ferritin',

'Base excess (arterial blood gas analysis)',

'HCO3 (arterial blood gas analysis)',

'Albumin',

'pCO2 (arterial blood gas analysis)',

'pO2 (arterial blood gas analysis)',

'Phosphor',

'pH (arterial blood gas analysis)',

'Arteiral Fio2',

'Hb saturation (arterial blood gases)',

'Creatine phosphokinase\xa0(CPK)\xa0',

'ctO2 (arterial blood gas analysis)',

'International normalized ratio (INR)',

'Relationship (Patient/Normal)',

'Total CO2 (arterial blood gas analysis)',

'Urine - Density',

'Lipase dosage',

'Lactic Dehydrogenase',

'Urine - Red blood cells',

'Promyelocytes'

]]
null_num = dataset_num.isnull().sum().to_frame()
null_num = dataset_num.isnull().sum().to_frame()

null_num.reset_index(level=0, inplace=True)

null_num = null_num.rename(columns={'index':'coluna',0:'qtd'})


dataset_num.fillna(0,inplace=True)
from sklearn.preprocessing import StandardScaler



# Classe responável pela normalização

scaler = StandardScaler()



# Convertendo todas variáveis para tipo float (necessário para normalização)

df03 = dataset_num.astype(float)



scaled_features = scaler.fit_transform(df03)

df_num = pd.DataFrame(scaled_features, columns = df03.columns)



df_num.head(10)
abt = pd.merge(df_nom,df_num, left_index=True , right_index=True)
abt = abt.merge(df['target'], left_index=True, right_index=True)
abt.rename(columns={"Urine - Crystals_Oxalato de Cálcio +++":"UOCALCIO_PPP",

                     "Urine - Crystals_Oxalato de Cálcio -++":"UOCALCIO_NPP",

                    "Urine - Crystals_Urato Amorfo +++":"UCUA_PPP",

                    "Urine - Crystals_Urato Amorfo --+":"UCUA_NNP"},inplace=True)
#for lgbm porblem

abt.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in abt.columns]
from sklearn.model_selection import train_test_split



explicativas = abt.drop(['target'], axis=1)

resposta = abt["target"]



x_train, x_test, y_train, y_test = train_test_split(explicativas, resposta, test_size = 0.30, random_state = 0)
# Gradient Boosting Classifier 

from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(loss='exponential', 

                                 learning_rate=0.01,

                                 n_estimators=500, 

                                 subsample=1.0, 

                                 criterion='friedman_mse',

                                 min_samples_split=2, 

                                 min_samples_leaf=1,

                                 min_weight_fraction_leaf=0.0,

                                 max_depth=2,

                                 min_impurity_decrease=0.0, 

                                 min_impurity_split=None, 

                                 init=None, 

                                 random_state=None,

                                 max_features=None,

                                 verbose=0, 

                                 max_leaf_nodes=None, 

                                 warm_start=False,

                                 validation_fraction=0.2, 

                                 n_iter_no_change=None,

                                 tol=0.0001)



gbc.fit(x_train, y_train)



# Treino

y_pred_gbc_train = gbc.predict(x_train)

y_score_gbc_train = gbc.predict_proba(x_train)[:,1]



# Teste

y_pred_gbc_test = gbc.predict(x_test)

y_score_gbc_test = gbc.predict_proba(x_test)[:,1]

# 1) Cálculo da acurácia

from sklearn.metrics import accuracy_score



#Treino

acc_gbc_train = round(accuracy_score(y_pred_gbc_train, y_train) * 100, 2)



#Teste

acc_gbc_test = round(accuracy_score(y_pred_gbc_test, y_test) * 100, 2)



# 2) Cálculo da área sob curva ROC e Gini

from sklearn.metrics import roc_curve, auc



# Treino

fpr_gbc_train, tpr_gbc_train, thresholds = roc_curve(y_train, y_score_gbc_train)

roc_auc_gbc_train = 100*round(auc(fpr_gbc_train, tpr_gbc_train), 2)

gini_gbc_train = 100*round((2*roc_auc_gbc_train/100 - 1), 2)



# Teste

fpr_gbc_test, tpr_gbc_test, thresholds = roc_curve(y_test, y_score_gbc_test)

roc_auc_gbc_test = 100*round(auc(fpr_gbc_test, tpr_gbc_test), 2)

gini_gbc_test = 100*round((2*roc_auc_gbc_test/100 - 1), 2)





# 3) Gráfico da curva ROC

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(12,6))



lw = 2



plt.plot(fpr_gbc_train, tpr_gbc_train, color='blue',lw=lw, label='ROC (Treino = %0.0f)' % roc_auc_gbc_train)

plt.plot(fpr_gbc_test, tpr_gbc_test, color='darkorange',lw=lw, label='ROC (Teste = %0.0f)' % roc_auc_gbc_test)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Falso Positivo', fontsize=15)

plt.ylabel('Verdadeiro Positivo', fontsize=15)

plt.legend(loc="lower right")

plt.legend(fontsize=20) 

plt.title('Curva ROC - Gradient Boosting', fontsize=20)

plt.show()



print('Acurácia, Gini e Área Curva ROC (Base de Treino): ',acc_gbc_train, gini_gbc_train, roc_auc_gbc_train)

print('Acurácia, Gini e Área Curva ROC (Base de Teste): ',acc_gbc_test, gini_gbc_test, roc_auc_gbc_test)
feat_imp_gradient = pd.DataFrame(gbc.feature_importances_, columns={'col'})

feat_imp_gradient =pd.merge(pd.DataFrame(x_train.columns),feat_imp_gradient, right_index=True, left_index=True)

feat_imp_gradient =feat_imp_gradient.sort_values('col',ascending=False)

feat_imp_gradient.rename(columns={0:'feature'},inplace=True)

ax = sns.barplot(x="col", y='feature',  data=feat_imp_gradient.iloc[0:20] ).set_title('Gradient Boosting Feature Importance')


from lightgbm import LGBMClassifier



lgbm= LGBMClassifier(boosting_type='gbrt',

                        num_leaves=3,

                        max_depth=2,

                        learning_rate=0.278,

                        n_estimators=1000,

                        subsample_for_bin=200000,

                        objective='binary',

                        class_weight=None,

                        is_unbalance = True,

                        min_split_gain=2,

                        min_child_weight=0.0001,

                        min_child_samples=4,

                        subsample=1.0, 

                        subsample_freq=0,

                        colsample_bytree=1.0, 

                        reg_alpha=0.5,

                        reg_lambda=0.7, 

                        random_state=37,

                        n_jobs=-1,

                        silent=True,

                        importance_type='gain'

                        )



lgbm.fit(x_train, y_train)



# Treino

y_pred_lgbm_train = lgbm.predict(x_train)

y_score_lgbm_train = lgbm.predict_proba(x_train)[:,1]



# Teste

y_pred_lgbm_test = lgbm.predict(x_test)

y_score_lgbm_test = lgbm.predict_proba(x_test)[:,1]

# 1) Cálculo da acurácia

from sklearn.metrics import accuracy_score



#Treino

acc_lgbm_train = round(accuracy_score(y_pred_lgbm_train, y_train) * 100, 2)



#Teste

acc_lgbm_test = round(accuracy_score(y_pred_lgbm_test, y_test) * 100, 2)



# 2) Cálculo da área sob curva ROC e Gini

from sklearn.metrics import roc_curve, auc



# Treino

fpr_lgbm_train, tpr_lgbm_train, thresholds = roc_curve(y_train, y_score_lgbm_train)

roc_auc_lgbm_train = 100*auc(fpr_lgbm_train, tpr_lgbm_train)

gini_lgbm_train = 100*round((2*roc_auc_lgbm_train/100 - 1), 2)



# Teste

fpr_lgbm_test, tpr_lgbm_test, thresholds = roc_curve(y_test, y_score_lgbm_test)

roc_auc_lgbm_test = 100*auc(fpr_lgbm_test, tpr_lgbm_test)

gini_lgbm_test = 100*round((2*roc_auc_lgbm_test/100 - 1), 2)





# 3) Gráfico da curva ROC

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(12,6))



lw = 2



plt.plot(fpr_lgbm_train, tpr_lgbm_train, color='blue',lw=lw, label='ROC (Treino = %0.0f)' % roc_auc_lgbm_train)

plt.plot(fpr_lgbm_test, tpr_lgbm_test, color='darkorange',lw=lw, label='ROC (Teste = %0.0f)' % roc_auc_lgbm_test)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Falso Positivo', fontsize=15)

plt.ylabel('Verdadeiro Positivo', fontsize=15)

plt.legend(loc="lower right")

plt.legend(fontsize=20) 

plt.title('Curva ROC Covid_Hosp', fontsize=20)

plt.show()



print('Acurácia, Gini e Área Curva ROC (Base de Treino): ',acc_lgbm_train, gini_lgbm_train, roc_auc_lgbm_train)

print('Acurácia, Gini e Área Curva ROC (Base de Teste): ',acc_lgbm_test, gini_lgbm_test, roc_auc_lgbm_test)
feat_imp_lgbm = pd.DataFrame(lgbm.feature_importances_, columns={'col'})
feat_imp_lgbm =pd.merge(pd.DataFrame(x_train.columns),feat_imp_lgbm, right_index=True, left_index=True)
feat_imp_lgbm =feat_imp_lgbm.sort_values('col',ascending=False)
feat_imp_lgbm.rename(columns={0:'feature'},inplace=True)
ax = sns.barplot(x="col", y='feature',  data=feat_imp_lgbm[feat_imp_lgbm['col'] > 20.0] ).set_title('LightGBM Feature Importance')


from xgboost import XGBClassifier

xgb = XGBClassifier(base_score=0.2, colsample_bylevel=1, colsample_bytree=1,

       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=1,

       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,

       objective='binary:logistic', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=0, silent=True, subsample=1)

xgb.fit(x_train, y_train)





# Treino

y_pred_xgb_train = xgb.predict(x_train)

y_score_xgb_train = xgb.predict_proba(x_train)[:,1]



# Teste

y_pred_xgb_test = xgb.predict(x_test)

y_score_xgb_test = xgb.predict_proba(x_test)[:,1]

# 1) Cálculo da acurácia

from sklearn.metrics import accuracy_score



#Treino

acc_xgb_train = round(accuracy_score(y_pred_xgb_train, y_train) * 100, 2)



#Teste

acc_xgb_test = round(accuracy_score(y_pred_xgb_test, y_test) * 100, 2)



# 2) Cálculo da área sob curva ROC e Gini

from sklearn.metrics import roc_curve, auc



# Treino

fpr_xgb_train, tpr_xgb_train, thresholds = roc_curve(y_train, y_score_xgb_train)

roc_auc_xgb_train = 100*round(auc(fpr_xgb_train, tpr_xgb_train), 2)

gini_xgb_train = 100*round((2*roc_auc_xgb_train/100 - 1), 2)



# Teste

fpr_xgb_test, tpr_xgb_test, thresholds = roc_curve(y_test, y_score_xgb_test)

roc_auc_xgb_test = 100*round(auc(fpr_xgb_test, tpr_xgb_test), 2)

gini_xgb_test = 100*round((2*roc_auc_xgb_test/100 - 1), 2)





# 3) Gráfico da curva ROC

import matplotlib.pyplot as plt



plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(12,6))



lw = 2



plt.plot(fpr_xgb_train, tpr_xgb_train, color='blue',lw=lw, label='ROC (Treino = %0.0f)' % roc_auc_xgb_train)

plt.plot(fpr_xgb_test, tpr_xgb_test, color='darkorange',lw=lw, label='ROC (Teste = %0.0f)' % roc_auc_xgb_test)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Falso Positivo', fontsize=15)

plt.ylabel('Verdadeiro Positivo', fontsize=15)

plt.legend(loc="lower right")

plt.legend(fontsize=20) 

plt.title('Curva ROC Covid_Hosp', fontsize=20)

plt.show()



print('Acurácia, Gini e Área Curva ROC (Base de Treino): ',acc_xgb_train, gini_xgb_train, roc_auc_xgb_train)

print('Acurácia, Gini e Área Curva ROC (Base de Teste): ',acc_xgb_test, gini_xgb_test, roc_auc_xgb_test)
feat_imp_xgb = pd.DataFrame(xgb.feature_importances_, columns={'col'})
feat_imp_xgb =pd.merge(pd.DataFrame(x_train.columns),feat_imp_xgb, right_index=True, left_index=True)
feat_imp_xgb= feat_imp_xgb.sort_values('col',ascending=False)
feat_imp_xgb.rename(columns={0:'feature'},inplace=True)
ax = sns.barplot(x="col", y='feature',data=feat_imp_xgb.iloc[0:20]  ).set_title('XGBoost Feature Importance')