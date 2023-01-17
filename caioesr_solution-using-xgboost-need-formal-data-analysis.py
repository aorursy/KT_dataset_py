import pandas as pd

import numpy as np
# lendo dataset

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()
df_aux = df.select_dtypes(include=['float32', 'float64']).apply(lambda x: x.fillna(x.median()),axis=0)

drop_columns = []

for i in df_aux.columns:

    drop_columns.append(i)

df = df.drop(drop_columns, axis = 1) 



df = pd.concat([df, df_aux], axis=1, sort=False)

df.head()
df['Influenza A'] = np.where(df['Influenza A'] == "detected",1,0)

df['Influenza B'] = np.where(df['Influenza B'] == "detected",1,0)

df['Influenza'] = np.where((df['Influenza A'] + df['Influenza B']) >= 1,1,0)

df['Parainfluenza 1'] = np.where(df['Parainfluenza 1'] == "detected",1,0)

df['Parainfluenza 2'] = np.where(df['Parainfluenza 2'] == "detected",1,0)

df['Parainfluenza 3'] = np.where(df['Parainfluenza 3'] == "detected",1,0)

df['Parainfluenza 4'] = np.where(df['Parainfluenza 4'] == "detected",1,0)

df['Parainfluenza'] = np.where((df['Parainfluenza 1']+ df['Parainfluenza 2'] + df['Parainfluenza 3']

                                + df['Parainfluenza 4']) >= 1,1,0)

df['Coronavirus229E'] = np.where(df['Coronavirus229E'] == "detected",1,0)

df['CoronavirusNL63'] = np.where(df['CoronavirusNL63'] == "detected",1,0)

df['Coronavirus HKU1'] = np.where(df['Coronavirus HKU1'] == "detected",1,0)

df['CoronavirusOC43'] = np.where(df['CoronavirusOC43'] == "detected",1,0)

df['Coronavirus'] = np.where((df['Coronavirus HKU1']+ df['Coronavirus229E'] + df['CoronavirusNL63']

                              + df['CoronavirusOC43']) >= 1,1,0)

df = df.drop(['Influenza A','Influenza B' ,'Parainfluenza 1','Parainfluenza 2','Parainfluenza 3','Parainfluenza 4'

             ,'Coronavirus229E','CoronavirusNL63','Coronavirus HKU1','CoronavirusOC43'], axis = 1) 
nan_pct = (df.isna().sum()/len(df))

nan_pct = dict(nan_pct)
columns_to_drop = []

for feature in nan_pct:

    if nan_pct[feature] > 0.9:

        columns_to_drop.append(feature)

df = df.drop(columns_to_drop, axis=1)
from category_encoders.one_hot import OneHotEncoder

oh = OneHotEncoder(cols= ['Respiratory Syncytial Virus','Rhinovirus/Enterovirus' ,'Chlamydophila pneumoniae'

                            ,'Metapneumovirus','Bordetella pertussis','Patient age quantile','Influenza A, rapid test'

                            ,'Influenza B, rapid test','Adenovirus','Inf A H1N1 2009'

                            ],use_cat_names=True)

oh.fit(df)

df = oh.transform(df)

df = df.replace({'positive':1,'negative':0})

df
# criando dataframes de treino e teste

from sklearn.model_selection import train_test_split



X = df.drop(['Patient ID','SARS-Cov-2 exam result'], axis=1)

y = df['SARS-Cov-2 exam result']



#20% test e 80% train

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.metrics import f1_score,accuracy_score

from sklearn.svm import SVC



svc = SVC(gamma='auto',random_state=42)

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)

print('f1_score:' , f1_score(y_pred_svc,y_test,average='weighted'))

print('accuracy:' , accuracy_score(y_pred_svc,y_test))
import xgboost as xgb

xgboost = xgb.XGBClassifier(learning_rate = 0.2

                            ,max_depth = 5

                            ,colsample_bytree = 0.9

                            ,n_estimators = 100

                            ,random_state=42

                            ,class_weight='balanced'

                           )



xgboost.fit(X_train,y_train)

y_pred_xg = xgboost.predict(X_test)

print('f1_score:' , f1_score(y_pred_xg,y_test,average='weighted'))

print('accuracy:' , accuracy_score(y_pred_xg,y_test))
patient_addmited_type = []

# vou criar um código formado pelo input em cada coluna para uma mesma linha

# por exemplo, se um paciente deu zero para ambos os tipos de admissão, então o código seria 000

# e ele receberia o valor 0 na coluna Patient addmited type

for i in df.index:

    code = str(df['Patient addmited to intensive care unit (1=yes, 0=no)'][i]) + str(df['Patient addmited to regular ward (1=yes, 0=no)'][i]) + str(df['Patient addmited to semi-intensive unit (1=yes, 0=no)'][i])

    if code == '000':

        patient_addmited_type.append(0)

    elif code == '100':

        patient_addmited_type.append(3)

    elif code == '010':

        patient_addmited_type.append(1)

    elif code == '001':

        patient_addmited_type.append(2)

df['Patient addmited type'] = patient_addmited_type
df = df.drop(['Patient addmited to intensive care unit (1=yes, 0=no)',

             'Patient addmited to regular ward (1=yes, 0=no)',

             'Patient addmited to semi-intensive unit (1=yes, 0=no)'], axis=1)

df.head()
X_2 = df.drop(['Patient ID','Patient addmited type'], axis=1)

y_2 = df['Patient addmited type']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2,test_size=0.2,random_state=42)
svc2 = SVC(gamma='auto',random_state=42)

svc2.fit(X_train_2, y_train_2)

y_pred_svc_2 = svc2.predict(X_test_2)

print('f1_score:' , f1_score(y_pred_svc_2,y_test_2,average='weighted'))

print('accuracy:' , accuracy_score(y_pred_svc_2,y_test_2))
xgboost2 = xgb.XGBClassifier(learning_rate = 0.2

                            ,max_depth = 5

                            ,colsample_bytree = 0.9

                            ,n_estimators = 100

                            ,random_state=42

                            ,class_weight='balanced'

                           )



xgboost2.fit(X_train_2,y_train_2)

y_pred_xg_2 = xgboost2.predict(X_test_2)

print('f1_score:' , f1_score(y_pred_xg_2,y_test_2,average='weighted'))

print('accuracy:' , accuracy_score(y_pred_xg_2,y_test_2))