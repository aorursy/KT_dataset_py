import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

from datetime import datetime



%matplotlib inline
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
print("There are exists {} columns".format(len(df.columns)))
df.columns.tolist()
plt.rcParams["figure.figsize"] = (20,6) #change size of figure

sns.heatmap(df.isnull(), cbar=False)
def renameColumns(column):

    column = re.sub('[^a-z A-z 0-9]', '', column) #remove non-alphanumeric characters

    column = re.sub(' +', ' ', column) #replace multiple spaces to just one



    return column.strip().replace(' ', '_').lower() #apply snake_case pattern
%%time

df.rename(columns=renameColumns, inplace=True)
target_columns = ['sarscov2_exam_result']

identifier_column = 'patient_id'
%%time

columns_null_index = (df.isnull().sum()/df.shape[0]).where(lambda x : x==1).dropna().index
print("Null columns")

columns_null_index
%%time

df.drop(columns_null_index, axis=1, inplace=True)

print("{} columns where removed".format(len(columns_null_index)))
%%time

df.dtypes.groupby(df.dtypes).count()
%%time

non_numerical_columns = df.dtypes.where(lambda x: x=='object').dropna().index.tolist()

numerical_columns = df.dtypes.where(lambda x: x!='object').dropna().index.tolist()
%%time

non_numerical_columns.remove(identifier_column)



for column in non_numerical_columns:

    print(df.groupby(column).count().index)
%%time

df.describe()
df['sarscov2_exam_result'] = df['sarscov2_exam_result'].map({'negative': 0, 'positive': 1})

df['influenza_a_rapid_test'] = df['influenza_a_rapid_test'].map({'negative': 0, 'positive': 1})

df['influenza_b_rapid_test'] = df['influenza_b_rapid_test'].map({'negative': 0, 'positive': 1})

df['strepto_a'] = df['strepto_a'].map({'negative': 0, 'positive': 1})



detect_or_not_columns = [

    'respiratory_syncytial_virus','influenza_a','influenza_b','parainfluenza_1','coronavirusnl63','rhinovirusenterovirus','coronavirus_hku1','parainfluenza_3',

    'chlamydophila_pneumoniae','adenovirus','parainfluenza_4','coronavirus229e','coronavirusoc43','inf_a_h1n1_2009','bordetella_pertussis','metapneumovirus',

    'urine_protein', 'urine_hyaline_cylinders', 'urine_granular_cylinders', 'urine_yeasts', 'urine_esterase', 'urine_protein', 'urine_nitrite', 'urine_urobilinogen',

    'urine_bile_pigments', 'urine_ketone_bodies', 'parainfluenza_2', 'urine_hemoglobin'

]



df[detect_or_not_columns]= df[detect_or_not_columns].replace({'not_detected': 0, 'not_done': 0.5, 'detected': 1, 'absent': 2, 'normal': 3, 'present': 4})



df['urine_ph'] = df['urine_ph'].replace({"Não Realizado": 0}).astype('float64')

df['urine_crystals'] = df['urine_crystals'].map({'Ausentes': 0, 'Oxalato de Cálcio +++': 1, 'Oxalato de Cálcio -++': 2,

       'Urato Amorfo +++': 3, 'Urato Amorfo --+': 4})

df['urine_color'] = df['urine_color'].map({'citrus_yellow': 1, 'light_yellow': 2, 'orange': 3, 'yellow': 4})

df['urine_aspect'] = df['urine_aspect'].map({'altered_coloring': 1, 'clear': 2, 'cloudy': 3, 'lightly_cloudy': 4})



df['urine_leukocytes'] = df['urine_leukocytes'].replace({'<1000': 500}).astype('float64')

df.dtypes.groupby(df.dtypes).count()
df_target = pd.DataFrame(df[target_columns])



df_valid = df.fillna(999999999)



del df_valid[identifier_column]
plt.rcParams["figure.figsize"] = (20,6)

sns.heatmap(df.isnull(), cbar=False)
df[numerical_columns].describe()
plt.rcParams["figure.figsize"] = (15,5)

fig, (ax1, ax2) = plt.subplots(ncols=2)



sns.distplot(df[df['sarscov2_exam_result'] == 1]['patient_age_quantile'], hist=True, label='Positive', ax=ax2)

sns.distplot(df[df['sarscov2_exam_result'] == 0]['patient_age_quantile'], hist=True, label='Negative', ax=ax2)

sns.distplot(df['patient_age_quantile'], hist=False, label='Geral', ax=ax1)

plt.legend()

%%time

newDf = df[[identifier_column] + target_columns].groupby(target_columns).count().reset_index().replace({0: "Negative", 1: "Postive"})
%%time

b = sns.barplot(y="patient_id", data=newDf, x="sarscov2_exam_result")

b.set_title("SARS-Cov-2 Exam",fontsize=25)

b.set_xlabel("",fontsize=15)

b.set_ylabel("Quantity",fontsize=15)

# b.set_xticklabels(['Negative', 'Postive'])
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import accuracy_score

%%time





X = df_valid.drop(target_columns, axis=1)

Y = df_valid[target_columns]
model = RandomForestClassifier(n_estimators=200)
%%time

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=55)
%%time

model.fit(X_train, Y_train.values.ravel())
%%time

pred = model.predict(X_test)
print(classification_report(Y_test, pred))
print(confusion_matrix(Y_test, pred))
accuracy = accuracy_score(Y_test, pred)



print(f'Mean accuracy score: {accuracy:.3}')