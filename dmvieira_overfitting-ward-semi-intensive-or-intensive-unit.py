# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectFromModel

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import *

from tpot import TPOTClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")

df.head()
df.describe()
null_series = df.isnull().sum()
null_series[null_series > 0]
good_cols = null_series[null_series < df.shape[0]/20].reset_index()

good_cols.head(10)
df_goods = df[good_cols["index"]]

df_goods.head()
df_goods.describe(include="all")
sn.heatmap(df_goods.corr(), annot=True)

plt.show()
corr_matrix = df.corr().abs()



#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

                 .stack()

                 .sort_values(ascending=False))

sol[sol > 0.5]
df.groupby("Patient addmited to regular ward (1=yes, 0=no)").count()
df.groupby("Patient addmited to intensive care unit (1=yes, 0=no)").count()
df.groupby("Patient addmited to semi-intensive unit (1=yes, 0=no)").count()
df['Urine - pH'].replace('Não Realizado', np.nan, inplace=True)
df['Urine - Leukocytes'].replace('<1000', '999', inplace=True)
df['Urine - pH'] = df['Urine - pH'].astype("float64")
df['Urine - Leukocytes'] = df['Urine - Leukocytes'].astype("float64")
df.dtypes[(df.dtypes == "object")].index
df_nop = df.drop([

    "Patient ID",

    'SARS-Cov-2 exam result'

], axis=1)
df_result = pd.concat([df_nop, pd.get_dummies(df_nop[df_nop.dtypes[(df_nop.dtypes == "object")].index])], axis=1).drop(

    df_nop.dtypes[(df_nop.dtypes == "object")].index, axis=1).drop([

    'Respiratory Syncytial Virus_not_detected',

    'Influenza A_not_detected',

    'Influenza B_not_detected',

    'Parainfluenza 1_not_detected',

    'CoronavirusNL63_not_detected',

    'Rhinovirus/Enterovirus_not_detected',

    'Coronavirus HKU1_not_detected',

    'Parainfluenza 3_not_detected',

    'Chlamydophila pneumoniae_not_detected',

    'Adenovirus_not_detected',

    'Parainfluenza 4_not_detected',

    'Coronavirus229E_not_detected',

    'CoronavirusOC43_not_detected',

    'Inf A H1N1 2009_not_detected',

    'Bordetella pertussis_not_detected',

    'Metapneumovirus_not_detected',

    'Influenza B, rapid test_negative',

    'Influenza A, rapid test_negative',

    'Urine - Esterase_not_done',

    'Urine - Hemoglobin_not_done',

    'Urine - Hemoglobin_absent',

    'Strepto A_not_done',

    'Urine - Bile pigments_not_done',

    'Urine - Ketone Bodies_not_done',

    'Urine - Nitrite_not_done',

    'Urine - Urobilinogen_not_done',

    'Urine - Protein_not_done'

], axis=1)
list(df_result.columns)
def make_pipeline(x, y, model=RandomForestClassifier(random_state=42, n_estimators=5)):



    pipeline = Pipeline([

        ('inputer', SimpleImputer(missing_values=np.nan, strategy='median')),

        ('normalizer', StandardScaler()),

        ('feature', SelectFromModel(RandomForestClassifier(random_state=42, n_estimators=5))),

        ('clf', model)

    ])

    sampler = RandomUnderSampler(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(x, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled,

                                                   y_resampled,

                                                   test_size = 0.3,

                                                   random_state = 42)

    return (pipeline.fit(X_train, y_train), X_test, y_test)
df_result_final = df_result.drop([

    'Patient addmited to semi-intensive unit (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)'

], axis=1)

x, y = (df_result_final.drop('Patient addmited to regular ward (1=yes, 0=no)', axis=1), df_result_final['Patient addmited to regular ward (1=yes, 0=no)'])
rfmodel, X_test, y_test = make_pipeline(x, y)
rfmodel.score(X_test, y_test)
print(classification_report(rfmodel.predict(X_test), y_test))
tpot = TPOTClassifier(generations=1, verbosity=2, random_state=42)

tmodel, X_test, y_test = make_pipeline(x, y, tpot)
tmodel.score(X_test, y_test)
print(classification_report(tmodel.predict(X_test), y_test))
tmodel["clf"].export("best_model.py")
with open("best_model.py") as best_model:

    print(best_model.read())
from sklearn.ensemble import GradientBoostingClassifier

best_model = GradientBoostingClassifier(

    learning_rate=0.01, max_depth=10, max_features=0.3, min_samples_leaf=7, min_samples_split=4, n_estimators=5, subsample=0.7000000000000001)

bmodel, X_test, y_test = make_pipeline(x, y, best_model)
bmodel.score(X_test, y_test)
print(classification_report(bmodel.predict(X_test), y_test))
df_result_final = df_result.drop([

    'Patient addmited to regular ward (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)'

], axis=1)

x, y = (df_result_final.drop('Patient addmited to semi-intensive unit (1=yes, 0=no)', axis=1), df_result_final['Patient addmited to semi-intensive unit (1=yes, 0=no)'])
rfmodel, X_test, y_test = make_pipeline(x, y)

print(rfmodel.score(X_test, y_test))

print(classification_report(rfmodel.predict(X_test), y_test))
from sklearn.ensemble import GradientBoostingClassifier

best_model = GradientBoostingClassifier(

    learning_rate=0.01, max_depth=10, max_features=0.3, min_samples_leaf=7, min_samples_split=4, n_estimators=5, subsample=0.7000000000000001)

bmodel, X_test, y_test = make_pipeline(x, y, best_model)

print(bmodel.score(X_test, y_test))

print(classification_report(bmodel.predict(X_test), y_test))
df_result_final = df_result.drop([

    'Patient addmited to regular ward (1=yes, 0=no)',

    'Patient addmited to semi-intensive unit (1=yes, 0=no)'

], axis=1)

x, y = (df_result_final.drop('Patient addmited to intensive care unit (1=yes, 0=no)', axis=1), df_result_final['Patient addmited to intensive care unit (1=yes, 0=no)'])
rfmodel, X_test, y_test = make_pipeline(x, y)

print(rfmodel.score(X_test, y_test))

print(classification_report(rfmodel.predict(X_test), y_test))