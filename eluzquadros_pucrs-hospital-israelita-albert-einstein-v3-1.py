# The new versions of Pandas and Matplotlib bring several warning messages to the developer.

# Let's disable this



import sys

import warnings

import matplotlib.cbook

if not sys.warnoptions:

    warnings.simplefilter("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)



# Imports

import numpy as np

import pandas as pd

import numpy as np

import pymc3

import arviz as az

import altair as alt



import statsmodels.api as sm

import scikitplot as skplt



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold







import seaborn as sns

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



%matplotlib inline



#https://discuss.analyticsvidhya.com/t/how-to-display-full-dataframe-in-pandas/23298/3

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 1000)

!pip install watermark



# Packages for this jupyter notebook

%reload_ext watermark

%watermark -a "PUCRS - INSCER" --iversions

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Set a path

#path = '/kaggle/input/covid19'
# Get data

df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
# Copy original dataset

df_input = df
# Print dataset

print(df.shape);df.head()
# Drop column Patient ID

df.drop(['Patient ID'],inplace=True,axis=1,errors='ignore')
# Check missing

missing_values = df.isnull().sum().sort_values(ascending = False)

missing_values
# Check missing percent

missing_values = missing_values[missing_values > 0]/df.shape[0] 

print(f'{missing_values * 100} %')
# Find positive

positive = df[df['SARS-Cov-2 exam result'] == 'positive']

positive
# Save posotive

#positive.to_csv(path+"positive.csv")
# Print positive

print(positive.shape);positive.head()
# Find pnegative

negative = df[df['SARS-Cov-2 exam result'] == 'negative']

negative
# Print negative

print(negative.shape);negative.head()
# Save negative

#negative.to_csv(path+"negative.csv")
# Find Patient addmited to regular ward (1=yes, 0=no) == 1

Patientaddmitedregularward = df[df['Patient addmited to regular ward (1=yes, 0=no)'] == 1]

Patientaddmitedregularward.shape
# Find Patient addmited to semi-intensive unit (1=yes, 0=no) == 1

Patientaddmitedtosemiintensiveunit = df[df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1]

Patientaddmitedtosemiintensiveunit.shape
# Find Patient addmited to intensive care unit (1=yes, 0=no)

Patientaddmitedtointensivecareunit = df[df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1]

Patientaddmitedtointensivecareunit.shape
# Create df_addmited dataframe

df_addmited = pd.concat([Patientaddmitedregularward, Patientaddmitedtosemiintensiveunit, Patientaddmitedtointensivecareunit])
# Print Patients with more completed diognostcs

print(df_addmited.shape);df_addmited.head()
# Explore classes result

df_addmited['SARS-Cov-2 exam result'].value_counts()
# Drop columns if have more than 20% of missing

df_addmited = df_addmited.dropna(thresh=0.20*len(df_addmited), axis=1)
# print addmited

print(df_addmited.shape);df_addmited.head()
## Get categorical

features_cat = []

for col in df_addmited.columns:

    if df_addmited[col].dtype == 'O':

        features_cat.append(col)

features_cat
# Fillna to Zero

df_addmited = df_addmited.apply(lambda x: x.fillna(0),axis=0)
# Fatorize dataframe

for (name, series) in df_addmited.iteritems():

    if series.dtype == 'O':

        #for objects: factorize

        df_addmited[name], tmp_indexer = pd.factorize(df_addmited[name])

        #but now we have -1 values (NaN)

    else:

        #for int or float: fill NaN

        tmp_len = len(df_addmited[series.isnull()])

        if tmp_len>0:

            #print "mean", train_series.mean()

            df_addmited.loc[series.isnull(), df_addmited] = 0
# print df fatorized

print(df_addmited.shape);df_addmited.head()
# Save df_addmited fatorized

#df_addmited.T.to_csv(path+"df_addmited.csv")
# Info dataframe

df_addmited.dtypes
# Number of each type of column

df_addmited.dtypes.value_counts()
# Check percentual missing

missing_values = missing_values[missing_values > 0]/df_addmited.shape[0] 

print(f'{missing_values * 100} %')
# Describe target

df_addmited['SARS-Cov-2 exam result'].describe()
# Target have a null?

df_addmited['SARS-Cov-2 exam result'].isnull().sum() 
# Check SARS-Cov-2 exam result for Task 1

SARSCov2examresult_count = df_addmited['SARS-Cov-2 exam result'].value_counts()

SARSCov2examresult_count
# Check Patient addmited to regular ward (1=yes, 0=no) for Task 2

regularward_count = df_addmited['Patient addmited to regular ward (1=yes, 0=no)'].value_counts()

regularward_count
# Check Patient addmited to semi-intensive unit (1=yes, 0=no) for Task 2

semintensivecareunit_count = df_addmited['Patient addmited to semi-intensive unit (1=yes, 0=no)'].value_counts()

semintensivecareunit_count
# Check Patient addmited to intensive care unit (1=yes, 0=no) for Task 2

intensivecareunit_count = df_addmited['Patient addmited to intensive care unit (1=yes, 0=no)'].value_counts()

intensivecareunit_count
# Create target dataframe

Patients_target = pd.DataFrame()
#Patients_targets

Patients_target['SARSCov2examresult'] = df_addmited['SARS-Cov-2 exam result']

Patients_target['regularward'] = df_addmited['Patient addmited to regular ward (1=yes, 0=no)']

Patients_target['semintensivecareunit'] = df_addmited['Patient addmited to semi-intensive unit (1=yes, 0=no)']

Patients_target['intensivecareunit'] = df_addmited['Patient addmited to intensive care unit (1=yes, 0=no)']
# print Patients_target

print(Patients_target.shape);Patients_target.head()
# Get df_addmited columns

cols = df_addmited.columns
scaler = MinMaxScaler()

df_norm = np.around(scaler.fit_transform(df_addmited), decimals=6)
standardization = StandardScaler()



sta = standardization.fit_transform(df_norm)

df_sta = pd.DataFrame(sta, columns = cols)
# Get data prep StandardScaler

df_prep = df_sta
print(df_prep.shape); df_prep.head()
print(df_prep.shape);df_prep.head(1)
# Get target - Task 1 e 2

target_SARSCov2examresult = Patients_target['SARSCov2examresult']

target_regularward = Patients_target['regularward']

target_semintensivecareunit = Patients_target['semintensivecareunit']

target_intensivecareunit = Patients_target['intensivecareunit']
# Copy df to train

df_SARSCov2examresult = df_prep

df_regularward = df_prep

df_semintensivecareunit = df_prep

df_intensivecareunit = df_prep
# print SARSCov2examresult sample

df_SARSCov2examresult.head()
# print regularward sample

df_regularward.head()
# Drop variables as we don't want to train

df_SARSCov2examresult = df_SARSCov2examresult.drop(['SARS-Cov-2 exam result'], axis=1)

df_regularward = df_regularward.drop(['Patient addmited to regular ward (1=yes, 0=no)'], axis=1)

df_semintensivecareunit= df_semintensivecareunit.drop(['Patient addmited to semi-intensive unit (1=yes, 0=no)'], axis=1)

df_intensivecareunit = df_intensivecareunit.drop(['Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1)
df_SARSCov2examresult.head(1)
#df_regularward['SARSCov2examresult'] = Patients_target['SARSCov2examresult']

df_regularward.head(1)
#df_semintensivecareunit['SARSCov2examresult'] = Patients_target['SARSCov2examresult']

df_semintensivecareunit.head(1)
#df_intensivecareunit['SARSCov2examresult'] = Patients_target['SARSCov2examresult']

df_intensivecareunit.head(1)
print(df_SARSCov2examresult.shape);df_SARSCov2examresult.head(1)
!pip install imblearn


import imblearn as im

im.__version__
# Import

# !pip install imblearn

from imblearn.over_sampling import SMOTE



# Seed result

seed = 100



# Set X e y

X = df_SARSCov2examresult

y = target_SARSCov2examresult
# Create SMOTE

smote_bal = SMOTE(random_state = seed)



# Aply smote

X_SARSCov2examresult, y_SARSCov2examresult = smote_bal.fit_resample(X, y)
# print y

y_SARSCov2examresult
# print X

print(X_SARSCov2examresult.shape);X_SARSCov2examresult.head()
# Plot

sns.countplot(y_SARSCov2examresult, palette = "OrRd")

plt.box(False)

plt.xlabel('SARS-Cov-2 exam result: Negativo (0) / Positivo (1)', fontsize = 11)

plt.ylabel('Total Pacientes', fontsize = 11)

plt.title('Contagem de Classes\n')

plt.show()
# Import

# !pip install imblearn

from imblearn.over_sampling import SMOTE



# Seed result

seed = 100



# Set X e y

X = df_regularward

y = target_regularward





# Criaate SMOTE

smote_bal = SMOTE(random_state = seed)



# Aply smote

X_regularward, y_regularward = smote_bal.fit_resample(X, y)
# Plot

sns.countplot(y_regularward, palette = "OrRd")

plt.box(False)

plt.xlabel('Patient addmited to regular ward (1=yes, 0=no)', fontsize = 11)

plt.ylabel('Total Pacientes', fontsize = 11)

plt.title('Contagem de Classes\n')

plt.show()
# Import

# !pip install imblearn

from imblearn.over_sampling import SMOTE



# Seed result

seed = 100



# Set X e y

X = df_semintensivecareunit

y = target_semintensivecareunit





# Criate SMOTE

smote_bal = SMOTE(random_state = seed)



# Aplyo smote

X_semintensivecareunit, y_semintensivecareunit = smote_bal.fit_resample(X, y)
# Plot

sns.countplot(y_semintensivecareunit, palette = "OrRd")

plt.box(False)

plt.xlabel('Patient addmited to semi-intensive unit (1=yes, 0=no)', fontsize = 11)

plt.ylabel('Total Pacientes', fontsize = 11)

plt.title('Contagem de Classes\n')

plt.show()
# Import

# !pip install imblearn

from imblearn.over_sampling import SMOTE



# Seed result

seed = 100



# Set X e y

X = df_intensivecareunit

y = target_intensivecareunit





# Criate SMOTE

smote_bal = SMOTE(random_state = seed)



# Aply smote

X_intensivecareunit, y_intensivecareunit = smote_bal.fit_resample(X, y)
# Plot

sns.countplot(y_intensivecareunit, palette = "OrRd")

plt.box(False)

plt.xlabel('SPatient addmited to intensive care unit (1=yes, 0=no)', fontsize = 11)

plt.ylabel('Total Pacientes', fontsize = 11)

plt.title('Contagem de Classes\n')

plt.show()
# create df

df_gluon_SARSCov2examresult_v1 = pd.DataFrame()

df_gluon_SARSCov2examresult_v1['SARSCov2examresult'] = y_SARSCov2examresult
# concat df

df_gluon_SARSCov2examresult_v2 = pd.concat([df_gluon_SARSCov2examresult_v1, X_SARSCov2examresult], axis=1)
# create df concatenate

gluon = df_gluon_SARSCov2examresult_v2
# print gluon dataframe

print(gluon.shape); gluon.head()
#Correlation between features

corr = gluon.corr('pearson')



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(30, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Get feature importances for SARSCov2examresult with ExtraTreesClassifier

y = gluon["SARSCov2examresult"]

X = gluon.drop(["SARSCov2examresult"], axis=1)



extc = ExtraTreesClassifier(n_estimators=750,random_state=0)



# Fit model

extc.fit(X, y)

  

for num,item in enumerate(extc.feature_importances_):

    print(X.columns[num],item)
# print

print(gluon.shape); gluon.head()
!pip install autogluon
import autogluon as ag

from autogluon import TabularPrediction as task
# setting up testing and training sets

msk = np.random.rand(len(gluon)) < 0.85
train_data = gluon[msk]

test_data = gluon[~msk]
# get target label

label_column = 'SARSCov2examresult'



# Get Gluon task.dataset

train_data = task.Dataset(train_data)

test_data = task.Dataset(test_data)



# Model: AutoGluon - vanila



dir = 'agModels-SARS-Cov-2examresult' # specifies folder where to store trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)
y_test = test_data[label_column]  # values to predict

test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating



print(test_data_nolab.head())
predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file



y_pred = predictor.predict(test_data_nolab)

print("Predictions:  ", y_pred)

perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print("AutoGluon infers problem type is: ", predictor.problem_type)

print("AutoGluon categorized the features as: ", predictor.feature_types)
predictor = task.fit(train_data=train_data, label=label_column)

performance = predictor.evaluate(test_data)
#results = predictor.fit_summary() # Esta linha está quebrando aqui, então tivemos que comentar. Rode em seu computador para ver a saída!
X_SARSCov2examresult.head()
#y_SARSCov2examresult
from sklearn.model_selection import cross_val_score



# Carregando os dados SARSCov2examresult

X = np.asarray(X_SARSCov2examresult)

y = np.asarray(y_SARSCov2examresult)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 99)



extc = ExtraTreesClassifier(n_estimators=750,max_features= None, criterion= 'entropy',min_samples_split= 10,

                           max_depth= 35, min_samples_leaf= 10, n_jobs = 25)



# Treinamento do Modelo

extc.fit(X_train, y_train)



# Score

scores = cross_val_score(extc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Imprimindo o resultado

print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Fazendo previsões

y_pred = extc.predict(X_test)



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print (confusionMatrix)



# Acurácia em teste

print("Acurácia em Teste:", accuracy_score(y_test, y_pred))



# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)



# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))
# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# 8.3 MODELO 2:  Machine learing with ExtraTreeClassifier - version 2

## Optimize Hyperparameters with Randomized Search



# Import

from sklearn.model_selection import RandomizedSearchCV



# Carregando os dados SARSCov2examresult

X = np.asarray(X_SARSCov2examresult)

y = np.asarray(y_SARSCov2examresult)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 99)





# Definição dos parâmetros

param_dist = {"max_depth": [7, 10, 20, None],

              "max_features": [15, 25, 30],

              "min_samples_split": [15, 20, 25],

              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],

              "bootstrap": [True, False]}



# Para o classificador criado com ExtraTrees, testamos diferentes combinações de parâmetros

rsearch = RandomizedSearchCV(extc, param_distributions = param_dist, n_iter = 50, return_train_score = True)  



# Aplicando o resultado ao conjunto de dados de treino e obtendo o score

rsearch.fit(X_train, y_train)



# Resultados 

rsearch.cv_results_



# Imprimindo o melhor estimador

bestextc = rsearch.best_estimator_

print (bestextc)



# Aplicando o melhor estimador para realizar as previsões

y_pred = bestextc.predict(X_test)



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print(confusionMatrix)



# Score

scores = cross_val_score(extc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Avaliando as previsões em treino

print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Avaliando as previsões em teste

accuracy = accuracy_score(y_test, y_pred)

print("ExtraTreesClassifier -> Acurácia em Teste: %.2f%%" % (accuracy * 100))



# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)



# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))



# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()



# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))



# Obtendo o grid com todas as combinações de parâmetros

#rsearch.cv_results_
# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# Obtendo o grid com todas as combinações de parâmetros

rsearch.cv_results_
# Imports

from sklearn.datasets import make_hastie_10_2

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier



# Carregando os dados SARSCov2examresult

X = np.asarray(X_SARSCov2examresult)

y = np.asarray(y_SARSCov2examresult)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99)



# Cria o classificador

gbc = GradientBoostingClassifier(n_estimators = 750, max_depth = None)



# Cria o modelo

gbc.fit(X_train, y_train)



# Previsões das classes (labels)

y_pred = gbc.predict(X_test)



# Previsão das probabilidades das classes

gbc.predict_proba(X_test)[0]



# Classificador - Método Ensemble (Gradient Boosting Classifier)

gbc



# Estimador Base

gbc.estimators_[0, 0]



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print(confusionMatrix)



# Score

scores = cross_val_score(gbc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Avaliando as previsões em treino

print ("GradientBoostingClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Avaliando as previsões em teste

accuracy = accuracy_score(y_test, y_pred)

print("GradientBoostingClassifier -> Acurácia em Teste: %.2f%%" % (accuracy * 100))





# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)





# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))

# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# Import dos módulos

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier



# Carregando os dados

X = np.asarray(X)

y = np.asarray(y)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 99)



# Criando o modelo

# Define parameters and grid search

# Definição dos parâmetros

param_dist = {"n_estimators": [100, 300, 500, 700],

              "max_depth": [1, 3, 7, 8, 12, None],

              "max_features": [8, 9, 10, 11, 16, 22],

              "min_samples_split": [8, 10, 11, 14, 16, 19],

              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],

              "bootstrap": [True, False]}



xgbc = XGBClassifier()



# Para o classificador criado com XGBOOST, testamos diferentes combinações de parâmetros

rsearch = RandomizedSearchCV(xgbc, param_distributions = param_dist, n_iter = 25, return_train_score = True)



# Aplicando o resultado ao conjunto de dados de treino e obtendo o score

rsearch.fit(X_train, y_train)



# Resultados 

rsearch.cv_results_



# Imprimindo o melhor estimador

bestextc = rsearch.best_estimator_

print (bestextc)



# Aplicando o melhor estimador para realizar as previsões

y_pred = bestextc.predict(X_test)

previsoes = [round(value) for value in y_pred]



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print(confusionMatrix)



# Score

scores = cross_val_score(xgbc, X_train, y_train, cv = 3, scoring = 'roc_auc', n_jobs = -1)



# Avaliando as previsões em treino

print ("GradientBoostingClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Avaliando as previsões em teste

accuracy = accuracy_score(y_test, y_pred)

print("GradientBoostingClassifier -> Acurácia em Teste: %.2f%%" % (accuracy * 100))





# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
X_regularward.head()
# create df

df_gluon_regularward_v1 = pd.DataFrame()

df_gluon_regularward_v1['Patient addmited to regular ward (1=yes, 0=no)'] = y_regularward



# concat df

df_gluon_regularward_v2 = pd.concat([df_gluon_regularward_v1, X_regularward], axis=1)



# create df concatenate

gluon = df_gluon_regularward_v2



# print gluon dataframe

print(gluon.shape); gluon.head()
# setting up testing and training sets

gluon = df_gluon_regularward_v2

msk = np.random.rand(len(gluon)) < 0.6



train_data = gluon[msk]

test_data = gluon[~msk]
label_column = 'Patient addmited to regular ward (1=yes, 0=no)'



train_data = task.Dataset(train_data)

test_data = task.Dataset(test_data)



# Model: AutoGluon



dir = 'agModels-SARS-Cov-admissao-enfermaria-geral' # specifies folder where to store trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)



y_test = test_data[label_column]  # values to predict

test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating

print(test_data_nolab.head())



predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file



y_pred = predictor.predict(test_data_nolab)

print("Predictions:  ", y_pred)

perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



print("AutoGluon infers problem type is: ", predictor.problem_type)

print("AutoGluon categorized the features as: ", predictor.feature_types)



predictor = task.fit(train_data=train_data, label=label_column)

performance = predictor.evaluate(test_data)



#results = predictor.fit_summary() # Esta linha está quebrando aqui, então tivemos que comentar. Rode em seu computador para ver a saída!
from sklearn.model_selection import cross_val_score



# Carregando os dados SARSCov2examresult

X = np.asarray(X_regularward)

y = np.asarray(y_regularward)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 99)



extc = ExtraTreesClassifier(n_estimators=750,max_features= 10, criterion= 'entropy',min_samples_split= 10,

                           max_depth= 10, min_samples_leaf= 10, n_jobs = 25)



# Treinamento do Modelo

extc.fit(X_train, y_train)



# Score

scores = cross_val_score(extc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Imprimindo o resultado

print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Fazendo previsões

y_pred = extc.predict(X_test)



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print (confusionMatrix)



# Acurácia em teste

print("Acurácia em Teste:", accuracy_score(y_test, y_pred))



# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)



# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))
# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# Drop variables as we don't want to train

df_gluon_semintensivecareunit_v1 = pd.DataFrame()

df_gluon_semintensivecareunit_v1['Patient addmited to semi-intensive unit (1=yes, 0=no)'] = y_semintensivecareunit.values
df_gluon_semintensivecareunit_v2 = pd.concat([df_gluon_semintensivecareunit_v1, X_semintensivecareunit], axis=1)
print(df_gluon_semintensivecareunit_v2.shape);df_gluon_semintensivecareunit_v2.head()
# setting up testing and training sets

gluon = df_gluon_semintensivecareunit_v2

msk = np.random.rand(len(gluon)) < 0.6



train_data = gluon[msk]

test_data = gluon[~msk]
label_column = 'Patient addmited to semi-intensive unit (1=yes, 0=no)'



train_data = task.Dataset(train_data)

test_data = task.Dataset(test_data)



# Model: AutoGluon - vanila



dir = 'agModels-SARS-Cov-admissao-semi-intensive-unit' # specifies folder where to store trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)



y_test = test_data[label_column]  # values to predict

test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating

print(test_data_nolab.head())



predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file



y_pred = predictor.predict(test_data_nolab)

print("Predictions:  ", y_pred)

perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



print("AutoGluon infers problem type is: ", predictor.problem_type)

print("AutoGluon categorized the features as: ", predictor.feature_types)



predictor = task.fit(train_data=train_data, label=label_column)

performance = predictor.evaluate(test_data)



results = predictor.fit_summary()
from sklearn.model_selection import cross_val_score



# Carregando os dados SARSCov2examresult

X = np.asarray(X_semintensivecareunit)

y = np.asarray(y_semintensivecareunit)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 99)



extc = ExtraTreesClassifier(n_estimators=750,max_features= 10, criterion= 'entropy',min_samples_split= 10,

                           max_depth= 10, min_samples_leaf= 10, n_jobs = 25)



# Treinamento do Modelo

extc.fit(X_train, y_train)



# Score

scores = cross_val_score(extc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Imprimindo o resultado

print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Fazendo previsões

y_pred = extc.predict(X_test)



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print (confusionMatrix)



# Acurácia em teste

print("Acurácia em Teste:", accuracy_score(y_test, y_pred))



# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)



# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))
# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# Obtendo o grid com todas as combinações de parâmetros

#rsearch.cv_results_
# Drop variables as we don't want to train

df_gluon_intensivecareunit_v1 = pd.DataFrame()

df_gluon_intensivecareunit_v1['Patient addmited to intensive care unit (1=yes, 0=no)'] = y_intensivecareunit.values
df_gluon_intensivecareunit_v2 = pd.concat([df_gluon_intensivecareunit_v1, X_intensivecareunit], axis=1)
print(df_gluon_intensivecareunit_v2.shape);df_gluon_intensivecareunit_v2.head()
# setting up testing and training sets

gluon = df_gluon_intensivecareunit_v2

msk = np.random.rand(len(gluon)) < 0.6



train_data = gluon[msk]

test_data = gluon[~msk]
label_column = 'Patient addmited to intensive care unit (1=yes, 0=no)'



train_data = task.Dataset(train_data)

test_data = task.Dataset(test_data)



# Model: AutoGluon - vanila



dir = 'agModels-SARS-Cov-admissao_intensive_care_unit' # specifies folder where to store trained models

predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)



y_test = test_data[label_column]  # values to predict

test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating

print(test_data_nolab.head())



predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file



y_pred = predictor.predict(test_data_nolab)

print("Predictions:  ", y_pred)

perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)



print("AutoGluon infers problem type is: ", predictor.problem_type)

print("AutoGluon categorized the features as: ", predictor.feature_types)



predictor = task.fit(train_data=train_data, label=label_column)

performance = predictor.evaluate(test_data)



#results = predictor.fit_summary() # Esta linha está quebrando aqui, então tivemos que comentar. Rode em seu computador para ver a saída!
from sklearn.model_selection import cross_val_score



# Carregando os dados SARSCov2examresult

X = np.asarray(X_intensivecareunit)

y = np.asarray(y_intensivecareunit)



# Divisão de dados de treino e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 99)



extc = ExtraTreesClassifier(n_estimators=750,max_features= 10, criterion= 'entropy',min_samples_split= 10,

                           max_depth= 10, min_samples_leaf= 10, n_jobs = 25)



# Treinamento do Modelo

extc.fit(X_train, y_train)



# Score

scores = cross_val_score(extc, X_train, y_train, cv = 3, scoring = 'accuracy', n_jobs = -1)



# Imprimindo o resultado

print ("ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f Desvio Padrão = %0.3f" % 

       (np.mean(scores), np.std(scores)))



# Fazendo previsões

y_pred = extc.predict(X_test)



# Confusion Matrix

confusionMatrix = confusion_matrix(y_test, y_pred)

print (confusionMatrix)



# Acurácia em teste

print("Acurácia em Teste:", accuracy_score(y_test, y_pred))



# Relatório de classificação

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score



classe_names = ['SARS-Cov-2 exam result']

report = classification_report(y_test, y_pred)

print(report)



# Score AUC

print("Acurácia AUC: ",roc_auc_score(y_test, y_pred))
# Calcula a Curva ROC para cada classe

y_probs = extc.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_probs)

plt.show()
# Coeficiente de Correlação Matthews

print(matthews_corrcoef(y_test, y_pred))
# Obtendo o grid com todas as combinações de parâmetros

#rsearch.cv_results_