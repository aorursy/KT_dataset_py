import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', 60) #Display 60 columns for better visualitation. 
from pandas.api.types import CategoricalDtype
import warnings
warnings.simplefilter("ignore")
df = pd.read_csv('../input/compas-scores/compas-scores.csv')
df = pd.DataFrame(df)
df.head(4)
df.info()
#'num_vr_cases', 'num_r_cases', son columnas completamente vacias que se deben eliminar
#Empty columns
df = df.drop(['num_vr_cases', 'num_r_cases'], axis =1)
#Selection of Columns that will be useful to find the desired insights 
#Seleccion de Columnas que seran de utilidad para encontrar los insights buscados

df1 = df[['id', 'name', 'sex', 'age', 'age_cat', 'is_recid', 'is_violent_recid','decile_score', 'v_decile_score', 'vr_offense_date', 
'r_offense_date', 'race', 'score_text', 'v_score_text', 'priors_count', 'compas_screening_date']].copy()
df1.head(3)
def describe_columna(df, col):
    print(f'Columna: {col}  -  Tipo de datos: {df[col].dtype}')
    print(f'Número de valores nulos: {df[col].isnull().sum()}  -  Número de valores distintos: {df[col].nunique()}')
    print('Valores más frecuentes:')
    for i, v in df[col].value_counts().iloc[:10].items() :
        print(i, '\t', v)
describe_columna(df1, 'id')
#Columna se encuentra completa, con solo valores unicos, utilizables como indices e identificadores.
describe_columna(df1, 'name')
df1[df1.name.duplicated(keep = False)].sort_values(by = 'name')
describe_columna(df1, 'sex')
# Columna integra, valida y sin nulos.
print(describe_columna(df1, 'age'))
print(df1.age.min() )
print(df1.age.max()) 
print(describe_columna(df1, 'age_cat'))
# Columna sin valores nulos, integra y valida.
print(describe_columna(df1, 'is_recid'))
#Esos 719 valores sin antecedentes de reincidencia se deben eliminar
df1 = df1.drop(df1[df1['is_recid']==-1].index)
print(describe_columna(df1, 'is_recid')) 
# Transform the columns that should be categorical
# Transformar a categoricos las columnas que deberian serlo

for col in ['sex', 'age_cat']:
    df1[col] = df1[col].astype('category')
   # 'decile_score', 'v_decile_score''is_recid', 'is_violent_recid',
print(describe_columna(df1, 'is_violent_recid'))
#COlumna integra y valida
print(describe_columna(df1, 'decile_score'))
#Columna integra y valida sin valores nulos
print(describe_columna(df1, 'v_decile_score'))
#Columna integra y valida sin valores nulos
print(describe_columna(df1, 'r_offense_date'))
# 3703 filas con informacion las cuales cuadran perfectamente con la cantidad de casos reincidentes
#que son 3703
#Tranformar to Date Time r_offense_date y vr_offense_date
df1.r_offense_date = pd.to_datetime(df1.r_offense_date, format="%Y-%m-%d")
df1.vr_offense_date = pd.to_datetime(df1.vr_offense_date, format="%Y-%m-%d")
print(describe_columna(df1, 'vr_offense_date'))
#Column complete, complete and valid. It transforms to Date Time
#Columna completa, integta y valida. Se tranforma a Date Time

df1.compas_screening_date = pd.to_datetime(df1.compas_screening_date, format="%Y-%m-%d")

print(describe_columna(df1, 'compas_screening_date'))
# Min and max dates of our dataframe to determine how to use them
#Fechas min y max de nuestro dataframe para determinar como utilizarlas

rod = df1.r_offense_date.min(), df1.r_offense_date.max()
vrod = df1.vr_offense_date.min(), df1.vr_offense_date.max()
csd = df1.compas_screening_date.min(), df1.compas_screening_date.max()
print('r_offense_date \n', {rod})
print('vr_offense_date \n', {vrod})
print('compas_screening_date \n', {csd})
#Distribución de decile score segun puntaje.
test = df1.groupby('decile_score')['id'].count()
print(test)
violent = df1[df1.is_violent_recid == 1]
violent.groupby('decile_score')['id'].count()
#Pearson correlation between numerical variables.
#Correlacion Pearson entre variables numericas.

correlation = df1.corr()
f, ax = plt.subplots(figsize=(4,4))
plt.title('Correlation of numerical attributes', size=10)
sns.heatmap(correlation)
plt.show()
correlation['v_decile_score'].sort_values(ascending=False).head(10)
correlation['decile_score'].sort_values(ascending=False).head(10)
df1.shape
#A variable is created with the date that will be filtered in the DF
#Se crea variable con la fecha que se filtrara en el DF
from datetime import datetime
fecha = '2014:01:01'
fecha_limite = datetime.strptime(fecha, '%Y:%M:%S')
def modelo1(caso):
    if  caso['is_violent_recid'] == 1 and caso['race'] == ('African-American') and caso['vr_offense_date'] > fecha_limite and caso['age_cat'=='Less than 25']:
        return 1
    else:
        return 0
score1 = df1.apply(lambda x: modelo1(x), axis = 1)
roc_auc_score((df1['v_score_text'] == 'High'), score1)
tc = pd.crosstab(score1, (df1['v_decile_score'] >= 7))
tc
#Calcular prediccion de la feature para IS_RECID
#Predictions for IS_recid
def modelo2(caso):
    if  caso['is_recid'] == 1 or caso['is_violent_recid'] == 1 and caso['r_offense_date'] > fecha_limite and caso['age_cat'=='Less than 25'] and caso['sex'] == 'Male':
        return 1
    else:
        return 0
score2 = df1.apply(lambda x: modelo2(x), axis = 1).rename('Predicción')
roc_auc_score((df1['decile_score'] >= 7), score2)

#Feature que nos entrega un valor de aciertos cercano al 61%. 
tc1 = pd.crosstab(score2, (df1['decile_score'] >= 7))
tc1
#Decile_score and v_decile_score -1 values are removed to plot
#Se eliminan valores -1 de decile_score y v_decile_score para graficar
df1 = df1.drop(df1[df1['v_decile_score']==-1].index)
df1 = df1.drop(df1[df1['decile_score']==-1].index)


#RACE
dfgb = df1.groupby("race")
race_count = df1.groupby("race")["name"].count()

fig, ax = plt.subplots(3, figsize=(14, 8))

for (i, race) in enumerate(["African-American", "Caucasian", "Hispanic"]):
    (
        (dfgb
            .get_group(race)
            .groupby("decile_score")["name"].count() / race_count[race]
        )
        .plot(kind="bar", ax=ax[i], color="#353535")
    )
    ax[i].set_ylabel(race)
    ax[i].set_xlabel("")
   
    ax[i].set_ylim(0, 0.32)

fig.suptitle("Score Frequency by Race")
plt.show()
#Sex
dfgb = df1.groupby("sex")
race_count = df1.groupby("sex")["name"].count()

fig, ax = plt.subplots(2, figsize=(14, 8))

for (i, sex) in enumerate(["Male", "Female"]):
    (
        (dfgb
            .get_group(sex)
            .groupby("decile_score")["name"].count() 
         #/ race_count[race]
        )
        .plot(kind="bar", ax=ax[i], color="#353535")
    )
    ax[i].set_ylabel(sex)
    ax[i].set_xlabel("")
   
 #   ax[i].set_ylim(0, 0.)

fig.suptitle("Score Frequency by Race")
plt.show()
#Age
dfgb = df1.groupby("age_cat")
race_count = df1.groupby("age_cat")["name"].count()

fig, ax = plt.subplots(3, figsize=(14, 8))

for (i, age_cat) in enumerate(["Less than 25", "25 - 45", "Greater than 45"]):
    (
        (dfgb
            .get_group(age_cat)
            .groupby("decile_score")["name"].count() 
        )
        .plot(kind="bar", ax=ax[i], color="#353535")
    )
    ax[i].set_ylabel(age_cat)
    ax[i].set_xlabel("")
   
 #   ax[i].set_ylim(0, 0.)

fig.suptitle("Score Frequency by age category")
plt.show()