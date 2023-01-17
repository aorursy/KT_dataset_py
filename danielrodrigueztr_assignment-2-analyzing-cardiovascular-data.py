# Import all required modules
import pandas as pd
import numpy as np

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import plotting modules
import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
%matplotlib inline
# Tune the visual settings for figures in `seaborn`
sns.set_context(
    "notebook", 
    font_scale=1.5,       
    rc={ 
        "figure.figsize": (11, 8), 
        "axes.titlesize": 18 
    }
)

from matplotlib import rcParams
rcParams['figure.figsize'] = 11, 8
df = pd.read_csv('../input/mlbootcamp5_train.csv')
print('Dataset size: ', df.shape)
df.head()
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active', 'cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', data=df_uniques, kind='bar', size=12);

df_uniques.variable.unique()
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active'], 
                     id_vars=['cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 
                                              'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               col='cardio', data=df_uniques, kind='bar', size=9);
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
# vemos el promedio de la altura por el campo 'gender' para determinar cual es masculino/femenino
df.groupby(['gender'])['height'].mean()
# usamos melt para tranponer los valores
df1 = pd.melt(frame=df, value_vars=['gender'])
# Realizamos un count para ver la distribucion
df1 = pd.DataFrame(df1.groupby(['variable', 'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()
#Renombramos los codigos teniendo en cuenta el promedio de altura revisado previamente.
df1['value']=['Women' if value==1 else 'Men' for value in df1['value']]
df1
# Realizamos un grafico con la informacion trabajada previamente
sns.factorplot(x='variable', y='count',hue='value', data=df1, kind='bar', size=5);
#transponemos usando melt y agregamos un columna con las filas que tienen valor "alco".
df1 = pd.melt(frame=df, value_vars=['gender'],id_vars=['alco'])
# agrupamos para ver la distribucion
df1 = pd.DataFrame(df1.groupby(['alco', 'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()
#renombramos los codigos con su descripcion de genero.
df1['value']=['Women' if value==1 else 'Men' for value in df1['value']]
df1
# realizamos un grafico de la informacion obtenida
sns.factorplot(x='value', y='count', hue='value', 
               col='alco', data=df1, kind='bar', size=5);
#transponemos usando melt y agregamos un columna con las filas que tienen valor "alco".
df1 = pd.melt(frame=df, value_vars=['gender'],id_vars=['smoke'])
# agrupamos para ver la distribucion
df1 = pd.DataFrame(df1.groupby(['smoke', 'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()
#renombramos los codigos con su descripcion de genero.
df1['value']=['Women' if value==1 else 'Men' for value in df1['value']]
df1
# realizamos un grafico de la informacion obtenida
sns.factorplot(x='value', y='count', hue='value', 
               col='smoke', data=df1, kind='bar', size=5);
df.age.unique()
divisor = 365
print('edad minima' , df.age.min()/divisor)
print('edad maxima' , df.age.max()/divisor)
print( 'el campo edad esta expresado en dias')
df.head()
df['age_meses']=round(df['age'] /365.25*12,0)
df.groupby(['smoke'])['age_meses'].median()
df['age_year']=round(df['age'] /365.25,0)
#df['level_cho']=[4 if cholesterol == 1 else 0 for weekday in datatrain[colname+"_weekday"]]
n = {1 : 4, 2 : 6,3 : 8}
df['level_cho'] = df['cholesterol'].map(n)
df.head()
#Calcule la fracción de las personas con ECV para los dos segmentos descritos anteriormente. ¿Cuál es la proporción de estas dos fracciones?
#100 * df['cardio'].value_counts(dropna = False).values / len(df)
#df.ap_hi.unique()
bins = [0,120, 140, 160,180] 
df['ap_hi_bin'] = pd.cut(df['ap_hi'], bins = bins,right = False)
bins = [0,40, 50, 55,60,65] 
df['age_bin'] = pd.cut(df['age_year'], bins = bins,right = False)
df.head()
#df.age_bin.unique()
df1=df[(df['age_year']>=60) & (df['age_year']<65)].groupby(['ap_hi_bin','level_cho'])['age'].count()
#df1.age_bin.unique()
df1#.info()
df1=df[(df['age_year']>=60) & (df['age_year']<65) & (df['smoke']==1)& (df['gender']==2)]
#df1.head()
pd.crosstab(df1['level_cho'], df1['ap_hi_bin']).T
# You code here
# You code here
# You code here
# You code here
# You code here