!ls ../input/dataset-with-tramps-predict-earning-potential/
# Manejo de datos
import pandas as pd
import numpy as np

# Graficas
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['figure.figsize'] = 15,10
# Inspeccion de datos rapida
salary_df = pd.read_csv('../input/dataset-with-tramps-predict-earning-potential/Datos_Con_Trampa.csv')       
salary_df.head()
salary_df.info()
salary_df.columns
# Eliminamos los espacios innecesarios del nombre de cada columna y
# Renombramos las columnas que ya sabemos.

salary_df.rename(columns = lambda x: x.lstrip(), inplace=True)
salary_df.rename(columns = {'State-gov': 'Company_type', 'Bachelors': 'Education', 'Never-married': 'Marital_status',
                            'Adm-clerical': 'Job_position', 'Not-in-family': 'Family_rol', 'White': 'Race', 'Male': 'Gender',
                            'United-States': 'Country', '<=50K': 'Income'}, inplace=True)

salary_df.columns
# Seleccionar unicamente las columnas de tipo 'object'
df_obj = salary_df.select_dtypes(['object'])
# Encontrar el porcentaje de repitencia de cada dato por columna
for col in df_obj:
    length = salary_df[col].count()
    unique = dict(salary_df[col].value_counts())
    print(col.upper())
    
    for k, v in unique.items():
        print("\t'{}' ({}%)".format(k, round((v*100)/length, 2)))
        
    print('\n')
# Eliminamos los espacios innecesarios de los datos.
salary_df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
# Reemplazamos los datos desconocidos, de '?' a 'Unknown'
salary_df['Company_type'].replace('?', 'Unknown', inplace=True)
salary_df['Job_position'].replace('?', 'Unknown', inplace=True)
salary_df['Country'].replace('?', 'Unknown', inplace=True)
salary_df.head(0)
# Cambiamos los datos categoricos de Income a numericos(0 y 1)
salary_df['Income'].replace({'<=50K': 0, '>50K': 1}, inplace=True)
salary_df['Income'].unique()
others = salary_df['Country'].unique()[1:]
salary_df['Country'].replace(others, 'Other', inplace=True)
salary_df['Country'].unique()
salary_df.select_dtypes(['int64']).hist();
# Desplegar los datos numericos como decimales en lugar de notacion cientifica
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

# Analisis estadistico de las columnas de tipo numerico
salary_df.describe()
# Renombramos la columna a su nombre correcto.
salary_df.rename(columns={'39': 'Age'}, inplace=True)
salary_df[salary_df['77516'] == 1484705]
salary_df.sample(n=2)
salary_df[salary_df['77516'] == 122272]
len(salary_df[salary_df.duplicated(keep='first') == True])
salary_df['77516'].sum()
print('Antes:\t', len(salary_df))
salary_df.drop_duplicates(inplace=True)
print('Despues:', len(salary_df))
salary_df.rename(columns={'77516': 'P_same_features'}, inplace=True)
for obj in set(zip(salary_df['Education'], salary_df['13'])):
    print(obj)
salary_df.rename(columns={'13':'Education_rank'}, inplace=True)
salary_df.drop(labels='Education', axis=1,inplace=True)
salary_df[salary_df['2174'] > 50000]['Income'].unique()
salary_df.rename(columns={'2174': 'Extra_income'}, inplace=True)
salary_df.rename(columns={'0': 'Val_desc'}, inplace=True)
salary_df['Extra_income'] = salary_df['Extra_income'] - salary_df['Val_desc']
salary_df.drop(labels=['Val_desc'], axis=1, inplace=True)
salary_df['40'].mean()
salary_df.rename(columns={'40': 'Hours/week'}, inplace=True)
# Exportamos nuestro dataset limpio a un nuevo archivo.
salary_df.to_csv('Datos_Sin_Trampa.csv', index=False)
import seaborn as sns
sns.set(palette='pastel')

salary_df = pd.read_csv('Datos_Sin_Trampa.csv')
# Dividimos nuestros datos
more_than_50K = salary_df[salary_df['Income'] == 1]
less_than_50K = salary_df[salary_df['Income'] == 0]
salary_df.corr()
sns.distplot(less_than_50K['Age'], kde=False, bins=72, color='b');
sns.distplot(more_than_50K['Age'], kde=False, bins=72, color='r');
sns.distplot(less_than_50K['P_same_features']);
sns.distplot(more_than_50K['P_same_features']);
sns.catplot(x='Education_rank',hue='Income', kind='count', data=salary_df);
less_than_50K['Extra_income'].hist(alpha=0.5, color='b');
more_than_50K['Extra_income'].hist(alpha=0.5, color='r');
less_than_50K['Hours/week'].hist(alpha=0.5, color='b');
more_than_50K['Hours/week'].hist(alpha=0.5, color='r');
sns.catplot(x='Company_type', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
sns.catplot(x='Marital_status', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
sns.catplot(x='Job_position', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
sns.catplot(x='Family_rol', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
sns.catplot(x='Race', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
sns.catplot(x='Gender', hue='Income', kind='count', data=salary_df);
sns.catplot(x='Country', hue='Income', kind='count', data=salary_df).set_xticklabels(rotation=90);
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
X = salary_df.drop(labels='Income', axis=1)
Y = salary_df['Income']
# Convertimos los datos categoricos a numericos
X = X.apply(LabelEncoder().fit_transform)
X = StandardScaler().fit_transform(X)
# Dividimos nuestro dataset un 70% para Entrenamiento y 30% par test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)
forest = RandomForestClassifier()
forest.fit(X_train, Y_train)
Y_prediction = forest.predict(X_test)
print("RandomForestClassifier score: ", round(accuracy_score(Y_test, Y_prediction) * 100, 2), '%')