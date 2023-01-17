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
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.patches as patches

%matplotlib inline

plt.rcParams['figure.figsize'] = (16, 9)

plt.style.use('ggplot')

from scipy.stats import linregress
fifa19_df = pd.read_csv('/kaggle/input/fifa19/data.csv')
#cantidad de registros

fifa19_df.shape
#primeros registros del dataset

fifa19_df.head()
fifa19_df.describe()
#Estadistivos de Valor

fifa19_df['Value'].describe()
fifa19_df.columns
fifa19_df.dtypes
fifa19_df.isnull().sum()
fifa19_df['Position'].isnull()
#Cuantos nulos en valor del jugador

pd.isnull(fifa19_df['Value']).values.ravel().sum()
#Cuantos nulos en Salario del jugador

pd.isnull(fifa19_df['Wage']).values.ravel().sum()
fifa19_df[['Wage','Value']]
fifa19_df[['Wage','Value']].dtypes
#vamos con el salario

fifa19_df['Wage'] = fifa19_df['Wage'].str.replace('€','')

fifa19_df['Wage'] = fifa19_df['Wage'].str.replace('M',' 1000000')

fifa19_df['Wage'] = fifa19_df['Wage'].str.replace('K',' 1000')

fifa19_df['Wage'] = fifa19_df['Wage'].str.split(' ', expand=True)[0].astype(float) * fifa19_df['Wage'].str.split(' ', expand=True)[1].astype(float)
#vamos con el valor

fifa19_df['Value'] = fifa19_df['Value'].str.replace('€','')

fifa19_df['Value'] = fifa19_df['Value'].str.replace('M',' 1000000')

fifa19_df['Value'] = fifa19_df['Value'].str.replace('K',' 1000')

fifa19_df['Value'] = fifa19_df['Value'].str.split(' ', expand=True)[0].astype(float) * fifa19_df['Value'].str.split(' ', expand=True)[1].astype(float)


fifa19_df[['Wage','Value']]
fifa19_df[['Wage','Value']].dtypes
fifa19_df[['Wage','Value']].describe()
fifa19_df.plot(kind = "scatter", x = "Wage", y = "Value")
#para el valor

#BINS calculados con la regla de sturges

k = int(1 + np.log2(fifa19_df.shape[0])) 

plt.hist(fifa19_df["Value"], bins = k)

plt.xlabel("Valor")

plt.ylabel("Frecuencia")

plt.title("Histograma Valor Jugadores FIFA19")
#para el Salario

k = int(1 + np.log2(fifa19_df.shape[0])) #regla de sturges

plt.hist(fifa19_df["Wage"], bins = k)

plt.xlabel("Salario")

plt.ylabel("Frecuencia")

plt.title("Histograma Salario Jugadores FIFA19")
fifa19_cost = pd.DataFrame(fifa19_df[['Wage','Value']])
fifa19_cost
corrmat = fifa19_cost.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(5,5))

g=sns.heatmap(fifa19_cost[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat
np.corrcoef(fifa19_cost['Value'], fifa19_cost['Wage'])
linregress(fifa19_cost['Value'], fifa19_cost['Wage'])
fifa19_df.describe().columns
#Variables categoricas

fifa19_cat = ['Nationality', 'Club', 'Preferred Foot', 'Work Rate', 'Body Type', 'Position']
#Variables Cuantitativas sin stats

fifa19_qty = ['Age', 'Value','Overall', 'Potential', 'Value', 'Wage', 'International Reputation', 'Weak Foot', 'Skill Moves']
#Variables Cuantitativas solo stats

fifa19_stats = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 

              'Volleys', 'Dribbling','Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 

              'SprintSpeed','Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',

             'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 

             'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving','GKHandling', 'GKKicking', 

            'GKPositioning', 'GKReflexes']
#validacion de nulos para las columnas con caracteristicas tecnicas del jugador

fifa19_df[fifa19_stats].isnull().sum()
#validacion de nulos para las columnas con caracteristicas tecnicas del jugador

fifa19_df[fifa19_qty].isnull().sum()
fifa19_df['TotalStats'] = fifa19_df[fifa19_stats].sum(axis=1)
fifa19_df.isnull().sum()
fifa19_df['TotalStats']
fifa19_variables = ['ID','Name','Age', 'Value','Overall', 'Potential', 'Wage', 'International Reputation', 'Weak Foot', 'Skill Moves','TotalStats']
fifa19_df_stats = pd.DataFrame(fifa19_df[fifa19_variables])
fifa19_df_stats.tail()
fifa19_df_stats.isnull().sum()
fifa19_df_stats = fifa19_df_stats.dropna()
fifa19_df_stats.shape
#DataSet Ordenado por Mejor acumulado de Stats y su valor

fifa19_df_stats.sort_values(by=['TotalStats', 'Value'], ascending=False)
corrmat = fifa19_df_stats.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

g=sns.heatmap(fifa19_df_stats[top_corr_features].corr(),annot=True,cmap="RdYlGn")
fifa19_df_stats.plot(kind = "scatter", x = "International Reputation", y = "Value")
fifa19_df_stats.plot(kind = "scatter", x = "Overall", y = "Value")
fifa19_df_stats.plot(kind = "scatter", x = "Potential", y = "Value")
fifa19_df_stats.plot(kind = "scatter", x = "Overall", y = "Potential")
fifa19_df_subvalorados = fifa19_df_stats[(fifa19_df_stats.Overall <= 70) & (fifa19_df_stats.Potential >=80)]

fifa19_df_subvalorados
fifa19_df_subvalorados.describe()
from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
#Modelo de regresion con las regresoras International Reputation, Overall, Potential, TotalStats

y = fifa19_df_stats['Value']

X = fifa19_df_stats[['International Reputation','Overall', 'Potential', 'TotalStats', 'Skill Moves']]
#Separamos datos de entrenamiento 80% y de prueba 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
#Generamos el modelo de regresion lineal

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
#Para obtener el intercepto:

print(regr.intercept_)

#Para obtener la pendiente

print(regr.coef_)
y_pred = regr.predict(X_test)
print('R2: ', r2_score(y_test,y_pred))
from xgboost import XGBRegressor
#Modelo con hyperparametros ajustados

#parametro max_depth de entre 3 4 5 6, el mejor es 3

#parametro n_estimators de entre 500 700 y 1000, el mejor es 700

model = XGBRegressor(n_jobs=-1, learning_rate = .5, max_depth=3,colsample_bytree = 1, verbosity=2,subsample=1, n_estimators=700)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("R2: ", r2_score(y_test, predictions))
sns.pairplot(fifa19_df_stats[['International Reputation','Overall', 'Potential', 'TotalStats', 'Skill Moves']], palette='deep')
fig = plt.figure()

ax = fig.add_subplot(111)

ax.title.set_text('Jugadores Recomendados a Comprar FIFA19')

ax.scatter(

    x=[fifa19_df_stats.Overall],

    y=[fifa19_df_stats.Potential],

    marker='o',

    alpha=0.9

)



#Area resaltada de jugadores recomendados a comprar

ax.add_patch(

    patches.Rectangle(

        xy=(60, 80),

        width=10,

        height=10,

        linewidth=1,

        color='blue',

        fill=False

    )

)



plt.show()
#Filtro por Rating Overall

fifa19_df_sugeridos = fifa19_df_stats[(fifa19_df_stats['Overall'] >= 60) & (fifa19_df_stats['Overall'] <= 70)]
#Filtro por Potencial

fifa19_df_sugeridos = fifa19_df_sugeridos[(fifa19_df_sugeridos['Potential'] >= 80) & (fifa19_df_sugeridos['Potential'] <= 90)]
fifa19_df_sugeridos