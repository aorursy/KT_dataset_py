import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

dados = pd.read_csv('../input/beer-consumption-sao-paulo/Consumo_cerveja.csv',thousands='.', decimal=',')

dados = dados.dropna()

dados = dados.reset_index(drop=True)

dados['Data'] = pd.to_datetime(dados['Data']).dt.strftime('%d/%m/%Y')

print(dados.shape)

dados
dados.describe().round(2)
dados.corr().round(4)
def plotar_grafico(dado,title,ylabel,xlabel):

    fig, ax = plt.subplots(figsize=(20, 6)) 

    ax.set_title(title, fontsize=20)

    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=16)

    ax = dado.plot(fontsize=14)

    

    

plotar_grafico(dados['Consumo de cerveja (litros)'], 'Consumo de Cerveja', 'Litros', 'Dias')

plotar_grafico(dados['Temperatura Media (C)'], 'Temperatura', 'Celsius', 'Dias')

plotar_grafico(dados['Precipitacao (mm)'], 'Chuva', 'mililitros', 'Dias')
plotar_grafico(dados['Consumo de cerveja (litros)'][:36], 'Consumo de Cerveja', 'Litros', 'Dias')

plotar_grafico(dados['Temperatura Media (C)'][:36], 'Temperatura', 'Celsius', 'Dias')

plotar_grafico(dados['Precipitacao (mm)'][:36], 'Chuva', 'mililitros', 'Dias')
import seaborn as sns



ax = sns.boxplot(y='Final de Semana', x='Consumo de cerveja (litros)', data=dados, orient='h', width=0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Consumo de Cerveja', fontsize=20)

ax.set_xlabel('Litros', fontsize=16)

ax.set_ylabel('Final de Semana', fontsize=16)

ax
ax = sns.distplot(dados['Consumo de cerveja (litros)'])

ax.figure.set_size_inches(12, 6)

ax.set_title('Distrubuição de Frequências', fontsize=20)

ax.set_ylabel('Consumo de Cerveja(Litros)', fontsize=16)

ax
ax = sns.pairplot(dados, kind='reg', y_vars='Consumo de cerveja (litros)', x_vars=['Temperatura Media (C)', 'Precipitacao (mm)', 'Final de Semana'])

ax
from sklearn.model_selection import train_test_split

X= dados[['Temperatura Media (C)', 'Precipitacao (mm)', 'Final de Semana']]

Y= dados['Consumo de cerveja (litros)']

x_treino, x_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.33, random_state=150)
ax = sns.jointplot(x='Temperatura Media (C)', y='Consumo de cerveja (litros)', data=dados, kind='reg')

ax.fig.suptitle('Dispersão - Consumo X Temperatura', fontsize=18, y=1.05)
from sklearn.linear_model import LinearRegression

from sklearn import metrics

modelo = LinearRegression()
modelo.fit(x_treino, y_treino)

y_previsto = modelo.predict(x_teste)

print('R² = {}'.format(metrics.r2_score(y_teste, y_previsto).round(2)))
temp_med=35

chuva=3

fds=0

entrada=[[temp_med, chuva, fds]]

print('{0:.2f} litros'.format(modelo.predict(entrada)[0]))
%%time



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.linear_model import (

    BayesianRidge,

    SGDRegressor,

)



# split the dataset beteween train and test

def split(x, y, plot=False):

    # seed

    # train_test_split

    train_x, test_x, train_y, test_y = train_test_split(

        x, y, test_size=0.1, random_state=42367,

    )

    if plot:

        print(

            "sizes: train (x,y) and test (x,y)",

            train_x.shape,

            train_y.shape,

            test_x.shape,

            test_y.shape,

        )

    return train_x, test_x, train_y, test_y





# Just train and valid the model

def run_reg_linear(train_x, test_x, train_y, test_y, model, plot=False):

    model.fit(train_x, train_y)

    test_pred = model.predict(test_x)



    mse = mean_squared_error(test_y, test_pred)

    mae = mean_absolute_error(test_y, test_pred)

    r2 = r2_score(test_y, test_pred)



    if plot:

        print("*" * 40)

        print("r2 score", r2)

        print("mse", mse)

        print("mae", mae)

        print("*" * 40)



    return r2, mse





# Train with all models then return a table with scores

def train_test_show(train_x, test_x, train_y, test_y):

    valores = []

    models = [

        ("BayesianRidge", BayesianRidge(n_iter=3000,compute_score=True)),

        ("MLPRegressor", MLPRegressor(learning_rate='adaptive',max_iter=10000)),

        ("RandomForestRegressor", RandomForestRegressor(n_jobs=-1)),

    ]

    for model in models:

        print(model[0])

        valores.append(

            (model[0], *run_reg_linear(train_x, test_x, train_y, test_y, model[1]))

        )

    valores = pd.DataFrame(valores, columns=["Model", "R2", "MSE"])

    return valores.style.background_gradient(cmap="Reds", low=0, high=1)





train_test_show(x_treino, x_teste, y_treino, y_teste)
from sklearn.model_selection import RandomizedSearchCV

import numpy as np



tol = [round(x/1000000.,5) for x in np.linspace(start=0, stop=100000, num=2000)]

alpha_1 = [round(x/100000.,4) for x in np.linspace(start=0, stop=100000, num=2000)]

alpha_2 = [round(x/100000.,4) for x in np.linspace(start=0, stop=100000, num=2000)]

lambda_1 = [round(x/100000.,4) for x in np.linspace(start=0, stop=100000, num=2000)]

lambda_2 = [round(x/100000.,4) for x in np.linspace(start=0, stop=100000, num=2000)]

fit_intercept = [True,False]

normalize = [True,False]

n_iter = [10000]

    

random_grid = {

    "tol": tol,

    "alpha_1": alpha_1,

    "alpha_2": alpha_2,

    "lambda_1": lambda_1,

    "lambda_2": lambda_2,   

    "fit_intercept": fit_intercept,  

    "normalize": normalize,

    'n_iter':n_iter





}



br = BayesianRidge()

br_random = RandomizedSearchCV(

    estimator=br,

    param_distributions=random_grid,

    n_iter=2000,

    cv=5,

    verbose=2,

    random_state=42,

    n_jobs=-1,

)



br_random.fit(

   x_treino, y_treino

)
p = br_random.best_params_

print(p)

score= run_reg_linear(x_treino, x_teste, y_treino, y_teste, BayesianRidge(

    tol=p['tol'],

    alpha_1= p['alpha_1'],

    alpha_2= p['alpha_2'],

    lambda_1= p['lambda_1'],

    lambda_2= p['lambda_2'],   

    fit_intercept= p['fit_intercept'],  

    normalize= p['normalize'],

    n_iter=p['n_iter']

), True)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=1, stop=1000, num=100)]

# Number of features to consider at every split

max_features = ["auto", "sqrt"]

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 1100, num=110)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 20,25,30]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4,6,8,10]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {

    "n_estimators": n_estimators,

    "max_features": max_features,

    "max_depth": max_depth,

    "min_samples_split": min_samples_split,

    "min_samples_leaf": min_samples_leaf,

    "bootstrap": bootstrap,

}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor(n_jobs=-1)

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=random_grid,

    n_iter=500,

    cv=4,

    verbose=2,

    random_state=42,

    n_jobs=-1,

)

# Fit the random search model

rf_random.fit(

   x_treino, y_treino

)
p = rf_random.best_params_

print(p)

score= run_reg_linear(x_treino, x_teste, y_treino, y_teste, RandomForestRegressor(

    n_estimators=p['n_estimators'],

    max_features= p['max_features'],

    max_depth= p['max_depth'],

    min_samples_split= p['min_samples_split'],

    min_samples_leaf= p['min_samples_leaf'],   

    bootstrap= p['bootstrap'],  

), True)


