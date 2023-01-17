import pandas as pd

energy = pd.read_excel('energy.xlsx')

print(energy.info())
energy.head(5)
treino = [1, 100, 500, 2000, 5000, 7654]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

features = ['AT', 'V', 'AP', 'RH']
target = 'PE'

train_sizes, train_scores, test_scores  = learning_curve( estimator = LinearRegression(), 
                                                   X = energy[features], y = energy[target],
                                                   train_sizes = treino, cv = 5,
                                                   scoring = 'neg_mean_squared_error')

train_scores = -train_scores.mean(axis=1) 
test_scores = -test_scores.mean(axis=1) 
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores, label = 'Erro no treino')
plt.plot(train_sizes, test_scores, label = 'Erro na validação')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Tamanho do conjunto de treino', fontsize = 14)
plt.title('Curvas de aprendizagem para a regressão linear', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,40)
treino = [1, 100, 500, 2000, 5000, 7654]

features = ['AT', 'V', 'AP', 'RH']
target = 'PE'

from sklearn.ensemble import RandomForestRegressor

train_sizes, train_scores, test_scores  = learning_curve( estimator = RandomForestRegressor(), 
                                                           X = energy[features], y = energy[target],
                                                           train_sizes = treino, cv = 5,
                                                           scoring = 'neg_mean_squared_error')

train_scores = -train_scores.mean(axis=1) 
test_scores = -test_scores.mean(axis=1) 

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores, label = 'Erro no treino')
plt.plot(train_sizes, test_scores, label = 'Erro na validação')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Tamanho do conjunto de treino', fontsize = 14)
plt.title('Curvas de aprendizagem para Random Forest', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,40)
# Regressão
import pandas as pd

energy = pd.read_csv('../input/addata/ad.data')

energy.head(5)

treino = [1, 15, 45, 70, 100, 120, 160]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

features = ['TV', 'Radio', 'Newspaper']
target = 'Sales'

train_sizes, train_scores, test_scores  = learning_curve( estimator = LinearRegression(), 
                                                   X = energy[features], y = energy[target],
                                                   train_sizes = treino, cv = 5,
                                                   scoring = 'neg_mean_squared_error')

train_scores = -train_scores.mean(axis=1) 
test_scores = -test_scores.mean(axis=1) 

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores, label = 'Erro no treino')
plt.plot(train_sizes, test_scores, label = 'Erro na validação')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Tamanho do conjunto de treino', fontsize = 14)
plt.title('Curvas de aprendizagem para a regressão linear', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,40)
#Regressão
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPRegressor

titanic = pd.read_csv('../input/addata/ad.data')

print(titanic.info())
titanic.head(5)

treino = [1, 15, 45, 70, 100, 120, 160]


from sklearn.model_selection import learning_curve

features = ['TV', 'Radio', 'Newspaper']
target = 'Sales'

train_sizes, train_scores, test_scores  = learning_curve( estimator = MLPRegressor(hidden_layer_sizes=(15), random_state=1), X = titanic[features], y = titanic[target], train_sizes = treino, cv = 5, scoring = 'neg_mean_squared_error')
train_scores = -train_scores.mean(axis=1) 
test_scores = -test_scores.mean(axis=1) 

import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('seaborn')

plt.plot(train_sizes, train_scores, label = 'Erro no treino')
plt.plot(train_sizes, test_scores, label = 'Erro na validação')

plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Tamanho do conjunto de treino', fontsize = 14)
plt.title('Curvas de aprendizagem para a regressão linear', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,40)
