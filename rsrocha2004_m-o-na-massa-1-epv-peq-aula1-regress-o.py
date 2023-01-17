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
# Notebook da Aula 1 (EPV PEQ Aula 1 - Regressão)

# BOSTON | MODELO DE FLORESTAS ALEATÓRIAS | MÉDIA DO ERRO ABSOLUTO E R2 (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

boston = sklearn.datasets.load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = sklearn.ensemble.RandomForestRegressor()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred) # Média do Erro Absoluto
r2 = sklearn.metrics.r2_score(y_test, y_test_pred) # Coeficiente de Correlação

print(f'MAE: {mae}')
print(f'R2: {r2}')
# Diabetes | Modelo Florestas Aleatórias | Média do Erro Quadrado & Variância Explicada (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

boston = sklearn.datasets.load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = sklearn.ensemble.RandomForestRegressor()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

# Usando outras métricas de regressão (Média do Erro Quadrado e Variância Explicada)

from sklearn.metrics import explained_variance_score

mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred) # Média do Erro Quadrado
var = explained_variance_score(y_test, y_test_pred) # Variância Explicada

print(f'MSE: {mse}') # Quanto mais próximo de ZERO, melhor.
print(f'VARIANCE: {var}') # Quanto mais próximo de 1, melhor.
# Diabetes | Modelo Florestas Aleatórias | Média do Erro Absoluto & R2 (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

diabetes = sklearn.datasets.load_diabetes()

X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = sklearn.ensemble.RandomForestRegressor()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred) # Média do Erro Absoluto
r2 = sklearn.metrics.r2_score(y_test, y_test_pred) # Coeficiente de Correlação

print(f'MAE: {mae}')
print(f'R2: {r2}') # Quanto mais próximo de 1, melhor.
# Diabetes | Modelo Florestas Aleatórias | Média do Erro Quadrado & Variância Explicada (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

diabetes = sklearn.datasets.load_diabetes()

X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = sklearn.ensemble.RandomForestRegressor()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

# Usando outras métricas de regressão (Média do Erro Quadrado e Variância Explicada)

from sklearn.metrics import explained_variance_score

mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred) # Média do Erro Quadrado
var = explained_variance_score(y_test, y_test_pred) # Variância Explicada

print(f'MSE: {mse}') # Quanto mais próximo de ZERO, melhor.
print(f'VARIANCE: {var}') # Quanto mais próximo de 1, melhor.
# Boston | Modelo Regressão Linear | Média do Erro Absoluto & R2 (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

# Acrescentado para chamar o modelo "Linear Regression"

from sklearn.linear_model import LinearRegression

boston = sklearn.datasets.load_boston()

X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = LinearRegression()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred) # Média do Erro Absoluto
r2 = sklearn.metrics.r2_score(y_test, y_test_pred) # Coeficiente de Correlação

print(f'MAE: {mae}')
print(f'R2: {r2}')
# Diabetes | Modelo Regressão Linear | Média do Erro Absoluto & R2 (Mão na Massa 1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble

# Acrescentado para chamar o modelo "Linear Regression"

from sklearn.linear_model import LinearRegression

diabetes = sklearn.datasets.load_diabetes()

X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1, random_state = 0)

m = LinearRegression()

m.fit(X_train, y_train)
y_test_pred = m.predict(X_test)

plt.plot(y_test, y_test_pred, '.')
plt.plot(plt.gca().get_ylim(),plt.gca().get_ylim())
plt.xlabel('y_test')
plt.ylabel('y_test_pred')

# Usando outras métricas de regressão (Média do Erro Quadrado e Variância Explicada)

from sklearn.metrics import explained_variance_score

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred) # Média do Erro Absoluto
r2 = sklearn.metrics.r2_score(y_test, y_test_pred) # Coeficiente de Correlação

print(f'MAE: {mae}')
print(f'R2: {r2}')