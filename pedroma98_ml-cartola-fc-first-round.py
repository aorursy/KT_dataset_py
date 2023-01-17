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
cartola = pd.read_csv("../input/cartola-pre/cartola_data.csv", encoding='latin1')
cartola = cartola[cartola.status == "Provável"]
print(f"DataFrame with 'Prováveis' shape: {cartola.shape}")
cartola.head()
posicao_dummie = pd.get_dummies(cartola["posicao"]).iloc[:, 1:-1]
X = pd.concat([posicao_dummie,cartola[["preco_antes_rodada", "jogando_casa", "prop_vitoria", "prop_vitoria_adv"]]], axis = 1)
y = cartola["pontos"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 28)
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from sklearn.metrics import mean_squared_error as mse


nn = range(1, 26)

mse_test_list = []
mse_train_list = []

for k in nn:
    #fitting model with k neighbors
    knn_model = knn_reg(n_neighbors = k)
    knn_model.fit(X_train, y_train)
    #pred for test and train
    y_pred_test = knn_model.predict(X_test)
    y_pred_train = knn_model.predict(X_train)
    #append to list
    mse_test_list.append(mse(y_test, y_pred_test))
    mse_train_list.append(mse(y_train, y_pred_train))
list(zip(mse_test_list, mse_train_list))
import matplotlib.pyplot as plt

plt.figure()
plt.plot(mse_test_list, label = "Test")
plt.plot(mse_train_list, label = "Train")
plt.legend()
plt.show()
