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
dataset = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")
dataset.head()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
dataset.shape
dataset = dataset.dropna(axis=0)
X = pd.DataFrame(np.c_[dataset["Temperatura Media (C)"]],columns=["Temperatura Media (C)"])
type(X)
X
y = dataset["Consumo de cerveja (litros)"]
y
media = np.array(dataset["Temperatura Media (C)"])
media = media.tolist()
type(media)
media2 = []
for i in media:
    media2.append(float(str(i).replace(",",".")))
media2
X = np.array(media2)
X
regression_model = LinearRegression()
X.shape
X = X.reshape(-1,1)
X.shape
regression_model.fit(X,y)
regression_model.coef_
regression_model.intercept_
y_prediction = regression_model.predict(X)
y_prediction
plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.plot(X,y_prediction, color ="r")
plt.show()
r2s = r2_score(y,y_prediction)
print("La prediccion del R2 score es de: ",r2s)
