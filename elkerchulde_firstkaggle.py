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
daaset = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")
daaset.head()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
daaset.shape
daaset=daaset.dropna(axis=0)
X = pd.DataFrame(np.c_[daaset["Temperatura Media (C)"]],columns=["Temperatura Media (C)"])
type(X)
X
Y= daaset["Consumo de cerveja (litros)"]
Y
media=np.array(daaset["Temperatura Media (C)"])
media=media.tolist()
type(media)
media2=[]
for i in media:
    media2.append(float(str(i).replace(",",".")))
media2
X=np.array(media2)
X
Regression_model = LinearRegression()
X.shape
X=X.reshape(-1,1)
X.shape
Regression_model.fit(X,Y)
Regression_model.coef_
Regression_model.intercept_
Y_prediction = Regression_model.predict(X)
Y_prediction
plt.scatter(X,Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X,Y_prediction, color="b")
plt.show()
r2sc= r2_score(Y,Y_prediction)
print("The prediction R2 score is:", r2sc)
