# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


df = pd.read_csv('../input/kc_house_data.csv')
print(df.describe())
cols = df.columns
# select some features
#print(cols[3])
fcols = cols[3:14]
fcols2 = fcols | cols[16:]
X = df[fcols2]

price = df['price']
# scale input features
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# divide data into training and test
x_train,x_ver,y_train,y_ver = train_test_split(Xs, price, random_state=3)
# test Neural Network regressor
numN= 16   # number of neurons per layer
maxNumLayers = 4
hLayer = (numN,)
scoreList = []
#for l in range(maxNumLayers+1):
#    nn = MLPRegressor(hidden_layer_sizes=hLayer,max_iter=5000,learning_rate='adaptive',solver='lbfgs')
#    nn.fit(x_train,y_train)
#    s = nn.score(x_ver,y_ver)
#    print("score", s)
#    scoreList.append(s)
#    hLayer = hLayer + (numN,)


# 2 layers appears best
nn = MLPRegressor(hidden_layer_sizes=(16,16,),max_iter=5000,learning_rate='adaptive',solver='lbfgs')
nn.fit(x_train,y_train)
s = nn.score(x_ver,y_ver)
print("score", s) 

y_predicted = nn.predict(x_ver)
plt.plot(y_ver,y_predicted,'o')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices by a NN")
plt.show()
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_ver,y_predicted)
rms = np.sqrt(mse)
print("RMS error in price prediction ",rms)
