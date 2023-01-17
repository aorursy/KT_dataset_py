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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



d_18 = pd.read_csv("/kaggle/input/2018qb/2018.csv")

d_17 = pd.read_csv("/kaggle/input/2017qb/2017.csv")

d_16 = pd.read_csv("/kaggle/input/2016qb/2016.csv")



df = pd.concat([d_18, d_17, d_16], ignore_index=True)

df2 = df.drop(columns = ['Rk', 'Tm', 'Age', 'Pos', 'G', 'GS', 'QBrec', 'Lng', 'GWD', '4QC', 'Cmp', 'Att', 'Rate'])

names = df2[["Player"]]

df3 = df2.drop(columns = ['Player'])

df3 = df3.astype(float)

df3 = df3.dropna().reindex()

df2
df2.describe()
%matplotlib inline

g = sns.PairGrid(df2)

g = g.map_upper(plt.scatter,marker='+')

g = g.map_lower(sns.kdeplot, cmap="hot",shade=True)

g = g.map_diag(sns.kdeplot, shade=True)
from sklearn.model_selection import train_test_split

import xgboost as xgb



X_train, X_test, y_train, y_test = train_test_split(df3[["Cmp%", "Yds","TD","TD%","Int","Int%","1D","Y/A","AY/A","Y/C","Y/G","Sk","Yds.1","NY/A","ANY/A","Sk%"]], df2[["QBR\n"]], test_size=0.2, random_state=0)



train = xgb.DMatrix(X_train, label=y_train)

test = xgb.DMatrix(X_test, label=y_test)
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot



iters = []

error = []

for i in np.linspace(0.1,1,15):

    for j in range(1,16,1):

        param = {

        'eval_metric':'rmse',

        'max_depth': j,

        'eta': i,

        'num_class': 1

        } 

        epochs = 25 



        model = xgb.train(param, train, epochs)

        predictions = model.predict(test)

        mse = mean_squared_error(y_test,predictions)

        error.append(mse)

        iters.append((i, j))



        

x_axis = np.linspace(0,86,225)

fig, ax = pyplot.subplots()

ax.plot(x_axis, error)

pyplot.ylabel('Mean Squared Error')

pyplot.title('Error v.s Iteration')

pyplot.show()
test2 = [x < 40 for x in error]

iter_index = [i for i in enumerate(error)]

iter_index

iters[120]

param_optimal = {

        'eval_metric':'rmse',

        'max_depth': 1,

        'eta': 0.6142857142857143,

        'num_class': 1

        } 

epochs_optimal = 20



model_optimal = xgb.train(param_optimal, train, epochs_optimal)

predictions_optimal = model_optimal.predict(test)

mse_optimal = mean_squared_error(y_test,predictions_optimal)



print(mse_optimal)

#print(predictions_optimal)

merged = names.merge(y_test, left_index=True, right_index=True, how='inner')

merged.head()

#names2 = names.merge(y_test,left_index= [y_test.index == names.index])
y_test.index

df_results = pd.merge(merged, X_test, how='outer', on = y_test.index)

df_results.insert(2, 'preds', predictions_optimal)

df_results
plt.scatter(df_results.index, df_results["preds"].values)

plt.scatter(df_results.index, df_results["QBR\n"].values)