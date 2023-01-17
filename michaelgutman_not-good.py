# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
data["rolling 21"] = data["Weighted_Price"].rolling(21).mean()
data["std"] = data["Weighted_Price"].rolling(21).std()
data["+1"] = data["rolling 21"] + data["std"]
data["-1"] = data["rolling 21"] - data["std"]
calc = data.loc[:, ["Timestamp", "Weighted_Price", "rolling 21", "+1", "-1"]]
calc["Target"] = calc["Weighted_Price"].shift(-60)
calc = calc.dropna(how="any")
#for col in calc.columns:
#    if col not in ["Timestamp", "Target"]:
#        plt.plot(calc["Timestamp"].tail(500).head(300), calc[col].tail(500).head(300), label=col)
#plt.legend()
#plt.show()
#plt.close()
#test, train = np.split(calc.tail(len(calc)-1), 2)
#sets = np.split(calc, 14)
#train = sets[0]
#test = sets[1]
#for i in range(2,14,2):
#    train = train.append(sets[i])
#    test = test.append(sets[i+1])
def get_mae(max_depth, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=10)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
train_x, test_x, train_y, test_y = train_test_split(calc.drop(columns=["Target", "Timestamp"]), calc["Target"], random_state=10)
for d in range(1,10):
    print(d, get_mae(d, train_x, test_x, train_y, test_y))

#clf = DecisionTreeRegressor(random_state=10, max_depth = d+1)
#clf.fit(train_x.drop(columns=["Target", "Timestamp"]), train_y)
#print(clf.score(test_x.drop(columns=["Target", "Timestamp"]), test_y))
