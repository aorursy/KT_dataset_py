import numpy as np # linear algebra
import pandas as pd # data processing
labels_list = ["Ilker", "Hakan", "Murat", "Serdar", "Erhan"]

data_list = [10, 20, 30, 40, 50]
pd.Series(data=data_list, index=labels_list)
pd.Series(data_list) # default index
np_array = np.array(data_list)

pd.Series(np_array) # Numpy arrays can also be given as data.
pd.Series(np_array, labels_list)
pd.Series(np_array, index=["A", "B", "C", "D", "E"])
data_dict = {"Ilker": 30, "Kemal": 80, "Mehmet": 60}

pd.Series(data_dict) # dictionary as Series
fruit2019 = pd.Series([3, 14, 22, 20], ["Apple", "Banana", "Mango", "Strawberry"])

fruit2019
fruit2020 = pd.Series([5, 10, 20, 40], ["Apple", "Banana", "Grapes", "Mango"])

fruit2020
total = fruit2019 + fruit2020

total
total["Mango"]
total["Grapes"]
try:
    print(total["Peach"])
except KeyError as e:
    print("A key error is thrown: {}".format(e))