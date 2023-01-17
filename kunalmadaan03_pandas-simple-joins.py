import numpy as np

import pandas as pd
pd.read_csv("../input/mydataset/cars1.csv")
pd.read_csv("../input/mydataset/cars2.csv")
cars1 = pd.read_csv("../input/mydataset/cars1.csv")

cars2 = pd.read_csv("../input/mydataset/cars2.csv")
cars1.info()
cars1 = cars1.drop(["Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12","Unnamed: 13"],axis = 1)

cars1.head()
print("Number of observation in Cars1 dataset: ",len(cars1))
print("Number of observation in Cars2 dataset: ",len(cars2))
cars = pd.concat([cars1,cars2],axis=0)
cars.info()
x = np.random.randint(15000,73000,size=(len(cars),1))
cars["owners"] = x
cars