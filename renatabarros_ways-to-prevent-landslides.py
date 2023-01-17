import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



data = pd.read_csv("../input/catalog.csv")

pd.DataFrame.describe(data, include = 'all')



data_np = np.array(data)

print(data_np)



# Defining a function that counts the number of fatalities associated with each type of landslide trigger


