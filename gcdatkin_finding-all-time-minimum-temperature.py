import numpy as np

import pandas as pd
data = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")
data
minimum_temperature = data['MinTemp'].min()
print(f"Minimum Temperature: {minimum_temperature} Degrees Celsius")