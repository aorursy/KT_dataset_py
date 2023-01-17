import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
df=pd.read_csv("../input/game_data.csv")
print(df.head(5))
print(df.shape)
df.describe

