import pandas as pd
import numpy as np
data = pd.read_csv("../input/battles.csv")
data.shape
data.head() 
data.dropna()
data.shape
data.dropna(inplace=True)
data.shape
data = pd.read_csv("../input/battles.csv")
data.dropna(how='all')
data.shape
data.dropna(how='all',axis=1)
data.shape
data['attacker_2'].dropna()


