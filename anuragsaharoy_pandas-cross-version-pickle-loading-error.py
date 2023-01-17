!pip install pandas==0.24.1
import pandas as pd

pd.__version__
import os

print(os.listdir("../input"))
pd.read_pickle("../input/test.pkl")