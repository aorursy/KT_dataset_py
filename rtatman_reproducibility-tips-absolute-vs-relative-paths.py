# library we'll need
import pandas as pd

# read in data using absolute path
data = pd.read_csv("/kaggle/input/WorldCupMatches.csv")
# read in data using relative path
data = pd.read_csv("../input/WorldCupMatches.csv")