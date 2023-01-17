import pandas as pd
import os
print(os.listdir("../input"))


df = pd.read_csv('../input/game_data_clean.csv')
df.head()