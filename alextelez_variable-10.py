import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
# Читаем данные из csv
# Переменная df является типом DataFrame
my_filepath = "../input/steam.csv"
df = pd.read_csv(my_filepath)
df.head()
grouped_df = df[df.owners == '10000000-20000000'].groupby('developer').count()[['appid']]
grouped_df.head()
grouped_df.plot.hist(by=['appid'])
#Создаем csv файл
grouped_df.to_csv("output.csv", index=True)