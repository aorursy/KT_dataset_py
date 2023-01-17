import pandas as pd
import numpy as np
df = pd.read_csv(r'../input/dummy-data/Dummy data.csv')
df
sample = []

for i in range(0, 101):
    sample.append([i,i+2])

df = df.rename(columns = {'numeber':'new_number'})
df[0:101] = pd.DataFrame(sample)
df
df.to_csv('new dummy data.csv', index = False) #把表格轉成csv檔