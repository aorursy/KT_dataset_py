%matplotlib inline



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/Salaries.csv")

df[:1]
df['JobTitle'] = df['JobTitle'].str.upper()

df.groupby('JobTitle').mean().sort('TotalPay', ascending = False)['TotalPay']
df[df['OvertimePay'] == 'Not Provided'] = 0.00

df['OvertimePay'] = df['OvertimePay'].astype(float)
