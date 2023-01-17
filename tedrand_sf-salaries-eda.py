import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/Salaries.csv",low_memory=False)
plt.hist(df['TotalPay'], bins=20, range=[0, 300000])
df['TotalPay'].mean()