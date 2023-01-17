import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
df = pd.read_excel('../input/Sudoku Stats.xlsx')
df.info()
df.describe()
ax = df['Time'].value_counts().head(20).plot.bar(
     figsize=(12,6),
     fontsize=14,
     color='aquamarine')
ax.set_title('Frequency of Sudoku Times', fontsize=20)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
sb.despine()
