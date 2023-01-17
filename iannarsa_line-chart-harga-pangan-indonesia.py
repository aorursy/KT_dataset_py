import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns
data_pangan = pd.read_csv('../input/pangan_19_20.csv', parse_dates = True)
list(data_pangan.columns)
plt.figure(figsize=(18,8))



plt.plot(data_pangan['Date'],data_pangan['Cabai Rawit'])

plt.xticks(rotation=90)



plt.show()