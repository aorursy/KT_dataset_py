%matplotlib inline

import numpy as np

import pandas as pd



df = pd.read_csv("../input/PastHires.csv")

df.head()
df.head(10)
df.tail(4)
df.shape
df.size
len(df)
df.columns
df['Hired']
df['Hired'][:5]
df['Hired'][5]
df[['Years Experience', 'Hired']]
df[['Years Experience', 'Hired']][:5]
df.sort_values(['Years Experience'])
degree_counts = df['Level of Education'].value_counts()

degree_counts
degree_counts.plot(kind='bar')
%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt



values = np.random.uniform(-10.0, 10.0, 100000)

plt.hist(values, 50)

plt.show()