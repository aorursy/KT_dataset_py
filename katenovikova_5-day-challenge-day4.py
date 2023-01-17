import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/scrubbed.csv")

sns.countplot(y = data['shape'])

plt.show()



