import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
nutrition = pd.read_csv('../input/starbucks_drinkMenu_expanded.csv')

nutrition.describe().transpose()
sodium = nutrition[' Sodium (mg)']
sns.distplot(sodium, kde=True, bins=9, color='green').set_title('Sodium in Starbucks Items')

plt.ylabel('Count')