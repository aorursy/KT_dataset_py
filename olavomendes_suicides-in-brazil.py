import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd



suicides = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
# Separating the number of suicides in Brazil

brazil = suicides[suicides['country'] == 'Brazil']



plt.figure(figsize=(12, 8))



sns.lineplot(x=suicides['year'], y=suicides['suicides/100k pop'], label='World')

sns.lineplot(x=brazil['year'], y=brazil['suicides/100k pop'], label='Brazil')



plt.title('Comparison of suicides between Brazil and the world')


