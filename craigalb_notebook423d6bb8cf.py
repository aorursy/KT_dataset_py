import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
menu = pd.read_csv('../input/Netflix Shows.csv',encoding='cp1252')

menu.info()
measuring=['user rating score']

for m in measuring:

    plot=sns.swarmplot(x='rating',y=m,data=menu)

    plt.setp(plot.get_xticklabels(),rotation=35)

    plt.title('TV ratings for dataset')

    plt.show()

That 