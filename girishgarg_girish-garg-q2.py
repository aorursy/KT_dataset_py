import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
data = {

    'Team' : ['CSK', 'KKR', 'DC', 'MI'],

    'Score': [102, 220, 165, 177]

}

dataset = pd.DataFrame(data)
bar_plot = plt.bar(dataset.iloc[:,0], dataset.iloc[:,1])

index_max_score = dataset.iloc[:,1].idxmax()

bar_plot[index_max_score].set_color('red')

plt.show()