import pandas as pd

import numpy as np

from math import log10

import matplotlib.pyplot as plt



referendum = pd.read_csv('../input/referendum.csv')



# Benford law

def first_digit_occurencies(col):

    return np.unique(referendum[col].astype(str).apply(lambda x: int(x[0])), return_counts=True)[1]



cols = ['Leave', 'Remain']

fdf = [0] * 9

for col in cols:

    fdf_col = first_digit_occurencies(col)

    fdf = [fdf[i] + fdf_col[i] for i in range(9)]



fdf = [fdf[i] / sum(fdf) for i in range(len(fdf))]

theoric_fdf = [log10(1 + 1 / (i + 1)) for i in range(9)]

signs = np.arange(1, 10)

ind = np.arange(9) #auxiliary variable so that the 'theoretical repartition' line doesn't get shifted

comp = {'sign': signs, 'Actual frequency': fdf, 'Benford frequency': theoric_fdf, 'ind': ind}

comp = pd.DataFrame(comp)



#plot

_, ax = plt.subplots()

comp[['ind', 'Benford frequency']].plot(x='ind', linestyle='-', marker='o', ax=ax)

comp[['sign', 'Actual frequency']].plot(x='sign', kind='bar', ax=ax)

plt.title("Actual vs. Benford's First Digit Frequency")

plt.show()