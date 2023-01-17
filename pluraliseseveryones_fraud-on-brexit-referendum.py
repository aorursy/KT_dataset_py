import pandas as pd

import numpy as np

from math import log10

import matplotlib.pyplot as plt



referendum = pd.read_csv('../input/referendum.csv')



THRESHOLD = 1800000



# Benford law

def first_digit_occurencies(col, threshold=0):

    filtered = referendum[(referendum['Electorate'] < threshold )]

    digits = np.unique(filtered[col].astype(str).apply(lambda x: int(x[0])), return_counts=True)

    values = []

    idx = 0

    for i in range(10):

        if not i in digits[0]:

            values.append(0)

        else:

            values.append(digits[1][idx])

            idx+=1

    print(values)    

    return np.array(values[1:])



cols = ['Leave', 'Remain']

fdf = [0] * 9

for col in cols:

    fdf_col = first_digit_occurencies(col, THRESHOLD)

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