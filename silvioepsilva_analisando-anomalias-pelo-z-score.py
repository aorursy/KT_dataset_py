import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
# Base de dados

xi = [3246, 3476, 3724, 3773, 3837, 3968, 4198, 4048, 4170, 4226, 4788, 4009, 3568, 4357]



# Base de dados com anomalia

xy = [6799, 3476, 3724, 3773, 3837, 3968, 4198, 4048, 4170, 4226, 4788, 4009, 3568, 4357]
# Definindo funcao para facilitar a demonstração dos valores

def print_full(x):

    pd.set_option('display.max_rows', len(x))

    pd.set_option('display.max_columns', len(x))

    print(x)

    pd.reset_option('display.max_rows')

    pd.reset_option('display.max_rows')



# Definindo funcao para facilitar os calculos

def calculos(x):

    mean   = np.mean(x)

    median = np.median(x)

    moda   = stats.mode(x)

    stddev = np.std(x, ddof = 1)

    zscore = (x - mean) / stddev



    print("-> Lista: {}".format(str(x)))

    print("Mean   : {}".format(mean))

    print("Median : {}".format(median))

    print("Moda   : {}".format(moda))

    print("Stddev : {}".format(stddev))

    print("Z-score: {}".format(zscore))



    return mean, median, moda,stddev,zscore



def plt_normal_distribution(x):

    stats.probplot(x, dist='norm', plot=plt)

    plt.show()
# Realizando os calculos

xi_mean, xi_median, xi_moda, xi_stddev, xi_zscore = calculos(xi)

print('\n')

xy_mean, xy_median, xy_moda, xy_stddev, xy_zscore = calculos(xy)
# Visualizando os dados

df = pd.DataFrame({'Base (sem anomalias)': xi, 'Z-score (sem anomalias)': xi_zscore,

                   'Base (Com anomalias)': xy, 'Z-score (com anomalias)': xy_zscore})

df.round(2)

cols = ['Base (sem anomalias)','Z-score (sem anomalias)','Base (Com anomalias)','Z-score (com anomalias)']

df = df[cols]

print_full(df)
print('Dados sem anomalias')

plt_normal_distribution(xi)

print('Dados com anomalias')

plt_normal_distribution(xy)