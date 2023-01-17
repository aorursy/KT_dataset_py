# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Ajouter chaque jour le nombre de nouveaux cas en France

nouveaux_cas = [2,4,20,19,43,30,61,21,73,138,190,336,177,286,372,497,595,785,838,924,1238,1081]

somme = np.sum(nouveaux_cas)

print(somme)
def cas_totaux(nouveaux_cas):

    cas_totaux = []

    hier = 0

    for cas in nouveaux_cas:

        cas_totaux.append(hier + cas)

        hier = cas_totaux[-1]

    return cas_totaux
def diff(log_cas):

    diff = [0]

    for i in range(1,len(log_cas)):

        diff.append(log_cas[i]-log_cas[i-1])

    return diff
cas = cas_totaux(nouveaux_cas)

log_cas = [math.log(x) for x in cas]

diff_list = diff(log_cas)
plt.plot([i for i in range(len(cas))], cas)
plt.plot([i for i in range(len(log_cas))], log_cas)
plt.plot([i for i in range(len(diff_list[1:]))], diff_list[1:])
trend_diff = diff_list[12:]



diff_mean = np.mean(trend_diff)

diff_max = max(trend_diff)

diff_min = min(trend_diff)



print(diff_min, diff_max, diff_mean)
def pred_cas(d, c, jour):

    cas_pred = [log_cas[-1]]

    for i in range(jour):

        cas_pred.append(cas_pred[-1] + d)

    return cas_pred[1:]
jour = 7



log_cas_pred_mean = log_cas + pred_cas(diff_mean, log_cas, jour) 

log_cas_pred_max = log_cas + pred_cas(diff_max, log_cas, jour) 

log_cas_pred_min = log_cas + pred_cas(diff_min, log_cas, jour) 
x = [i for i in range(len(log_cas_pred_mean))]



fig, ax = plt.subplots(1, figsize=(8, 6))



# Set the title for the figure

fig.suptitle('Log du nombre de cas', fontsize=15)



# Draw all the lines in the same plot, assigning a label for each one to be

# shown in the legend.

ax.plot(x, log_cas_pred_mean, color="blue", label="cas moyen")

ax.plot(x, log_cas_pred_min, color="green", label="meilleur cas")

ax.plot(x, log_cas_pred_max, color="red", label="pire cas")



# Add a legend, and position it on the lower right (with no box)

plt.legend(loc="upper left", frameon=False)



plt.show()
nb_cas_mean = [math.exp(log) for log in log_cas_pred_mean]

nb_cas_min = [math.exp(log) for log in log_cas_pred_min]

nb_cas_max = [math.exp(log) for log in log_cas_pred_max]
x = [i for i in range(len(nb_cas_mean))]



fig, ax = plt.subplots(1, figsize=(8, 6))



# Set the title for the figure

fig.suptitle('Nombre de cas en France', fontsize=15)



# Draw all the lines in the same plot, assigning a label for each one to be

# shown in the legend.

ax.plot(x, nb_cas_mean, color="blue", label="cas moyen")

ax.plot(x, nb_cas_min, color="green", label="meilleur cas")

ax.plot(x, nb_cas_max, color="red", label="pire cas")



# Add a legend, and position it on the lower right (with no box)

plt.legend(loc="upper left", frameon=False)



plt.show()
print(int(nb_cas_min[len(nouveaux_cas)]), int(nb_cas_mean[len(nouveaux_cas)]), int(nb_cas_max[len(nouveaux_cas)]))