import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import norm



plt.style.use('seaborn')



np.random.seed(42)
# Un dé a une valeur de 1 à 6

dice_values = [1, 2, 3, 4, 5, 6]
N = 1000

heads = np.zeros(N, dtype=int)

for i in range(N):

    heads[i] = np.random.choice(dice_values)

heads;
values, counts = np.unique(heads, return_counts=True)



_ = plt.stem(values, (counts/N * 100), use_line_collection=True)

_ = plt.xlabel('Valeur du dé')

_ = plt.ylabel('Probabilité du lancer (%)')

_ = plt.title('Fonction de Masse (PMF)')



plt.show()
sizes = np.linspace(-5, 5, 5000)



_ = plt.plot(sizes, norm.pdf(sizes))

_ = plt.xlabel('Une variable (une unité quelconque)')

_ = plt.ylabel('Densité')

_ = plt.title('Fonction de Densité (PDF)')



plt.show()
plt.clf()



# Affiche la courbe

_ = plt.plot(sizes, norm.pdf(sizes), color="slateblue", alpha=0.4)



# Affiche l'aire totale de la distribution (donc probabilité = 1)

_ = plt.fill_between(sizes, norm.pdf(sizes), color="skyblue", alpha=0.6)



# Affiche l'aire correspondant aux valeurs comprises entre 0 et 2

sub_values = sizes[(sizes >= 0) & (sizes <=2)]

_ = plt.fill_between(sub_values, norm.pdf(sub_values), color="lightpink", alpha=0.6, hatch="X")

_ = plt.xlabel('Une variable (une unité quelconque)')

_ = plt.ylabel('Densité')

_ = plt.title('Fonction de Densité (PDF)')



plt.show()
plt.clf()



_ = plt.hist(sizes, density=True, cumulative=True, label='CDF', alpha=0.5, color='lightgreen', bins = 30)



_ = plt.plot([-5, 5], [0.5, 0.5], alpha=0.6, color='slateblue', linestyle='--')

_ = plt.plot([-5, 5], [0.75, 0.75], alpha=0.6, color='slateblue', linestyle='--')

_ = plt.plot([0, 0], [0, 1], alpha=0.6, color='slateblue', linestyle='--')

_ = plt.plot([2, 2], [0, 1], alpha=0.6, color='slateblue', linestyle='--')



_ = plt.xlabel('Une variable (une unité quelconque)')

_ = plt.ylabel('Probabilité')

_ = plt.title('Fonction de Distribution Cumulée')



plt.show()