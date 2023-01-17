import itertools
import math

import seaborn as sns
import matplotlib.pyplot as plt

cards = 'ABCD'
n = 3

set_of_all = set(itertools.product(cards, repeat=n))
print('Given a deck of {} cards, there are {} different sequences of size {}'.format(len(cards),len(set_of_all),n))
print(set_of_all)

set_of_permutations = set(itertools.permutations(cards, n))
print('There are {} different sequences without repetitions'.format(len(set_of_permutations)))
print(set_of_permutations)
set_of_shared_card = set_of_all - set_of_permutations
print('There are {} sequences with a card repeated'.format(len(set_of_shared_card),))
print(set_of_shared_card)
prob = 100*len(set_of_shared_card)/len(set_of_all)
print('Given a set of {} cards, it takes {} people for a {}% chance of two of them sharing a card'.format(len(cards),n,prob))
def plot_data(n):
    return((365**n - math.factorial(365)/math.factorial(365-n))/365**n)

data = [plot_data(i) for i in range (100)]
#import seaborn as sns; sns.set(color_codes=True)

sns.set()

ax = plt.plot(data)

for n in range(3,40,5):
    print('For {:2} people, there is a {:4.{prec}f}% chance of at least two of them sharing a birthday'.format(n,100*plot_data(n),prec=1))

