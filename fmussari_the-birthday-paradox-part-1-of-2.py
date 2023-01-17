import itertools
import random

# Create a set {1, 2, ..., 365} with Set Comprehention
days_of_year = {i+1 for i in range(365)}

n = 9  # Lets take 9 persons

print(list(days_of_year)[0:10]+['....'])
random.seed(1.12)  
# Change 1.12 if you want to get different random numbers from «random.choice»
# But I find that with 1.12 two persons get the birthday number 27

# To simulate random people in a party, 
# Lets generate a dictionary of persons p1, p2, ..., pn 
# that can take one birthday number at random
dict = {"p"+str(i+1): random.choice(list(days_of_year)) for i in range(n)}
dict

cards = 'ABCD'
n = 2
set_of_all = itertools.product(cards, repeat=n)
set_of_all = {comb for comb in set_of_all}
print('Given a deck of {} cards, there are {} different sequences of size {}'.format(len(cards),len(set_of_all),n))
set_of_all
set_of_shared_card = {comb for comb in list(set_of_all) if comb[0] == comb[1]}
print('There are {} different sequences with a card repeated'.format(len(set_of_shared_card),))
set_of_shared_card

prob = 100*len(set_of_shared_card)/len(set_of_all)
print('Given a set of {} cards, it takes {} people for a {}% chance of two of them sharing a card'.format(len(cards),n,prob))
set_of_different_cards = set_of_all - set_of_shared_card
print('There are {} different sequences without repetitions'.format(len(set_of_different_cards),))
set_of_different_cards
set_of_permutations = itertools.permutations(cards, n)
set_of_permutations = {perm for perm in set_of_permutations}
print(len(set_of_permutations))
print(set_of_permutations == set_of_different_cards)
print(set_of_permutations)
