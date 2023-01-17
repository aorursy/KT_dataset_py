import itertools

import random

users = ["PB","MB","VM","RW","RL","PG","AY","ZB","KD","NT","CG","LM","JS","EV","TP","TN"]

random.shuffle(users)

hangouts = ["GREEN", "BLUE", "RED", "PURPLE"]

random.shuffle(hangouts)

hangouts = hangouts

result = {"GREEN": [], "BLUE": [], "RED": [], "PURPLE": []}

for i in range(len(users)):

    result[hangouts[i%4]].append(users[i])

print(result)