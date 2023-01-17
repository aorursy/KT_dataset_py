import random
import numpy as np
import matplotlib.pyplot as plt
rolls = []
#10,000 iterations of rolling 1000 dice
for i in range(10000):
    rolls.append(np.sum(random.choices(range(1,7),k=1000)))
print(rolls)
print(np.mean(rolls))
print(np.std(rolls))
plt.hist(rolls,bins = 100);
rolls2 = []
#10,000 iterations of rolling 1000 dice
for i in range(10000):
    rolls2.append(1000*(random.choice(range(1,7))))
print(rolls2)
print(np.mean(rolls2))
print(np.std(rolls2))
plt.hist(rolls2,bins = 20);