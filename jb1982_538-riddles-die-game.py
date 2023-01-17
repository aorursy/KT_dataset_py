# import modules

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
def die_game(no_of_sides, sims=1):

    

    # initialise list that will contain total number of rounds needed to meet required condition

    total_rounds = []

    # run for loop for required number of simulations

    for sim in range(sims):         

        # initialise a die and the no of rounds to zero

        die_values = list(range(1, (no_of_sides + 1)))

        no_of_rounds = 0        

        # then start a while loop that breaks when final the condition met

        while True:

            roll_results = []

            for x in range(no_of_sides):

                roll = np.random.choice(die_values)

                roll_results.append(roll)

            die_values = roll_results

            no_of_rounds += 1

            if all(elem == die_values[0] for elem in die_values):

                total_rounds.append(no_of_rounds)

                break

    return total_rounds
# run the simulation for a six sides die, with 10k simulations

total_rounds = die_game(6, 10000)

print(np.mean(total_rounds))
# plot the results of the above simulation

sns.set_style('ticks')

sns.distplot(total_rounds, bins = 72, kde = False)

sns.despine(top = True, right = True)

plt.xlabel("rounds")

plt.ylabel("count")

plt.title("Rounds to complete for six-sided die")

plt.show()
# run the simulation for dies having between two and ten sides to see what the results look like

sides = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for x in sides:

    total_rounds = die_game(x, 10000)

    print("The average number of rounds to finish with a " + str(x) + " sided die is " + str(np.mean(total_rounds)) + "\n")