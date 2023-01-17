# Import numpy & matplotlib
import numpy as np

import matplotlib.pyplot as plt
# Set seed in order to make the simulation reproducible
np.random.seed(1234)
# Initialize all walks
all_walks = []
# Simulate 1,000 random walks
for n in range(1000):

    random_walk = [0]

    for i in range(100):

        # Set step as last element in random_walk

        step = random_walk[-1]

    

        # Roll the die

        dice = np.random.randint(1,7)

    

        # Determine next step. They can't go underground so make sure step doesn't go below zero

        if dice <= 2:

            step = max(0, step - 1)

        elif dice <=5:

            step = step + 1

        else:

            step = step + np.random.randint(1,7)

            

        # Implement lack of self control: 0.2% chance of eating all the strawberries and going home

        if np.random.rand() <= 0.002:

            step = 0

        

        # Append the next step to random_walk

        random_walk.append(step)

    

    # Append random_walk to all_walks

    all_walks.append(random_walk)
# Convert all_walks to numpy array
np_all_walks = np.array(all_walks)
# Transpose np_all_walks so that each row represents the position after 1 throw for the 1,000 simulations 
np_all_walks_t = np.transpose(np_all_walks)
# Plot np_all_walks_t. Notice how some walks drop back down to zero when Danae & Shaelo eat the strawberries on the spot and return home. 
plt.plot(np_all_walks_t)

plt.xlabel("Die Rolls")

plt.ylabel("Steps")

plt.title("1,000 Random Walks")

plt.show()
# Select the last row from np_all_walks_t, which is the endpoint of all 1,000 simulations 
endpoints = np_all_walks_t[-1]
# Plot histogram of endpoints
plt.hist(endpoints)

plt.xlabel("Steps Reached at Endpoint")

plt.ylabel("Frequency")

plt.title("Histogram of 1,000 Random Walks")

plt.show()
# Calculate the chances that Danae and Shaelo reach Strawberry Summit
chances = np.mean(endpoints >= 100)

print(chances)