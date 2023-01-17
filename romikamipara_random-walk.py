import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
np.random.seed(123)
#Intializing steps

step = 50
#Throwing a die

dice = np.random.randint(1,7)
# Satisfying the condition

if dice <= 2 :

    step = step - 1

elif dice < 6 and dice > 2 :

    step += 1

else :

    step = step + np.random.randint(1,7)

print(dice,step)
random_walk = [0]

# Complete the ___

for x in range(100) :

    # Set step: last element in random_walk

    step = random_walk[-1]



    # Roll the dice

    dice = np.random.randint(1,7)



    # Determine next step

    if dice <= 2:

        step = step - 1

    elif dice <= 5:

        step = step + 1

    else:

        step = step + np.random.randint(1,7)



    # append next_step to random_walk

    random_walk.append(step)



# Print random_walk

print(random_walk)
# Initialize random_walk

random_walk = [0]



for x in range(100) :

    step = random_walk[-1]

    dice = np.random.randint(1,7)



    if dice <= 2:

        # Replace below: use max to make sure step can't go below 0

        step = max(0, step - 1)

    elif dice <= 5:

        step = step + 1

    else:

        step = step + np.random.randint(1,7)



    random_walk.append(step)



print(random_walk)
plt.plot(random_walk)

plt.show()
# Initialize all_walks

all_walks = []



# Simulate random walk 10 times

for i in range(10) :



    # Code from before

    random_walk = [0]

    for x in range(100) :

        step = random_walk[-1]

        dice = np.random.randint(1,7)



        if dice <= 2:

            step = max(0, step - 1)

        elif dice <= 5:

            step = step + 1

        else:

            step = step + np.random.randint(1,7)

        random_walk.append(step)



    # Append random_walk to all_walks

    all_walks.append(random_walk)



# Print all_walks

print(all_walks)
# Convert all_walks to Numpy array



np_all_walks = np.array(all_walks)
print(np_all_walks)
print(np_all_walks.shape)
plt.plot(np_all_walks)

plt.show()
# Transpose np_all_walks

np_all_walks_transpose = np.transpose(np_all_walks)
print(np_all_walks_transpose)
print(np_all_walks_transpose.shape)
plt.plot(np_all_walks_transpose)

plt.show()
# Simulate random walk 250 times

all_walks = []

for i in range(250) :

    random_walk = [0]

    for x in range(100) :

        step = random_walk[-1]

        dice = np.random.randint(1,7)

        if dice <= 2:

            step = max(0, step - 1)

        elif dice <= 5:

            step = step + 1

        else:

            step = step + np.random.randint(1,7)



        # Implement clumsiness

        if np.random.rand() <= 0.001:

            step = 0



        random_walk.append(step)

    all_walks.append(random_walk)
np_all_walks_transpose = np.transpose(np.array(all_walks))

plt.plot(np_all_walks_transpose)

plt.show()
# Select last row from transposed matrix

ends = np.array(np_all_walks_transpose[-1])

ends
# Plot histogram of ends, display plot

plt.hist(ends)

plt.show()
np.random.seed(123)

# Simulate random walk 500 times

all_walks = []

for i in range(500) :

    random_walk = [0]

    for x in range(100) :

        step = random_walk[-1]

        dice = np.random.randint(1,7)

        if dice <= 2:

            step = max(0, step - 1)

        elif dice <= 5:

            step = step + 1

        else:

            step = step + np.random.randint(1,7)

        if np.random.rand() <= 0.001 :

            step = 0

        random_walk.append(step)

    all_walks.append(random_walk)

# Create and plot np_aw_t

np_all_walks_transpose = np.transpose(np.array(all_walks))
plt.plot(np_all_walks_transpose)

plt.show()
# Select last row from transposed np array

ends = np.array(np_all_walks_transpose[-1])



# Plot histogram of ends, display plot

plt.hist(ends)

plt.show()
ends
np.count_nonzero(ends)
np.count_nonzero(ends>=60)
print((np.count_nonzero(ends>=60)/len(ends))*100)