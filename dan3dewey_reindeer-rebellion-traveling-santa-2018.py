import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import Counter
from time import time
from matplotlib import collections  as mc

sns.set_style('whitegrid')
df = pd.read_csv("../input/cities.csv")  #dd for running on kaggle
##df = pd.read_csv("./input/cities.csv")  #dd my local location
##dd df.head()
##dd plt.figure(figsize=(15, 10))
##dd plt.scatter(df.X, df.Y, s=1)
##dd plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="red")
##dd plt.grid(False)
##dd plt.show()
nb_cities = max(df.CityId)
print("Number of cities to visit : ", nb_cities)
##dd df.tail()
#dd: change primes to prime_flags
def sieve_eratosthenes(n):
    prime_flags = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (prime_flags[p] == True):
            for i in range(p * 2, n + 1, p):
                prime_flags[i] = False
        p += 1
    return prime_flags
prime_flags = np.array(sieve_eratosthenes(nb_cities)).astype(int)
df['Prime'] = prime_flags
#dd This is what Santa wants, but it is only applied to stepNumber % 10 == 0 cities ?!
#dd  "Not acceptable!" - R'deer
#dd This variable is only used in dist_matrix() below if penalize=True
penalization = 0.1 * (1 - prime_flags) + 1
df.head()
#dd Shows 179967 non-primes and 17802 primes
##dd plt.figure(figsize=(15, 10))
##dd sns.countplot(df.Prime)
##dd plt.title("Prime repartition : " + str(Counter(df.Prime)))
##dd plt.show()
#dd Change colors to green, red, and black(NP)
plt.figure(figsize=(15, 10))
plt.scatter(df[df['Prime'] == 0].X, df[df['Prime'] == 0].Y, s=1, alpha=0.3, c='green')
plt.scatter(df[df['Prime'] == 1].X, df[df['Prime'] == 1].Y, s=1, alpha=0.7, c='red')
plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="black")
plt.grid(False)
plt.title('Visualisation of cities')
plt.show()
# Put all the city corrdinates in an np array
coordinates = np.array([df.X, df.Y])
# Various routines

# Assign a distance measure from city i to all others
def dist_array(coords_in, i, RightLeft=False):
    begin = np.array([df.X[i], df.Y[i]])[:, np.newaxis]
    # if RightLeft then scale/reduce x,y coords/distances
    # that are more than some distance to the right of city i.
    # This encourages not leaving cities far behind (on the right)
    # and once on the right the path will have a general trend to the left.
    if RightLeft:
        # scale the X,Y values to be smaller if to the right of city i
        coords_mod = coords_in.copy()
        # Different values tried: 500.0, 700.0, 900.0, 600.0, 400.0, 250.0, 160.0, 40.0, 16.0, 20.0, 100.0
        x_width = 600.0  # 600 is best so far
        bound_right = begin[0] + x_width
        x_far_right = 1.0*((coords_in[0] - bound_right) > 0.0)
        coords_mod[0] = bound_right + (coords_mod[0]-bound_right)*(1.0-0.75*x_far_right)
        coords_mod[1] = begin[1] + (coords_mod[1]-begin[1])*(1.0-0.50*x_far_right)
        mat = coords_mod - begin
    else:
        mat =  coords_in - begin
    return np.linalg.norm(mat, ord=2, axis=0)

# return the index of the nearest available city
def get_next_city(dist, avail):
    return avail[np.argmin(dist[avail])]

def plot_path(path, coordinates):
    # Plot tour
    lines = [[coordinates[: ,path[i-1]], coordinates[:, path[i]]] for i in range(1, len(path))]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
    # add the North Pole location
    plt.scatter(coordinates[0][0], coordinates[1][0], s=150, c="red", marker="*", linewidth=3)
    # and first cities on the path
    plt.scatter(coordinates[0][path[1:10]], coordinates[1][path[1:10]], s=15, c="black")
    plt.show()
    
# Calculate the Score, Carrots, Length (RR=True to select the Reindeer Rebellion scoring)
def get_score(path, coords, prime_flags, RR=False):
    # RR=True calculates the Reindeer preferred scoring
    score = 0
    carrots = 0 
    length = 0
    steps_since_carrot = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coords[:, end] - coords[:, begin], ord=2)
        length += distance
        # Choose the scoring method:
        if not RR:
            # Usual scoring, is this one of the 10th-city steps?
            if i % 10 == 0:
                # if the starting city is prime then a carrot and no penalties
                if prime_flags[begin]:
                    carrots += 1
                # if not prime, no carrot and a penalty
                else:
                    distance *= 1.1
            score += distance
        else:
            # RR scoring
            steps_since_carrot += 1
            if prime_flags[end]:
                # got carrots here!
                carrots += 1
                steps_since_carrot = 0
            # any penalty?
            if steps_since_carrot > 10:
                distance *= (1.0 + 0.05*(steps_since_carrot - 10))
            score += distance
    return score, carrots, length
# Initialize the left_cities

# All cities:
Nth = 1;  city_start = 1

# Can use only every Nth city for quicker testing
##Nth = 37; city_start = 1
# primes: 1-0.09, 1+0.03
#    Rudolph Score: 242742  Penalty frac:  0.79 %  Carrots: 57   Length: 240818 .
#   Reindeer Score: 286664  Penalty frac: 19.03 %  Carrots: 492   Length: 240818 .
# No prime considerations
#    Rudolph Score: 243210 Penalty frac:  0.88 %  Carrots: 61  
#   Reindeer Score: 287413 Penalty frac: 19.22 %  Carrots: 492 (all of them)

# All odd cities - about twice as many carrots !
##Nth = 50;  city_start = 1
# All even cities - no carrots :(
##Nth = 50;  city_start = 4

# Select the cities desired
left_cities = np.array(df.CityId)[city_start: :Nth]

# Or put in a know set of cities to test scoring
# "The carrot run"
##left_cities = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67])
# "No carrots"
##left_cities = 2*np.array(list(range(1,20)))

print("Number of cities besides the NP: ", len(left_cities), "  Total primes:",
      sum(prime_flags[left_cities]))
# Initialize the path, etc.
path = [0]
current_city = 0
stepNumber = 1
t0 = time()

if len(left_cities) < 15000:
    show_every = 1000
else:
    show_every = 10000

# For Rudolph scoring:
# factor to reduce prime distance to account for prime's no-penalty advantage
prime_reduce = (1.0 - 0.09*prime_flags)
# factor to increase prime distance when a prime doesn't matter (save them for when it matters)
prime_increase = (1.0 + 0.03*prime_flags)

# And loop though the cities
while left_cities.size > 0:
    if stepNumber % show_every == 0: # We print the progress of the algorithm
        print(f"Time elapsed : {time() - t0} - Number of cities left : {left_cities.size}")
    # Compute the distance matrix
    ##distances = dist_array(coordinates, current_city, RightLeft=False) # same as Viel's
    distances = dist_array(coordinates, current_city, RightLeft=True) # modified distances
    # Encourage a prime every 10th city (%10==9) by reducing prime's distances
    if stepNumber % 10 == 9:
        distances = distances * prime_reduce  # reduce distance for primes
    else:
        distances = distances * prime_increase  # increase distance for primes
    # Get the closest city and go to it
    current_city = get_next_city(distances, left_cities)
    # Update the list of not visited cities
    left_cities = np.setdiff1d(left_cities, np.array([current_city]))
    # Append the city to the path
    path.append(current_city)
    # Add one step
    stepNumber += 1
    
# Add the North Pole and we're done
path.append(0)
print(f"Loop lasted {(time() - t0) // 60} minutes ")
# Show the path
plot_path(path, coordinates)
# Show the Rudolph Score results
score, carrots, length = get_score(path, coordinates, prime_flags, RR=False)
# and without going back to the NP
score_noNP, dummy1, dummy2 = get_score(path[:-1], coordinates, prime_flags, RR=False)

print("Rudolph Score:", int(score), "   Carrots:", carrots, "   Length:", int(length), ".\n" +
      " Penalty frac:", int(10000*(score-length)/length)/100,
      "%   Final step to NP has distance ", int(score - score_noNP))

# Show the *** Reindeer Rebellion *** Score results
score, carrots, length = get_score(path, coordinates, prime_flags, RR=True)
# and without going back to the NP
score_noNP, dummy1, dummy2 = get_score(path[:-1], coordinates, prime_flags, RR=True)

print("Reindeer Score:", int(score), "  Carrots:", int(carrots), "  Length:", int(length), ".\n" +
      " Penalty frac:", int(10000*(score-length)/length)/100,
      "%   Final step to NP has distance ", int(score - score_noNP))
# Output the path to a file that we can submit
if True:
    submission = pd.DataFrame({"Path": path})
    submission.to_csv("submission.csv", index=None)


