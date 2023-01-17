# importing the core library

import numpy as np



# print multiple output in single cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Question: Compute the one-hot encodings (dummy binary variables for each unique value in the array)

np.random.seed(101) 

arr = np.random.randint(1,4, size=6)

arr



np.array([[1 if i == j else 0 for i in list(set(arr))] for j in arr])
# Question: Create row numbers grouped by a categorical variable. Use the following sample from iris species as input.



species = np.genfromtxt("../input/iris/Iris.csv", delimiter=',', dtype='str', usecols=5, skip_header=1)

species_small = np.sort(np.random.choice(species, size=20))

species_small



[i for val in np.unique(species_small) for i, grp in enumerate(species_small[species_small==val])]
# Question: Create group ids based on a given categorical variable. Use the following sample from iris species as input.



# Question: Create the ranks for the given numeric array a.
# Question: Create a rank array of the same shape as a given numeric array a.

# Question: Compute the maximum for each row in the given array.



# Question: Compute the min-by-max for each row for given 2d numpy array.
# Question: Find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False.

# Question: Find the mean of a numeric column grouped by a categorical column in a 2D numpy array

iris = np.genfromtxt("../input/iris/Iris.csv", delimiter=',', dtype='object', skip_header=1)

sepallength = iris[:,1].astype(float)

targets = iris[:,5]

[[target, np.mean(sepallength[np.where(targets == target)])] for target in np.unique(targets)]
# Question: Import the image from the following URL and convert it to a numpy array.



from PIL import Image

I = Image.open("../input/exercise60/Denali_Mt_McKinley.jpg")

np.array(I)
# Question:
# Question:
# Question:
# Question:
# Question:
# Question:
# Question:
# Question: Create a numpy array of length 10, starting from 5 and has a step of 3 between consecutive numbers



#end = len * step + starting point

end = (10 * 3) + 5



np.arange(5, end, 3)
# Question:
arr = np.arange(15) 

arr



# Desired Output

# > [[ 0  1  2  3]

# >  [ 2  3  4  5]

# >  [ 4  5  6  7]

# >  [ 6  7  8  9]

# >  [ 8  9 10 11]

# >  [10 11 12 13]]



# Solution



print(np.array([arr[i:i+4]for i in range(0,12,2)]))   