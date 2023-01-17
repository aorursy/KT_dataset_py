# Imports

import numpy as np # Standard numpy import

from matplotlib import pyplot as plt # Standard pyplot import

import matplotlib.patches as mpatches # For legend for pyplot

from sklearn.datasets import load_iris # Our dataset comes with scikitlearn

from sklearn.neighbors import KNeighborsClassifier # KNN from scikitlearn
# Mode for a 1d array with numpy is a bit messy, so define a lambda function for convenience

mode = lambda x: np.argmax(np.bincount(x.astype(int)))



# Utility function for chart; this will create a list of colours which corresponds to the spieces of the different datapoints

def int_to_colour(d):

    colours = []

    for x in d:

        if x == 0: colours.append('red')

        if x == 1: colours.append('green')

        if x == 2: colours.append('blue')

    return colours
# This gives us our dataset

iris = load_iris()



# Let's look at what data we have

print(iris.data.shape)

print(iris.feature_names)

print(iris.data[:5])
# Reduce the dimensions from four to two by taking the column vectors which relate to the lengths (and not the widths) of the petals and sepals respectively

sepal_length = iris.data[:,0]

petal_length = iris.data[:,2]

species = iris.target



# Colours for chart

colours = int_to_colour(species)



# The below sets up the legend for the chart

patches = [mpatches.Patch(color=int_to_colour([x])[0], label=iris.target_names[x]) for x in range(3)]



# Create plot of sepals and petal lengths

plt.figure(figsize=(12,6))

plt.legend(handles=patches)

plt.xlabel('Sepal length')

plt.ylabel('Petal length')

plt.scatter(sepal_length, petal_length, color=colours)

plt.show()
# New numpy array which has the two feature dimensions -- here we are packaging the data in a way that scikit-learn likes

X = np.c_[sepal_length, petal_length]

y = species



# Train

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X[:120],y[:120])
# Predict

x_p1 = np.array([[0,0]])

y_hat = knn.predict(x_p1)

print(iris.target_names[y_hat])



x_p2 = np.array([[6,4]])

y_hat = knn.predict(x_p2)

print(iris.target_names[y_hat])



x_p3 = np.array([[8,6.5]])

y_hat = knn.predict(x_p3)

print(iris.target_names[y_hat])
y_hats = knn.predict(X[120:])

print(iris.target_names[y_hats])
correct = sum([(i[0] == i[1]) for i in zip(y_hats,y[120:])])

pct_correct = correct / len(y_hats) * 100

print("{} correct which represents {:.0f}%".format(correct,pct_correct))
# Argsort twice for rank example

my_npa = np.array([3,1,2])

sort_indices = np.argsort(my_npa)

sort_rank = np.argsort(sort_indices)

print(my_npa)

print(sort_indices)

print(sort_rank + 1) # Note the rank is 0-indexed; let's make this indexed from 1 by adding 1



# Conclusion: to rank in numpy, use argsort twice and optionally add 1 to the result if you wish the rank to start from 1 rather than 0
# Euclidian distance in two dimensions

def euc_dist_2d(x_p,X,y):

    deltas = X - x_p # Matrix subtraction, deltas will be a matrix of the differences

    euc_dists = np.sum(deltas ** 2, axis=1) ** 0.5 # Sum across the rows (i.e. across the 2 dimensions) of the squared deltas, and then square root these sums

    a_sort = np.argsort(euc_dists) # Argsort twice for rank

    ranks = np.argsort(a_sort)



    # Returns a matrix consisting of the following:

    # ranks, euclidian disance, X, class

    

    # np.c_[...] splices the columns vectors together into a matrix, which we then return

    return np.c_[ranks,euc_dists,X,y]



# Own KNN function, only for 2 dimensions

# Return y_hat



def knn_2d(x_p,X,y,k):

    if k < 2:

        print('n must be larger than 1') # 1 is minimum value of k

        return

    

    distance_info = euc_dist_2d(x_p,X,y) # Matrix of euclidean distances from x_p to all other points 



    # Get rank 1 i.e. the closest nearest neighbours

    for i in range(len(distance_info)):

        if int(distance_info[i][0]) == 0: # i.e. is the rank 0

            knns = distance_info[i] # if it is, our knns matrix starts with this

        

    # Add rank 1 to n from there i.e. the 2nd to kth nearest neighbours

    if k > 1:

        for r in range(2, k + 1):

            for i in range(len(distance_info)):

                if int(distance_info[i][0]) == (r-1): # i.e. is the rank r-1

                    knns = np.vstack((knns,distance_info[i])) # if it is, add as a row vector to the knns matrix

    #print(knns)

    return knns[:,4]
result = knn_2d(x_p1,X[:120],y[:120],5)

print(iris.target_names[mode(result)])



result = knn_2d(x_p2,X[:120],y[:120],5)

print(iris.target_names[mode(result)])



result = knn_2d(x_p3,X[:120],y[:120],5)

print(iris.target_names[mode(result)])
y_hats = [mode(knn_2d(x_p,X[:120],y[:120],5)) for x_p in X[120:]]

print(iris.target_names[y_hats])
correct = sum([(i[0] == i[1]) for i in zip(y_hats,y[120:])])

pct_correct = correct / len(y_hats) * 100

print("{} correct which represents {:.0f}%".format(correct,pct_correct))