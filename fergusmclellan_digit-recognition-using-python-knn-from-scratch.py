# Import Python modules which are needed for data exploration

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting/graphical functionality

import seaborn as sns # seaborn can be used to extend matplotlib functionality

%matplotlib inline





# The input data files from Kaggle are available in the "../input/" directory.

import os

print(os.listdir("../input"))



# the os.listdir will output the input files which are available in the input directory
df_train_data = pd.read_csv('../input/digit-recognizer/train.csv')

df_test_data = pd.read_csv('../input/digit-recognizer/test.csv')

df_train_data.head(10)
# The training data is in a format whereby the label (the digit represented, 0 through to 9) is in the first column.

# This is followed by 784 columns of pixels, pixel0 through to pixel783.

# Each image is 28 pixels by 28 pixels, therefore, a grid of 28x28 means a total number of 784 pixels.



# Separate out the pixel data from the labels for the training data:

df_train_labels = df_train_data['label']

df_train_data = df_train_data.drop(['label'], axis=1)



# The original images of the handwritten digits can be viewed using plt and reshaping each row 

# from the dataframe into a 2D image with a size of 28 by 28 pixels.

# This for loop will display the first 10 images from the training data, along with their labels:

for i in range(10):

    ax= plt.subplot(1,10 ,i+1)

    im=ax.imshow(df_train_data.iloc[i].values.reshape(28,28))

    plt.xlabel=''

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)

    plt.title(df_train_labels.iloc[i])
# Point A is [1,1]

# Point B is [5,4]



# Draw 2D right angle triangle

fig = plt.figure()

fig, ax = plt.subplots()

ax.plot([1,5], [1,4], linestyle='solid')

ax.text(0.82,0.9, "A", color='red', size=20)

ax.text(5.01,3.92, "B", color='red', size=20)

ax.text(2,2, "Hypotenuse", color='blue', rotation=32.5, size=20)

# Add the x, y sides of the triangle

ax.plot([1,5], [1,1], linestyle='dashed')

ax.plot([5,5], [1,4], linestyle='dashed')



ax.set_xlabel("x")

ax.set_ylabel("y")

plt.show()
# Draw 3D right angle pyramid

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update(plt.rcParamsDefault)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# Point A is [1,1,1]

# Point B is [4,4,4]

ax.plot([1,4], [1,4], [1,4], linestyle='solid')

ax.text(0.9,0.9,0.9, "A", color='red')

ax.text(4.1,4.1,4.1, "B", color='red')

# Add the x, y and z sides of the triangular pyramid

ax.plot([1,4], [1,1], [1,1], linestyle='dashed')

ax.plot([4,4], [1,4], [1,1], linestyle='dashed')

ax.plot([4,4], [4,4], [1,4], linestyle='dashed')

ax.plot([1,4], [1,4], [1,1], linestyle='dashed')

ax.set_xlabel("x")

ax.set_ylabel("y")

ax.set_zlabel("z")

plt.show()
# With points 1,1,1 and 4,4,4, the length of each side is 4 - 1 = 3

# The length of the x side is 3, the length of the y side is 3, and the z side is 3

print("The Euclidean distance from A to B is equal to")

print(np.sqrt( 3**2 + 3**2 + 3**2 ))
# We can define a function for 2 points in multiple dimensions



def euclid_distance(p1, p2):



    # Create an integer to store the running total sum

    sum_of_square_distances = 0

    for a,b in zip(p1,p2):

        # for each dimension, find the distance between the 2 points in that dimension, and square it

        # a and b represent the coordinate in the current dimension for points p1 and p2

        distance_in_this_dimension_squared = (a - b)**2

        sum_of_square_distances = sum_of_square_distances + distance_in_this_dimension_squared

    

    # The euclidean distance is the square root of the sum of all of the (distances in each dimension squared)

    euclidean_distance = np.sqrt(sum_of_square_distances)

    

    return euclidean_distance



# For 2 points which have coordinates in 6 dimensions:

point1 = (5,6,7,8,9,10)

point2 = (1,2,3,4,5,6)



print(euclid_distance(point1, point2))
def labels_of_nearest_neighbours(np_list_of_distances, np_list_of_labels, k):

    

    """

    Arguments:

        np_list_of_distances: a numpy array of distances between the new point and the known/training points

        np_list_of_labels: a numpy array of labels for the known/training points

        k: integer of the number of nearest neighbours to consider

    Returns: the labels corresponding to the neighbours with the smallest distances.

    """

    # reshape both input lists from a row to a column

    np_list_of_distances = np_list_of_distances.reshape(-1,1)

    np_list_of_labels = np_list_of_labels.reshape(-1,1)

    

    # can now join the 2 columns together into an array and convert into a dataframe

    np_distances_and_labels = np.concatenate((np_list_of_distances,np_list_of_labels), axis = 1)

    labels_df = pd.DataFrame(np_distances_and_labels, columns=['distance','label'])

    

    # sort the entries in the dataframe by distance

    labels_df = labels_df.sort_values('distance')

    return labels_df['label'].head(k).values

from collections import Counter



def most_common_label(np_arr_of_labels):

    """

    Arguments:

        np_list_of_labels: a numpy array of labels. 

    Returns: the most common label    

    """

    mostCommon = Counter(np_arr_of_labels).most_common(1)

    return mostCommon[0][0]

def simple_KNN( new_point, existing_points, labels_for_existing_points, k=2):

    """

    Arguments:

        new_point: a pandas Series corresponding to the new point with an unknown label.

        existing_points: a pandas DataFrame containing the known points

        labels_for_existing_points: a pandas Series containing the labels for the known points

        k: the number of nearest neighbours to consider when identifying the label for the new point

        If k is not specified, it defaults to 2

    Returns: the predicted label for the new point

    """

    row_count_of_existing_points = existing_points.shape[0]

    # Create an empty list of the distances

    list_of_distances = []

    

    # For each known point, calculate the euclidean distance between the known point and the new point

    # add the euclidean distance to the list_of_distances

    for i in range(row_count_of_existing_points):

        this_distance = euclid_distance(new_point, existing_points.iloc[i])

        list_of_distances.append(this_distance)

    

    # Using the list_of_distances, and the list of labels, identify the labels of the points with the 

    # smallest distance to the new point

    labels_of_k_nearest_neighbours = labels_of_nearest_neighbours(np.array(list_of_distances), np.array(labels_for_existing_points), k)

    

    # Identify the most common label in the nearest neighbours. This is the predicted label for the new point

    predicted_label_of_new_point = most_common_label(labels_of_k_nearest_neighbours)

    return predicted_label_of_new_point



# new point which is closest to the red dots

new_point = pd.Series([2,3])

# number of nearest neighbours (k) to use to classify a point: set to 2 for simplicity

k = 2
# We can create a very simple 2 dimensional sample of points which are obviously in 2 groups

# and plot the points and colour them based on their grouping



# Create data

data1 = np.array([[0, 1],[1, 0],[1, 1],[1, 2],[2, 2], [2, 1]])

data2 = data1 + 5

dataframe1 = pd.DataFrame(data1)

dataframe2 = pd.DataFrame(data2)



# Create a scatterplot with the dataframe1 points in red, and the dataframe2 points in blue

ax1 = dataframe1.plot.scatter(0,1, color='red')

dataframe2.plot.scatter(0,1, color='blue', ax=ax1)

# Join the 2 dataframes together into a single dataframe called X_train

X_train = dataframe1

X_train = X_train.append(dataframe2, ignore_index=True)



# Create a set of labels for the data called y_train, which labels the points in X_train as red or blue

red_arr = np.array("red")

blue_arr = np.array("blue")

y_train = np.concatenate((np.repeat(red_arr,6), np.repeat(blue_arr,6)), axis=None)

y_train = pd.Series(y_train)

print(y_train)
# Create a new point which is deliberately closest to the red dots

new_point = pd.Series([2,3])

# number of nearest neighbours (k) to use to classify a point: set to 2 for simplicity

k = 2

print(simple_KNN(new_point, X_train, y_train, k))
df_test_data.head(10)
# Display the first images in the test set

for i in range(10):

    ax= plt.subplot(1,10 ,i+1)

    im=ax.imshow(df_test_data.iloc[i].values.reshape(28,28))

    plt.xlabel=''

    ax.axes.get_xaxis().set_visible(False)

    ax.axes.get_yaxis().set_visible(False)
#set new_point equal to the first entry in the test data

# (from the above we can see that the first image is a number 2)

new_point = df_test_data.iloc[0]
# test by comparing the test image against the first 1000 entries 

# in the training data 

print(simple_KNN(new_point, df_train_data.head(1000), df_train_labels.head(1000), k))
from scipy import spatial



tree = spatial.cKDTree(df_train_data.head(1000))

mindist, minid = tree.query(df_test_data.head(1))

# minid is the ID of the lowest distance, therefore the nearest neighbour

print(mindist, minid)

# we can use the minid to reference the label of the nearest point to our test point (the first digit in our test data is the number 2)

print(df_train_labels[minid])
from scipy import spatial



def faster_KNN(X_test, X_train, y_labels):

    predictions_list = []

    tree = spatial.cKDTree(X_train)

    number_of_test_data_points = len(X_test)

    for point in range(number_of_test_data_points):

        mindist, minid = tree.query(X_test.iloc[point])

        this_prediction = y_labels[minid]



        output_file.write(str(point+1) + "," + str(this_prediction) + "\n")

    


output_file = open("predictions.csv", "w")

output_file.write("ImageId,Label\n")

faster_KNN(df_test_data,df_train_data.head(2000), df_train_labels.head(2000))

# You can run the KNN against all of the training data by uncommenting the following line

# faster_KNN(df_test_data,df_train_data, df_train_labels)

output_file.close()