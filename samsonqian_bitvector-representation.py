import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

%matplotlib inline
objects = pd.DataFrame({"Object name": ["chair", "desk", "shoe", "car", "person"], 

                     "Object coordinate": [(2, 1), (1, 2), (-0.5, 1), (-1, -1), (5, -5)]})

objects["X-coordinate"] = objects["Object coordinate"].apply(lambda x: x[0])

objects["Y-coordinate"] = objects["Object coordinate"].apply(lambda x: x[1])

print(objects)
fig = plt.gcf()

fig.set_size_inches(8, 8)

sns.scatterplot(x="X-coordinate", y="Y-coordinate", hue="Object name", data=objects)

plt.title("Object Coordinates")
def euclidean_distance(x1, y1, x2, y2):

    """

    Calculates the euclidean distance between two points.

    :param x1: x-coordinate of first point

    :param y1: y-coordinate of first point

    :param x2: x-coordinate of second point

    :param y2: y-coordinate of second point

    :return: euclidean distance between points

    """

    return np.sqrt((x2-x1)**2 + (y2-y1)**2)



def distances(x, y, data):

    """

    Returns euclidean distances of inputted object coordinates and

    all other objects in the inputted data.

    :param x: x-coordinate of object

    :param y: y-coordinate of object

    :param data: dataframe of objects

    :return: list of distances between objects

    """

    data["Distances"] = euclidean_distance(x, y, data["X-coordinate"], data["Y-coordinate"])

    temp = data.sort_values(by="Distances")

    return [list(l) for l in zip(temp["Object name"], temp["Distances"])]





chair = distances(2, 1, objects)



desk = distances(1, 2, objects)



shoe = distances(-0.5, 1, objects)



car = distances(-1, -1, objects)



person = distances(5, -5, objects)



print("chair: " + str(chair))

print("desk: " + str(desk))

print("shoe: " + str(shoe))

print("car: " + str(car))

print("person: " + str(person))
def bitvector(data):

    """

    Returns a series of each object's bitvector representation within

    the inputted data.

    :param data: dataframe of objects

    :return: series of bitvectors

    """

    def transform_x(x):

        if x < 0:

            return "0"

        else:

            return "1"

        

    def transform_y(y):

        if y < 0:

            return "0"

        else:

            return "1"

    

    x_bit = objects["X-coordinate"].apply(transform_x)

    y_bit = objects["Y-coordinate"].apply(transform_y)

    bitvector_representations = x_bit + y_bit

    data["Bitvector"] = bitvector_representations

    bitvector_representations.index = data["Object name"]

    return bitvector_representations



bitvector(objects)
def hamming_distances(object_bit, data):

    """

    Returns hamming distances between inputted object's bitvector and

    the bitvectors of the objects in the inputted data.

    :param object_bit: bitvector of object

    :param data: dataframe of objects

    :return: list of hamming distances

    """

    def hamming(other_bit):

        return sum(object_bit[i] != other_bit[i] for i in range(len(object_bit)))

    data["Hamming distance"] = data["Bitvector"].apply(hamming)

    temp = data.sort_values(by="Hamming distance")

    return [list(l) for l in zip(temp["Object name"], temp["Hamming distance"])]

   



chair = hamming_distances("11", objects)



desk = hamming_distances("11", objects)



shoe = hamming_distances("01", objects)



car = hamming_distances("00", objects)



person = hamming_distances("10", objects)



print("chair: " + str(chair))

print("desk: " + str(desk))

print("shoe: " + str(shoe))

print("car: " + str(car))

print("person: " + str(person))