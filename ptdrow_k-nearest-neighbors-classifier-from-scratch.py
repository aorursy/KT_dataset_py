import math

import pandas as pd

from collections import Counter
def euclidean_dist(xi,xj):

    """

    Calculates the euclidean distance in a N-dimensional space between the points xi and xj

    

    Parameters:

        xi, xj: numeric list or tuple representing each point.

        

    Returns:

        distance (float): the euclidean distance between the points.

    """

    

    distance_squared = 0

    for d in range(len(xi)):

        distance_squared += (xj[d] - xi[d]) ** 2

    return math.sqrt(distance_squared)
x1 = (2,1,1)

x2 = (1,0,0)

euclidean_dist(x1,x2)
def calc_all_distances(xi, data, columns):

    """

    Calculates the Euclidean distance between a point and all the points in the data.

    

    Parameters:

        xi      : Numeric list or tuple representing the point.

        data    : Pandas dataframe with the training data.

                  Each row will represent a point in the N-dimensional space.

        columns : A list of the columns name to take as the dimensions of the Euclidean space. 

        

    Returns:

        distances (Series): the euclidean distances between the points.

        

    """

    #Convert data frame to points in n-dimensional space

    points = data[columns].apply(tuple,axis=1)

    #Calculate distance for each row

    distances = points.apply(lambda x: euclidean_dist(xi,x))

    return distances
def predict_knn(xi, data, columns, target_column, k):

    """

    Predicts the label for a given point by taking votes from its k-nearest neigbors

    

    Parameters:

        xi           : Numeric list or tuple representing the point.

        data         : Pandas dataframe with the training data.

                       Each row will represent a point in the N-dimensional space.

        columns      : A list of the columns names to take as the dimensions of the Euclidean space. 

        target_column: The name of the target feature

        k            : The number of nearest neighbors to take into account

        

    Returns:

        prediction (str): the predicted label for the point

    

    """

    

    all_distances = calc_all_distances(xi,data,columns)

    k_nearest = list(all_distances.sort_values()[:k].index)

    k_nearest_labels = data.loc[k_nearest, target_column]

    poll = Counter(k_nearest_labels)

    return poll.most_common()[0][0]
iris_data = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris_data.head()
feature_columns = list(iris_data.columns[1:5])

print(feature_columns)

target_column = iris_data.columns[5]

print(target_column)
flower_point = (5.0, 4.0, 2.0, 1.5)

k = 3

print(predict_knn(flower_point,

                  iris_data, 

                  feature_columns,

                  target_column,

                  k))
predictions = list()

for row in iris_data.itertuples():

    dict_row = row._asdict()

    xi = list()

    for col in feature_columns:

        xi.append(dict_row[col])

    predictions.append(predict_knn(xi,

                                   iris_data.drop(index=row.Index), 

                                   feature_columns, 

                                   target_column, 

                                   k))

        

print(predictions)
correct_predictions = iris_data.loc[:,target_column] == predictions

accuracy = sum(correct_predictions)/len(predictions)

print(accuracy)