# Ignoring warning
import warnings
warnings.simplefilter('ignore')
# Importing useful libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
# Fixing random state for reproducibility
np.random.seed(160010010)
iris = datasets.load_iris()
iris_data = pd.DataFrame(iris.data)
iris_data['target'] = pd.Series(iris.target)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
target_classes = ['setosa','versicolor','virginica']
print("The number of observations are :", iris_data.shape[0])
print("Name of columns are :", iris_data.columns.values)
print("Here are some rows from our final dataframe:")
print(iris_data.head())
# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i , *args: target_classes[i])
plt.scatter(x=iris_data.sepal_length,y=iris_data.sepal_width,c=iris_data.species)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.scatter(x=iris_data.petal_length,y=iris_data.petal_width,c=iris_data.species)
plt.colorbar(ticks=[0,1,2],format=formatter)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
# Giving each species name in our dataframe
iris_data_with_labels = iris_data.copy(deep=True)
iris_data_with_labels.species = pd.Series([target_classes[x] for x in iris_data.species])
print("New dataframe :")
print(iris_data_with_labels.head())
sns.pairplot(iris_data_with_labels, hue="species")  # making matrix plot between each variables and coloring points based on its
# category
corr = iris_data.corr()
sns.heatmap(corr,annot=True)
# Creating histogram
iris_data_with_labels.hist(bins=10)
sns.boxplot(x='species', y='sepal_length', data=iris_data_with_labels, order=["virginica", "versicolor", "setosa"])
sns.boxplot(x='species', y='sepal_width', data=iris_data_with_labels, order=["virginica", "versicolor", "setosa"])
sns.boxplot(x='species', y='petal_length', data=iris_data_with_labels, order=["virginica", "versicolor", "setosa"])
sns.boxplot(x='species', y='petal_width', data=iris_data_with_labels, order=["virginica", "versicolor", "setosa"])
# Since petal width, petal length and sepal length are correlated with the species of flower, we are going
# to plot these variables on the 3 axes
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data.petal_width, iris_data.petal_length, iris_data.sepal_length, c=iris_data.species)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
plt.show()
import random
import numpy as np

def EuclidianDistance(x,y):
    # This function will return Euclidian distance between x and y,
    # where x and y are n-dimensional vector
    xi = np.array(list(x))
    yi = np.array(list(y))
    return np.sqrt(np.sum(np.square(xi - yi)))

def Calculate_Mean_Square_Error(assignment_of_nodes,current_centers,dataframe):
    # This funciton will calculate the mean square error or J 
    # When provided with centers, assignment of nodes and dataframe
    result = 0
    length_of_dataframe = dataframe.shape[0]
    for x in range(length_of_dataframe):
        result += EuclidianDistance(dataframe.loc[dataframe.index[x], :],
                current_centers.loc[current_centers.index[int(assignment_of_nodes[x])], :]) ** 2
    result = result / length_of_dataframe
    return result

def KmeansCluster(dataframe, number_of_clusters, maximum_number_of_iteration = 100000):
    # This function will assign a group to every data point, and then it will return
    # the assignment, all the assignments that was calculated in each iteration, and 
    # a list of value of J in each iteration.
    length_of_dataframe = dataframe.shape[0]
    width_of_dataframe = dataframe.shape[1]
    # choose k random points and make them centers
    random_indices = random.sample(list(range(length_of_dataframe)), number_of_clusters)
    current_centers = dataframe.loc[random_indices, :]
    # assign label to each of the observation points
    all_assignments = []
    mean_square_list = []
    assignment_of_nodes = np.zeros(length_of_dataframe)
    previous_assignment_of_nodes = np.copy(assignment_of_nodes)
    # iterate till maximum number of times or when the assignment of nodes is not changing
    for ix in range(maximum_number_of_iteration):
        # assign group to every data point 
        for i in range(length_of_dataframe):
            current_assignment = 0
            for j in range(number_of_clusters):
                current_distance = EuclidianDistance(dataframe.loc[dataframe.index[i], :],
                        current_centers.loc[current_centers.index[current_assignment], :])
                new_distance = EuclidianDistance(dataframe.loc[dataframe.index[i], :],
                         current_centers.loc[current_centers.index[j], :])
                if new_distance < current_distance:
                    current_assignment = j
            assignment_of_nodes[i] = current_assignment
        this_assignment = list(assignment_of_nodes)
        this_assignment = [int(x) for x in this_assignment]
        all_assignments.append(this_assignment)
        mean_square_list.append(Calculate_Mean_Square_Error(assignment_of_nodes,
                                                            current_centers,dataframe))
        if np.sum(previous_assignment_of_nodes == assignment_of_nodes) == dataframe.shape[0]:
            break
        previous_assignment_of_nodes = np.copy(assignment_of_nodes)
        # calculating the center again
        for i in range(number_of_clusters):
            current_centers.loc[current_centers.index[i]] = dataframe.loc[assignment_of_nodes == i, :].mean(0)
    # post-processing results
    assignment_of_nodes = list(assignment_of_nodes)
    assignment_of_nodes = [int(x) for x in assignment_of_nodes]
    return (assignment_of_nodes, all_assignments, mean_square_list)
clusters, all_assignment, mean_square_list = KmeansCluster(iris_data[list(iris_data.columns[:-1])],3)
# Ploting final cluster assignment
newframe = iris_data[list(iris_data.columns[:-1])]
label_classes = ['class-0','class-1','class-2']
newframe["clusters"] = pd.Series([label_classes[i] for i in clusters])
sns.pairplot(newframe, hue='clusters')
%matplotlib notebook
import time

#initialise the graph and settings
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()
ax = fig.gca(projection='3d')
number_of_iteration_completed = 0
for current_cluster in all_assignment:
    ax.clear() # - Clear
    current_frame = iris_data[list(iris_data.columns[:-1])]
    current_frame["clusters"] = pd.Series(current_cluster)
    ax.scatter(current_frame.petal_width, current_frame.petal_length, 
               current_frame.sepal_length, c=current_frame.clusters)
    ax.set_xlabel('petal width (cm)')
    ax.set_ylabel('petal length (cm)')
    ax.set_zlabel('sepal length (cm)')
    number_of_iteration_completed += 1
    title = "After " + str(number_of_iteration_completed) + " iterations"
    ax.set_title(title)
    fig.canvas.draw()
    time.sleep(2)
# Plotting final assignment
final_frame = iris_data[list(iris_data.columns[:-1])]
final_frame["clusters"] = pd.Series(clusters)
%matplotlib inline
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(final_frame.petal_width, final_frame.petal_length, final_frame.sepal_length, c=final_frame.clusters)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title('Final Assignment')
number_of_iterations = len(mean_square_list)
for i in range(number_of_iterations):
    print("Value of J after " + str(i+1) + " iterations is " + str(mean_square_list[i]))
%matplotlib inline
plt.plot(list(range(1,len(mean_square_list) + 1)), mean_square_list, 'k')
plt.xlabel('Number of iterations')
plt.ylabel('Value of J(error)')
plt.title('How value of J is changing with the number of iterations.')
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

iris_data_without_labels = iris_data[list(iris_data.columns[:-1])]

# Run kmeans algorithm and print the result
kmeans_model = KMeans(n_clusters=3)
kmeans_result = kmeans_model.fit(iris_data_without_labels)
print("Labels assigned to our data by kmeans : ", kmeans_result.labels_)

# Run agglomerative clustering algorithm and print the result
agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3)
agglomerative_clustering_result = agglomerative_clustering_model.fit(iris_data_without_labels)
print("Labels assigned to our data by agglomerative clustering : ", agglomerative_clustering_result.labels_)

# Printing True labels
print("True labels : ",np.array(iris_data.species))
print("Result of kmeans : ")
kmeans_unique_class , kmeans_unique_class_counts = np.unique(kmeans_result.labels_, return_counts=True)
for x , y in zip(kmeans_unique_class, kmeans_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)

print("Result of agglomerative clustering : ")
agglomerative_clustering_unique_class , agglomerative_clustering_unique_class_counts = np.unique(agglomerative_clustering_result.labels_, return_counts=True)
for x , y in zip(agglomerative_clustering_unique_class, agglomerative_clustering_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)

print("True labels :")
target_classes = ['setosa','versicolor','virginica']
true_unique_class , true_unique_class_counts = np.unique(np.array(iris_data.species), return_counts=True)
for x , y in zip(true_unique_class, true_unique_class_counts):
    print("The number of observations assigned class",target_classes[x],"is",y)
# we are taking petal width, petal length and sepal length as our 3 axes

# Creating plot for kmeans
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=pd.Series(kmeans_result.labels_))
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("kmeans")
plt.show()

# Creating plot for agglomerative clustering
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=pd.Series(agglomerative_clustering_result.labels_))
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("agglomerative clustering")
plt.show()

# Creating plot with true labels
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=iris_data.species)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("true labels")
plt.show()
# Case 1 - All initial points are same
case_1_init = np.array([(iris_data_without_labels.loc[0,:]) for x in range(3)])
case_1_kmeans_model = KMeans(n_clusters=3,init=case_1_init,n_init=1)
case_1_kmeans_model_result = case_1_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_1_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("All initial points are same")
plt.show()

# Case 2 - All initial points are in same class
case_2_init = np.array([(iris_data_without_labels.loc[x,:]) for x in range(3)])
case_2_kmeans_model = KMeans(n_clusters=3,init=case_2_init,n_init=1)
case_2_kmeans_model_result = case_2_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_2_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("All initial points are in same class")
plt.show()

# Case 3 - With init = 'random'
case_3_kmeans_model = KMeans(n_clusters=3,init='random',n_init=1)
case_3_kmeans_model_result = case_3_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_3_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("With init = 'random'")
plt.show()

# Case 4 - With init = 'k-means++'
case_4_kmeans_model = KMeans(n_clusters=3,init='k-means++',n_init=1)
case_4_kmeans_model_result = case_4_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_4_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("With init = 'k-means++'")
plt.show()
# Case 1 - All initial points are same
case_1_init = np.array([(iris_data_without_labels.loc[0,:]) for x in range(3)])
case_1_kmeans_model = KMeans(n_clusters=3,init=case_1_init,n_init=1, max_iter=5)
case_1_kmeans_model_result = case_1_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_1_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("All initial points are same")
plt.show()

# Case 2 - All initial points are in same class
case_2_init = np.array([(iris_data_without_labels.loc[x,:]) for x in range(3)])
case_2_kmeans_model = KMeans(n_clusters=3,init=case_2_init,n_init=1, max_iter=5)
case_2_kmeans_model_result = case_2_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_2_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("All initial points are in same class")
plt.show()

# Case 3 - With init = 'random'
case_3_kmeans_model = KMeans(n_clusters=3,init='random',n_init=1, max_iter=5)
case_3_kmeans_model_result = case_3_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_3_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("With init = 'random'")
plt.show()

# Case 4 - With init = 'k-means++'
case_4_kmeans_model = KMeans(n_clusters=3,init='k-means++',n_init=1, max_iter=5)
case_4_kmeans_model_result = case_4_kmeans_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_4_kmeans_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("With init = 'k-means++'")
plt.show()
# Case 1 - Choosing 'ward' linkage
case_1_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='ward')
case_1_agglomerative_clustering_model_result = case_1_agglomerative_clustering_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_1_agglomerative_clustering_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("Case 1 - Choosing 'ward' linkage")
plt.show()

# Case 2 - Choosing 'complete' linkage
case_2_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='complete')
case_2_agglomerative_clustering_model_result = case_2_agglomerative_clustering_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_2_agglomerative_clustering_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("Case 2 - Choosing 'complete' linkage")
plt.show()

# Case 3 - Choosing 'average' linkage
case_3_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='average')
case_3_agglomerative_clustering_model_result = case_3_agglomerative_clustering_model.fit(iris_data_without_labels)
# Creating plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(iris_data_without_labels.petal_width, 
           iris_data_without_labels.petal_length, 
           iris_data_without_labels.sepal_length, 
           c=case_3_agglomerative_clustering_model_result.labels_)
ax.set_xlabel('petal width (cm)')
ax.set_ylabel('petal length (cm)')
ax.set_zlabel('sepal length (cm)')
ax.set_title("Case 3 - Choosing 'average' linkage")
plt.show()
# Case 1 - Choosing 'ward' linkage
case_1_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='ward')
case_1_agglomerative_clustering_model_result = case_1_agglomerative_clustering_model.fit(iris_data_without_labels)
print("Case 1 - Choosing 'ward' linkage")
case_1_agglomerative_clustering_unique_class , case_1_agglomerative_clustering_unique_class_counts = np.unique(case_1_agglomerative_clustering_model_result.labels_, return_counts=True)
for x , y in zip(case_1_agglomerative_clustering_unique_class , case_1_agglomerative_clustering_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)
    

# Case 2 - Choosing 'complete' linkage
case_2_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='complete')
case_2_agglomerative_clustering_model_result = case_2_agglomerative_clustering_model.fit(iris_data_without_labels)
print("Case 2 - Choosing 'complete' linkage")
case_2_agglomerative_clustering_unique_class , case_2_agglomerative_clustering_unique_class_counts = np.unique(case_2_agglomerative_clustering_model_result.labels_, return_counts=True)
for x , y in zip(case_2_agglomerative_clustering_unique_class , case_2_agglomerative_clustering_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)

# Case 3 - Choosing 'average' linkage
case_3_agglomerative_clustering_model = AgglomerativeClustering(n_clusters=3,linkage='average')
case_3_agglomerative_clustering_model_result = case_3_agglomerative_clustering_model.fit(iris_data_without_labels)
print("Case 3 - Choosing 'average' linkage")
case_3_agglomerative_clustering_unique_class , case_3_agglomerative_clustering_unique_class_counts = np.unique(case_3_agglomerative_clustering_model_result.labels_, return_counts=True)
for x , y in zip(case_3_agglomerative_clustering_unique_class , case_3_agglomerative_clustering_unique_class_counts):
    print("The number of observations assigned class",x,"is",y)
error_list_for_different_k = []
for number_of_clusters in range(1,11):
    # Run kmeans algorithm and save the error to list
    kmeans_model = KMeans(n_clusters=number_of_clusters)
    kmeans_result = kmeans_model.fit(iris_data_without_labels)
    error_list_for_different_k.append(kmeans_result.inertia_)
plt.plot(list(range(1,11)), error_list_for_different_k)
plt.show()