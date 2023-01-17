from csv import reader
import matplotlib.pyplot as plt; plt.rcdefaults()
import random
from math import sqrt

data = list()
with open('../input/iris-data/iris.data.txt', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        data.append(row)
        
for i in range(len(data[0]) - 1):
    for row in data:
        row[i] = float(row[i].strip())
        
for row in data:
    if row[-1] == "Iris-setosa":
        row[-1] = 0
    elif row[-1] == "Iris-versicolor":
        row[-1] = 1
    else:
        row[-1] = 2

data[0:5]

random.shuffle(data)

dev = data[:100]
test = data[100:]

print(dev[:5])
print(test[:5])
# helper functions from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    normalized = []
    for row in dataset:
        temp = []
        for i in range(len(row)):
            temp.append((row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]))
        normalized.append(temp)
    return normalized

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):    
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Reference https://stackoverflow.com/a/18424953/4451655
def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/sqrt(sumxx*sumyy)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, algo):
    distances = list()
    for train_row in train:
        if train_row == test_row:
            continue
        if algo == "euclidean":           
            dist = euclidean_distance(test_row, train_row)
        elif algo == "normalized_euclidean":
            dist = euclidean_distance(test_row, train_row)
        elif algo == "test_data":
            dist = cosine_similarity(test_row, train_row)
        elif algo == "cosine":
            dist = cosine_similarity(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    if algo == "cosine" or algo == "test_data":
        distances.reverse() # reverse if cosine
    if algo != "test_data":
        distances.pop(0) # remove distance from itself if this is dev
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict(data, test_row, k, algo):
    neighbors = get_neighbors(data, test_row, k, algo)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
# num_neighbors = [1, 3, 5, 7] # mentioned in assignment
num_neighbors = range(1, 11) # to find optimal k

accuracy = []

for k in num_neighbors:
    count = 0
    for row in dev:
        label = predict(dev, row[:-1], k, "euclidean")
        if label == row[-1]:
            count += 1
    accuracy.append(count / len(dev))
    print("For k = " + str(k) + " Accuracy is : " + str(count / len(dev)))
    

plt.bar(num_neighbors, accuracy)
plt.show()


normalized = normalize_dataset(dev, dataset_minmax(dev))
accuracy = []
for k in num_neighbors:
    count = 0
    for row in normalized:
        label = predict(normalized, row[:-1], k, "normalized_euclidean")
        if label == row[-1]:
            count += 1
    accuracy.append(count / len(normalized))
    print("For k = " + str(k) + " Accuracy is : " + str(count / len(normalized)))
    
plt.bar(num_neighbors, accuracy)
plt.show()

accuracy = []

for k in num_neighbors:
    count = 0
    for row in dev:
        label = predict(dev, row[:-1], k, "cosine")
        if label == row[-1]:
            count += 1
    accuracy.append(count / len(dev))
    print("For k = " + str(k) + " Accuracy is : " + str(count / len(dev)))
    
plt.bar(num_neighbors, accuracy)
plt.show()

count = 0
for row in test:
    label = predict(dev, row[:-1], 5, "test_data")
    if label == row[-1]:
        count += 1
accuracy = count / len(test)

print("Final accuracy is " + str(accuracy))

# Please note that as we are shuffling everytime this runs (and dataset is same), we may get different results for best k 