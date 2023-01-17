import math
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
def plot_data(tiles, n, data, heat_map, title, x_min, x_max, y_min, y_max):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    getColor = {0:'red', 1:'blue'}

    x_pos = data["X"].values
    y_pos = data["Y"].values
    color = data["class"].apply(lambda value: getColor[value])
    
    tile = tiles[1][n]
    
    tile.set_aspect('equal')
    tile.set_title(title)
    tile.set_xlabel("X")
    tile.set_ylabel("Y")
    tile.set_xlim(x_min, x_max)
    tile.set_ylim(y_min, y_max)
    
    if heat_map != 0:
        xx, yy, grid_values = heat_map
        tile.pcolormesh(xx, yy, grid_values, cmap=cmap_light)
        
    tile.scatter(x = x_pos, y = y_pos, c = color) 
def get_decision_boundary(model, x_min, x_max, y_min, y_max, step_size):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    
    points = np.c_[xx.ravel(),yy.ravel()]
    Z = np.zeros(len(points))
    for i in range(len(points)):
        temp_x = points[i][0]
        temp_y = points[i][1]
        point = pd.Series({'X' : temp_x, 'Y' : temp_y})
        Z[i] = model.predict(point)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    return (xx, yy, Z)
def get_accuracy(classified_data, predicted_data):
    correct_prediction_count = 0
    for index, data in predicted_data.iterrows():
        if data['class'] == classified_data.iloc[index]['class']:
            correct_prediction_count += 1
        
    return correct_prediction_count / len(predicted_data.index)
train_data = pd.read_csv(r"../input/train.csv")
classified_data = pd.read_csv(r"../input/test.csv")
test_data = classified_data.drop(columns="class")
def get_distance(point1, point2):
    return math.sqrt((point1['X'] - point2['X']) ** 2 + (point1['Y'] - point2['Y']) ** 2)
class NN_Classifier:
    neighborhood = 0
    
    def __init__(self, data_structure):
        self.neighborhood = data_structure
        
    def train(self, train_data):
        for index, data in train_data.iterrows():
            self.neighborhood.fill(data)
    
    def predict(self, point):
        nearest_neighbor = self.neighborhood.get_nearest_neighbor(point)
        return nearest_neighbor['class']
class Simple_Neighborhood:
    points = []
    
    def __init__(self):
        pass
    
    def fill(self, point):
        self.points.append(point)
        
    def get_nearest_neighbor(self, point):
        distance = 9999999
        nearest_neighbor = -1
        for i in range(0, len(self.points)):
            new_distance = get_distance(self.points[i], point)
            if new_distance < distance:
                distance = new_distance
                nearest_neighbor = i
        
        return self.points[nearest_neighbor]
neighborhood = Simple_Neighborhood()

nn_classifier = NN_Classifier(neighborhood)

nn_classifier.train(train_data[1:30])  
predicted_data = pd.DataFrame(columns = ['X', 'Y', 'class'])

for index, data in test_data.iterrows():
    prediction = nn_classifier.predict(data)
    predicted_data = predicted_data.append(pd.Series({'X' : data['X'], 'Y' : data['Y'], 'class' : prediction}), ignore_index=True)
accuracy = get_accuracy(classified_data, predicted_data)

print("accuracy: ", accuracy)
    
tiles = plt.subplots(1, 3, figsize=(30, 10))

decision_boundary = get_decision_boundary(nn_classifier, -1.5, 1.5, -1.5, 1.5, 0.01) 

plot_data(tiles, 0, train_data, 0, "train data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 1, classified_data, 0, "classified data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 2, predicted_data, decision_boundary, "predicted data", -1.5, 1.5, -1.5, 1.5)
class TwoD_Node:
    level = 0
    split = 0
    child1 = 0
    child2 = 0
    
    def __init__(self, point, level):
        self.split = point
        self.level = level
    
    def fill(self, point):
        if self.level % 2 == 0:
            if point['X'] < self.split['X']:
                if self.child1 != 0:
                    self.child1.fill(point)
                else:
                    self.child1 = TwoD_Node(point, self.level + 1)
            else:
                if self.child2 != 0:
                    self.child2.fill(point)
                else:
                    self.child2 = TwoD_Node(point, self.level + 1)
        else: 
            if point['Y'] < self.split['Y']:
                if self.child1 != 0:
                    self.child1.fill(point)
                else:
                    self.child1 = TwoD_Node(point, self.level + 1)
            else:
                if self.child2 != 0:
                    self.child2.fill(point)
                else:
                    self.child2 = TwoD_Node(point, self.level + 1)
                    
    def get_nearest_neighbor(self, point):
        if self.level % 2 == 0:
            if point['X'] < self.split['X']:
                if self.child1 != 0:
                    nn = self.child1.get_nearest_neighbor(point)
                    if abs(nn['X'] - point['X']) < abs(self.split['X'] - point['X']):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
            else:
                if self.child2 != 0:
                    nn = self.child2.get_nearest_neighbor(point)
                    if abs(nn['X'] - point['X']) < abs(self.split['X'] - point['X']):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
        else:
            if point['Y'] < self.split['Y']:
                if self.child1 != 0:
                    nn = self.child1.get_nearest_neighbor(point)
                    if abs(nn['Y'] - point['Y']) < abs(self.split['Y'] - point['Y']):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
            else:
                if self.child2 != 0:
                    nn = self.child2.get_nearest_neighbor(point)
                    if abs(nn['Y'] - point['Y']) < abs(self.split['Y'] - point['Y']):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
                    
    def print_node(self, level):
        for x in range(0, level):
            print("   ", end="")
        print(self.split['X'], self.split['Y'])
        if self.child1 != 0:
            self.child1.print_node(level + 1)
        if self.child2 != 0:
            self.child2.print_node(level + 1)
                    
class TwoD_Tree:
    root_node = 0
    
    def __init__(self):
        pass
    
    def fill(self, point):
        if self.root_node == 0:
            self.root_node = TwoD_Node(point, 0)
        else:
            self.root_node.fill(point)
        
    def get_nearest_neighbor(self, point):
        return self.root_node.get_nearest_neighbor(point)
    
    def print_tree(self):
        self.root_node.print_node(0)
tree = TwoD_Tree()

nn_classifier = NN_Classifier(tree)

nn_classifier.train(train_data)  
predicted_data = pd.DataFrame(columns = ['X', 'Y', 'class'])

for index, data in test_data.iterrows():
    prediction = nn_classifier.predict(data)
    predicted_data = predicted_data.append(pd.Series({'X' : data['X'], 'Y' : data['Y'], 'class' : prediction}), ignore_index=True)
accuracy = get_accuracy(classified_data, predicted_data)

print("accuracy: ", accuracy)
    
tiles = plt.subplots(1, 3, figsize=(30, 10))

decision_boundary = get_decision_boundary(nn_classifier, -1.5, 1.5, -1.5, 1.5, 0.01) 

plot_data(tiles, 0, train_data, 0, "train data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 1, classified_data, 0, "classified data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 2, predicted_data, decision_boundary, "predicted data", -1.5, 1.5, -1.5, 1.5)
class FourD_Node:
    split = 0
    child1 = 0
    child2 = 0
    child3 = 0
    child4 = 0
    
    def __init__(self, point):
        self.split = point
    
    def fill(self, point):
        if point['X'] < self.split['X']:
            if point['Y'] < self.split['Y']:
                if self.child1 != 0:
                    self.child1.fill(point)
                else:
                    self.child1 = FourD_Node(point)
            else:
                if self.child2 != 0:
                    self.child2.fill(point)
                else:
                    self.child2 = FourD_Node(point)
        else: 
            if point['Y'] < self.split['Y']:
                if self.child3 != 0:
                    self.child3.fill(point)
                else:
                    self.child3 = FourD_Node(point)
            else:
                if self.child4 != 0:
                    self.child4.fill(point)
                else:
                    self.child4 = FourD_Node(point)
                    
    def get_nearest_neighbor(self, point):
        if point['X'] < self.split['X']:
            if point['Y'] < self.split['Y']:
                if self.child1 != 0:
                    nn = self.child1.get_nearest_neighbor(point)
                    if get_distance(nn, point) < get_distance(self.split, point):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
            else:
                if self.child2 != 0:
                    nn = self.child2.get_nearest_neighbor(point)
                    if get_distance(nn, point) < get_distance(self.split, point):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
        else:
            if point['Y'] < self.split['Y']:
                if self.child3 != 0:
                    nn = self.child3.get_nearest_neighbor(point)
                    if get_distance(nn, point) < get_distance(self.split, point):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
            else:
                if self.child4 != 0:
                    nn = self.child4.get_nearest_neighbor(point)
                    if get_distance(nn, point) < get_distance(self.split, point):
                        return nn;
                    else:
                        return self.split
                else:
                    return self.split
                    
    def print_node(self, level):
        for x in range(0, level):
            print("   ", end="")
        print(self.split['X'], self.split['Y'])
        if self.child1 != 0:
            self.child1.print_node(level + 1)
        if self.child2 != 0:
            self.child2.print_node(level + 1)
        if self.child3 != 0:
            self.child3.print_node(level + 1)
        if self.child4 != 0:
            self.child4.print_node(level + 1)
                    
class FourD_Tree:
    root_node = 0
    
    def __init__(self):
        pass
    
    def fill(self, point):
        if self.root_node == 0:
            self.root_node = FourD_Node(point)
        else:
            self.root_node.fill(point)
        
    def get_nearest_neighbor(self, point):
        return self.root_node.get_nearest_neighbor(point)
    
    def print_tree(self):
        self.root_node.print_node(0)
tree = FourD_Tree()

nn_classifier = NN_Classifier(tree)

nn_classifier.train(train_data)    
predicted_data = pd.DataFrame(columns = ['X', 'Y', 'class'])

for index, data in test_data.iterrows():
    prediction = nn_classifier.predict(data)
    predicted_data = predicted_data.append(pd.Series({'X' : data['X'], 'Y' : data['Y'], 'class' : prediction}), ignore_index=True)
accuracy = get_accuracy(classified_data, predicted_data)

print("accuracy: ", accuracy)
    
tiles = plt.subplots(1, 3, figsize=(30, 10))

decision_boundary = get_decision_boundary(nn_classifier, -1.5, 1.5, -1.5, 1.5, 0.01) 

plot_data(tiles, 0, train_data, 0, "train data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 1, classified_data, 0, "classified data", -1.5, 1.5, -1.5, 1.5)
plot_data(tiles, 2, predicted_data, decision_boundary, "predicted data", -1.5, 1.5, -1.5, 1.5)