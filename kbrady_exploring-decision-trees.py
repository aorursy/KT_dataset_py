# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv')

print(training_data.columns)
training_data.head()
from collections import Counter

from matplotlib import pyplot as plt

%matplotlib inline



# put the feature names into a list

features = training_data.columns[1:-1]



# Look at feature values

value_counters = {}

value_types = {}



for f in features:

    col = Counter(training_data[f])

    value_counters[f] = col

    unique_values = col.keys()

    vt = [type(k) for k in unique_values if k == k]

    value_types[f] = vt[0]

    

print('The types of variables are ', list(set(value_types.values())))



num_categories_per_str = Counter([len(value_counters[f]) for f in features if value_types[f] == str])

print('The maximum number of unique values in a string variable is ', max(num_categories_per_str))



def in_range(val, lower, upper):

    return val >= lower and val <= upper



def find_bin(val, bins):

    for i in range(len(bins)):

        if in_range(val, bins[i][0], bins[i][1]):

            return i

    return -1



bins = [(0,25),(26,60),(61,250),(251,2000)]

int_distribution = Counter([find_bin(len(value_counters[f]), bins) for f in features if value_types[f] == np.int64])

labels = [str(b) for b in bins]

values = [int_distribution[i] for i in range(len(labels))]

plt.bar(range(len(labels)), values, align='center')

plt.xticks(range(len(labels)), labels)

plt.title('Distribution of #values in integer columns')

plt.show()
y = list(training_data['SalePrice'])

plt.hist(y, bins=int(np.sqrt(len(y))))

plt.title('Sale Price Histogram')
from collections import defaultdict



def get_variance(guess, data):

    return np.mean([(guess - d) ** 2 for d in data])



def evaluate_split(y_1, y_2):

    guess_1 = np.mean(y_1)

    error_1 = get_variance(guess_1, y_1)

    guess_2 = np.mean(y_2)

    error_2 = get_variance(guess_2, y_2)

    return error_1, error_2, guess_1, guess_2

        

def is_catagorical(f):

    if value_types[f] == str:

        return True

    if value_types[f] == np.float64:

        return False

    return len(value_counters[f]) <= 25



def build_long_term_lambda(current_list, current_index, catagorical=True):

    if catagorical:

        deep_list_copy = []

        for i in range(current_index):

            deep_list_copy.append(current_list[i])

        return lambda val : val in deep_list_copy

    else:

        cut_off_val = current_list[current_index]

        return lambda val : val < cut_off_val



def maximize(f, training):

    # get the value to predict and the feature

    y = list(training['SalePrice'])

    x = list(training[f])

    # zip x and y together to make sure indexes don't get off

    data = list(zip(x, y))

    # if the data is catagorical split the data into groups for each value

    # order these groups by their mean y value so we don't go through all

    # combinations in our split function

    if is_catagorical(f):

        groups = defaultdict(list)

        null_data = []

        for d in data:

            if type(d[0]) is not str and np.isnan(d[0]):

                null_data.append(d[1])

                continue

            groups[d[0]].append(d[1])

        x_set = [(k, np.mean(groups[k])) for k in groups]

        x_set.sort(key=lambda v: v[1])

        x_set = [v[0] for v in x_set]

    # if the data is numerical take out the NaN values (we will deal with these at the end)

    else:

        null_data = [d[1] for d in data if np.isnan(d[0])]

        data = [d for d in data if not np.isnan(d[0])]

        x_set = list(set([d[0] for d in data]))

    # variables to keep track of the minimum error and the best split

    best_lambda = None

    min_error = -1

    best_guess = (0,0)

    best_pos_desc = ""

    best_neg_desc = ""

    # loop through the possible splits we can make in the data

    # (for the moment we only consider binary splits)

    for index in range(len(x_set)):

        if is_catagorical(f):

            # we need to make a deep copy of the list of values to include in the split or 

            # the function will change when index gets updated

            this_lambda = build_long_term_lambda(x_set, index)

            # descriptions are good for keeping track of what the tree is doing

            pos_description = str(x_set[:index])

            neg_description = str(x_set[index:])

        else:

            this_lambda = build_long_term_lambda(x_set, index, catagorical=False)

            pos_description = "< "+str(x_set[index])

            neg_description = ">= "+str(x_set[index])

        y_1 = [d[1] for d in data if this_lambda(d[0])]

        y_2 = [d[1] for d in data if not this_lambda(d[0])]

        # If a split puts all the data on one side (i.e. is a minimum or maximum)

        # skip the split and continue with the next option

        if len(y_1) == 0 or len(y_2) == 0:

            continue

        error_1, error_2, guess_1, guess_2 = evaluate_split(y_1, y_2)

        combined_error = (error_1 * len(y_1) + error_2 * len(y_2)) / len(data)

        if min_error < 0 or combined_error < min_error:

            min_error = combined_error

            best_lambda = this_lambda

            # We need to keep track of the guesses made for each split so we can quickly 

            # figure out where to put the NaN values at the end

            best_guess = (guess_1, guess_2)

            # descriptions are good for keeping track of what the tree is doing

            best_pos_desc = pos_description

            best_neg_desc = neg_description

    # add in the capicty to reason about NaN values

    if len(null_data) != 0:

        mean_null = np.mean(null_data)

        if abs(mean_null - best_guess[0]) < abs(mean_null - best_guess[1]):

            best_lambda = lambda val : np.isnan(val) or best_lambda(val)

            best_pos_desc = best_pos_desc + " or NaN"

        else:

            best_lambda = lambda val : not np.isnan(val) and best_lambda(val)

            best_neg_desc = best_neg_desc + " or NaN"

    return best_lambda, min_error, best_pos_desc, best_neg_desc



error_values = []

split_functions = {}

data = training_data

print(len(data))

for f in features:

    split_fun, error, pos_desc, neg_desc = maximize(f, data)

    if error > 0:

        error_values.append((f, error))

        split_functions[f] = (split_fun, pos_desc, neg_desc)

error_values.sort(key=lambda p:p[1])

print(error_values)
def show_split(attr_name, training, split_functions):

    y = list(training['SalePrice'])

    x = list(training[attr_name])

    data = list(zip(x, y))

    split_fun = split_functions[attr_name][0]

    pos_desc = split_functions[attr_name][1]

    neg_desc= split_functions[attr_name][2]

    y_1 = [d[1] for d in data if split_fun(d[0])]

    y_2 = [d[1] for d in data if not split_fun(d[0])]

    

    min_val = min(y)

    max_val = max(y)

    

    num_bins = 10

    bins = [min_val + i*(max_val - min_val)/num_bins for i in range(num_bins+1)]

    ind = np.array(bins[:-1])

    width = 0.4 * (bins[1] - bins[0])

    

    hist_1 = plt.hist(y_1, bins=bins)

    hist_2 = plt.hist(y_2, bins=bins)

    plt.cla()

    

    plt.bar(ind, hist_1[0], width, color='green', align='center', edgecolor='none',

            label=str(len(y_1))+' observations')

    plt.bar(ind+width, hist_2[0], width, color='blue', align='center', edgecolor='none',

            label=str(len(y_2))+' observations')

    plt.title('Split on '+attr_name)

    plt.legend(loc=1)

    plt.show()

    print("green: "+pos_desc)

    print("blue: "+neg_desc)



show_split('Neighborhood', training_data, split_functions)
def find_best_feature(training, features):

    error_values = []

    split_functions = {}

    for f in features:

        split_fun, error, pos_desc, neg_desc = maximize(f, training)

        if error > 0:

            error_values.append((f, error))

            split_functions[f] = (split_fun, pos_desc, neg_desc)

    error_values.sort(key=lambda p:p[1])

    if len(error_values) == 0:

        return None, -1

    best_feature = error_values[0][0]

    return best_feature, split_functions[best_feature]



class Node:

    def __init__(self, parent, data, features, desc = "", depth = 0, stopping_criteria = lambda n : n.depth >= 2):

        self.parent = parent

        self.data = data

        self.features = features

        self.desc = desc

        self.depth = depth

        self.stopping_criteria = stopping_criteria

        self.guess = np.mean(self.data['SalePrice'])

        self.error = np.sqrt(get_variance(self.guess, self.data['SalePrice']))

        self.split_fun = None

        self.feature = None

        self.children = []

        self.build_children()

    

    def __repr__(self):

        my_str = "\n"

        for i in range(self.depth):

            my_str += " "

        my_str += "{points: "+str(len(self.data))+", depth: "

        my_str += str(self.depth)+", error: "+str(self.error)

        my_str += ", desc:"+self.desc+", children:"+str(self.children)+"}"

        return my_str

    

    def build_children(self):

        if self.stopping_criteria(self):

            return

        best_feature, split_info = find_best_feature(self.data, self.features)

        if best_feature is None:

            return

        split_fun = split_info[0]

        if split_fun is None:

            return

        pos_desc = best_feature + " " + split_info[1]

        neg_desc= best_feature + " " + split_info[2]

        pos_child_data = self.data[[split_fun(val) for val in self.data[best_feature]]]

        neg_child_data = self.data[[not split_fun(val) for val in self.data[best_feature]]]

        pos_child = Node(self, pos_child_data, features, pos_desc, self.depth+1, self.stopping_criteria)

        neg_child = Node(self, neg_child_data, features, neg_desc, self.depth+1, self.stopping_criteria)

        self.children = [pos_child, neg_child]

        self.split_fun = split_fun

        self.feature = best_feature

    

    def predict(self, data_points):

        ids = list(data_points['Id'])

        if self.split_fun is None:

            return list(zip(ids, [self.guess] * len(data_points)))

        pos_child_points = data_points[[self.split_fun(val) for val in data_points[self.feature]]]

        neg_child_points = data_points[[not self.split_fun(val) for val in data_points[self.feature]]]

        return self.children[0].predict(pos_child_points) + self.children[1].predict(neg_child_points)

    

    def tree_error(self, data_points=None):

        if data_points is None:

            data_points = self.data

        predictions = dict(self.predict(data_points))

        actual = dict(zip(list(data_points['Id']), list(data_points['SalePrice'])))

        return np.sqrt(np.mean([(actual[k] - predictions[k]) ** 2 for k in actual.keys()]))

            

my_tree = Node(None, training_data, features, desc="", depth=0, stopping_criteria = lambda n : n.depth > 3 or len(n.data) <= 100 or n.error < 4000)

print(my_tree)

print(my_tree.tree_error())



base_tree = Node(None, training_data, features, desc="", depth=0, stopping_criteria = lambda n : n.depth >= 0)

print(base_tree)

print(base_tree.tree_error())
k = 10

k_trees = []

length = len(training_data)/k

error_measurements = []

for index in range(k):

    testing = training_data[int(index*length):int((index+1)*length)]

    training = pd.concat([training_data[:int(index*length)], training_data[int((index+1)*length):]])

    this_tree = Node(None, training, features, stopping_criteria = lambda n : n.depth > 2 or len(n.data) <= 100 or n.error < 4000)

    error_measurements.append(this_tree.tree_error(testing))

    k_trees.append(this_tree)

    print(error_measurements[-1])
class DescriptiveNode:

    def __init__(self, node_list, depth=0):

        self.feature_set = Counter([n.feature for n in node_list])

        self.num_nodes = len(node_list)

        self.depth = depth

        left_list = [n.children[0] for n in node_list if len(n.children) > 0]

        right_list = [n.children[1] for n in node_list if len(n.children) > 0]

        self.children = [self.build_child(left_list), self.build_child(right_list)]

    

    def build_child(self, child_list):

        if len(child_list) > 0:

            child = DescriptiveNode(child_list, self.depth+1)

        else:

            child = ""

        return child

    

    def __repr__(self):

        my_str = "\n"

        for i in range(self.depth):

            my_str += " "

        my_str += "{feature_set: "+str(self.feature_set)+", points: "+str(self.num_nodes)

        my_str += ", children:"+str(self.children)+"}"

        return my_str



print(DescriptiveNode(k_trees))