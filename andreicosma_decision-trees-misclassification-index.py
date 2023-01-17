import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv_dataset = pd.read_csv("../input/species_classification.csv")
csv_dataset.describe()
csv_dataset.plot.hist(alpha=0.25)
plt.show()
class MisclassificationTreeNode:
    parent = None
    branch = None
    attr_index = None
    permitted_attributes = None
    permitted_entries = None
    children = None
    classes_dist_dict = None
    class_label = None

    def __init__(self, parent=None,
                 attr_index=None,
                 children=None,
                 permitted_attributes=None,
                 permitted_entries=None,
                 classes_dist_dict=None,
                 branch=None,
                 class_type=None):
        self.class_label = class_type
        self.parent = parent
        self.attr_index = attr_index
        self.children = children
        self.permitted_attributes = permitted_attributes
        self.permitted_entries = permitted_entries
        self.classes_dist_dict = classes_dist_dict
        self.branch = branch
def attr_distribution(x, y, distinct_classes, attr_index, permitted_entries_indexes):
    branches_class_distribution = {}
    branches = []
    for entry in permitted_entries_indexes:
        if x[entry][attr_index] not in branches:
            branches.append(x[entry][attr_index])
    for branch in branches:
        class_distribution_over_a_branch = {}
        for i in distinct_classes:
            class_distribution_over_a_branch.update({i: 0})
        total_of_a_single_branch = 0
        for permitted_entry_index in permitted_entries_indexes:
            if branch == x[permitted_entry_index][attr_index]:
                total_of_a_single_branch += 1
                class_distribution_over_a_branch[y[permitted_entry_index]] = \
                    class_distribution_over_a_branch[(
                        y[permitted_entry_index])] + 1
        branches_class_distribution.update({branch: class_distribution_over_a_branch})
    return branches_class_distribution
def split_attr(x, y, distinct_classes, attr_index, permitted_entries_indexes):
    attr_dist = attr_distribution(x, y, distinct_classes, attr_index, permitted_entries_indexes)
    attr_split = 0
    for dictionary in attr_dist.values():
        attr_split += (1 - (max(dictionary.values()) / sum(dictionary.values()))) * (
                sum(dictionary.values()) / len(permitted_entries_indexes))
    return attr_split, attr_dist
def activate_node(x, y, distinct_classes, class_column, node):
    entries = []
    attributes = []
    if node.parent is not None:
        for entry in node.parent.permitted_entries:
            if x[entry][node.parent.attr_index] == node.branch:
                entries.append(entry)
        for attr in node.parent.permitted_attributes:
            attributes.append(attr)
    else:
        for i in range(0, len(x)):
            entries.append(i)
        for i in range(0, len(x[0])):
            attributes.append(i)
    node.permitted_entries = entries
    attributes_split_index_dictionary = {}
    attributes_dist_dict = {}
    for attribute in attributes:
        split_index, attr_dist = split_attr(x, y, distinct_classes, attribute, entries)
        attributes_split_index_dictionary.update({attribute: split_index})
        attributes_dist_dict.update({attribute: attr_dist})
    attr_index = min(attributes_split_index_dictionary,
                        key=attributes_split_index_dictionary.get)
    attributes.remove(attr_index)
    children = []
    node.attr_index = attr_index
    node.permitted_attributes = attributes
    node.classes_dist_dict = attributes_dist_dict.get(attr_index)
    for key, value in attributes_dist_dict.get(attr_index).items():
        child = MisclassificationTreeNode(parent=node, branch=key)
        child.classes_dist_dict = value
        if sum(value.values()) == max(value) or len(attributes) == 0:
            child.attr_index = class_column
            for key_1, value_1 in value.items():
                if value_1 == max(value.values()):
                    child.class_type = key
        children.append(child)
    node.children = children
class_column = 7
db = csv_dataset.iloc[:, :].values.astype(np.int_)
np.random.shuffle(db)
y = db[:, class_column]
distinct_classes = np.unique(y)
x = np.delete(db, [class_column], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
root = MisclassificationTreeNode(parent=None)
nodes_to_activate = queue.Queue()
nodes_to_activate.put(root)
while not nodes_to_activate.empty():
    candidate = nodes_to_activate.get()
    activate_node(x_train, y_train, distinct_classes, class_column, candidate)
    for node in candidate.children:
        if node.attr_index is None:
            nodes_to_activate.put(node)
q = queue.Queue()
q.put(root)
count_nodes = 1
count_leafs = 0
while not q.empty():
    candidate = q.get()
    count_nodes += 1
    if candidate.attr_index is class_column:
        count_leafs += 1
    if candidate.children is not None:
        for node in candidate.children:
            q.put(node)
print("The tree has %s nodes and %s leafs." % (count_nodes, count_leafs))
def decide(root, class_column, candidate):
    start = root
    while class_column is not start.attr_index:
        if start.attr_index == class_column:
            return start.class_type
        if candidate[start.attr_index] >= len(start.children):
            return -1
        else:
            start = start.children[candidate[start.attr_index]]
    return start.class_label
test_results = []
for entry in x_test:
    test_results.append(decide(root, class_column, entry))