import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

csv_dataset = pd.read_csv("../input/bcwp_1995.csv")
csv_dataset.loc[:,['class']].plot()
print("Database shape rows: %s columns:%s \n" % np.shape(csv_dataset))
print(csv_dataset.describe())
plt.show()
def create_one_hot_encoding(classes, shape):
    one_hot_encoding = np.zeros(shape)
    for i in range(0, len(one_hot_encoding)):
        one_hot_encoding[i][int(classes[i])] = 1
    return one_hot_encoding
def train(weights, x, y):
    h = x.dot(weights)
    h = np.maximum(h, 0, h)
    return np.linalg.pinv(h).dot(y)
def soft_max(layer):
    soft_max_output_layer = np.zeros(len(layer))
    for i in range(0, len(layer)):
        numitor = 0
        for j in range(0, len(layer)):
            numitor += np.exp(layer[j] - np.max(layer))
        soft_max_output_layer[i] = np.exp(layer[i] - np.max(layer)) / numitor
    return soft_max_output_layer

def matrix_soft_max(matrix_):
    soft_max_matrix = []
    for i in range(0, len(matrix_)):
        soft_max_matrix.append(soft_max(matrix_[i]))
    return soft_max_matrix
def check_network_power(o, o_real):
    count = 0
    for i in range(0, len(o)):
        count += 1 if np.argmax(o[i]) == np.argmax(o_real[i]) else 0
    return count
def test(weights, beta, x, y):
    h = x.dot(weights)
    h = np.maximum(h, 0, h)  # ReLU
    o = matrix_soft_max(h.dot(beta))
    return check_network_power(o, y) / len(y)
class_column = 0
test_size = 0.1
db = csv_dataset.iloc[:, :].values.astype(np.float)
np.random.shuffle(db)
y = db[:, class_column]
y -= np.min(y)
output_layer_perceptron_count = len(np.unique(y))
y = create_one_hot_encoding(y, (len(y), len(np.unique(y))))
x = np.delete(db, [class_column], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
hidden_layer_perceptron_count = len(y_test)
x = preprocessing.normalize(x)
weights = np.random.random((len(x[0]), hidden_layer_perceptron_count))
beta = train(weights, x_train, y_train)
print("Accuracy = %s." % test(weights, beta, x_test, y_test))