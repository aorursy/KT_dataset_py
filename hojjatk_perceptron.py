import numpy as np

#
# Perceptron Demo
# If you're not familiar with Perceptron or would like to know more about it
# Please, see: https://en.wikipedia.org/wiki/Perceptron
#
class Perceptron(object):
    def __init__(self, input_size, epochs = 50, learning_rate = 0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate        
        #self.weights = np.zeros(input_size + 1) 
        self.weights = np.random.normal(0, 0.01, input_size + 1)
        
    def step(self, x):
        y = 1 if x > 0 else 0 
        return y
    
    def activate(self, X):        
        y = np.dot(X, self.weights[1:]) + self.weights[0]
        y = np.asscalar(y)
        return self.step(y)    
    
    
    def train(self, X_train, y_train):
        errors = []
        total_errors = []
        for epoch in range(self.epochs):
            total_error = 0                       
            for (x, y) in zip(X_train, y_train):                                
                pred = self.activate(x)
                error = y - pred                
                self.weights[1:] = self.weights[1:] + self.learning_rate * error * x
                self.weights[0] = self.weights[0] + self.learning_rate * error
                errors.append(error)
                total_error += error
            total_errors.append(total_error)
        return (total_errors, errors)
    
    
    def predict(self, X_test):
        preds = []
        for x in X_test:
            preds.append(self.activate(x))
        return preds
%matplotlib inline
import matplotlib.pyplot as plt

#
# Generate some test data
#
def gen2d_cluster(center, distance, size = 100):
    cluster_x1 = np.random.uniform(center[0], center[0] + distance, size=(size,))
    cluster_x2 = np.random.normal(center[1], distance, size=(size,)) 
    cluster_data = np.array(list(zip(cluster_x1, cluster_x2)))
    return cluster_data


center1 = (50, 60)
center2 = (80, 20)
distance = 20
cluster1 = gen2d_cluster(center1, distance)
cluster1_y = np.zeros(len(cluster1))

cluster2 = gen2d_cluster(center2, distance)
cluster2_y = np.ones(len(cluster2))

x_dataset = np.concatenate((cluster1, cluster2))
y_dataset = np.concatenate((cluster1_y, cluster2_y))

dataset = np.array(list(zip(x_dataset, y_dataset))) 
np.random.shuffle(dataset)
dataset_size = dataset.shape[0]
training_size = int(0.8 * dataset_size)

training_data = dataset[0:training_size, :]
test_data = dataset[training_size:dataset_size, :]

X_train = np.array([d[0] for d in training_data])
y_train = np.array([d[1] for d in training_data])

X_test = np.array([d[0] for d in test_data])
y_test = np.array([d[1] for d in test_data])

plt.scatter(cluster1[:, 0], cluster1[:, 1])
plt.scatter(cluster2[:, 0], cluster2[:, 1])
plt.show()
def normalize(X):
    X_normalize = X - np.min(X) / (np.max(X) - np.min(X))
    return X_normalize    

X_train_normalized = normalize(X_train)
X_test_normalized = normalize(X_test)
perceptron = Perceptron(2, epochs = 20, learning_rate = 0.1)
print('start training perceptron...')

(total_errors, errors) = perceptron.train(X_train_normalized, y_train)
plt.plot(total_errors)
plt.title("Total Error per epoch")
plt.show()

correct = 0
preds = perceptron.predict(X_test_normalized)
for (pred, y) in list(zip(preds, y_test)):
    if (pred == y):
        correct += 1

print(f'Test Accuracy = {correct}/{y_test.shape[0]} = {correct/y_test.shape[0]} ', )
