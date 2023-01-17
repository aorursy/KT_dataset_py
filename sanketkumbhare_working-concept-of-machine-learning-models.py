# importing required libraries 

from sklearn import datasets   

import matplotlib.pyplot as plt

import math
# loadind data from dataset

iris = datasets.load_iris()
print(iris.DESCR)
iris.data[:5]
iris.target
# Select all rows and only first two columns (sepal length/width)

X = iris.data[:, :2]



# Target will be used to plot samples in different colors for different species

Y = iris.target



plt.scatter(X[:,0], X[:,1], c=Y)

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Sepal Width (cm)')

plt.title('Sepal size distribution')
# Select all rows and only last two columns (sepal length/width)

X = iris.data[:, 2:]



# Target will be used to plot samples in different colors for different species

Y = iris.target



plt.scatter(X[:,0], X[:,1], c=Y)

plt.xlabel('Petal Length')

plt.ylabel('Petal Width')

plt.title('Petal size distribution (cm)')
def sigmoid(z):

  return 1.0/(1 + math.e ** (-z))



x = [i * 0.1 for i in range(-50, 51)]

y = [sigmoid(z) for z in x]

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Logistic Sigmoid')
def predict(sample):

  result  = 0.0

  for i in range(len(sample)):

    result = result + weights[i] * sample[i]

    

  result = result + bias

  return sigmoid(result)
def loss(y_train, y_predicted):

  return -(y_train * math.log(y_predicted) + (1.0 - y_train) * math.log(1 - y_predicted))
y_train = 0.9

x = [i * 0.1 for i in range(1, 9)]

y = [loss(y_train, yp) for yp in x]

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Loss near %0.2f' % y_train)
def parabola(x):

    return x**2 + x/2.0



x = [i * 0.1 for i in range(-10, 11)]

y = [parabola(xi) for xi in x]

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.title('Simple function')
x_k = 0.0



learning_rate = 0.5



def derivative(x):

  return 2*x + 0.5



for i in range(5):

    gradient = derivative(x_k)

    x_k = x_k - learning_rate*gradient



print('Estimated minimum %0.2f, %0.2f' % (x_k, parabola(x_k)))

print('Derivative (gradient) %0.2f' % gradient)



x = [i * 0.1 for i in range(-10, 11)]

y = [parabola(xi) for xi in x]

plt.plot(x, y)

plt.xlabel('x')

plt.ylabel('y')

plt.plot(x_k, parabola(x_k), 'ro')

line_x = [x_k - 0.5, x_k + 0.5]

line_y = [gradient*(xi-x_k)+parabola(x_k) for xi in line_x]

plt.plot(line_x, line_y)

plt.title('Simple function')
num_features = iris.data.shape[1]



def train_one_epoch(x_train_samples, y_train_samples):

  cost = 0.0

  dw = [0.0] * num_features

  db = 0.0



  global bias, weights



  m = len(x_train_samples)

  for i in range(m):

    x_sample = x_train_samples[i]

    y_sample = y_train_samples[i]

    predicted = predict(x_sample)

    cost = cost + loss(y_sample, predicted)

    

    # dz is the derivative of the loss function

    dz = predicted - y_sample

    

    for j in range(len(weights)):

      dw[j] = dw[j] + x_sample[j] * dz

    db = db + dz

  

  cost = cost / m

  db = db / m

  bias = bias - learning_rate*db

  for j in range(len(weights)):

    dw[j] = dw[j] / m

    weights[j] = weights[j] - learning_rate*dw[j]

  

  return cost
# Model will "learn" values for the weights and biases



weights = [0.0] * num_features

bias = 0.0

 

learning_rate = 0.1



epochs = 2000



x_train_samples = iris.data



# here we are training the model according to Iris-Virginica we change it by changing the value of 'y' to '0', '1', '2'

# '0' for Iris-Iris-Setosa, '1' for Iris-Versicolour, '2' for Iris-Virginica

y_train_samples = [1 if y == 2 else 0 for y in iris.target] 



loss_array = []

for epoch in range(epochs):

  loss_value = train_one_epoch(x_train_samples, y_train_samples)

  loss_array.append(loss_value)



plt.plot(range(epochs), loss_array)

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.title('Loss vs. Epoch')

plt.show()
predictions = []



m = len(x_train_samples)

correct = 0

for i in range(m):

  sample = x_train_samples[i]

  value = predict(sample)

  predictions.append(value)

  if value >= 0.5:

    value = 1

  else:

    value = 0

  if value == y_train_samples[i]:

    correct = correct + 1.0



plt.plot(range(m), predictions, label='Predicted')

plt.plot(range(m), y_train_samples, label='Ground truth')

plt.ylabel('Prediction')

plt.xlabel('Sample')

plt.legend(loc='best')

plt.show()



print('Accuracy: %.2f %%' % (100 * correct/m))
weights
bias