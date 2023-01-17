import numpy as np
A=3

B=5

X = np.linspace(0,100,101)

Y = A*X + B

print ("A few data samples ")

for i in range(0,51,10):

    print (X[i],Y[i])
def loss_function(y_actual,y_predicted):

    return (abs(y_actual - y_predicted))
initial_weight_a = 0

initial_weight_b = 0

w = [initial_weight_a,initial_weight_b]

print (w)
learning_rate = 0.0002

iterations = 2000
for i in range(iterations):

    np.random.shuffle(X)

    for x in X:

        y_actual = A*x + B

        y_predicted = w[0]*x + w[1]

        loss = loss_function(y_actual,y_predicted)

        w[0] = w[0] - learning_rate*(y_predicted - y_actual)*x

        w[1] = w[1] - learning_rate*(y_predicted - y_actual)*1



print (w)        