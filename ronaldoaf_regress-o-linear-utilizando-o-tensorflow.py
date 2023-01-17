import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 



print(tf)
np.random.seed(101) 

tf.set_random_seed(101) 



# Genrating random linear data 

# There will be 50 data points ranging from 0 to 50 

x = np.linspace(0, 50, 50) 

y = np.linspace(0, 50, 50) 

  

# Adding noise to the random linear data 

x += np.random.uniform(-4, 4, 50) 

y += np.random.uniform(-4, 4, 50) 

  

n = len(x) # Number of data points 



print('x=', x)

print('y=', y)
# Plot of Training Data 

plt.scatter(x, y) 

plt.xlabel('x') 

plt.xlabel('y') 

plt.title("Training Data") 

plt.show() 
#Vamos começar a criar nosso modelo definindo os  placeholders X e Y, para podemos alimentarmos com os dados de treinamento.

X = tf.placeholder("float") 

Y = tf.placeholder("float") 





#Vamos declarar duas Variáveis do Tensorflow para Weights(pesos) e Bias(viés) iniciando-ao aleatoriamente usando np.random.randn()

W = tf.Variable(np.random.randn(), name = "W") 

b = tf.Variable(np.random.randn(), name = "b") 





#Definimos os hiperparametros do modelo, Learning Rate (taxa de aprendizagem) e o número de Epochs (interações)

learning_rate = 0.01

training_epochs = 1000





#Vamos construir a Hypothesis (hipótese), a Cost Function (função custo) e o Optimizer (optimizador).

#Não vamos implementar o Gradient Descent Optimizer manualmente, pois ele é criado dentro do Tensorflow.

#Depois disso, estaremos inicializando as variáveis.



# Hypothesis 

y_pred = tf.add(tf.multiply(X, W), b) 



# Mean Squared Error Cost Function 

cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 



# Gradient Descent Optimizer 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 



# Global Variables Initializer 

init = tf.global_variables_initializer() 

# Starting the Tensorflow Session 

with tf.Session() as sess: 

      

    # Initializing the Variables 

    sess.run(init) 

      

    # Iterating through all the epochs 

    for epoch in range(training_epochs): 

          

        # Feeding each data point into the optimizer using Feed Dictionary 

        for (_x, _y) in zip(x, y): 

            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 

          

        # Displaying the result after every 50 epochs 

        if (epoch + 1) % 50 == 0: 

            # Calculating the cost a every epoch 

            c = sess.run(cost, feed_dict = {X : x, Y : y}) 

            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

      

    # Storing necessary values to be used outside the Session 

    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 

    weight = sess.run(W) 

    bias = sess.run(b) 
# Calculating the predictions 

predictions = weight * x + bias 

print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 
# Plotting the Results 

plt.plot(x, y, 'ro', label ='Original data') 

plt.plot(x, predictions, label ='Fitted line') 

plt.title('Linear Regression Result') 

plt.legend() 

plt.show() 
