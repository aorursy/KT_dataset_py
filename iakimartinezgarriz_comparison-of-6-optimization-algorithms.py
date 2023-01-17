from IPython.display import display, Math, Latex



display(Math(r'y_p = \theta\cdot X'))
import numpy as np

import matplotlib.pyplot as plt
np.random.seed(42)

m = 1000

n = 2

x = np.linspace(0, 1, m)

ones = np.ones(m)

X = np.column_stack((ones, x))

Y = 5 * x + 4

y = 5 * x + 4 + np.random.randn(len(x))/5

y = y.reshape(m, 1)
labels = ['Instances', 'Initial model']



plt.plot(x, y, 'go')

plt.plot(x, Y, 'r')

plt.legend(labels)

plt.show()
np.random.seed(42)

theta0 = np.array([np.random.randn(), np.random.randn()])

theta0 = theta0.reshape(n, 1)



m0 = np.array([0, 0])

m0 = m0.reshape(n, 1)



s0 = np.array([0, 0])

s0 = s0.reshape(n, 1)
display(Math(r'MSE(\theta) = \frac{1}{m} \cdot \sum_{i=1}^{m}(X^{(i)} \theta - y^{(i)})^2'))
display(Math(r'\nabla_\theta MSE(\theta) = \frac{2}{m} X^T (X\theta - y)'))
mse = lambda x: (1/m) * np.sum((X @ x - y) ** 2)



grad_mse = lambda x: (2/m) * (X.T @ (X @ x - y))
learning_rate = 0.005

max_epochs = 100000
thetas = []

_losses = []

n_epochs = []
display(Math(r'\theta = (X^T X)^{-1}X^T y'))
mathematical_solution = np.linalg.inv(X.T @ X) @ X.T @ y

mathematical_solution
display(Math(r'\theta_{next step} = \theta - \alpha \nabla_\theta MSE(\theta)'))
def gradient_descent(epochs, learning_rate, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    last_loss = 9999

    

    for epoch in range(epochs):

        loss = mse(theta)

        gradients = grad_mse(theta)

        theta = theta - (learning_rate * gradients)

        

        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if(abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)

                return theta, losses, epoch

            else:

                last_loss = loss

    

    losses = np.array(losses)

    print('Max. number of iterations without converging')   

    return theta, losses, epochs
p1, losses, epoch = gradient_descent(max_epochs, learning_rate)

thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)
plt.plot(_losses[0][0:, 1], _losses[0][0:, 0])

plt.show()
plt.plot(_losses[0][50:, 1], _losses[0][50:, 0])

plt.show()
display(Math(r'1.\quad m \leftarrow \beta m - \alpha \nabla_\theta MSE(\theta)'))

display(Math(r'2.\quad \theta = \theta + m'))
def momentum_optimization(epochs, learning_rate, beta, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    m = m0

    last_loss = 9999

    

    for epoch in range(epochs):

        loss = mse(theta)

        gradients = grad_mse(theta)

        m = beta * m + learning_rate * gradients

        theta = theta - m

        

        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if(abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)

                return theta, losses, epoch

            else:

                last_loss = loss

            

    losses = np.array(losses)

    print('Max. number of iterations without converging')  

    return theta, losses, epochs
p1, losses, epoch = momentum_optimization(max_epochs, learning_rate, 0.9)

thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)



plt.plot(_losses[1][:, 1], _losses[1][:, 0])

plt.show()
plt.plot(_losses[1][3:, 1], _losses[1][3:, 0])

plt.show()
display(Math(r'1.\quad m \leftarrow \beta m - \alpha \nabla_\theta MSE(\theta + \beta m)'))

display(Math(r'2.\quad \theta = \theta + m'))
def nesterov_accelerated_gradient(epochs, learning_rate, beta, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    m = m0

    last_loss = 9999

    

    for epoch in range(epochs):

        loss = mse(theta)

        gradients = grad_mse(theta + beta * m)

        m = beta * m + learning_rate * gradients

        theta = theta - m

        

        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if(abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)        

                return theta, losses, epoch

            else:

                    last_loss = loss

            

    losses = np.array(losses)

    print('Max. number of iterations without converging')

    return theta, losses, epochs
p1, losses, epoch = nesterov_accelerated_gradient(max_epochs, learning_rate, 0.9)



thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)



plt.plot(_losses[1][:, 1], _losses[1][:, 0])

plt.show()
plt.plot(_losses[1][3:, 1], _losses[1][3:, 0])

plt.show()
display(Math(r'1.\quad s \leftarrow s + \nabla_\theta MSE(\theta)\otimes \nabla_\theta MSE(\theta)'))

display(Math(r'2. \quad \theta \leftarrow \theta - \alpha \nabla_\theta MSE(\theta) \oslash \sqrt{s + \epsilon}'))
def adaGrad(epochs, learning_rate, epsilon=10**-10, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    s = s0

    last_loss = 9999

    

    for epoch in range(epochs):

        loss = mse(theta)

        gradients = grad_mse(theta)

        s = s + gradients * gradients

        theta = theta - (learning_rate * gradients) / (np.sqrt(s+ epsilon))

        

        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if (abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)

                return theta, losses, epoch

            else:

                last_loss = loss

            

    losses = np.array(losses)

    print('Max. number of iterations without converging')

    return theta, losses, epochs
p1, losses, epoch = adaGrad(max_epochs, learning_rate)



thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)



plt.plot(_losses[3][:, 1], _losses[3][:, 0])

plt.show()
display(Math(r'1.\quad s \leftarrow \beta s + (1 - \beta) \nabla_\theta MSE(\theta)\otimes \nabla_\theta MSE(\theta)'))

display(Math(r'2. \quad \theta \leftarrow \theta - \alpha \nabla_\theta MSE(\theta) \oslash \sqrt{s + \epsilon}'))
def RMSProp(epochs, learning_rate, beta, epsilon=10**-10, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    s = s0

    last_loss = 9999

    

    for epoch in range(epochs):

        loss = mse(theta)

        gradients = grad_mse(theta)

        s = beta * s + (1 - beta) * gradients * gradients

        theta = theta - (learning_rate * gradients) / (np.sqrt(s + epsilon))

        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if(abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)

                return theta, losses, epoch

            else:

                last_loss = loss

            

    losses = np.array(losses)

    print('Max. number of iterations without converging')    

    return theta, losses, epochs
p1, losses, epoch = RMSProp(max_epochs, learning_rate, 0.9)



thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)



plt.plot(_losses[4][:, 1], _losses[4][:, 0])

plt.show()
display(Math(r'1. \quad m \leftarrow \beta_1 m - (1 - \beta_1)\nabla_\theta MSE(\theta)'))

display(Math(r'2. \quad s \leftarrow \beta_2 s + (1 - \beta_2)\nabla_\theta MSE(\theta) \otimes\nabla_\theta MSE(\theta)'))

display(Math(r'3. \quad \hat{m} \leftarrow \frac{m}{1 - \beta_1^T}'))

display(Math(r'4. \quad \hat{s} \leftarrow \frac{s}{1 - \beta_2^T}'))

display(Math(r'5. \quad \theta \leftarrow \theta + \alpha \hat{m} \oslash \sqrt{\hat{s} + \epsilon}'))

print('Where T represents the iteration number')
def adam_opt(epochs, learning_rate, beta1, beta2, epsilon=10**-10, loss_variation=10**-5):

    

    theta = theta0

    losses = []

    s = s0

    m = m0

    last_loss = 9999

    

    for epoch in range(epochs):

        e = epoch

        loss = mse(theta)

        gradients = grad_mse(theta)



        m = beta1 * m + (1 - beta1) * gradients

        s = beta2 * s + (1 - beta2) * gradients * gradients



        m2 = m / (1 - beta1**(epoch+1))

        s2 = s / (1 - beta2**(epoch+1))



        theta = theta - (learning_rate * m2 )/ (np.sqrt(s2 + epsilon))



        if epoch % 20 == 0:

            losses.append((loss, epoch))

            if(abs(last_loss - loss) < loss_variation):

                losses = np.array(losses)

                return theta, losses, epoch

            else:

                last_loss = loss

        

    losses = np.array(losses)

    print('Max. number of iterations without converging') 

    return theta, losses, epochs
p1, losses, epoch = adam_opt(max_epochs, learning_rate, 0.9, 0.9)



thetas.append(p1)

_losses.append(losses)

n_epochs.append(epoch)



plt.plot(_losses[5][:, 1], _losses[5][:, 0])

plt.show()
labels = ['Gradient Descent', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSProp', 'Adam']

y_pos = np.arange(len(n_epochs))



fig = plt.figure(figsize=(15,6))

plt.bar(y_pos, n_epochs)

plt.xticks(y_pos, labels)

plt.show()
labels.remove('AdaGrad')

n_epochs = np.delete(n_epochs, 3)

y_pos = np.arange(len(n_epochs))



fig = plt.figure(figsize=(15,6))

plt.bar(y_pos, n_epochs)

plt.xticks(y_pos, labels)

plt.show()
fig = plt.figure(figsize=(15,6))

# Instances

plt.plot(x, y, 'go')

# First model

plt.plot(x, Y, 'r')

# Mathematical solution

ms = mathematical_solution[0] + mathematical_solution[1] * x

plt.plot(x, ms, 'y')

# Gradient descent

gd = thetas[0][0] + thetas[0][1] * x

plt.plot(x, gd, 'b')

# Momentum

mo = thetas[1][0] + thetas[1][1] * x

plt.plot(x, mo, 'cyan')

# Nesterov

no = thetas[2][0] + thetas[2][1] * x

plt.plot(x, no, 'salmon')

# AdaGrad

ag = thetas[3][0] + thetas[3][1] * x

plt.plot(x, ag, 'black')

# RMSProp

rms = thetas[4][0] + thetas[4][1] * x

plt.plot(x, rms, 'orange')

# Adam

adam = thetas[4][0] + thetas[4][1] * x

plt.plot(x, adam, 'purple')



labels = ['Instances', 'Initial model', 'Mathematical solution', 'Gradient descent',

         'Momentum optimizer', 'Nesterov optimizer', 'AdaGrad optmizer', 'RMSProp optimizer', 'Adam optimizer']



plt.legend(labels)



plt.show()