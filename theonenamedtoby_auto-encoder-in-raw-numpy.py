import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = (x_train / 255).reshape(-1, 784)

x_test = (x_test / 255).reshape(-1, 784)
def sigmoid(z, d=False):

    return sigmoid(z) * (1 - sigmoid(z)) + 1e-12 if d else 1 / (1 + np.exp(-z))



def relu(z, d=False):

    return (z > 0)+1e-12 if d else  z * (z > 0)
layers = [

    # activation, shape:(out,in)

    {"act":relu, "shape":(1024,784)},

    {"act":relu, "shape":(50,1024)},

    {"act":relu, "shape":(1024,50)},

    {"act":sigmoid, "shape":(784,1024)}

]
# global properties

l, errors, epochs = len(layers), [], 30

# adam properties

lr, b1, b2 = 0.002, 0.9, 0.999

rw,mw,rb,mb = {},{},{},{}

# layer properties

a,w,b,f, = {},{},{},{}

for i, layer in zip(range(1,l+1), layers):

    n_out, n_in = layer["shape"]

    f[i] = layer["act"]

    # Xavier Initialization of weights

    w[i] = np.random.randn(n_out, n_in) / n_in**0.5

    b[i], rb[i], mb[i] = [np.zeros((n_out,1)) for i in [1,2,3]]

    rw[i], mw[i] = [np.zeros((n_out, n_in)) for i in [1,2]]
for t in range(1, epochs+1):

    # Train

    for batch in np.split(x_train, 30):

        # Forward pass

        a[0] = batch.T

        for i in range(1,l+1):

            a[i] = f[i]((w[i] @ a[i-1]) + b[i])

        # Backpropagation

        dz,dw,db = {},{},{}

        for i in range(1,l+1)[::-1]:

            d = w[i+1].T @ dz[i+1] if l-i else 0.5*(a[l]-a[0])

            dz[i] = d * f[i](a[i],d=1)

            dw[i] = dz[i] @ a[i-1].T

            db[i] = np.sum(dz[i], 1, keepdims=True)

        # Adam updates

        def adam(m, r, z, dz, i):

            m[i] = b1 * m[i] + (1 - b1) * dz[i]

            r[i] = b2 * r[i] + (1 - b2) * dz[i]**2

            m_hat = m[i] / (1. - b1**t)

            r_hat = r[i] / (1. - b2**t) 

            z[i] -= lr * m_hat / (r_hat**0.5 + 1e-12)

        for i in range(1,l+1):

            adam(mw, rw, w, dw, i)

            adam(mb, rb, b, db, i)

    # Validate

    a[0] = x_test.T

    for i in range(1,l+1):

        a[i] = f[i]((w[i] @ a[i-1]) + b[i])

    errors += [np.mean((a[l]-a[0])**2)]

    print("Val loss - ", errors[-1])
import matplotlib.pyplot as plt

y_pred = []

a[0] = x_train[:20].T

#forward pass

for i in range(1,l+1):

    a[i] = f[i](w[i] @ a[i-1] + b[i])

y_pred = a[l]



plt.figure(figsize=(20,5))



for i in range(20):

    plt.subplot(3, 20, i + 1)

    plt.imshow(x_train[i].reshape(28,28), cmap="gray")

    plt.axis("off")

    plt.grid(b=False)



for i in range(20):

    plt.subplot(3, 20, i + 1 + 20)

    plt.imshow(a[l-2].T[i].reshape(5,-1), cmap="gray")

    plt.axis("off")

    plt.grid(b=False)

    

for i in range(20):

    plt.subplot(3, 20, i + 1 + 40)

    plt.imshow(y_pred.T[i].reshape(28,28), cmap="gray")

    plt.axis("off")

    plt.grid(b=False)



plt.show()