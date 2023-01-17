import numpy as np
import torch, torch.nn as nn
import seaborn as sns
import math
from matplotlib import pyplot as plt
print(torch.__version__)

def generate_data(size, k, b, noise=0.):
    data = np.random.rand(size, 3)
    for i in range(size):
        if i <= size * (1 - noise):
            data[i, 2] = 1 if k * data[i, 0] + b > data[i, 1] else 0
        else:
            data[i, 2] = np.random.randint(0, 2)
    np.random.shuffle(data)
    return data

def q1_check():
    print("Right" if abs(sigmoid(3) - 0.9525741268224331) < 1e-5 else "Wrong")
    
def q1_hint():
    print("""def sigmoid(x):
    return 1 / (1 + math.e ** (-x))""")

def q2_hint():
    print("""model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)""")
    
def q3_hint():
    print("""optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # See https://pytorch.org/docs/stable/optim.html#
X = torch.tensor(data[:, 0:2], dtype=torch.float)
Y = torch.tensor(data[:, 2].reshape((-1, 1)), dtype=torch.float)
print("X.shape =", X.shape, "Y.shape =", Y.shape)

epochs = 1000
loss_function = nn.L1Loss()
for epoch in range(epochs):
    optimizer.zero_grad()
    
    prediction = model(X)
    loss = loss_function(prediction, Y)
    loss.backward()
    
    optimizer.step()
    
plot_decision_boundary(model, X, Y)""")
    

def plot_decision_boundary(model, X, Y):
    from matplotlib import pyplot as plt
    # Generate a grid of points with distance h between them
    xx, yy = torch.meshgrid(torch.arange(-0.1, 1.1, 0.1), torch.arange(-0.1, 1.1, 0.1))
    # Predict the function value for the whole grid
    pred = model(torch.stack([xx.flatten(), yy.flatten()], dim=1))
    pred = pred.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx.detach(), yy.detach(), pred.detach(), cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.Spectral)
    plt.show()
def sigmoid(x):
    
    out = None
    
    return out

q1_check()
# q1_hint()
model = nn.Sequential(
    None
)

# q2_hint()
# run the code below to inspect the data
data = generate_data(100, 1, 0, 0.2)
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue = data[:, 2])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # See https://pytorch.org/docs/stable/optim.html#
X = torch.tensor(data[:, 0:2], dtype=torch.float)
Y = torch.tensor(data[:, 2].reshape((-1, 1)), dtype=torch.float)
print("X.shape =", X.shape, "Y.shape =", Y.shape)

epochs = None
loss_function = None
for epoch in range(epochs):
    optimizer.zero_grad()
    
    pass
    
    optimizer.step()
    
plot_decision_boundary(model, X, Y)

# q3_hint()