from pylab import *
from sklearn.datasets import load_boston
boston_dataset = load_boston()
import pandas as pd
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston['MEDV'] = boston_dataset.target

boston.head()
scatter(boston['RM'], boston['MEDV'], alpha=0.3)

show()
X = array(boston['RM'])

Y = array(boston['MEDV'])
import torch
print(X.shape, Y.shape)
XT = torch.Tensor(X.reshape(506, 1))

YT = torch.Tensor(Y.reshape(506, 1))
print(XT.shape, YT.shape)
class LinearRegretion(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        

        self.F = torch.nn.Linear(input_dim, 1)

        self.loss = None

        self.accuracy = None

        

    def forward(self, x):

        return self.F(x)

    

    def fit(self, x, y, epochs=1, lr=0.01):

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        

        self.train()

        

        for i in range(0, epochs):

            y_ = self.forward(x)

            loss = loss_fn(y_, y)

            

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        self.loss = loss.detach().numpy()

        self.accuracy = self.loss / x.shape[0]
h = LinearRegretion(1)
hy = array(h(XT).detach())

scatter(X, Y, alpha=0.3)

plot(X, hy, c="brown")

show()
print(h.F.weight)
print(h.F.bias)
h.fit(XT, YT, epochs=50000, lr=0.003)



hy = array(h(XT).detach())

scatter(X, Y, alpha=0.3)

plot(X, hy, c="brown")

show()
for param in h.parameters():

    print(param)
print(h.F.weight)
print(h.F.bias)
print("Loss: ", h.loss)

print("Accuracy: ", h.accuracy)
print((h(torch.Tensor([6]))*1000).detach().numpy())
print(h)
print(h.state_dict())
torch.save(h.state_dict(), "linear_state_dict.pt")