import numpy as np

from matplotlib import pyplot as plt

import torch

import torch.nn

import torch.optim
def draw_circle(x2, y2, r2):

    num_samples = 100

    theta = np.linspace(0, 2*np.pi, num_samples)

    r = np.random.rand(num_samples)

    x, y = r * np.cos(theta), r * np.sin(theta)

    return x*r2+x2, y*r2+y2
X1, Y1 = draw_circle(1, 1, 3)

X2, Y2 = draw_circle(4, 4, 3)

X3, Y3 = draw_circle(-4, 4, 3)

X4, Y4 = draw_circle(3, -4, 3)

X5, Y5 = draw_circle(-5, -4, 3)
X = np.concatenate((X1, X2, X3, X4, X5), axis=0)

Y = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=0)

XY = np.hstack([X, Y])

XY = np.reshape(XY, (500, 2), order='F')

XY = torch.FloatTensor(XY)
r1 = np.zeros(100)

r2 = np.ones(100)

r3 = np.ones(100) * 2

r4 = np.ones(100) * 3

r5 = np.ones(100) * 4

result = np.concatenate((r1, r2, r3, r4, r5), axis=0)

result = torch.LongTensor(result)
linear1 = torch.nn.Linear(2, 4, bias=True)

linear2 = torch.nn.Linear(4, 8, bias=True)

linear3 = torch.nn.Linear(8, 5, bias=True)

sigmoid = torch.nn.Sigmoid()

softmax = torch.nn.Softmax()



model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, softmax)



criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1)
for step in range(1001):

    optimizer.zero_grad()

    hypothesis = model(XY)

    cost = criterion(hypothesis, result)

    cost.backward()

    optimizer.step()



    if step % 100 == 0:

        print(step, ' : ', cost.item())

        with torch.no_grad():

            XX = [i*0.5 for i in range(-20, 20)]

            YY = [i*0.5 for i in range(-20, 20)]

            ZZ = np.zeros((40, 40))

            for yy in range(40):

                for xx in range(40):

                    val = torch.FloatTensor([XX[xx], YY[yy]])

                    ZZ[yy][xx] = (model(val) > 0.5).max(dim=0)[1]



            plt.title("Contour plots")

            plt.contourf(XX, YY, ZZ, alpha=.75, cmap='jet')

            plt.contour(XX, YY, ZZ, colors='black')



            plt.scatter(X1, Y1, color='black')

            plt.scatter(X2, Y2, color='red')

            plt.scatter(X3, Y3, color='green')

            plt.scatter(X4, Y4, color='Yellow')

            plt.scatter(X5, Y5, color='blue')



            plt.pause(0.001)

            plt.clf()