%matplotlib inline

from fastai.basics import *
n=100
x = torch.ones(n,2) 

x[:,0].uniform_(-1.,1)

x[:5]
a = tensor(3.,2); a
y = x@a + uniform(-0.5,0.5,n)
plt.scatter(x[:,0], y);
def mse(y_hat, y): return ((y_hat-y)**2).mean()
a = tensor(-1.,1)
y_hat = x@a

mse(y_hat, y)
plt.scatter(x[:,0],y)

plt.scatter(x[:,0],y_hat);
a = nn.Parameter(a); a
def update():

    y_hat = x@a

    loss = mse(y, y_hat)

    if t % 10 == 0: print(loss)

    loss.backward()

    with torch.no_grad():

        a.sub_(lr * a.grad)

        a.grad.zero_()
lr = 1e-1

for t in range(100): update()
plt.scatter(x[:,0],y)

plt.scatter(x[:,0],x@a);

a
from matplotlib import animation, rc

rc('animation', html='jshtml')
a = nn.Parameter(tensor(-1.,1))



fig = plt.figure()

plt.scatter(x[:,0], y, c='orange')

line, = plt.plot(x[:,0], x@a)

plt.close()



def animate(i):

    update()

    line.set_ydata(x@a)

    return line,



animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)
a = tensor(-1.,1)

y = x@tensor(3.,2) + uniform(-0.5,0.5,n)

lr = 1e-1



fig = plt.figure()

scat = plt.scatter(x[:,0], y, c='orange')

line, = plt.plot(x[:,0], x@a)

plt.close()



def updateFlex(whatToUpdate):

    y_hat = x@a

    loss = mse(y, y_hat)

    if t % 10 == 0: print(loss)

    loss.backward()

    with torch.no_grad():

        whatToUpdate.sub_(lr * whatToUpdate.grad)

        whatToUpdate.grad.zero_()



def animate(i):

    global a,y,lr

    if i < 50 :

        if i == 0 :

            a = nn.Parameter(a)

        updateFlex(a)

        line.set_ydata(x@a)

    else :

        if i == 50 :

            a = a.data

            y = nn.Parameter(y)

            lr *= 3

        updateFlex(y)

        scat.set_offsets(torch.cat((x[:,0].unsqueeze(1),y.unsqueeze(1)),1))



animation.FuncAnimation(fig, animate, np.arange(0, 150), interval=50)