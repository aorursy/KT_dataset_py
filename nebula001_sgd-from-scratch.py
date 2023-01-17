%matplotlib inline

from fastai.basics import *
n=100
x = torch.ones(n,2) 

x[:,0].uniform_(-1.,1)

x[:5]
a = tensor(3.,2); a
y = x@a + torch.rand(n)
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