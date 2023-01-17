%matplotlib inline

from fastai.basics import *
n = 100
x = torch.ones(n,3) 

x[:,1].uniform_(-3.,3)

x[:,0] = x[:,1]**2

x[:5]
a = tensor(2.,1, -2); a #tensor of a1 a2 and a3
y = x@a + torch.rand(n) 

y[:5]
plt.scatter(x[:,1], y);
def mse(y_hat, y): return ((y_hat-y)**2).mean()
a = tensor(-3.,1,10)

y_hat = x@a

mse(y_hat, y)
plt.scatter(x[:,1],y)

plt.scatter(x[:,1],y_hat);
a = nn.Parameter(a); a
def update():

    y_hat = x@a

    loss = mse(y, y_hat)

    loss.backward()

    if t % 50 == 0: 

        print(loss)

    with torch.no_grad():

        a.sub_(lr * a.grad)

        a.grad.zero_()
lr = 5e-2

for t in range(150): update()
plt.scatter(x[:,1],y)

plt.scatter(x[:,1],x@a);

from matplotlib import animation, rc

rc('animation', html='jshtml')
a = nn.Parameter(tensor(-3.,1,10))



fig = plt.figure()



def animate(i):

    update() 

    plt.clf()

    plt.scatter(x[:,1], y, c='orange')

    return plt.scatter(x[:,1], x@a)



animation.FuncAnimation(fig, animate, np.arange(0, 150), interval=10)