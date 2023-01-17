%matplotlib inline

from fastai.basics import *
# number of rows

n=100
# create a tensor with n rows and 2 cols

# all the values in this tensor will be 1s

x = torch.ones(n,2) 

x[:5]
# fill the first col with uniform values b/w -1 and 1

x[:,0].uniform_(-1.,1)

x[:5]
# initialize the coefficients

a = tensor(3.,2); a
# tensor multiplication and add a little randomness to it

y = x@a + torch.rand(n)
plt.scatter(x[:,0], y);
def mse(y_hat, y): return ((y_hat - y) ** 2).mean()
# now we want to find a1 and a2 i.e tensor a in such a way

# that the line we draw minimizes the error or the loss function

# which in this case is the mse
# let's start with an initial value of -1,1

# note that all numbers need to be floating point

a = tensor(-1.,1)
# we calculate our predictions and mse

y_hat = x@a

mse(y_hat, y)
plt.scatter(x[:,0], y);

plt.scatter(x[:,0],y_hat);
# the line is horribly wrong
# we have our model (linear regression) and we have

# our evalution metrics i.e mse. Now we need a way

# to optimize that i.e. to find the best line

# this is where gradient descent comes into the picture

a = nn.Parameter(a); a
# GD implemented in pytorch

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
# our error is now down to 0.09

plt.scatter(x[:,0],y)

plt.scatter(x[:,0],x@a);
# the values of a1 and a2 as determined by our model

a
# what would happen if our learning rate was too high

a = tensor(-1.,1)
a = nn.Parameter(a); a
lr = 2

for t in range(100): update()
# the model keeps getting worse instead of getting better

# and the error goes so high it is represented as inf
# learning rate too low

a = tensor(-1.,1)

a = nn.Parameter(a); a
lr = 1e-4

for t in range(100): update()
# the error reduces very gradually

# and would need a lot of epochs before learning well