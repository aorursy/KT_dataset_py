import numpy as np # linear algebra
W = np.array([1,-1,0,0.5]).transpose()
Xi = [np.array([1,-2,1.5,0]).transpose(),np.array([1,-0.5,-2,-1.5]).transpose(), np.array([0,1,-1,1.5]).transpose()]
c = 1  
Iteration = 0
for i in range(len(Xi)):
    net = sum(W.transpose()*Xi[i])
    Fnet = np.sign(net)
    dw = c * Fnet * Xi[i]
    W = W + dw
    Iteration += 1
print("Final weight matrix : {}".format(W))
print("Iterations : {}".format(Iteration))
