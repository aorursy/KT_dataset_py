import numpy as np
import matplotlib.pyplot as plt

#creating linspace data
x = np.linspace(-10, 10, num=100, endpoint=True, retstep=False, dtype=None, axis=0)

def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

plt.plot(x, list(map(lambda x: bipolar_sigmoid(x),x)), label="Bipolar_Sigmoid")
plt.xlabel("net")
plt.ylabel("sigm")
plt.title("Bipolar_Sigmoid",fontweight="bold")
plt.grid()
plt.show()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plt.plot(x, list(map(lambda x: sigmoid(x),x)), label="softmax")
plt.xlabel("net")
plt.ylabel("sigm")
plt.title("Unipolar_Sigmoid",fontweight="bold")
plt.show()
plt.plot(x,list(map(lambda x: 1 if x > 0 else -1,x)),label="Bipolar")
plt.xlabel("net")
plt.ylabel("signum")
plt.title("Bipolar Signum Function",fontweight="bold")
plt.show()
plt.plot(x,list(map(lambda x: 1 if x > 0 else 0,x)),label="Unipolar")
plt.xlabel("net")
plt.ylabel("signum")
plt.title("Unipolar Signum Function",fontweight="bold")
plt.show()
def Ramp(x) :
    return max(x, 0)

plt.plot(x, list(map(lambda x: Ramp(x),x)), label="relu")
plt.xlabel("net")
plt.ylabel("ramp1")
plt.title("Ramp Function",fontweight="bold")
plt.show()