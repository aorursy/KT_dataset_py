import numpy as np
import matplotlib.pyplot as plt

data = np.array([[0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],
                 [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])
data = data[data[:,0].argsort()]
print(data)
x = data[:,0] 
y = data[:,1] 

plt.scatter(x,y)
plt.xlabel("Fernsehdauer")
plt.ylabel("Tiefschlafdauer")
plt.show()
x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = 0
denominator = 0
 
for i in range(8):
     numerator += (x[i] - x_mean)*(y[i] - y_mean)
     denominator += (x[i] - x_mean)**2
 
b_1 = numerator / denominator   
b_0 = y_mean - b_1*x_mean

print("estimated linear function: f(x) = ",b_1, "x +", b_0)
plt.scatter(x,y)
plt.xlabel("Fernsehdauer")
plt.ylabel("Tiefschlafdauer")
plt.plot([0, 4], [b_1*0 + b_0, b_1*4+b_0])
plt.show()
def f(x):
    return b_1 * x + b_0 if x >= 0.3 and x <= 4 else None

values = []
i = 0.5
while (i <= 3):
    values.append([i,f(i)])
    i += 0.5 

values = np.array(values)    
print(values)
plt.plot(values[:,0], values[:, 1], "o")
plt.plot(x, y, "o")
plt.xlabel("Fernsehdauer")
plt.ylabel("Tiefschlafdauer")
plt.show()