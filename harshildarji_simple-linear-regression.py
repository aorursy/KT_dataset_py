X = [5, 7, 15, 28]

Y = [30530, 90000, 159899, 270564]
x_mean = sum(X) / 4

y_mean = sum(Y) / 4

print('x_mean: {:.3f}\ny_mean: {:.3f}'.format(x_mean, y_mean))
cov = var = 0

for i in range(4):

    cov += ((Y[i] - y_mean) * (X[i] - x_mean))

    var += (X[i] - x_mean) ** 2

print('Cov: {}\nVar: {}'.format(cov, var))
w1 = cov / var

w0 = y_mean - (w1 * x_mean)

print('Weights:\nw0: {:.3f}\nw1: {:.3f}'.format(w0, w1))
def predict(x):

    return w0 + (w1 * x)
age = 15

print('Mileage for a {}-year old car is {:.3f} km'.format(age, predict(age)))
import matplotlib.pyplot as plt



plt.scatter(X, Y, color = 'g')



prediction = []

for i in range(len(X)):

    prediction.append(predict(X[i]))

    x1 = (X[i], X[i])

    y1 = (Y[i], prediction[i])

    plt.plot(x1, y1, color = 'r')

    

plt.plot(X, prediction, color = 'black')

plt.scatter(X, prediction, color = 'r')

plt.show()