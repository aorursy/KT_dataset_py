import numpy as np
import matplotlib.pyplot as plt

N = 50000
D = 2
lr = 10**-20
epoch = 5
l2 = 1000

sample = int((90/100) * N)

X = np.linspace(0, 5*np.pi, N)
Y = X*3 + np.random.randn(N)
# Y = X**3 + np.random.randn(N)
# Y = np.sin(X)

# adding the bias term --> 1s
X = np.vstack([np.ones(N), X]).T

X_train = X[:sample]
Y_train = Y[:sample]

X_test = X[sample:]
Y_test = Y[sample:]

# -----OUTLIERS------
# Y_train[-1] += 30
# Y_train[-2] += 30

# Y_test[-1] += 30
# Y_test[-2] += 30

w_ml = np.linalg.solve(X_train.T.dot(X_train), X_train.T.dot(Y_train))
w_map = np.linalg.solve(l2*np.eye(2) + X_train.T.dot(X_train), X_train.T.dot(Y_train))
w_gd = np.random.randn(D) / np.sqrt(D)

Yhat_ml = X_train.dot(w_ml)
Yhat_map = X_train.dot(w_map)
Yhat_gd = X_train.dot(w_gd)
# costs = []

for _ in range(epoch):
    Yhat_gd = X_train.dot(w_gd)
    # print("Yhat_gd: ", Yhat_gd[_])
    # print("Y_train: ", Y_train[_])
    print("w_ml:", w_ml)
    print("w_map:", w_map)
    print("w_gd:", w_gd)
    delta =  Yhat_gd - Y_train
    w_gd = w_gd - (lr*X_train.T.dot(delta))
    print(delta)
    print(lr*X_train.T.dot(delta))
    # mse = delta.dot(delta)
    # costs.append(mse)
    
# plt.plot(costs, label="MSE")
# plt.legend()
# plt.show()
    
SSres_ml = (Y_train - Yhat_ml).dot(Y_train - Yhat_ml)
SSres_map = (Y_train - Yhat_map).dot(Y_train - Yhat_map)
SSres_gd = (Y_train - Yhat_gd).dot(Y_train - Yhat_gd)

SStot = (Y_train - Y_train.mean()).dot(Y_train - Y_train.mean())

print('\nON TRAIN SAMPLES')
print('R-Squared-ML: ', 1 - SSres_ml/SStot)
print('R-Squared-MAP: ', 1 - SSres_map/SStot)
print('R-Squared-GD: ', 1 - SSres_gd/SStot)


print("\n\nCOMPARING HYPOTHESIS / PREDICTIONS ON TRAIN SAMPLE\n")
print("Y_train[3999] (actual): ", Y_train[3999])
print("Yhat_ml[3999] (max likelihood): ", Yhat_ml[3999])
print("Yhat_map[3999] (L2-regularization): ", Yhat_map[3999])
print("Yhat_gd[3999] (gradient descent): ", Yhat_gd[3999])
    

plt.scatter(X_train[:,1], Y_train)
# plt.plot(X_train[:,1], Y_train, label="Real Solution")
plt.plot(X_train[:,1], Yhat_ml, label="Maximum Likelihood")
plt.plot(X_train[:,1], Yhat_map, label="L2 Regularization (map)")
# plt.plot(X_train[:,1], Yhat_gd, label="Gradient Descent")
plt.legend()
plt.show()


Yhat_ml = X_test.dot(w_ml)
Yhat_map = X_test.dot(w_map)
Yhat_gd = X_test.dot(w_gd)

for _ in range(epoch):
    Yhat_gd = X_test.dot(w_gd)
    delta =  Yhat_gd - Y_test
    w_gd = w_gd - (lr*X_test.T.dot(delta))

SSres_ml = (Y_test - Yhat_ml).dot(Y_test - Yhat_ml)
SSres_map = (Y_test - Yhat_map).dot(Y_test - Yhat_map)
SSres_gd = (Y_test - Yhat_gd).dot(Y_test - Yhat_gd)

SStot = (Y_test - Y_test.mean()).dot(Y_test - Y_test.mean())

print('\nON TEST SAMPLES')
print('R-Squared-ML: ', 1 - SSres_ml/SStot)
print('R-Squared-MAP: ', 1 - SSres_map/SStot)
print('R-Squared-GD: ', 1 - SSres_gd/SStot)

print("\n\nCOMPARING HYPOTHESIS / PREDICTIONS ON TEST SAMPLE\n")
print("Y_test[499] (actual): ", Y_test[499])
print("Yhat_ml[499] (max likelihood): ", Yhat_ml[499])
print("Yhat_map[499] (L2-regularization): ", Yhat_map[499])
print("Yhat_gd[499] (gradient descent): ", Yhat_gd[499])
    

plt.scatter(X_test[:,1], Y_test)
# plt.plot(X_test[:,1], Y_test, label="Real Solution")
plt.plot(X_test[:,1], Yhat_ml, label="Maximum Likelihood")
plt.plot(X_test[:,1], Yhat_map, label="L2 Regularization (map)")
# plt.plot(X_test[:,1], Yhat_gd, label="Gradient Descent")
plt.legend()
plt.show()
    
