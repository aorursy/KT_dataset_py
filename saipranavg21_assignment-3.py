import numpy as np
import matplotlib.pyplot as plt
def batch_gradient_descent(X,Y,learning_rate,iterations):
    
    W = np.array([0.0 , 0.0])
    T = np.array([0.0 , 0.0])
    J = np.array([0.0] * iterations)
    
    # Gradient Descent Algorithm
    
    for i in range(iterations):
        H = np.array(W[0] + W[1]*X)
        T[0] = W[0] - learning_rate * ((1/len(X))*np.sum(np.subtract(H,Y)))
        T[1] = W[1] - learning_rate * ((1/len(X))*np.sum(np.multiply(np.subtract(H,Y),X)))
        W = T
        J[i] = (1/(2*len(X)))*np.sum(np.square(H - Y))
    
    # Plotting Learning Curves
    
    print('                  ================== Learning Curve ==================')
    plt.plot(J)
    plt.title("Learning Curve")
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost Function")
    plt.show()
    
    # Plotting Regression line
    
    print('                 ================== Regression Plot ==================')
    Y_pred = np.add(X*W[1] , W[0])
    plt.plot(X,Y,'rs')
    plt.plot(X,Y_pred,'b')
    plt.title("Regression Plot")
    plt.xlabel("feature value")
    plt.ylabel("target value")
    plt.show()
    return W
def stochastic_gradient_descent(X,Y,learning_rate,iterations):
    
    W = np.array([0.0 , 0.0])
    T = np.array([0.0 , 0.0])
    J = np.array([0] * iterations)
    
    # Stochastic Gradient Descent Algorithm
    
    for j in range(iterations):
        for i in range(len(X)):
            H = np.array(W[0] + W[1]*X)
            T[0] = W[0] - learning_rate * (H[i] - Y[i])
            T[1] = W[1] - learning_rate * (H[i] - Y[i]) * X[i]
            W = T
        J[j] = (1/(2*len(X)))*np.sum(np.square(H - Y))
    
    # Plotting Learning Curves
    
    print('                  ================== Learning Curve ==================')
    plt.plot(J)
    plt.title("Learning Curve")
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost Function")
    plt.show()
    
    # Plotting Regression line
    
    print('                 ================== Regression Plot ==================')
    Y_pred = np.add(X*W[1] , W[0])
    plt.plot(X,Y,'rs')
    plt.plot(X,Y_pred,'b')
    plt.title("Regression Plot")
    plt.xlabel("feature value")
    plt.ylabel("target value")
    plt.show()
    return W
def mini_batch_gradient_descent(X,Y,learning_rate,iterations,batch_size):
    
    W = np.array([0.0 , 0.0])
    T = np.array([0.0 , 0.0])
    J = np.array([0] * iterations)
    
    # Mini-Batch Gradient Descent Algorithm
    
    for j in range(int(len(X)/batch_size)):
        X_ = X[j*batch_size:j*batch_size + batch_size]
        Y_ = Y[j*batch_size:j*batch_size + batch_size]
        for i in range(iterations):
            H = np.array(W[0] + W[1]*X_)
            T[0] = W[0] - learning_rate * ((1/len(X_))*np.sum(np.subtract(H,Y_)))
            T[1] = W[1] - learning_rate * ((1/len(X_))*np.sum(np.multiply(np.subtract(H,Y_),X_)))
            W = T
            J[i] = (1/(2*len(X)))*np.sum(np.square(H - Y_))
    
    # Plotting Learning Curves
    
    print('                  ================== Learning Curve ==================')
    plt.plot(J)
    plt.title("Learning Curve")
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost Function")
    plt.show()
    
    # Plotting Regression line
    
    print('                 ================== Regression Plot ==================')
    Y_pred = np.add(X*W[1] , W[0])
    plt.plot(X,Y,'rs')
    plt.plot(X,Y_pred,'b')
    plt.title("Regression Plot")
    plt.xlabel("feature value")
    plt.ylabel("target value")
    plt.show()
    return W
X = np.array([1,2,3,4,5])
Y = np.array([14,27,44,55,63])
batch_gradient_descent(X,Y,0.01,90)
stochastic_gradient_descent(X,Y,0.023,60)
mini_batch_gradient_descent(X,Y,0.05,60,3)