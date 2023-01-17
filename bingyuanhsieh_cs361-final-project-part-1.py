%matplotlib inline
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import torch

dim_theta = 10
data_num = 1000
scale = .1

theta_true = np.ones((dim_theta,1))
print('True theta:', theta_true.reshape(-1))

A = np.random.uniform(low=-1.0, high=1.0, size=(data_num,dim_theta))
y_data = np.matmul(A,theta_true)+np.random.normal(loc=0.0, scale=scale, size=(data_num,1))

A_test = np.random.uniform(low=-1.0, high=1.0, size=(50,dim_theta))
y_test = np.matmul(A_test,theta_true)+np.random.normal(loc=0.0, scale=scale, size=(50,1))
theta_pred = (la.inv(A.T@A)@A.T).dot(y_data)
print('Empirical theta', theta_pred.reshape(-1))
batch_size = 1
max_iter = 1000
lr = 0.001
theta_init = np.random.random((10,1)) * 0.1
theta_hat = np.ones((10,1))
# Noisy function that can be used in place of noisy_val_grad for the SGD variants (optional)
# def noisy_mse_val_grad(theta_hat, data_, label_, deg_=2.):
    # Atheta_minus_y_idx=np.matmul(data_,theta_hat) - label_
    # gradient = 2*np.matmul(np.transpose(data_), Atheta_minus_y_idx)
    # loss = np.sum(Atheta_minus_y_idx**2)
    # return loss / data_.shape[0],gradient / data_.shape[0]

def noisy_poly_val_grad(theta_hat, data_, label_, deg_=2.):
    gradient = np.zeros_like(theta_hat, dtype=np.float64)
    loss = 0.0
    
    for i in range(data_.shape[0]):
        x_ = data_[i, :].reshape(-1,1)
        y_ = label_[i, 0]
        err = np.sum(x_ * theta_hat) - y_
        
#         print('Not implemented. Student needs to find the gradient and loss')
        if (x_.T @ theta_hat - y_) >= 0:
            grad = deg_ * (x_.T @ theta_hat - y_)**(deg_-1) * x_
        else:
            grad = -1 * deg_ * (y_ - x_.T @ theta_hat)**(deg_-1) * x_   
        l = np.abs(x_.T @ theta_hat - y_) ** deg_

        loss += l / data_.shape[0]
        gradient += grad / data_.shape[0]
    
    return loss, gradient
deg_ = 2
num_rep = 10
max_iter = 1000
fig, ax = plt.subplots(figsize=(10,10))
best_vals = dict()
test_exp_interval = 50
grad_artificial_normal_noise_scale = 0.
deg_
la_ada = 0.1
lr = 0.001
deg_ = 5
for method_idx, method in enumerate(['adam', 'sgd', 'adagrad']):
    test_loss_mat = []
    train_loss_mat = []
    
    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(method, replicate)
            
        if method == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = 0
            nu = 0

        if method == 'adagrad':
            epsilon = 1e-8
            Gt = 0
#       Optional
#         if method == 'rmsprop':
#             print('This is not mandatory. Implement it for your own fun.')
#             print('Not implemented. Student has to initialize and create necessary hyperparameters/moments of RMSPROP')
#             ?????
            
        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []
        
        for t in range(max_iter):
            idx = np.random.choice(data_num,batch_size)
            train_loss, gradient = noisy_poly_val_grad(theta_hat, A[idx,:], y_data[idx,:], deg_=deg_)
            artificial_grad_noise = np.random.randn(10,1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)
            
            if t % test_exp_interval == 0:
                test_loss, _ = noisy_poly_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=deg_)
                test_loss_list.append(test_loss)                
            
            if method == 'adam':
                t_incremented = t + 1
                m = beta1 * m + (1-beta1) * gradient
                nu = beta2 * nu + (1-beta2) * gradient**2
                m_hat = m / (1 - beta1**t_incremented)
                nu_hat = nu / (1 - beta2**t_incremented)
                theta_hat = theta_hat - lr * m_hat / (np.sqrt(nu_hat) + epsilon)
            elif method == 'adagrad':
                if la_ada == -1:
                    la_ada = lr
                Gt += np.multiply(gradient, gradient)
                theta_hat = theta_hat - la_ada * (gradient / np.sqrt(Gt + epsilon))
            elif method == 'sgd':
                theta_hat = theta_hat - lr * gradient
                
#           Optional
#             elif method=='rmsprop':
#                 print('Not implemented. Student has to implement the core rmsprop algorithm (this is optional)')
#                 ????
#                 theta = theta - lr * ????

        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        
    print(method, 'done')
    print("final set of parameters:\n", theta_hat[:,0])
    theta_final = theta_hat
    x_axis = np.arange(max_iter)[::test_exp_interval]
    
    test_loss_np = np.array(test_loss_mat)
    
#     print('Not implemented. Student has to create the test loss mean and the unbiased standard error vectors.')
#     print('test_loss_np is a 2d array with num_rep rows and some number of columns, where each column denotes a specific update stage in training.')
#     print('The elements of test_loss_np are the test loss values computed in each replicate and training stage.')
    test_loss_mean = np.mean(test_loss_np, axis = 0)
    test_loss_se = np.std(test_loss_np[:,:,0,0], axis = 0, ddof=1)
    plt.errorbar(x_axis, test_loss_mean, yerr=2.5*test_loss_se, label=method)
    best_vals[method] = min(test_loss_mean)

best_vals = {k: int(v*1000)/1000. for k,v in best_vals.items()} # A weird way to round numbers
plt.title(f'Test Loss \n(objective degree: {deg_},  best values: {best_vals})')
plt.ylabel('Test Loss')
plt.legend()
plt.xlabel('Updates')
# Run ADAM under different gammas
lr = 0.001

for method_idx, degree in enumerate([0.4, 0.7, 1, 2, 3, 5]):
    test_loss_mat = []
    train_loss_mat = []
    
    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(degree, replicate)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m = 0
        nu = 0

        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []
        
        for t in range(max_iter):
            idx = np.random.choice(data_num,batch_size)
            train_loss, gradient = noisy_poly_val_grad(theta_hat, A[idx,:], y_data[idx,:], deg_=degree)
            artificial_grad_noise = np.random.randn(10,1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)
            
            if t % test_exp_interval == 0:
                test_loss, _ = noisy_poly_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=degree)
                test_loss_list.append(test_loss)                
            
            t_incremented = t + 1
            m = beta1 * m + (1-beta1) * gradient
            nu = beta2 * nu + (1-beta2) * gradient**2
            m_hat = m / (1 - beta1**t_incremented)
            nu_hat = nu / (1 - beta2**t_incremented)
            theta_hat = theta_hat - lr * m_hat / (np.sqrt(nu_hat) + epsilon)

        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        
    print('gamma =', degree, 'done')
    print("final set of parameters:\n", theta_hat[:,0])
    theta_final = theta_hat
    x_axis = np.arange(max_iter)[::test_exp_interval]
    
    test_loss_np = np.array(test_loss_mat)
    
    test_loss_mean = np.mean(test_loss_np, axis = 0)
    test_loss_se = np.std(test_loss_np[:,:,0,0], axis = 0, ddof=1)
    plt.errorbar(x_axis, test_loss_mean, yerr=2.5*test_loss_se, label=degree)
    best_vals[degree] = min(test_loss_mean)

best_vals = {k: int(v*1000)/1000. for k,v in best_vals.items()} # A weird way to round numbers
plt.title(f'Test Loss \n(objective degree: {degree},  best values: {best_vals})')
plt.ylabel('Test Loss')
plt.legend()
plt.xlabel('Updates')
# Run ADAM and SGD under different gammas
la_ada = 0.1
lr = 0.0001
gamma = 3
num_rep = 10
best_vals = {}
max_iter = 1000
for method_idx, method in enumerate(['adam', 'sgd']):
    test_loss_mat = []
    train_loss_mat = []
    
    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(method, replicate)
            
        if method == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = 0
            nu = 0

        if method == 'adagrad':
            epsilon = 1e-8
            Gt = 0
            
        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []
        
        for t in range(max_iter):
            idx = np.random.choice(data_num,batch_size)
            train_loss, gradient = noisy_poly_val_grad(theta_hat, A[idx,:], y_data[idx,:], deg_=gamma)
            artificial_grad_noise = np.random.randn(10,1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)
            
            if t % test_exp_interval == 0:
                test_loss, _ = noisy_poly_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=gamma)
                test_loss_list.append(test_loss)                
            
            if method == 'adam':
                t_incremented = t + 1
                m = beta1 * m + (1-beta1) * gradient
                nu = beta2 * nu + (1-beta2) * gradient**2
                m_hat = m / (1 - beta1**t_incremented)
                nu_hat = nu / (1 - beta2**t_incremented)
                theta_hat = theta_hat - lr * m_hat / (np.sqrt(nu_hat) + epsilon)
            elif method == 'adagrad':
                if la_ada == -1:
                    la_ada = lr
                Gt += np.multiply(gradient, gradient)
                theta_hat = theta_hat - la_ada * (gradient / np.sqrt(Gt + epsilon))
            elif method == 'sgd':
                theta_hat = theta_hat - lr * gradient

        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        
    print(method, 'done')
    print("final set of parameters:\n", theta_hat[:,0])
    theta_final = theta_hat
    x_axis = np.arange(max_iter)[::test_exp_interval]
    
    test_loss_np = np.array(test_loss_mat)
    
    test_loss_mean = np.mean(test_loss_np, axis = 0)
    test_loss_se = np.std(test_loss_np[:,:,0,0], axis = 0, ddof=1)
    plt.errorbar(x_axis, test_loss_mean, yerr=2.5*test_loss_se, label=method)
    best_vals[method] = min(test_loss_mean)

best_vals = {k: int(v*1000)/1000. for k,v in best_vals.items()} # A weird way to round numbers
plt.title(f'Test Loss \n(objective degree: {gamma},  best values: {best_vals})')
plt.ylabel('Test Loss')
plt.legend()
plt.xlabel('Updates')
# Making ADAGRAD perform better than ADAM but worse than SGD
la_ada = 0.1
lr = 0.001
deg_ = 5
for method_idx, method in enumerate(['adam', 'sgd', 'adagrad']):
    test_loss_mat = []
    train_loss_mat = []
    
    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(method, replicate)
            
        if method == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            m = 0
            nu = 0

        if method == 'adagrad':
            epsilon = 1e-8
            Gt = 0
#       Optional
#         if method == 'rmsprop':
#             print('This is not mandatory. Implement it for your own fun.')
#             print('Not implemented. Student has to initialize and create necessary hyperparameters/moments of RMSPROP')
#             ?????
            
        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []
        
        for t in range(max_iter):
            idx = np.random.choice(data_num,batch_size)
            train_loss, gradient = noisy_poly_val_grad(theta_hat, A[idx,:], y_data[idx,:], deg_=deg_)
            artificial_grad_noise = np.random.randn(10,1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)
            
            if t % test_exp_interval == 0:
                test_loss, _ = noisy_poly_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=deg_)
                test_loss_list.append(test_loss)                
            
            if method == 'adam':
                t_incremented = t + 1
                m = beta1 * m + (1-beta1) * gradient
                nu = beta2 * nu + (1-beta2) * gradient**2
                m_hat = m / (1 - beta1**t_incremented)
                nu_hat = nu / (1 - beta2**t_incremented)
                theta_hat = theta_hat - lr * m_hat / (np.sqrt(nu_hat) + epsilon)
            elif method == 'adagrad':
                if la_ada == -1:
                    la_ada = lr
                Gt += np.multiply(gradient, gradient)
                theta_hat = theta_hat - la_ada * (gradient / np.sqrt(Gt + epsilon))
            elif method == 'sgd':
                theta_hat = theta_hat - lr * gradient
                
#           Optional
#             elif method=='rmsprop':
#                 print('Not implemented. Student has to implement the core rmsprop algorithm (this is optional)')
#                 ????
#                 theta = theta - lr * ????

        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        
    print(method, 'done')
    print("final set of parameters:\n", theta_hat[:,0])
    theta_final = theta_hat
    x_axis = np.arange(max_iter)[::test_exp_interval]
    
    test_loss_np = np.array(test_loss_mat)
    
#     print('Not implemented. Student has to create the test loss mean and the unbiased standard error vectors.')
#     print('test_loss_np is a 2d array with num_rep rows and some number of columns, where each column denotes a specific update stage in training.')
#     print('The elements of test_loss_np are the test loss values computed in each replicate and training stage.')
    test_loss_mean = np.mean(test_loss_np, axis = 0)
    test_loss_se = np.std(test_loss_np[:,:,0,0], axis = 0, ddof=1)
    plt.errorbar(x_axis, test_loss_mean, yerr=2.5*test_loss_se, label=method)
    best_vals[method] = min(test_loss_mean)

best_vals = {k: int(v*1000)/1000. for k,v in best_vals.items()} # A weird way to round numbers
plt.title(f'Test Loss \n(objective degree: {deg_},  best values: {best_vals})')
plt.ylabel('Test Loss')
plt.legend()
plt.xlabel('Updates')