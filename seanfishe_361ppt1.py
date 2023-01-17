%matplotlib inline
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
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
#print('Not implemented.')
def svdsolve(a,b):
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
    w = np.linalg.solve(np.diag(s),c)
    x = np.dot(v.T,w)
    return x
'''
Hints:
1. Use np.matmul() and la.inv() to solve for x in Ax = b.
2. Use the defined variable A in Ax = b. Use y_data as b. Use theta_pred as x.
'''
theta_pred = la.pinv(A)@y_data

print('Empirical theta', theta_pred.reshape(-1))
batch_size = 1
max_iter = 1000
lr = 0.001
theta_init = np.random.random((10,1)) * 0.1
def noisy_val_grad(theta_hat, data_, label_, deg_=2.):
    gradient = np.zeros_like(theta_hat)
    loss = 0
    
    for i in range(data_.shape[0]):
        x_ = data_[i, :].reshape(-1,1)
        y_ = label_[i, 0]
        err = np.sum(x_ * theta_hat) - y_
        
        #print('Not implemented.')

        '''
        Hints:
        1. Find the gradient and loss for each data point x_.
        2. For grad, you need err, deg_, and x_.
        3. For l, you need err and deg_ only.
        '''
        grad = np.dot(x_,err)*deg_
        l = len(data_)/deg_*np.sum(np.square(err))
        
        loss += l / data_.shape[0]
        gradient += grad / data_.shape[0]
        
    return loss, gradient

#noisy_val_grad = noisy_poly_val_grad
deg_ = 2.
num_rep = 10
max_iter = 1000
fig, ax = plt.subplots(figsize=(10,10))
best_vals = dict()
test_exp_interval = 50
grad_artificial_normal_noise_scale = 0.
for method_idx, method in enumerate(['adam', 'sgd', 'adagrad']):
    test_loss_mat = []
    train_loss_mat = []
    
    for replicate in range(num_rep):
        if replicate % 20 == 0:
            print(method, replicate)
            
        if method == 'adam':
            #print('Not implemented.')
            beta_1 = 0.9
            beta_2 = 0.999
            m = 0
            v = 0
            epsilon = 10**(-8)

        if method == 'adagrad':
            #print('Not implemented.')
            epsilon = 10**(-8)
            squared_sum = 0
            
        theta_hat = theta_init.copy()
        test_loss_list = []
        train_loss_list = []
        
        for t in range(max_iter):
            idx = np.random.choice(data_num,batch_size)
            train_loss, gradient = noisy_val_grad(theta_hat, A[idx,:], y_data[idx,:], deg_=deg_)
            artificial_grad_noise = np.random.randn(10,1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.
            gradient = gradient + artificial_grad_noise
            train_loss_list.append(train_loss)
            
            if t % test_exp_interval == 0:
                test_loss, _ = noisy_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=deg_)
                test_loss_list.append(test_loss)                
            
            if method == 'adam':
                #print('Not implemented.')
                m = beta_1*m  + (1-beta_1)*gradient
                v = beta_2*v + (1-beta_2)*np.square(gradient)
                m_hat = m / (1-beta_1**t)
                v_hat = v / (1-beta_2**t)
                theta_hat = theta_hat - lr * m_hat / (v_hat**(1/2)+epsilon)
            
            elif method == 'adagrad':
                #print('Not implemented.')
                squared_sum = squared_sum + np.square(gradient)
                theta_hat = theta_hat - lr * gradient / (squared_sum+epsilon)**.5
            
            elif method == 'sgd':
                theta_hat = theta_hat - lr * gradient
        
        test_loss_mat.append(test_loss_list)
        train_loss_mat.append(train_loss_list)
        
    print(method, 'done')
    x_axis = np.arange(max_iter)[::test_exp_interval]
    
    print('test_loss_np is a 2d array with num_rep rows and each column denotes a specific update stage in training')
    print('The elements of test_loss_np are the test loss values computed in each replicate and training stage.')
    test_loss_np = np.array(test_loss_mat)
    
    #print('Not implemented.')
    '''
    Hints:
    1. Use test_loss_np in np.mean() with axis = 0
    '''
    test_loss_mean = np.mean(test_loss_np, axis=0)

    '''
    Hints:
    1. Use test_loss_np in np.std() with axis = 0 
    2. Divide by np.sqrt() using num_rep as a parameter
    '''
    test_loss_se = np.std(test_loss_np)/np.sqrt(num_rep)

    plt.errorbar(x_axis, test_loss_mean, yerr=2.5*test_loss_se, label=method)
    best_vals[method] = min(test_loss_mean)
best_vals = {k: int(v*1000)/1000. for k,v in best_vals.items()} # A weird way to round numbers
plt.title(f'Test Loss \n(objective degree: {deg_},  best values: {best_vals})')
plt.ylabel('Test Loss')
plt.legend()
plt.xlabel('Updates')