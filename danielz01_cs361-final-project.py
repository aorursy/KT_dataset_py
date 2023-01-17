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



A = np.random.uniform(low=-1.0, high=1.0, size=(data_num, dim_theta))

y_data = np.matmul(A,theta_true) + np.random.normal(loc=0.0, scale=scale, size=(data_num,1))



A_test = np.random.uniform(low=-1.0, high=1.0, size=(50, dim_theta))

y_test = np.matmul(A_test, theta_true) + np.random.normal(loc=0.0, scale=scale, size=(50,1))
print('Not implemented. Student needs to create theta_pred using A and y_data from above')

theta_pred = np.matmul(np.linalg.pinv(y_data), y_data)

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

def noisy_val_grad(theta_hat, data_, label_, deg_=2.):

    gradient = np.zeros_like(theta_hat)

    loss = 0

    

    for i in range(data_.shape[0]):

        x_ = data_[i, :].reshape(-1, 1)

        y_ = label_[i, 0]

        err = np.sum(x_ * theta_hat) - y_

        

        # print('Not implemented. Student needs to find the gradient and loss')

        # grad = deg_ * np.sign(np.matmul(x_.T, theta_hat) - y_) * np.abs(np.matmul(x_.T, theta_hat) - y_) ** (deg_ - 1) * x_

        # l = np.abs(np.matmul(x_.T, theta_hat) - y_) ** deg_

        grad = deg_ * np.sign(err) * np.abs(err) ** (deg_ - 1) * x_

        l = np.abs(err) ** deg_

        

        loss += l / data_.shape[0]

        gradient += grad / data_.shape[0]

        

    return loss, gradient



# ??? noisy_val_grad = noisy_poly_val_grad
deg_ = 2

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

            lr = 1e-1 # lr

            beta_1 = 0.9 # lr decay rate

            beta_2 = 0.999

            episilon = 1e-8 # division-by-zero control constant

            moments = {}

        if method == 'adagrad':

            lr = 1e-1

            episilon = 1e-8

            grad_square_sum = 0

        if method == 'sgd':

            lr = 1e-1

#         if method == 'rmsprop':

#             print('This is not mandatory. Implement it for your own fun.')

#             print('Not implemented. Student has to initialize and create necessary hyperparameters/moments of RMSPROP')

#             ?????

            

        theta_hat = theta_init.copy()

        test_loss_list = []

        train_loss_list = []

        

        for t in range(max_iter):

            idx = np.random.choice(data_num, batch_size)

            train_loss, gradient = noisy_val_grad(theta_hat, A[idx, :], y_data[idx,:], deg_=deg_)

            artificial_grad_noise = np.random.randn(10, 1) * grad_artificial_normal_noise_scale + np.sign(np.random.random((10,1))-0.5) * 0.

            gradient = gradient + artificial_grad_noise

            train_loss_list.append(train_loss)

            

            

            if t % test_exp_interval == 0:

                test_loss, _ = noisy_val_grad(theta_hat, A_test[:,:], y_test[:,:], deg_=deg_)

                test_loss_list.append(test_loss)

            

            if method == 'adam':

                g_t = gradient

                if 'm_t' not in moments:

                    m_t = np.zeros_like(g_t)

                if 'v_t' not in moments:

                    v_t = np.zeros_like(g_t)

                    

                m_t = beta_1 * m_t + (1 - beta_1) * g_t

                v_t = beta_2 * v_t + (1 - beta_2) * (g_t ** 2)

                

                moments['m_t'] = m_t

                moments['v_t'] = v_t

                

                m_t_hat = m_t / (1 - (beta_1 ** (t + 1)))

                v_t_hat = v_t / (1 - (beta_2 ** (t + 1)))

                theta_hat = theta_hat - lr * m_t_hat / (np.sqrt(v_t_hat) + episilon)

            elif method == 'adagrad':

                g_t = gradient

                grad_square_sum += g_t * g_t

                theta_hat = theta_hat - (lr * g_t) / np.sqrt(grad_square_sum + episilon)

            elif method == 'sgd':

                theta_hat = theta_hat - lr * gradient

                

                

#            elif method=='rmsprop':

#                theta = theta - lr * ????

        

        test_loss_mat.append(test_loss_list)

        train_loss_mat.append(train_loss_list)

        print('{}/{}\n'.format(replicate, method), theta_hat)

        

    print(method, 'done')

    x_axis = np.arange(max_iter)[::test_exp_interval]

    

    test_loss_np = np.array(test_loss_mat)

    

    test_loss_mean = np.sum(test_loss_np, axis=0).reshape(-1) / np.sqrt(num_rep)

    test_loss_se = np.std(test_loss_np, axis=0).reshape(-1) / np.sqrt(num_rep)

    

    # print(test_loss_se)

    



    plt.errorbar(x_axis, test_loss_mean, yerr=2.5 * test_loss_se, label=method)

    best_vals[method] = min(test_loss_mean)

    

best_vals = {k: int(v * 1000) / 1000. for k,v in best_vals.items()} # A weird way to round numbers

plt.title(f'Test Loss \n(objective degree: {deg_},  best values: {best_vals}')

plt.ylabel('Test Loss')

plt.legend()

plt.xlabel('Updates')