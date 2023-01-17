%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
data = pd.read_csv("../input/sarcos_inv.csv", header=None)
data = np.array(data)
sarcos_x_raw = data[:,:-1]
sarcos_y_raw = data[:,-1]
split = int(sarcos_x_raw.shape[0] * 0.75)

np.random.seed(0)
shuffle = np.random.permutation(sarcos_x_raw.shape[0])

sarcos_x = sarcos_x_raw[shuffle]
sarcos_y = sarcos_y_raw[shuffle]

sarcos_x_train = sarcos_x[:split]
sarcos_y_train = sarcos_y[:split]
sarcos_x_test = sarcos_x[split:]
sarcos_y_test = sarcos_y[split:]
plt.figure(figsize=[10,8])
plt.bar([i for i in range(21)], np.sum(sarcos_x_train==0, axis=0))
plt.grid(True)
plt.xlabel('Predictors')
plt.ylabel('Number of 0 Values')
np.mean(sarcos_x_train, axis=0)
plt.figure(figsize=[10,8])
plt.bar([i for i in range(21)], np.mean(sarcos_x_train, axis=0))
plt.grid(True)
plt.xlabel('Predictors')
plt.ylabel('Mean Value')
plt.figure(figsize=[10,8])
plt.bar([i for i in range(21)], np.var(sarcos_x_train, axis=0))
plt.grid(True)
plt.xlabel('Predictors')
plt.ylabel('Variance Value')
plt.title('Predictor Variances in training data')
var_x = np.var(sarcos_x_train, axis=0)
var_x
f = lambda x: 0.5 * x[0]
f2 = lambda x: -0.5 * x[0] + np.sin(x[1])
f3 = lambda x: -0.5 * x[0] - np.exp(x[1])
def toy_prob(domains, n, function = f, seed = 1):
    np.random.seed(seed)
    X = np.zeros((n, len(domains)), dtype = "float64")
    for i in range(n):
        for j in range(len(domains)):
            dom = domains[j]
            r = dom[1] - dom[0]
            X[i,j] = (r * np.random.rand() + dom[0])
    Y = function(X.T)
    
    return X, Y
x_train, y_train  = toy_prob(np.array([(-20, 20)]), 750, f)
x_test, y_test = toy_prob(np.array([(-20, 20)]), 250, f)
plt.figure(figsize=[10,8])
plt.plot(x_train, y_train, 'b+')
plt.xlabel('x_train')
plt.ylabel('y_train = f(x_train)')
x_train2, y_train2  = toy_prob(np.array([(-10, 10), (-10, 10)]), 750, f2)
x_test2, y_test2 = toy_prob(np.array([(-10, 10), (-10, 10)]), 250, f2)
plt.figure(figsize=[10,8])
plt.plot(x_train2, y_train2, 'r+')
plt.xlabel('x_train2')
plt.ylabel('y_train2 = f2(x_train2)')
x_train3, y_train3  = toy_prob(np.array([(-20, 20), (-20, 20)]), 750, f3)
x_test3, y_test3 = toy_prob(np.array([(-20, 20), (-20, 20)]), 250, f3)
plt.figure(figsize=[10,8])
plt.plot(x_train3, y_train3, 'g+')
plt.xlabel('x_train3')
plt.ylabel('y_train3 = f3(x_train)')
def r2_score(y, y_pred):
    mean_y = np.mean(y)
    bottom = np.sum(np.square(y - mean_y))
    top = np.sum(np.square(y - y_pred))
    r2 = 1 - (top / bottom)
    return r2

def mse(y, y_pred):
    err = np.sum((y-y_pred)**2)/ len(y_pred)
    return err
# Closed form solution
def linear_regression_fit(x_train, y_train):
    
    # insert offset var
    x_train = np.insert(x_train, 0, 1, axis=1)
    # calculate coefficients using closed-form solution
    
    k = np.linalg.inv(np.matmul(np.transpose(x_train), x_train))
    
    k2 = np.matmul(np.transpose(x_train), y_train)

    w = np.matmul(k, k2)
    
    return w
    

def predict_linear_regression(w, x_test):
    x_test = np.insert(x_test, 0, 1, axis=1)
    return np.matmul(x_test, w)
s_time = time.time()
w = linear_regression_fit(x_train, y_train)
y_pred = predict_linear_regression(w, x_test)
lr_t1 = time.time() - s_time
print(t)
s_time2 = time.time()
w2 = linear_regression_fit(x_train2, y_train2)
y_pred2 = predict_linear_regression(w2, x_test2)
lr_t2 = time.time() - s_time2
print(t)
s_time3 = time.time()
w3 = linear_regression_fit(x_train3, y_train3)
print(w3)
y_pred3 = predict_linear_regression(w3, x_test3)
lr_t3 = time.time() - s_time3
print(t)
lr_r1 = r2_score(y_pred, y_test)
lr_r2 = r2_score(y_pred2, y_test2)
lr_r3 = r2_score(y_pred3, y_test3)
s_time = time.time()
w = linear_regression_fit(sarcos_x_train, sarcos_y_train)
sarcos_y_pred = predict_linear_regression(w, sarcos_x_test)
t = time.time() - s_time
print(t)
r2_score(sarcos_y_pred, sarcos_y_test)
def knn3(x_train, y_train, x_test, k, chunk_size=400):
    
    n_chunks = x_test.shape[0] // chunk_size + 1
    
    all_distances = np.full([x_train.shape[0], x_test.shape[0]], np.nan)
    
    for n in range(n_chunks):
                
        small_test = x_test[(n*chunk_size):min(((n+1)*chunk_size), x_test.shape[0]), :]
        
        rep_train = np.repeat(x_train[:, :, None], small_test.shape[0], axis=2)
        rep_test = np.repeat(np.transpose(small_test[None, :, :], [0, 2, 1]), x_train.shape[0], axis=0)
        
        distances = np.linalg.norm(rep_test - rep_train, axis=1)
        all_distances[:, (n*chunk_size):min(((n+1)*chunk_size), x_test.shape[0])] = distances
        
    k_smallest = np.apply_along_axis(lambda x: np.argpartition(x, k)[:k], axis=0, arr=all_distances)
    
    y_pred = np.apply_along_axis(lambda x: np.mean(y_train[x], axis=0), axis=0, arr=k_smallest)
    
    return y_pred
s_time = time.time()
knn_y_pred = knn3(x_train, y_train, x_test, k=3, chunk_size=300)
knn_t1 = time.time() - s_time
print('time:', knn_t1)
knn_r1 = r2_score(knn_y_pred, y_test)
print('r2_score:', knn_r1)
s_time = time.time()
knn_y_pred2 = knn3(x_train2, y_train2, x_test2, k=10, chunk_size=300)
knn_t2 = time.time() - s_time
print('time:', knn_t2)
knn_r2 = r2_score(knn_y_pred2, y_test2)
print('r2_score:', knn_r2)
s_time = time.time()
knn_y_pred = knn3(x_train, y_train, x_test, k=10, chunk_size=300)
knn_t3 = time.time() - s_time
print('time:', knn_t3)
knn_r3 = r2_score(knn_y_pred, y_test)
print('r2_score:', knn_r3)
'''s_time = time.time()
sarcos_knn_y_pred = knn3(sarcos_x_train, sarcos_y_train, sarcos_x_test, k=10, chunk_size=400)
t = time.time() - s_time
print('time:', t)

print('r2_score:', r2_score(sarcos_knn_y_pred, sarcos_y_test))'''
'''perform_results = {}
for k1 in range(15, 20):
    s_time = time.time()
    y_pred = knn3(sarcos_x_test, sarcos_y_train, sarcos_x_test, k=k1, chunk_size=400)
    time_taken = time.time() - s_time
    perform_results[k1] = [mse(y_pred, sarcos_y_test), r2_score(y_pred, sarcos_y_test), time_taken]   '''
def find_split(x, y):
    """Given a dataset and its target values, this finds the optimal combination
    of feature and split point that gives the maximum information gain."""
    
    # Need the starting variance so we can measure improvement...
    #start_variance = np.var(y)
    
    # Best thus far, initialised to a dud that will be replaced immediately...
    best = {'variance' : np.inf}
    
    # Loop every possible split of every dimension...
    for i in range(x.shape[1]):
        for split in np.unique(x[:,i]):
            # **************************************************************** 5 marks
            
            # find the indices of each partition
            left_indices = [j for j in range(x.shape[0]) if x[j,i] <= split]
            right_indices = [j for j in range(x.shape[0]) if x[j,i] > split] 
            
            # split the data points
            lt = x[left_indices]
            gt = x[right_indices]
            
            # calculate the variance of both children
            var_left = np.var(y[left_indices]) * len(lt)/len(x)
            var_right = np.var(y[right_indices]) * len(gt)/len(x)
            
            # sub into formula
            new_var = var_left + var_right
            
            # only collect best variance
            if new_var < best['variance']:
                best = {'feature' : i,
                        'split' : split,
                        'variance' : new_var, 
                        'left_indices' : left_indices,
                        'right_indices' : right_indices}
    return best

def build_tree(x, y, max_depth = np.inf):
    # Check if either of the stopping conditions have been reached. If so generate a leaf node...
    if max_depth==1 or (y==y[0]).all():
        # Generate a leaf node...
        #values, counts = np.unique(y, return_counts=True)
        
        return {'leaf' : True, 'target' : np.mean(y)}
    
    else:
        move = find_split(x, y)

        left = build_tree(x[move['left_indices'],:], y[move['left_indices']], max_depth - 1)
        right = build_tree(x[move['right_indices'],:], y[move['right_indices']], max_depth - 1)
        
        return {'leaf' : False,
                'feature' : move['feature'],
                'split' : move['split'],
                'variance' : move['variance'],
                'left' : left,
                'right' : right}

def predict_one(tree, sample):
    """Does the prediction for a single data point"""
    if tree['leaf']:
        return tree['target']
    
    else:
        if sample[tree['feature']] <= tree['split']:
            return predict_one(tree['left'], sample)
        else:
            return predict_one(tree['right'], sample)

def predict(tree, samples):
    """Predicts target for every entry of a data matrix."""
    ret = np.empty(samples.shape[0], dtype=int)
    ret.fill(-1)
    indices = np.arange(samples.shape[0])
    
    def tranverse(node, indices):
        nonlocal samples
        nonlocal ret
        
        if node['leaf']:
            ret[indices] = node['target']
        
        else:
            going_left = samples[indices, node['feature']] <= node['split']
            left_indices = indices[going_left]
            right_indices = indices[np.logical_not(going_left)]
            
            if left_indices.shape[0] > 0:
                tranverse(node['left'], left_indices)
                
            if right_indices.shape[0] > 0:
                tranverse(node['right'], right_indices)
    
    tranverse(tree, indices)
    return ret


def random_forest(x_train, y_train, max_depth, n_trees, n_sample, n_feat):
    trees = n_trees*[None]
    for i in range(n_trees):
        samp_x, samp_y = sample(x_train, y_train, n_sample, n_feat)
        trees[i] = build_tree(samp_x, samp_y, max_depth = max_depth)
    return trees

def sample(x_train, y_train, n_sample, n_feat = np.ceil(np.sqrt(21))):
    idc = np.random.choice(np.arange(x_train.shape[0]), n_sample)
    feats = np.random.choice(np.arange(x_train.shape[1]), n_feat, replace=False)
    return x_train[idc][:,feats], y_train[idc]

def predict_forest(forest, x_test):
    return np.mean([predict(tree, x_test) for tree in forest], axis = 0)
    

s_time = time.time()
forest = random_forest(x_train, y_train, max_depth=5, n_trees=3, n_sample=200, n_feat=1)
rf_y_pred = predict_forest(forest, x_test)
rf_t1 = time.time() - s_time
print('time:', rf_t1)
rf_r1 = r2_score(rf_y_pred, y_test)
print('r2_score:', rf_r1)
# unpredictable with values ranging from -1.13038 to 0.71560
s_time = time.time()
forest2 = random_forest(x_train2, y_train2, max_depth=10, n_trees=4, n_sample=200, n_feat=2)
rf_y_pred2 = predict_forest(forest2, x_test2)
rf_t2 = time.time() - s_time
print('time:', rf_t2)
rf_r2 = r2_score(rf_y_pred2, y_test2)
print('r2_score:', rf_r2)
s_time = time.time()
forest3 = random_forest(x_train3, y_train3, max_depth=5, n_trees=5, n_sample=500, n_feat=2)
rf_y_pred3 = predict_forest(forest3, x_test3)
rf_t3 = time.time() - s_time
print('time:', rf_t3)
rf_r3 = r2_score(rf_y_pred3, y_test3)
print('r2_score:', rf_r3)
def kernel(a, b, param = 0.1):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def gp(x_train, y_train, x_test, noise=1e-7, kernel=kernel, pred_size=5):
    
    # Apply kernel function
    k_train = kernel(x_train, x_train) # nxn
    k_test = kernel(x_test, x_test)  # mxm
    k_mix = kernel(x_test, x_train) # mxn
    v_comp = np.linalg.inv(k_train + noise * np.eye(k_train.shape[0])) # nxn
    
    mu = np.dot(k_mix, np.dot(v_comp, y_train)) #mxn
    cov = k_test - np.matmul(k_mix, np.matmul(v_comp, np.transpose(k_mix))) #mxm
    
    pred_sample = np.random.multivariate_normal(mu, cov, size=pred_size)
    
    return np.mean(pred_sample, axis=0)  
s_time = time.time()
gp_y_pred = gp(x_train, y_train, x_test)
gp_t1 = time.time() - s_time
print('time:', gp_t1)
gp_r1 = r2_score(gp_y_pred, y_test)
print('r2_score:', gp_r1)
s_time = time.time()
gp_y_pred2 = gp(x_train2, y_train2, x_test2)
gp_t2 = time.time() - s_time
print('time:', gp_t2)
gp_r2 = r2_score(gp_y_pred2, y_test2)
print('r2_score:', gp_r2)
s_time = time.time()
gp_y_pred3 = gp(x_train3, y_train3, x_test3)
gp_t3 = time.time() - s_time
print('time:', gp_t3)
gp_r3 = r2_score(gp_y_pred3, y_test3)
print('r2_score:', gp_r3)
'''s_time = time.time()
sarcos_gp_y_pred = gp(sarcos_x_train[:12000], sarcos_y_train[:12000], sarcos_x_test)
gp_t = time.time() - s_time
print('time:', t)

print('r2_score:', r2_score(gp_sarcos_y_pred, sarcos_y_test))'''
r2_score(sarc_y_pred_gp, y_test)
sarcos_x_train.shape

runtimes = pd.DataFrame({
    "Function ID": [1, 2, 3],
    "Linear regression": [lr_t1, lr_t2, lr_t3],
    "Random forest": [rf_t1, rf_t2, rf_t3],
    "k-nearest neighbours": [knn_t1, knn_t2, knn_t3],
    "Gaussian processes": [gp_t1, gp_t2, gp_t3]
})


r2_scores = pd.DataFrame({
    "Function ID": [1, 2, 3],
    "Linear regression": [lr_r1, lr_r2, lr_r3],
    "Random forest": [rf_r1, rf_r2, rf_r3],
    "k-nearest neighbours": [knn_r1, knn_r2, knn_r3],
    "Gaussian processes": [gp_r1, gp_r2, gp_r3]
})
runtimes
r2_scores
