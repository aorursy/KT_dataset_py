import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.ndimage.filters import generic_filter
from scipy.ndimage.measurements import find_objects, label

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#EEEEEE', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow, 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

plot_size=2
# reusable list of tick for performance optimization
max_ticks = [x-0.5 for x in range(32)]

def plot_image(ax, matrix, title):
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
    ax.set_yticks(max_ticks[:1+matrix.shape[0]])
    ax.set_xticks(max_ticks[:1+matrix.shape[1]])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_hist(ax, matrices, title):
    matrices = [m.ravel() for m in matrices]
    ax.hist(matrices, bins=max_ticks[:11])
    ax.set_xticks(range(10))
    ax.set_title(title)

def plot_hist_relative(ax, matrices, title):
    matrices = [m.ravel() for m in matrices]
    weights=[np.zeros_like(m) + 100. / m.size for m in matrices]
    ax.hist(matrices, bins=max_ticks[:11], weights=weights)
    ax.set_xticks(range(10))
    ax.set_title(title)

def plot_task_item(plot_func, axis, prefix, index, item, same_plot = False, prediction_func = None, plot_out = True):
    t_in, t_out, t_pred = np.array(item["input"]), np.array(item["output"]), None
    if prediction_func is not None:
        t_pred = prediction_func(t_in.copy())
    if same_plot:
        matrices = [t_in, t_out, t_pred] if t_pred is not None else [t_in, t_out]
        plot_func(axis[index], matrices, f'{prefix}')
    else:
        plot_func(axis[0][index], t_in, f'{prefix} in')
        plot_func(axis[1][index], t_out if plot_out else np.zeros(t_out.shape), f'{prefix} out')
        if t_pred is not None:
            plot_func(axis[2][index], t_pred, f'{prefix} pred')
        
def plot_task(task, plot_func, vscale = 1., same_plot = False, prediction_func = None):
    num_train = len(task["train"])
    num_test = len(task["test"])
    num_all = num_train + num_test
    num_rows = 1 if same_plot else 2 if prediction_func is None else 3 
    fig, axis = plt.subplots(num_rows, num_all, figsize=(num_all*plot_size,num_rows*vscale*plot_size), dpi=50)
    
    for index, item in enumerate(task["train"]):
        plot_task_item(plot_func, axis, f'Train-{index}', index, item, same_plot, prediction_func, True)
        
    for index, item in enumerate(task["test"]):
        plot_task_item(plot_func, axis, f'Test-{index}', num_train + index, item, same_plot, prediction_func, True)

    plt.tight_layout()
    plt.show()
indices = [
    [1,2,2,3,3,3],
    [0,1,0,2,1,0]
]
size_vars = ['K','L','M','N']

def get_prop_ratio_info(target, ratio):
    return target if ratio == 1. else f'{int(ratio)}*{target}' if int(ratio) == ratio else f'{ratio:.2f}*{target}'

def get_prop_divmod_info(target, quotient, reminder):
    return f'{target}+{reminder}' if quotient == 1. else f'{int(quotient)}*{target}+{reminder}'

def get_prop_divmodinv_info(target, quotient, reminder):
    return f'{target}-{reminder}' if quotient == 1. else f'({target}-{reminder})/{int(quotient)}'

def get_prop_info(prop, variables):
    n = prop.shape[1]
    nn = n * (n-1) // 2
    prop0 = prop[:, indices[0][:nn]]
    prop1 = prop[:, indices[1][:nn]]

    ratio = prop0 / prop1
    quotient1, reminder1 = np.divmod(prop0, prop1)
    quotient2, reminder2 = np.divmod(prop1, prop0)

    same_ratio = np.all(ratio == ratio[0], axis=0)
    same_divmod1 = np.all((quotient1 > 0) * (quotient1 == quotient1[0]) * (reminder1 == reminder1[0]), axis=0)
    same_divmod2 = np.all((quotient2 > 0) * (quotient2 == quotient2[0]) * (reminder2 == reminder2[0]), axis=0)
    same_prop = np.all(prop == prop[0], axis=0)

    info = np.where(same_prop, prop[0], variables[:n])
    for i in range(nn):
        i0, i1 = indices[0][i], indices[1][i]
        if same_prop[i0]:
            continue
        elif same_ratio[i]:
            info[i0] = get_prop_ratio_info(info[i1], ratio[0][i])
        elif same_divmod1[i]:
            info[i0] = get_prop_divmod_info(info[i1], quotient1[0][i], reminder1[0][i])
        elif same_divmod2[i]:
            info[i0] = get_prop_divmodinv_info(info[i1], quotient2[0][i], reminder2[0][i])

    factor = np.gcd.reduce(np.absolute(prop - prop[0]), axis=0)
    reminder = prop[0] % np.maximum(factor, 1)

    constraints = []
    
    expand = np.all(ratio >= 1, axis=0)
    shrink = np.all(ratio <= 1, axis=0)
    for i in range(nn):
        i0, i1 = indices[0][i], indices[1][i]
        if info[i0] == variables[i0] and info[i1] == variables[i1]:
            if expand[i]:
                constraints.append(f'{info[i1]} < {info[i0]}')
            if shrink[i]:
                constraints.append(f'{info[i1]} > {info[i0]}')

    lower_variables = [v.lower() for v in variables]
    for i0 in range(n):
        if info[i0] == variables[i0] and factor[i0] > 1:
            constraints.append(f'{info[i0]} = {factor[i0]}*{lower_variables[i0]}' + ('' if reminder[i0] == 0 else f'+{reminder[i0]}'))

    return info, constraints
def load_task(task_index, task_name, base_path):
    task_file = str(base_path / task_name)
    
    with open(task_file, 'r') as f:
        task = json.load(f)
        
    print(task_index)
    print(task_name)
    return task
    
def describe_task(task):
    size_prop = np.array([[len(t["input"][0]), len(t["input"]), len(t["output"][0]), len(t["output"])] for t in task["train"]])
    info, constraints = get_prop_info(size_prop, size_vars)
    print(f'Size pattern: {info[0]} x {info[1]} -> {info[2]} x {info[3]}')
    if len(constraints):
        constraints_str =  '; '.join(constraints)
        print(f'Size constraints: {constraints_str}')
w3 = np.ones([3,3])
w3_cross = np.array([[0,1,0],[1,1,1],[0,1,0]])

def solver0(x):
    x = np.kron(x, x != 0)
    return x

def solver1(x):
    x = np.where(x == 0, 4, x)
    for i in range((x.size+1)//2):
        x = generic_filter(x, lambda y: y[2] if y[2] != 4 or all(y) else 0, footprint=w3_cross, mode='constant', cval=0)
    return x

def solver2(x):
    for i in range(1,x.shape[0]):
        if np.all(x[:-i] == x[i:]):
            x = x[:i]
            break
    x = np.tile(x, (30,30))[:9,:3]
    x = np.where(x == 1, 2, x)
    return x

def filter4(x):
    x = x.reshape((5,5))
    c = np.max(x[1:4,1:4])
    return c if np.sum(x==c) == np.sum(x[1:4,1:4]==c) and np.sum(x==c) > 3 else 0

def solver4(x):
    m, n = x.shape
    i, j = np.unravel_index(np.argmax(generic_filter(x, filter4, size=5, mode='constant', cval=0)), x.shape)
    for di in range(-1,2):
        for dj in range(-1,2):
            c = np.max(x[max(0,i+4*di-1):max(0,i+4*di+2),max(0,j+4*dj-1):max(0,j+4*dj+2)])
            for k in range(1,5):
                si, sj = 4*k*di, 4*k*dj
                i_min, i_max = min(m, max(0,i+si-1)), min(m, max(0,i+si+2))
                j_min, j_max = min(n, max(0,j+sj-1)), min(n, max(0,j+sj+2))
                x[i_min:i_max,j_min:j_max] = np.where(x[i_min-si:i_max-si,j_min-sj:j_max-sj], c, 0)
    return x

def solver5(x):
    x = x[:,:3] & x[:,4:]
    x = np.where(x == 1, 2, x)
    return x

def solver6(x):
    y = x.ravel()[:-1].reshape((-1,3))
    y = np.max(y, axis=0)
    x = np.tile(y, x.size)[:x.size].reshape(x.shape)
    return x

def solver7(x):
    m2,n2 = np.nonzero(x == 2)
    m8,n8 = np.nonzero(x == 8)
    if m2.min() > m8.max()+1:
        x = np.split(x,(m8.max()+1,m2.min()), axis=0)
        x = np.concatenate((x[0],x[2],x[1]), axis=0)
    if m8.min() > m2.max():
        x = np.split(x,(m2.max()+1,m8.min()), axis=0)
        x = np.concatenate((x[1],x[0],x[2]), axis=0)
    if n2.min() > n8.max()+1:
        x = np.split(x,(n8.max()+1,n2.min()), axis=1)
        x = np.concatenate((x[0],x[2],x[1]), axis=1)
    if n8.min() > n2.max():
        x = np.split(x,(n2.max()+1,n8.min()), axis=1)
        x = np.concatenate((x[1],x[0],x[2]), axis=1)
    return x

def filter8(x):
    x = x.reshape((19,19))
    if x[9,9]:
        return x[9,9]
    i1 = np.argmax(x[8::-1,9]>0)
    i2 = np.argmax(x[10:,9]>0)
    j1 = np.argmax(x[9,8::-1]>0)
    j2 = np.argmax(x[9,10:]>0)
    if x[8-i1,9] > 0 and x[8-i1,9] == x[10+i2,9]:
        return x[8-i1,9]
    if x[9,8-j1] > 0 and x[9,8-j1] == x[9,10+j2]:
        return x[9,8-j1]
    return 0

def solver8(x):
    y = x[0::3,0::3]
    y = generic_filter(y, filter8, size=19, mode='constant', cval=0)
    x[0::3,0::3] = y
    x[1::3,0::3] = y
    x[0::3,1::3] = y
    x[1::3,1::3] = y
    return x

def solver9(x):
    y = label(x, structure=w3)[0]
    return y

def solver10(x):
    y = np.array([[np.sum(x[4*i:4*i+3,4*j:4*j+3] > 0) for i in range(3)] for j in range(3)])
    j, i = np.unravel_index(np.argmin(y), y.shape)
    y = x[4*i:4*i+3,4*j:4*j+3]
    y = np.kron(y, np.ones([4,4]))[:-1,:-1]
    y[3::4,:] = x[3::4,:]
    y[:,3::4] = x[:,3::4]
    return y

def filter11(x):
    n = np.sum(x[1::2]>0)
    if n>1:
        return np.max(x[n%2::2])
    return x[12]

def solver11(x):
    x = generic_filter(x, filter11, size=5, mode='constant', cval=0)
    return x

def solver12(x):
    a = 0 if x.shape[0] < x.shape[1] else 1 
    y = np.max(x, axis=a)
    z, = np.nonzero(y)
    d = z[-1] - z[0]
    for i in range(z.size):
        y[z[i]::2*d] = y[z[i]]
    y = np.tile(y, (x.shape[a], 1))
    if a: y = y.T
    return y

def solver13(x):
    y = [y for y in find_objects(x) if y is not None and x[y].size != x.size][0]
    return x[y]

solvers = [
    solver0,
    solver1,
    solver2,
    None,
    solver4,
    solver5,
    solver6,
    solver7,
    solver8,
    solver9,
    solver10,
    solver11,
    solver12,
    solver13,
]

# solvers = []

def default_solver(x):
    y = label(x, structure=w3)[0]%10
    return y

def get_solver(task, task_index):
    return solvers[task_index] if task_index < len(solvers) else default_solver

def get_solver_eval(task, task_index):
    return None
for task_index, task_name in enumerate(training_tasks[:]):
    task = load_task(task_index, task_name, training_path)
    describe_task(task)
    solver = get_solver(task, task_index)
    plot_task(task, plot_hist_relative, vscale = .5, same_plot=True)
    plot_task(task, plot_image, prediction_func = solver)
for task_index, task_name in enumerate(evaluation_tasks[:]):
    task = load_task(task_index, task_name, evaluation_path)
    describe_task(task)
    solver = get_solver_eval(task, task_index)
    plot_task(task, plot_hist_relative, vscale = .5, same_plot=True)
    plot_task(task, plot_image, prediction_func = solver)