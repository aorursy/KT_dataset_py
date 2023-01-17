# Install vaex & pythran from pip

!pip install vaex

!pip install pythran
# Import packages

import vaex

import vaex.ml



import numpy as np



import pylab as plt

import seaborn as sns



from tqdm.notebook import tqdm



import time
df = vaex.ml.datasets.load_iris_1e8()
# Get a preview of the data

df
def benchmark(func, reps=5):

    times = []

    for i in tqdm(range(reps), leave=False, desc='Benchmark in progress...'):

        start_time = time.time()

        res = func()

        times.append(time.time() - start_time)

    return np.mean(times), res
def some_heavy_function(x1, x2, x3, x4):

    a = x1**2 + np.sin(x2/x1) + (np.tan(x2**2) - x4**(2/3))

    b = (x1/x3)**(0.3) + np.cos(x1) - np.sqrt(x2) - x4**3

    return a**(2/3) / np.tan(b)
# Numpy

df['func_numpy'] = some_heavy_function(df.sepal_length, df.sepal_width, 

                                       df.petal_length, df.petal_width)



# Numba

df['func_numba'] = df.func_numpy.jit_numba()



# Pythran

df['func_pythran'] = df.func_numpy.jit_pythran()



# CUDA

df['func_cuda'] = df.func_numpy.jit_cuda()
# Calculation of the sum of the virtual columns - this forces their evaluation

duration_numpy, res_numpy =  benchmark(df.func_numpy.sum)

duration_numba, res_numba =  benchmark(df.func_numba.sum)

duration_pythran, res_pythran =  benchmark(df.func_pythran.sum)

duration_cuda, res_cuda =  benchmark(df.func_cuda.sum)
print(f'Result from the numpy sum {res_numpy:.5f}')

print(f'Result from the numba sum {res_numba:.5f}')

print(f'Result from the pythran sum {res_pythran:.5f}')

print(f'Result from the cuda sum {res_cuda:.5f}')
# Calculate the speed-up compared to the (base) numpy computation

durations = np.array([duration_numpy, duration_numba, duration_pythran, duration_cuda])

speed_up = duration_numpy / durations



# Compute

compute = ['numpy', 'numba', 'pythran', 'cuda']
# Let's visualise it



plt.figure(figsize=(16, 6))



plt.subplot(121)

plt.bar(compute, speed_up)

plt.tick_params(labelsize=14)



for i, (comp, speed) in enumerate(zip(compute, speed_up)):

    plt.annotate(s=f'x {speed:.1f}', xy=(i-0.1, speed+0.3), fontsize=14)

plt.annotate(s='(higher is better)', xy=(0, speed+2), fontsize=16)



plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)

plt.xlabel('Accelerators', fontsize=14)

plt.ylabel('Speed-up wrt numpy', fontsize=14)

plt.ylim(0, speed_up[-1]+5)



plt.subplot(122)

plt.bar(compute, durations)

plt.tick_params(labelsize=14)



for i, (comp, duration) in enumerate(zip(compute, durations)):

    plt.annotate(s=f'{duration:.1f}s', xy=(i-0.1, duration+0.3), fontsize=14)

plt.annotate(s='(lower is better)', xy=(2, durations[0]+3), fontsize=16)



plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)

plt.xlabel('Accelerators', fontsize=14)

plt.ylabel('Duration [s]', fontsize=14)

plt.ylim(0, durations[0]+5)





plt.tight_layout()

plt.show()
def arc_distance(theta_1, phi_1, theta_2, phi_2):

    temp = (np.sin((theta_2-theta_1)/2*np.pi/180)**2

           + np.cos(theta_1*np.pi/180)*np.cos(theta_2*np.pi/180) * np.sin((phi_2-phi_1)/2*np.pi/180)**2)

    distance = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1-temp))

    return distance * 3958.8
# Numpy

df['arc_distance_numpy'] = arc_distance(df.sepal_length, df.sepal_width, 

                                       df.petal_length, df.petal_width)



# Numba

df['arc_distance_numba'] = df.arc_distance_numpy.jit_numba()



# Pythran

df['arc_distance_pythran'] = df.arc_distance_numpy.jit_pythran()



# CUDA

df['arc_distance_cuda'] = df.arc_distance_numpy.jit_cuda()
# Calculation of the sum of the virtual columns - this forces their evaluation

duration_numpy, res_numpy =  benchmark(df.arc_distance_numpy.sum)

duration_numba, res_numba =  benchmark(df.arc_distance_numba.sum)

duration_pythran, res_pythran =  benchmark(df.arc_distance_pythran.sum)

duration_cuda, res_cuda =  benchmark(df.arc_distance_cuda.sum)
print(f'Result from the numpy sum {res_numpy:.5f}')

print(f'Result from the numba sum {res_numba:.5f}')

print(f'Result from the pythran sum {res_pythran:.5f}')

print(f'Result from the cuda sum {res_cuda:.5f}')
# Calculate the speed-up compared to the (base) numpy computation

durations = np.array([duration_numpy, duration_numba, duration_pythran, duration_cuda])

speed_up = duration_numpy / durations
# Let's visualise it



plt.figure(figsize=(16, 6))



plt.subplot(121)

plt.bar(compute, speed_up)

plt.tick_params(labelsize=14)



for i, (comp, speed) in enumerate(zip(compute, speed_up)):

    plt.annotate(s=f'x {speed:.1f}', xy=(i-0.1, speed+0.3), fontsize=14)

plt.annotate(s='(higher is better)', xy=(0, speed+2), fontsize=16)



plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)

plt.xlabel('Accelerators', fontsize=14)

plt.ylabel('Speed-up wrt numpy', fontsize=14)

plt.ylim(0, speed_up[-1]+5)



plt.subplot(122)

plt.bar(compute, durations)

plt.tick_params(labelsize=14)



for i, (comp, duration) in enumerate(zip(compute, durations)):

    plt.annotate(s=f'{duration:.1f}s', xy=(i-0.1, duration+0.3), fontsize=14)

plt.annotate(s='(lower is better)', xy=(2, durations[0]+3), fontsize=16)



plt.title('Evaluation of complex virtual columns with Vaex', fontsize=16)

plt.xlabel('Accelerators', fontsize=14)

plt.ylabel('Duration [s]', fontsize=14)

plt.ylim(0, durations[0]+5)





plt.tight_layout()

plt.show()