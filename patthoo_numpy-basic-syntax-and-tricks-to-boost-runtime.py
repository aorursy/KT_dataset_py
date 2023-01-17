import numpy as np

array = np.arange(10)

array
# create 2 dimensional array with random values = [0,25]

array2 = np.arange(25).reshape(5,5)

array2
array3d = np.arange(36).reshape(3,3,4)

array3d
np.zeros((3,4))
np.ones((3,4))
np.empty((2,3))
np.full((2,2), 4)
np.linspace(0, 10, num=5)
np.linspace(0, 10, num=6)
array = np.array([(1,2,3), (4,5,6)])

array
my_num = [0,3,5,6,8]

np.array([my_num, my_num])
np.random.random((2,2))
np.random.randint(low=5, high=10, size=8)
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.uniform.html

# create an array with random numbers from a uniform distribution

np.random.uniform(low = 0, high = 1, size=5)
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html

# create an array with random numbers from a normal distribution

np.random.normal(loc=0.0, scale=1.0, size=4)
import time

start_time = time.time()



entries = range(1000000) # 1 million entries

results = np.array((0,)) # empty array



for entry in entries:

  processed_entry = entry + 5 # do something

  np.append(results, [processed_entry])

    

elapsed_time = time.time() - start_time

elapsed_time
start_time = time.time()



entries = range(1000000) # 1 million entries

results = np.zeros((len(entries),)) # prefilled array



for idx, entry in enumerate(entries):

  processed_entry = entry + 5 # do something

  results[idx] = processed_entry

    

elapsed_time = time.time() - start_time

elapsed_time
# results = np.ones((1000,1000,1000,5))

results = np.ones((500,500,500,5)) # this one eats out 4GB of RAM



# do something...

results[100, 25, 1, 4] = 42
import h5py



hdf5_store = h5py.File("./cache.hdf5", "a")

results = hdf5_store.create_dataset("results", (500,500,500,5), compression="gzip")



# do something...

results[100, 25, 1, 4] = 42
hdf5_store = h5py.File("./cache.hdf5", "r")



print(hdf5_store["results"][100, 25, 1, 4]) # 42.0
start = time.time()



some_array = np.ones((100, 200, 300))



for _ in range(10000000):

    some_array[50, 12, 199] # get some value some_array



runtime = time.time() - start

runtime
start = time.time()



some_array = np.ones((100, 200, 300))



the_value_I_need = some_array[50, 12, 199] # access some_array



for _ in range(10000000):

    the_value_I_need

    

runtime = time.time() - start

runtime