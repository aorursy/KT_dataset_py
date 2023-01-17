from sys import getsizeof, getrefcount



i = 1

ch = 'c'

st = "asasdf"

int_list = [x for x in range(10)]

str_list = [str(x) for x in range(10)]



print("Size of an int: {} bytes".format(getsizeof(i)))

print("Size of a char: {} bytes".format(getsizeof(ch)))

print("Size of a string: {} bytes".format(getsizeof(st)))

print("Size of list of ints: {} bytes".format(getsizeof(int_list)))

print("Size of list of strings: {} bytes".format(getsizeof(str_list)))
print(int_list)
del i

del ch

del st

del int_list, str_list
# Make sure that all of the references deleted.

print("Number of references to variable `int_list`: {}".format(getrefcount(int_list)))
import pyarrow.parquet as pq
%%timeit

parq = pq.read_pandas('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet').to_pandas()
import numpy as np



images = np.random.randint(low=0, high=255, size=(200840, 137 * 236), dtype=np.uint8)

print("{0:.2f}Gb".format(images.nbytes / (1024*1024*1024)))
float_images = 255.0 * np.random.rand(1, 137 * 236)

print("Number of bytes per (float64) image: {}".format(float_images.nbytes))

print("Number of bytes per (float32) image: {}".format(float_images.astype(np.float32).nbytes))



print("Memory usage of the full dataset (float32): {0:.2f}Gb".format(float_images.astype(np.float32).nbytes * 200840 / (1024*1024*1024)))