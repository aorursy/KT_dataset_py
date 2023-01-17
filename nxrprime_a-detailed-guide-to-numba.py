import numpy as np

from numba import jit
x = np.arange(102).reshape(17, 6)
x
@jit(nopython=True)

def example1(a): # Function is compiled to machine code when called the first time

    trace = 0

    for i in range(a.shape[0]):   

        trace += np.tanh(a[i, i]) 

    return a + trace              



print(example1(x))
x = {'a': [1, 2, 3], 'b': [20, 30, 40]}



import pandas as pd

@jit

def use_pandas(a): 

    df = pd.DataFrame.from_dict(a) # Numba doesn't know about pd.DataFrame

    df += 1                        # Numba doesn't understand what this is

    return df.cov()                # or this!



print(use_pandas(x))
import time



x = np.arange(100).reshape(10, 10)



@jit(nopython=True)

def go_fast(a): # Function is compiled and runs in machine code

    trace = 0

    for i in range(a.shape[0]):

        trace += np.tanh(a[i, i])

    return a + trace



# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!

start = time.time()

go_fast(x)

end = time.time()

print("Elapsed (with compilation) = %s" % (end - start))



# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE

start = time.time()

go_fast(x)

end = time.time()

print("Elapsed (after compilation) = %s" % (end - start))
from numba import vectorize, float64, int32, int64, float32



@vectorize([float64(float64, float64)])

def example2(x, y):

    return x + y
@vectorize([int32(int32, int32),

            int64(int64, int64),

            float32(float32, float32),

            float64(float64, float64)])

def f(x, y):

    return x + y
start = time.time()

f(9, 9.9)

end = time.time()

print("Elapsed (after compilation) = %s" % (end - start))
import numpy as np

from numba import jitclass          # import the decorator



spec = [

    ('value', int32),               # a simple scalar field

    ('array', float32[:]),          # an array field

]



@jitclass(spec)

class Bag(object):

    def __init__(self, value):

        self.value = value

        self.array = np.zeros(value, dtype=np.float32)



    @property

    def size(self):

        return self.array.size



    def increment(self, val):

        for i in range(self.size):

            self.array[i] = val

        return self.array
from numba import cfunc



@cfunc("float64(float64, float64)")

def add(x, y):

    return x + y

from numba import stencil



@stencil

def kernel1(a):

    return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])