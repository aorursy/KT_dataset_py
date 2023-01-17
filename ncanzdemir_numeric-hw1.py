import numpy as np
round((np.sqrt(2)**2)-2)

import time

print("wait 3 seconds")
time.sleep(3)
print("perfect")

%timeit sum(range(2))
def summ(K):
    return sum(range(K))

%prun summ(999999)
%load_ext memory_profiler
%memit summ(999999)