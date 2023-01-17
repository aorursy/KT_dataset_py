%time?
%time
%time 2*5
%time 3**8
%timeit?
import random
L = [random.random() for i in range(50000)]
%timeit L.sort()

import random
L = [random.random() for i in range(50000)]
%time L.sort()
y = None
%timeit y
%timeit -n 10 y
%timeit -r 5 y

%timeit -n 2 -r 2 y
%prun?