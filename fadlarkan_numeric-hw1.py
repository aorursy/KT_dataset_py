import numpy 

number=round((numpy.sqrt(2)**2)-2)

print(number)
%time number

%time sum(range(100000))
%timeit sum(range(100000))
def sum_of_lists(N):

    total = 0

    for i in range(5):

        L = [j ^ (j >> i) for j in range(N)]

        total += sum(L)

    return total
%prun sum_of_lists(1000000)
%load_ext line_profiler

%lprun -f sum_of_lists sum_of_lists(5000)
%load_ext memory_profiler

%memit sum_of_lists(1000000)
%%file homework2.py

import numpy 

number=round((numpy.sqrt(2)**2)-2)



def sum_of_lists(N):

    total = 0

    for i in range(5):

        L = [j ^ (j >> i) for j in range(N)]

        total += sum(L)

        del L # remove reference to L

    return total

from homework2 import sum_of_lists

%mprun -f sum_of_lists sum_of_lists(10000)