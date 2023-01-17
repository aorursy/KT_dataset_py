import numpy
abs((numpy.sqrt(2)**2)-2)
x=0.00000000001
if abs((numpy.sqrt(2)**2)-2) < x:

    r = round(abs((numpy.sqrt(2)**2)-2))

    print("\n", r)
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
%%file mprun_demo.py

def sum_of_lists(N):

    total = 0

    for i in range(5):

        L = [j ^ (j >> i) for j in range(N)]

        total += sum(L)

        del L # remove reference to L

    return total

from mprun_demo import sum_of_lists

%mprun -f sum_of_lists sum_of_lists(1000000)