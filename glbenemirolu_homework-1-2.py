%time?
import random

L = [random.random() for i in range(500000)]
print("sorting an unsorted list:")
%time L.sort()
print("sorting an already sorted list:")
%time L.sort()
%timeit?
%timeit sum(range(500))
%prun?
def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total

%prun sum_of_lists(1000000)
%load_ext line_profiler
%lprun?
%lprun -f sum_of_lists sum_of_lists(5000)
%load_ext memory_profiler

%memit?
%memit sum_of_lists(1000000)
%mprun?
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