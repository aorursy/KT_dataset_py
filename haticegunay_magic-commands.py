import numpy

print(round(numpy.sqrt(2)**2)-2)
#This magic command shows how long the function took to run.

%time print(3*5/28)
#This magic command runs the function many times and shows the average time.

%timeit sum([1,2,3,4,5])
#%prun lists the function calls where the execution spent the most time according to the total time.

def summ(a,b):

    c = a+b

    print(c)

    return c

%prun summ(5,6)
#%lprun runs code with the line-by-line profiler and shows how much time is spent on each line.

%load_ext line_profiler

%lprun -f summ summ(5,6)
#this magic command measures the memory use of a single statement.

%load_ext memory_profiler

%memit summ(5,6)
%%file mprun_demo.py

def summ(a,b):

    c = a+b

    print(c)

    return c

#this magic command shows how much each line affects the total memory usage.

from mprun_demo import summ

%mprun -f summ summ(5,6)