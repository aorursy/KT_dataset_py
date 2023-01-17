# HOMEWORK 1 - NUMERICAL ANALYSIS
 # %time : Time the execution of a one statement

    

import random



list = [random.random() for i in range(10000)]



%time list.sort()
# %timeit : Time repeated execution of a single statement for more accuracy



def factorial(n):

    if n < 2:

        return 1

    else:

        return n * factorial(n-1)



%timeit factorial(10)
 # &prun : Run code with the profiler

    

 # Keys : 

        

 #  ncalls for the number of calls,



 #  tottime for the total time spent in the given function (and excluding time made in calls to sub-functions),



 #  percall is the quotient of tottime divided by ncalls



 #  cumtime is the total time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.



 #  percall is the quotient of cumtime divided by primitive calls



def sumLists(rangeN):

    total = 0

    for i in range(10):

        list = [j ^ (j >> i) for j in range(rangeN)]

        total += sum(list)

    return total



%prun sumLists(1000000)



# Result 



# 24 function calls in 2.101 seconds



#   Ordered by: internal time



#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)

#       10    1.909    0.191    1.909    0.191 <ipython-input-101-ae9aa5937dc0>:18(<listcomp>)

#        1    0.097    0.097    2.088    2.088 <ipython-input-101-ae9aa5937dc0>:15(sumLists)

#       10    0.082    0.008    0.082    0.008 {built-in method builtins.sum}

#        1    0.013    0.013    2.101    2.101 <string>:1(<module>)

#        1    0.000    0.000    2.101    2.101 {built-in method builtins.exec}

#        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
# %lprun: Run code with the line-by-line profiler



%reload_ext line_profiler



%lprun -f sumLists sumLists(5000)



#Result : 



#Timer unit: 1e-06 s



#Total time: 0.027364 s

#File: <ipython-input-107-9fd7ef521db1>

#Function: sumLists at line 15



#Line #      Hits         Time  Per Hit   % Time  Line Contents

#==============================================================

#    15                                           def sumLists(rangeN):

#    16         1         33.0     33.0      0.1     total = 0

#    17        11         20.0      1.8      0.1     for i in range(10):

#    18        10      26809.0   2680.9     98.0         list = [j ^ (j >> i) for j in range(rangeN)]

#    19        10        500.0     50.0      1.8         total += sum(list)

#    20         1          2.0      2.0      0.0     return total
# %memit : Measure the memory use of a single statement



# Required library to %memit

    #pip install memory_profiler

    #You can load or reload_ext line_profiler with %load_ext line_profiler

    

%reload_ext line_profiler



# Using

%memit -r 5 [x for x in range(1000000)]



# Notes : 

# See how much memory a script uses overall. 

# %memit works a lot like %timeit except that the number of iterations is set with -r instead of -n.



#Result : peak memory: 191.03 MiB, increment: 37.10 MiB
# %mprun: Run code with the line-by-line memory profiler



%%file mprun_example.py



def sumLists(rangeN):

    total = 0

    for i in range(10):

        list = [j ^ (j >> i) for j in range(rangeN)]

        total += sum(list)

    return total



from mprun_example import sumLists

%mprun -f sumLists sumLists(1000000)



# We're getting errors because we can't define files on the platform