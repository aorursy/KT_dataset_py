import time 

import matplotlib.pyplot as plt 

import threading



def fib(n):

    if n == 0 or n == 1:

        return n 

    else:

        return fib(n-1) + fib(n-2)



def fibDP(n):

    F = []

    F.append(0)

    F.append(1)

    for i in range(2,n+1):

        F.append(F[i-1] + F[i-2])

    return F[n]





def getData(function,limit,printData=True):

    term_number = []

    t_taken = []

    for i in range(0,limit+1):

        t = time.time()

        term_number.append(i)

        x = function(i)

        t_taken.append(time.time() - t)

        if printData == True:

            print("Term Number:",i,", Fib Number:",x,"Time Taken:",t_taken[i])

    return term_number, t_taken
term_dp, time_dp = getData(fibDP,40,)

term_recur, time_recur = getData(fib,40)
plt.plot(term_recur,time_recur,c="red",label="Normal Recursion")       

plt.plot(term_dp,time_dp,c="green",label="Dynamic Programming")

plt.xlabel("Input Size")

plt.ylabel("Time Taken")

plt.legend()

plt.show()
term_dp, time_dp = getData(fibDP,1000, printData=False)

plt.plot(term_dp,time_dp,c="green",label="Dynamic Programming")

plt.xlabel("Input Size")

plt.ylabel("Time Taken")

plt.legend()

plt.show()