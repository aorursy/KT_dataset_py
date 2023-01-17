%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import random
import timeit
import math


def plot_monty_hall_noswitch(attempts):
    meanScores = []
    count = []
    
    for i in range(2, attempts+2):
        
        scores = []  
        
        for j in range(1, i):
            
            prize = random.randint(0, 3)
            guess = random.randint(0, 3)    
            if prize == guess:
                scores.append(1)  
            else:
                scores.append(0)
                
        meanScores.append(np.mean(scores))    
        count.append(i)
      
    plt.plot(count, meanScores)
    plt.xlabel("Attempts")
    plt.ylabel("Mean values")
    plt.title("Monty Hall no Switch")
    plt.show()
    print("Mean : " + str(round(np.mean(meanScores),3)))
    
    
    
plot_monty_hall_noswitch(1000)

def plot_monty_hall_switch(attempts):
    meanScores = []
    count = []

    for i in range(2, attempts+2):
        scores = []   
        for j in range(1, i):
            prize = random.randint(0, 3)
            guess = random.randint(0, 3)  
            
            newGuesses = [0,1,2]
            newGuesses.remove(guess)
            if newGuesses[0] == prize:
                del newGuesses[1]
            elif newGuesses[1] == prize:
                del newGuesses[0] 
            else:
                del newGuesses[random.randint(0, len(newGuesses))]
            guess = newGuesses[0]
        
            if prize == guess:
                scores.append(1)
            else:
                scores.append(0)
               
        meanScores.append(np.mean(scores))    
        count.append(i)
    plt.plot(count, meanScores)
    plt.xlabel("Attempts")
    plt.ylabel("Mean values")
    plt.title("Monty Hall Switch")
    plt.show()
    print("Mean: " + str(round(np.mean(meanScores),3)))
    


plot_monty_hall_switch(1000)
def monty_hall_revenge_noswitch(attempts):
    meanScores = []
    count = []
    for i in range(2, 2000):
        scores = []   
        for j in range(1, i):
            prize = random.randint(0, 3)
            guess = random.randint(0, 3)    
            coin_toss = random.randint(0,2) # Tails == 1, Heads == 0

            if prize == guess:
                scores.append(1)
            elif prize != guess:
                if coin_toss == 0:
                    scores.append(0)
                elif coin_toss == 1:
                    scores.append(0)
        meanScores.append(np.mean(scores))    
        count.append(i)
    plt.plot(count, meanScores)
    plt.xlabel("Attempts")
    plt.ylabel("Mean values")
    plt.title("Monty's Revenge no Switch")
    plt.show()
    print("Mean: " + str(round(np.mean(meanScores),3)))
    
monty_hall_revenge_noswitch(1000)
def monty_hall_revenge_switch(attempts):
    meanScores = []
    count = []
    for i in range(2, attempts):
        scores = []   
        for j in range(1, i):
            prize = random.randint(0, 3)
            guess = random.randint(0, 3) 
            coin_toss = random.randint(0,2)
            if coin_toss == 1:  # Tails == 1, Heads == 0
                if guess != prize:
                    scores.append(0)
                    continue
             
            newGuesses = [0,1,2]
            newGuesses.remove(guess)
            if newGuesses[0] == prize:
                del newGuesses[1]
            elif newGuesses[1] == prize:
                del newGuesses[0] 
            else:
                del newGuesses[random.randint(0, len(newGuesses))]
            guess = newGuesses[0]
        
            if prize == guess:
                scores.append(1)
            else:
                scores.append(0)
               
        meanScores.append(np.mean(scores))    
        count.append(i)
    plt.plot(count, meanScores)
    plt.xlabel("Attempts")
    plt.ylabel("Mean values")
    plt.title("Monty's Revenge Switch")
    plt.show()
    print("Mean: " + str(round(np.mean(meanScores),3)))
    

monty_hall_revenge_switch(1000)
    
def mc_integrate(f,a,b,N,exact,title):
    
    areas = np.array(np.zeros(N)) # 

    for i in range(N):
        
        x_rand_arr = np.array(np.random.uniform(a,b,N)) # Making the array of random numbers
        
        integral = f(x_rand_arr) 
        
        answer = (b-a)/float(N) * integral.sum() # MC integration algorithm
        areas[i] = answer
    
    plt.hist(areas, bins=30, ec ='black')
    plt.title(title)
    plt.show()
    
    error = np.abs(np.mean(areas) - exact) # Measunring the error between the exact value and the approximation
    
    print("M.C. Integral evaluation: " + str(np.mean(areas)))
    print("Exact value: " + str(exact))
    print("Error: " + str(error))
    
mc_integrate(lambda x: 1/(1 + x**2), 0, 5, 3000, np.arctan(5),r"MC Integration of $\frac{1}{1+x^2}$")
mc_integrate(lambda x: x**2, 0, 3, 10000, 9, r"MC integration of $x^2$") 
mc_integrate(lambda x: np.sqrt(1-x**2), 0, 1, 15000, np.pi/4, r"MC integration of $\frac{\pi}{4}$")
def trapz(f,a,b,N,exact,title):
    
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x) 
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N # Delta x
    T = (dx/2) * np.sum(y_right + y_left) # This is basicly our trapezoid formula but for all intervals
    # X and Y values for plotting y=f(x)
    X = np.linspace(a,b,N+1)
    Y = f(X)
    plt.plot(X,Y)

    for i in range(N):
        
        xs = [x[i],x[i],x[i+1],x[i+1]]
        ys = [0,f(x[i]),f(x[i+1]),0]
        plt.fill(xs,ys,'b',edgecolor='b',alpha=0.2)

    plt.title(title)
    plt.show()
    error = np.abs(exact - T)
    print("Trapezoid integration: " + str(T))
    print("Exact value: " + str(exact))
    print("Error:" + str(error))
trapz(lambda x: 1/(1 + x**2), 0, 5, 50, np.arctan(5),r"Trapezoid Rule for $\frac{1}{1+x^2}$")
trapz(lambda x: x**2, 0, 3, 500, 9,r"Trapezoid Rule for $x^2$")
trapz(lambda x: np.sqrt(1-x**2), 0, 1, 5000, np.pi/4,r"Trapezoid Rule for $\frac{\pi}{4}$")
def mc_integrate_time_test(f,a,b,N): # MC function 
    
    areas = np.array(np.zeros(N))

    for i in range(N):
        
        x_rand_arr = np.array(np.random.uniform(a,b,N))
        
        integral = f(x_rand_arr)
        
        answer = (b-a)/float(N) * integral.sum()
        areas[i] = answer
    return np.mean(areas)
def wrapper_mc(func, *args, **kwargs): # Decorator
    def wrapped_mc():
        return func(*args, **kwargs)
    return wrapped_mc

wrapped_mc = wrapper_mc(mc_integrate_time_test,lambda x: 1/(1 + x**2), 0, 5, 1000)
print("MC integration execute time: " + str(timeit.timeit(wrapped_mc,number=1)))
def trapz_time_test(f,a,b,N): # Trapezoid rule function
    
    x = np.linspace(a,b,N+1)
    y = f(x)
    y_right = y[1:] 
    y_left = y[:-1] 
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T
def wrapper_trapz(func, *args, **kwargs): # Decorator
    def wrapped_trapz():
        return func(*args, **kwargs)
    return wrapped_trapz

wrapped_trapz = wrapper_trapz(trapz_time_test, lambda x: 1/(1 + x**2), 0, 5, 1000)
print("Trapezoid rule execute time: " + str(timeit.timeit(wrapped_trapz,number=1)))