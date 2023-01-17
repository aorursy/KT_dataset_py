import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
def choice_cost(choice, n_people):

    if choice == 0:

        return 0

    elif choice == 1:

        return 50

    elif choice == 2:

        return 50 + (9*n_people)

    elif choice == 3:

        return 100 + (9*n_people)

    elif choice == 4:

        return 200 + (9*n_people)

    elif choice == 5:

        return 200 + (18*n_people)

    elif choice == 6:

        return 300 + (18*n_people)

    elif choice == 7:

        return 300 + (36*n_people)

    elif choice == 8:

        return 400 + (36*n_people)

    elif choice == 9:

        return 500 + ((36+199)*n_people)

    else:

        return 500 + ((36+398)*n_people)
x0 = [choice_cost(0, i) for i in range(2,9)]

x1 = [choice_cost(1, i) for i in range(2,9)]

x2 = [choice_cost(2, i) for i in range(2,9)]

x3 = [choice_cost(3, i) for i in range(2,9)]

x4 = [choice_cost(4, i) for i in range(2,9)]

x5 = [choice_cost(5, i) for i in range(2,9)]

x6 = [choice_cost(6, i) for i in range(2,9)]

x7 = [choice_cost(7, i) for i in range(2,9)]

x8 = [choice_cost(8, i) for i in range(2,9)]

x9 = [choice_cost(9, i) for i in range(2,9)]

x10 = [choice_cost(10, i) for i in range(2,9)]
y = [i for i in range(2,9)]
plt.plot(y,x0,marker='o')

plt.plot(y,x1,marker='o')

plt.plot(y,x2,marker='o')

plt.plot(y,x3,marker='o')

plt.plot(y,x4,marker='o')

plt.plot(y,x5,marker='o')

plt.plot(y,x6,marker='o')

plt.plot(y,x7,marker='o')

plt.plot(y,x8,marker='o')

plt.plot(y,x9,marker='o')

plt.plot(y,x10,marker='o')

plt.ylabel('Cost')

plt.xlabel('Number of People')

plt.title('Choice Penalty by Number of People')
def accounting_cost(nToday, nDayBefore):

    if nToday < 125 or nToday > 300 or nDayBefore < 125 or nDayBefore > 300:

        return 1e8

    else:

        x = (nToday - 125) / 400

        y = 0.5 + abs(nToday - nDayBefore) / 50 

        return x * nToday ** y

    

def day_one(nDayOne):

    return ((nDayOne-125)/400)*nDayOne**0.5
import random as rnd

def steps_change(n):

    result = []

    result.append(212)

    x = 1

    while len(result) < 100:

        if result[x-1] > (300 - n) or result[x-1] < (125 + n):

            result[x-1] = rnd.randint(min(result),max(result))

            result.append(result[x-1] + rnd.choice([-n,n]))

        else:

            result.append(result[x-1] + rnd.choice([-n,n]))

        x +=1

    return result

        
def results(n):

    x = steps_change(n)

    print('Min of steps is {}'.format(min(x)))

    print('Max of steps is {}'.format(max(x)))

    plt.plot(range(100), x, marker = 'o')

    plt.title('Number of People per day')

    plt.xlabel('Day')

    plt.ylabel('Number of People')

    plt.show()

    y = [day_one(x[0])]

    for i in range(1,100):

        y.append(accounting_cost(x[i],x[i-1]))

    print('Sum of Cost is {}'.format(sum(y)))

    plt.plot(range(100),y)

    plt.title('Accounting Cost per day')

    plt.xlabel('Day')

    plt.ylabel('Cost')

    plt.show()

    z =[y[0]]

    for i in range(1,100):

        z.append(z[i-1]+y[i])

    plt.plot(range(100),z)

    plt.title('Cumulative Cost per day')

    plt.xlabel('Day')

    plt.ylabel('Cost')

    plt.show()
for i in range(1,25):

    print('Max Change in People Count: {}'.format(i))

    results(i)

    print('-------------------------------------------------------------------')