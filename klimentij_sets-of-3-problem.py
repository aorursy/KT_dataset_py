import itertools

import numpy as np

import math
def array_info(a):

    print('Array len: ', len(a))

    print("Number of combinations of 3: ", len(list(itertools.combinations(a, 3))))

    print("The same using factorials (sanity check): ", int(math.factorial(len(a))/(math.factorial(3)*math.factorial(len(a)-3))))
def baseline(a):

    desc = "straight forward approach with 3 loops"

    found = False

    

    for i1 in range(len(a)):

        for i2 in range(len(a)):

            if i1 != i2:

                for i3 in range(len(a)):

                    if i1 != i3 and i2 != i3 and a[i1]+a[i2]+a[i3] == 0:

                        #print(i1,i2,i3)

                        found = True

    return found, desc

            

def try_index(a):

    desc = "index() with try-except instead of the most inner loop"

    found = False

    for i1 in range(len(a)):

        for i2 in range(len(a)):

            if i1 != i2:

                a3 = -a[i1]-a[i2]

                try:

                    i3 = a.index(a3)

                    if i3 != i2 and i3 != i1:

                        #print(i1,i2,i3)

                        found = True

                        break

                except:

                    continue

    return found, desc

def combos_to_numpy(a):

    desc = "combos to matrix, then sum along axis=1"

    C = np.array(list(itertools.combinations(a, 3)))

    sums = np.sum(C, axis=1)

    found = 0 in sums

    return found, desc



def loop_over_combos(a):

    desc = "Create combos of 3 using itertools and iterate over combos"

    found = False

    combs = list(itertools.combinations(a, 3)) 

    for c in combs:

        if sum(c) == 0:

            #print (c, sum(c))

            found = True

            break

    return found, desc
a = [0,3,2,-2,0,7,5,4,-1,8] #array with 0-sum triples

b = [0,1,2,3,4, 7,5,4,-121,8,1,7] #array without 0-sum triples



array_info(a)

print()

array_info(b)
%%timeit

baseline(a)
%%timeit

try_index(a) # 3x faster than baseline
%%timeit

combos_to_numpy(a) # 2.2x faster than baseline
%%timeit

loop_over_combos(a) # 25x faster than baseline
#just a sanity check



print('Should be Trues')

print (baseline(a))

print (try_index(a))

print (combos_to_numpy(a))

print (loop_over_combos(a))



print()

print('Should be Falses')

print (baseline(b))

print (try_index(b))

print (combos_to_numpy(b))

print (loop_over_combos(b))

size = int(100)

a_big = np.random.randint(-size, high=size, size=size) #array with 0-sum triples

b_big = np.random.randint(1, high=size, size=size) #array without 0-sum triples



%time print(baseline(a_big))

%time print(try_index(a_big))

%time print(combos_to_numpy(a_big))

%time print(loop_over_combos(a_big))



%time print(baseline(b_big))

%time print(try_index(b_big))

%time print(combos_to_numpy(b_big))

%time print(loop_over_combos(b_big))