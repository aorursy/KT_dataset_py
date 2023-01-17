import matplotlib.pyplot as plt

import time
def pattern_1(end):

    start1 = time.time()

    start = 1

    #end = 10000

    xs = []

    ys = []

    for val in range(start, end + 1): 

        if val > 1:

            for n in range(2, val): 

                if (val % n) == 0: 

                       break

            else:

                xs.append(val)

                ys.append(val)



    #xs = list(range(1, len(ys)))



    fig = plt.figure(figsize=(15, 15))

    

    for x, y in zip(xs, ys):

        plt.polar(x, y, 'b.')

    end1 = time.time()

    print('it was run for {} seconds'.format(end1-start1))
pattern_1(100)
pattern_1(1000)
pattern_1(10000)
pattern_1(100000)
pattern_1(1000000)
start = 1

end = 15000

ys = []

for val in range(start, end + 1): 

    if val > 1:

        for n in range(2, val): 

            if (val % n) == 0: 

                   break

        else:

            ys.append(val)



xs = list(range(1, len(ys)))



fig = plt.figure(figsize=(10, 10))

for x, y in zip(xs, ys):

    plt.polar(x, y, 'r.')