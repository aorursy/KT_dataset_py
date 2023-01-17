def bad_function(N):

    total = 0

    for n in range(N):

        if n % 3 == 0 or n % 5 == 0:

            total += n

    return total
from time import time

start_time = time()



def evaluate(N):

    totaal = 0

    x = 0

    b_list = [60, 5325, 525750, 52507500, 5250075000, 525000750000, 52500007500000] # More values could be added to crunch even bigger numbers easily.

    b = len(b_list) - 1

    while b > -1: #See the kernell below for an explanaition of this loop.

        a = 10**b    

        while x + 15 * a < N:

            totaal += b_list[b] + 7 * a * x 

            x += 15 * a

        b -= 1

    # Note that for this last part, we use the 'bad' algorithm from before. Everything up to x we have calculated more efficiently.

    for n in range(x+1, N):

        if n % 3 == 0 or n % 5 == 0:

            totaal += n

    return totaal



#Let's test our function for N = 1 000 000 000

print( evaluate(10**9) )



end_time = time()



print("Total running time:", end_time - start_time)
