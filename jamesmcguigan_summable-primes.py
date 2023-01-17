# TODO: add z3-solver to kaggle-docker image

! pip3 install -q z3-solver
import z3

import numpy as np

import itertools

import sys

import math

import time

import timeit

import signal

from typing import List

from numba import njit



notebook_start = time.perf_counter()
def generate_primes(count):

    primes = [2]

    for n in range(3, sys.maxsize, 2):

        if len(primes) >= count: break

        if all( n % i != 0 for i in range(3, int(math.sqrt(n))+1, 2) ):

            primes.append(n)

    return primes





#      10 primes generated in:    0.01 ms

#     100 primes generated in:    0.32 ms

#    1000 primes generated in:    6.99 ms

#  10,000 primes generated in:  153.21 ms

# 100,000 primes generated in: 4831.97 ms

print( generate_primes(42) )

print( f'     10 primes generated in: {timeit.timeit(lambda: generate_primes(     10), number=10)/10*1000:7.2f} ms' )

print( f'    100 primes generated in: {timeit.timeit(lambda: generate_primes(    100), number=10)/10*1000:7.2f} ms' )

print( f'   1000 primes generated in: {timeit.timeit(lambda: generate_primes(  1_000), number=10)/10*1000:7.2f} ms' )

print( f' 10,000 primes generated in: {timeit.timeit(lambda: generate_primes( 10_000), number=10)/10*1000:7.2f} ms' )

# print( f'100,000 primes generated in: {timeit.timeit(lambda: generate_primes(100_000), number=10)/10*1000:7.2f} ms' )
def z3_is_prime(x):

    y = z3.Int("y")

    return z3.Or([

        x == 2,              # 2 is the only even prime

        z3.And(

            x > 1,           # x is positive

            x % 1 == 0,      # x is int

            x % 2 != 0,      # x is odd

            z3.Not(z3.Exists([y], z3.And(

                y   > 1,     # y is positive

                y*y < x,     # y < sqrt(x)

                y % 2 != 0,  # y is odd

                x % y == 0   # y is not a divisor of x

            )))

        )

    ])





def z3_generate_primes(count):

    output = []



    number = z3.Int('prime')

    solver = z3.Solver()

    solver.add([ number > 1, number % 1 == 0 ])  # is positive int

    solver.add([ z3_is_prime(number) ])          # must be a prime, obviously

    solver.push()



    domain = 2

    while len(output) < count:

        solver.pop()

        solver.push()

        solver.add([ number < domain ])  # this helps prevent unsat

        solver.add(z3.And([ number != value for value in output ]))

        while solver.check() == z3.sat:

            value = solver.model()[number].as_long()

            solver.add([ number != value ])

            output.append( value )

            if len(output) >= count: break

        domain *= 2  # increment search space

    return sorted(output)



assert len(z3_generate_primes(24)) == 24

print( 'z3: ', z3_generate_primes(24) )

print( 'std:', generate_primes(24) )





#   2 primes generated in:    12.8 ms

#   4 primes generated in:    17.8 ms

#   8 primes generated in:    38.0 ms

#  16 primes generated in:    86.8 ms

#  32 primes generated in:   246.1 ms

#  64 primes generated in:  1655.1 ms

# 128 primes generated in: 14740.5 ms

for n in [2,4,8,16,32,64,128]:

    print( f'{n:3d} primes generated in: {timeit.timeit(lambda: z3_generate_primes(n), number=1)/1*1000:7.1f} ms' )
def z3_generate_summable_primes(size=50, combinations=2, timeout=0) -> List[int]:

    candidates = [ z3.Int(n) for n in range(size) ]

    summations = []

    for n_combinations in range(1,combinations+1):

        summations += [

            z3.Sum(group)

            for group in itertools.combinations(candidates, n_combinations)

        ]



    solver = z3.Solver()

    if timeout: solver.set("timeout", int(timeout * 1000/2.5))  # timeout is in milliseconds, but inexact and ~2.5x slow



    solver.add([ num > 1      for num in candidates ])

    solver.add([ num % 1 == 0 for num in candidates ])

    solver.add([ candidates[n] < candidates[n+1] for n in range(len(candidates)-1) ])  # sorted



    # solver.add([ z3_is_prime(num) for num in candidates ])

    # primes = z3_generate_primes(128)



    solver.add( z3.Distinct(candidates) )

    solver.add( z3.Distinct(summations) )

    solver.push()

    domain = 2

    while True:

        solver.pop()

        solver.push()



        primes = generate_primes(domain)

        solver.add([ z3.Or([ num == prime for prime in primes ])

                     for num in candidates ])

        solver.add([ num < domain for num in candidates ])



        if solver.check() != z3.sat:

            domain *= 2

        else:

            values = sorted([ solver.model()[num].as_long() for num in candidates ])

            return list(values)
try:

    timeout = 5*60*60

    def raise_timeout(signum, frame): raise TimeoutError    # DOC: https://docs.python.org/3.6/library/signal.html

    signal.signal(signal.SIGALRM, raise_timeout)            # Register a function to raise a TimeoutError on the signal.

    signal.alarm(timeout)                                   # Schedule the signal to be sent after ``time``.

    

    for size in range(10,50+1):

        for combinations in [size]:  # range(2,size+1):

            time_start      = time.perf_counter()

            hashable_primes = z3_generate_summable_primes(size=size, combinations=combinations, timeout=timeout)

            time_taken      = time.perf_counter() - time_start

            print(f'size = {size:2d} | combinations = {combinations} | time = {time_taken:7.1f}s | ', hashable_primes)        

        print()

except   TimeoutError as exception: print('timeout')

finally: signal.alarm(0)
primes_np = np.array( generate_primes(100_000), dtype=np.int64 )



@njit()

def generate_hashable_primes(size=50, combinations=2) -> np.ndarray:

    """

    Return a list of primes that have no summation collisions for N=2, for use in hashing

    NOTE: size > 50 or combinations > 2 produces no results

    """

    domain     = primes_np

    candidates = np.zeros((0,), dtype=np.int64)

    exclusions = set()

    while len(candidates) < size:  # loop until we have successfully filled the buffer

        # refill candidates ignoring exclusions

        for n in range(len(domain)):

            prime = np.int64( domain[n] )

            if not np.any( candidates == prime ):

                if prime not in exclusions:

                    candidates = np.append( candidates, prime )

                    if len(candidates) >= size: break

        else:

            return np.zeros((0,), dtype=np.int64)  # prevent infinite loop if we run out of primes



        # This implements itertools.product(*[ candidates, candidates, candidates ]

        collisions = set(candidates)

        indexes    = np.array([ 0 ] * combinations)

        while np.min(indexes) < len(candidates)-1:



            # Sum N=combinations candidate primes and check for duplicate collisions, then exclude and try again

            values = candidates[indexes]

            summed = np.sum(values)

            if summed in collisions:  # then remove the largest conflicting number from the list of candidates

                exclude    = np.max(candidates[indexes])

                candidates = candidates[ candidates != exclude ]

                exclusions.add(exclude)

                break  # pick a new set of candidates and try again

            collisions.add(summed)



            # This implements itertools.product(*[ candidates, candidates, candidates ]

            indexes[0] += 1

            for i in range(len(indexes)-1):

                while np.count_nonzero( indexes == indexes[i] ) >= 2:

                    indexes[i] += 1                    # ensure indexes are unique

                if indexes[i] >= len(candidates):      # overflow to next buffer

                    indexes[i]    = np.min(indexes)+1  # for triangular iteration

                    indexes[i+1] += 1

    return candidates
combinations = 2

for size in [5, 25, 50, 100, 150]:

    time_start = time.perf_counter()

    hashable_primes = generate_hashable_primes(size=size, combinations=combinations)

    time_taken = time.perf_counter() - time_start

    print(f'combinations = {combinations} | size = {size} | time = {time_taken:.0f}s\n', hashable_primes, '\n')

    if len(hashable_primes) == 0: break

        

combinations = 3

for size in range(2,25):

    time_start = time.perf_counter()

    hashable_primes = generate_hashable_primes(size=size, combinations=combinations)

    time_taken = time.perf_counter() - time_start

    print(f'combinations = {combinations} | size = {size} | time = {time_taken:.0f}s\n', hashable_primes, '\n')

    if len(hashable_primes) == 0: break