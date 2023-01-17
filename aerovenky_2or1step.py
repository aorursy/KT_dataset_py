def f(n):

    if n==1:

        return 1

    elif n==2:

        return 2

    else:

        return f(n-1) + f(n-2)
number_to_check = 30
%timeit f(number_to_check)
def fCache(n, cache={1:1,2:2}):

    """

    lets us introduce a cache, in this way we will only compute a value once

    There will be added cost of space

    In current implementation we are using a dictionary as cache

    """

    if n in cache:

        return cache[n]

    else:

        ans = fCache(n-1, cache) + fCache(n-2, cache)

        cache[n] = ans

        return ans
%timeit fCache(number_to_check)
def fLoop(n):

    if n==1:

        return 1

    elif n==2:

        return 2

    val_minus_1 = 2

    val_minus_2 = 1

    i = 2

    while(i<n):

        new_val = val_minus_1 + val_minus_2 

        val_minus_2 = val_minus_1

        val_minus_1 = new_val

        i = i + 1

    return new_val
for i in range(1, 10):

    print(i, fCache(i), fLoop(i), f(i))
%timeit fLoop(number_to_check)