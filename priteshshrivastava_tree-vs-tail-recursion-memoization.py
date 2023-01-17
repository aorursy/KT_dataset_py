def fib_tree(n):

    if n in [0,1]:

        return n

    else:

        return (fib_tree(n-1) + fib_tree(n-2))



fib_tree(5)
%%time

fib_tree(40)
def fib_iter(n): 

    def helper(a, b, counter):

        #print(f"Calling loop for counter = {counter}")

        if counter == 0:

            return b

        else:

            return helper(a+b, a, counter-1)

    return helper(1, 0, n)
%%time

fib_iter(40)
def fib_memo(n):

    memo = {0:0, 1:1}

    def helper(x):

        if x in memo:

            return memo[x]

        else:

            memo[x] = helper(x-1) + helper(x-2)

            return memo[x]

    return helper(n)
%%time

fib_memo(40)