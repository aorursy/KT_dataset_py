def my_function(arg1, arg2):   # Defines a new function

    return arg1 + arg2           # Function body (code to execute)
my_function(5, 10)
def sum_3_items(x, y, z, print_args = False): 

    if print_args:                        

        print(x,y,z)

    return x + y + z
sum_3_items(5,10,20)        # By default the arguments are not printed
sum_3_items(5,10,20,True)   # Changing the default prints the arguments
def sum_many_args(*args):

    print (type (args))

    return (sum(args))



    

sum_many_args(1, 2, 3, 4, 5)
def sum_keywords(**kwargs):

    print (type (kwargs))

    return (sum(kwargs.values()))



    

sum_keywords(mynum=100, yournum=200)
import numpy as np



def rmse(predicted, targets):

    """

    Computes root mean squared error of two numpy ndarrays

    

    Args:

        predicted: an ndarray of predictions

        targets: an ndarray of target values

    

    Returns:

        The root mean squared error as a float

    """

    return (np.sqrt(np.mean((targets-predicted)**2)))
lambda x, y: x + y
my_function2 = lambda x, y: x + y



my_function2(5,10)
# Example of using map() without a lambda function



def square(x):    # Define a function

    return x**2



my_map = map(square, [1, 2, 3, 4, 5])  # Pass the function to map()



for item in my_map:

    print(item)
# Example of using map() with a lambda function



my_map = map(lambda x: x**2, [1, 2, 3, 4, 5]) 



for item in my_map:

    print(item)