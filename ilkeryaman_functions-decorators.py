import time
def calculate_time(func):
    def decorated_function(var):
        start = time.time()
        result = func(var)
        end = time.time()
        print(func.__name__ + " takes " + str(end - start) + " second(s).")
        return result

    return decorated_function
@calculate_time
def calculate_square(numbers):
    result = list()
    for x in numbers:
        result.append(x ** 2)
    return result
@calculate_time
def calculate_cube(numbers):
    result = list()
    for x in numbers:
        result.append(x ** 3)
    return result
list1 = range(1, 100000)

calculate_square(list1)
calculate_cube(list1)