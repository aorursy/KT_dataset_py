def listproduct(lst):
    """
    Takes a list argument and returns the 
    product of all its elements.
    """
    product = 1
    for number in lst:
        product *= number
    return product
n = 20

numerator = listproduct([number for number in range(n+1, 2*n+1)])
denominator = listproduct([number for number in range(1, n+1)])
solution = numerator / denominator
print(int(solution))