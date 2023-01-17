def get_smallest(data):
    smallest = None
    #PUT YOUR CODE HERE
    # What are the steps?
    for n in data:
        if smallest is None or smallest > n:
            smallest = n
            print(n)
    return smallest

#Sample Data
p = [123,597,631,61,93,509]
p2 = [123,597,11,631,61,93,509,9]

#Test function
print(get_smallest(p))
print(get_smallest(p2))

