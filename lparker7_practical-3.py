# Q1

def find_largest(my_list, index = 0):

    if index == len(my_list) - 1:

        return my_list[index];

    else:

        comparable = find_largest(my_list, index + 1)

        if comparable > my_list[index]:

            return comparable

        else:

            return my_list[index]

        

find_largest([0,4,1,2,3])


# Q4

def multiply(m, n):

    if n == 1:

        return m

    else:

        return m + multiply(m, n - 1)

    

multiply(8,7)
# Q5

# def compute_average(n):

#     data = DynamicArray()

#     start = time()

#     for k in range(n):

#         data.append(None)

#     end = time()

#     return (end - start) / n



# n = 10

# n_max = 10000000



# while n <= n_max:

#     print(compute_average(n)))

#     n *= 10
# Q6

# class DynamicArray2(DynamicArray):

#     def __init__(self, resize_factor):

#         super().__init__()

#         self._resize_factor = resize_factor

#     def append(self, obj):

#         if self._n == self._capacity:

#             self._resize(int(self._resize_factor * self._capacity) + 1)

#         self._A[self._n] 

#         self._n += 1