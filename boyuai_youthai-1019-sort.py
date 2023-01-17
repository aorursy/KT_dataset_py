import random

import time
def bubble_sort(a):

    for i in range(0, len(a)):

        for j in range(0, len(a) - 1 - i):

            if (a[j] > a[j + 1]):

                a[j], a[j + 1] = a[j + 1], a[j]

    return a
def select_sort(a):

    for i in range(0, len(a)):

        mi = i

        for j in range(i + 1, len(a)):

            if a[j] < a[mi]:

                mi = j

        a[i], a[mi] = a[mi], a[i]

    return a
def merge(x, y):

    z = []

    i, j = 0, 0

    while i<len(x) and j<len(y):

        if x[i] < y[j]:

            z.append(x[i])

            i += 1

        else:

            z.append(y[j])

            j += 1

    if i == len(x):

        z += y[j:]

    else:

        z += x[i:]

    return z



def merge_sort(a):

    if (len(a) <= 1):

        return a

    mid = int(len(a) / 2)

    left = merge_sort(a[:mid])

    right = merge_sort(a[mid:])

    return merge(left, right)
def Partition(a, left, right):

    a[left], a[int((left + right) / 2)] = a[int((left + right) / 2)], a[left]

#     rand = random.randint(left, right)

#     a[left], a[rand] = a[rand], a[left]

    key = a[left]



    while left < right:

        while left < right and a[right] >= key:

            right -= 1

        a[left] = a[right]

        while left < right and a[left] <= key:

            left += 1

        a[right] = a[left]



    a[left] = key

    return left



def q_sort(a, left, right):

    if left < right:

        pivot = Partition(a, left, right)

        q_sort(a, left, pivot - 1)

        q_sort(a, pivot + 1, right)

    return a



def quick_sort(a):

    return q_sort(a, 0, len(a) - 1)
class sort:

    def bubble(a):

        st = time.time()

        ret = bubble_sort(a.copy())

        en = time.time()

        print("bubble_sort:", en - st)

        return ret

    

    def select(a):

        st = time.time()

        ret = select_sort(a.copy())

        en = time.time()

        print("select_sort:", en - st)

        return ret

    

    def merge(a):

        st = time.time()

        ret = merge_sort(a.copy())

        en = time.time()

        print("merge_sort:", en - st)

        return ret

    

    def quick(a):

        st = time.time()

        ret = quick_sort(a.copy())

        en = time.time()

        print("quick_sort:", en - st)

        return ret
a = [random.randint(0,1e5) for i in range(50)]

print(a)

print(sort.bubble(a))

print(sort.select(a))

print(sort.merge(a))

print(sort.quick(a))
a = [random.randint(0,1e5) for i in range(1000)]

print("random list:")

sort.bubble(a)

sort.select(a)

sort.merge(a)

b = sort.quick(a)

print(a[:30])

print("sorted list:")

sort.bubble(b)

sort.select(b)

sort.merge(b)

sort.quick(b)

print(b[:30])
a = [random.randint(0,1e5) for i in range(5000)]

print("random list:")

sort.bubble(a)

sort.select(a)

sort.merge(a)

b = sort.quick(a)

print(a[:30])

print("sorted list:")

sort.bubble(b)

sort.select(b)

sort.merge(b)

sort.quick(b)

print(b[:30])