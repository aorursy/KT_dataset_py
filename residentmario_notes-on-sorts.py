def bubble_sort(arr):

    swap_seen = True

    while swap_seen:

        i = 0

        swap_seen = False

        while i < len(arr) - 1:

            if arr[i] > arr[i + 1]:

                arr[i], arr[i + 1] = arr[i + 1], arr[i]

                swap_seen = True

            i += 1

    return arr
bubble_sort([3,2,1,0])
def selection_sort(arr):

    i = 0

    while i < len(arr):

        idx = i

        idxmin = None  # sentinel value

        while idx < len(arr):

            if idxmin is None or arr[idx] < arr[idxmin]:

                idxmin = idx

            idx += 1

        print(i, idxmin)

        arr[i], arr[idxmin] = arr[idxmin], arr[i]

        i += 1

    return arr
def insertion_sort(arr):

    for j in range(1, len(arr)):

        for k in range(j):

            if arr[k] > arr[j]:

                arr = arr[:k] + [arr[j]] + arr[k:j] + arr[j + 1:]

    return arr
insertion_sort([3,2,1])
def quicksort(arr):

    if arr is None or len(arr) <= 1:

        return arr

    

    pivot = len(arr) // 2

    pv = arr[pivot]

    left, right = [], []

    for i in range(pivot):

        if arr[i] > pv:

            right.append(arr[i])

        else:

            left.append(arr[i])

    for i in range(pivot + 1, len(arr)):

        if arr[i] < pv:

            left.append(arr[i])

        else:

            right.append(arr[i])

    

    if len(left) > 1:

        left = quicksort(left)

    if len(right) > 1:

        right = quicksort(right)

        

    return left + [pv] + right
quicksort([1,2,3])
def mergesort(arr):

    if arr is None or len(arr) <= 1:

        return arr

    

    if len(arr) == 2:

        return arr if arr[0] < arr[1] else arr[::-1]

    

    pivot = len(arr) // 2

    l, r = mergesort(arr[:pivot]), mergesort(arr[pivot:])

    return join(l, r)

    

def join(l, r):

    ll, rl = len(l), len(r)

    li = ri = 0

    out = []

    while li < ll and ri < rl:

        if l[li] < r[ri]:

            out.append(l[li])

            li += 1

        else:

            out.append(r[ri])

            ri += 1

    while li < ll:

        out.append(l[li])

        li += 1

    while ri < rl:

        out.append(r[ri])

        ri += 1

    return out
mergesort([5,3,1,7,2,4,6])
def countingsort(arr):

    m = {}

    for v in arr:

        if v not in m:

            m[v] = 1

        else:

            m[v] += 1

    

    out = []

    for v in m:

        for _ in range(m[v]):

            out.append(v)

    

    return out
def countingsortidx(arr):

    m = [[] for _ in range(max(arr))]

    for i, v in enumerate(arr):

        m[v - 1] += [i]

    return m



def radixsort(arr):

    maxlen = max([len(str(v)) for v in arr])

    for i in range(maxlen - 1, -1, -1):

        darr = []

        for v in arr:

            v_d = int(str(v).zfill(maxlen)[i])

            darr.append(v_d)

        darr_sorted_idxs = countingsortidx(darr)

        print(darr_sorted_idxs)

        new_arr = []

        for idxs in darr_sorted_idxs:

            for idx in idxs:

                new_arr.append(arr[idx])

        arr = new_arr

    return arr