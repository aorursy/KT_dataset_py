##递归法



def binary_search_re(alist, item):

    n=len(alist)

    if n < 1:

         return None        

    mid = n // 2

    if alist[mid] > item:

         return binary_chop(alist[0:mid], item)

    elif alist[mid] < item:       

        tmp=binary_chop(alist[mid+1:], item)        

        if tmp:         

            return mid+1+tmp

        else:

            return None

    else:

         return mid

## 循环法



def binary_search(alist,item):    

    low = 0    

    high = len(alist) - 1    

    while low <= high:        

        mid = (low + high) // 2        

        guess = alist[mid]        

        if guess == item:            

            return mid        

        elif guess > item:            

            high = mid - 1       

        else:            

            low = mid + 1  

    return None

list=[]

print(binary_search(list,8))

print(binary_search_re(list,8))