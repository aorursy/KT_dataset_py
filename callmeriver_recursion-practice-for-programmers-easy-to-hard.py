import math

def sum(A):
    if not A:
        return 0
    if len(A)==1:
        return A[0]
    if len(A)>1:
        return A[0]+sum(A[1:])

print(sum([1,2,3]))

def basicMin(A, currMin):
    if not A:
        return currMin
    if currMin>A[0]:
        currMin=A[0]
    return basicMin(A[1:],currMin)

# optimized version
def findMin(A, l, r):
    if l==r:
        return A[r]
    mid = math.floor((l+r)/2)
    leftMin = findMin(A,l , mid)
    rightMin = findMin(A, mid+1, r)
    return min(leftMin, rightMin)

print( findMin([3,1,2],0,2))
# checks if a string is a palindrom
def isPali(text):
    if len(text)== 1:
        return True
    elif len(text)==0 or text[0] != text[len(text)-1]:
        return False
    else:
        temp = text[1:-1]
        return isPali(temp)

print(  isPali('ssws') )
def reverseList(A,rev):
    if not A:
        return
    reverseList(A[1:],rev)
    rev.append(A[0])

rev = []
reverseList([3,2,1],rev)
print(rev)

# prints a subset without null values
def print_set(subset):
    temp = []
    for x in subset:
        if x!= None:
            temp.append(x)
    print(temp)
    
# allocate an empty subset with the right size. and call helper.
def all_subsets(given_array):
    subset = [None] * len(given_array)
    helper(given_array,subset,0)

#  Either add new item or add null, if we reached the end of the list, print the list.
def helper(given_array, subset, i):
    if i==len(given_array):
        print_set(subset)
    else:
        subset[i] = None
        helper(given_array,subset,i+1)
        subset[i] = given_array[i]
        helper(given_array,subset,i+1)

all_subsets([1,2,3])
