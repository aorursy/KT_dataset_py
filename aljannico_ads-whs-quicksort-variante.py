debug = True # if you want to see the swaps, set to False otherwise
vergleiche = 0 # wir zählen die Schlüsselvergleiche
# wenn sie beim Probieren mehrere QUICKSORTS nacheinander
# aufrufen, müssen sie das jeweils auf 0 zurücksetzen

def QUICKSORT(A,p,r):
    '''
    Inputs:
        A: an array (nummeriert von 0 bis n-1, n ist hierbei die Länge des Arrays)
        P,r: starting (p) and ending (r) indices of a subarray of A
    Result:
      The elements of the subarray A[p..r] are sorted into nondecreasing order
    '''
    if p >= r: return
    else:
        q = PARTITION(A,p,r)
        QUICKSORT(A,p,q-1)
        QUICKSORT(A,q+1,r)
        
def PARTITION(A,p,r):
    '''
    Inputs: siehe oben
    Output: Rearranges the elements of A[p..r] so that every element in
         [p..q-1] is less than or equal to A[q] and every element in A[q+1..r] is
         greater than A[q]. Returns the index q to the caller.
    '''
    global vergleiche # zum Zählen der Vergleiche
    q = p
    for u in range(p,r):
        vergleiche += 1 # Vergleiche zählen
        if A[u] <= A[r]: 
            A[q],A[u] = A[u],A[q] # swap
            if debug and u != q: print(" "*7,A)
            q += 1
    A[q],A[r] = A[r],A[q] # swap
    if debug and q != r: print(" "*7,A)
    return q
        
A = [2,8,20,1,24,21,16,22]

print("Input: ",A)
QUICKSORT(A,0,len(A)-1)
print("\n\nSorted:",A, "-- Vergleiche:",vergleiche)
def QUICKSORT(A,p,r):
    '''
    Inputs:
        A: an array (nummeriert von 0 bis n-1, n ist hierbei die Länge des Arrays)
        P,r: starting (p) and ending (r) indices of a subarray of A
    Result:
      The elements of the subarray A[p..r] are sorted into nondecreasing order
    '''
    if p < r: # nicer and logically equivalent
        q = PARTITION(A,p,r)
        QUICKSORT(A,p,q-1)
        QUICKSORT(A,q+1,r)
        
def PARTITION(A,p,r):
    '''
    Inputs: siehe oben
    Output: Rearranges the elements of A[p..r] so that every element in
         [p..q-1] is less than or equal to A[q] and every element in A[q+1..r] is
         greater than A[q]. Returns the index q to the caller.
    '''
    
    q = p
    for u in range(p,r):
        if A[u] <= A[r]: 
            A[q],A[u] = A[u],A[q] # swap
            q += 1
    A[q],A[r] = A[r],A[q] # swap
    return q
        
A = [2,8,20,1,24,21,16,22]

print("Input: ",A)
QUICKSORT(A,0,len(A)-1)
print("Sorted:",A)
