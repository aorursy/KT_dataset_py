a = [1,2,3,4]
for i in a:
    print(i**2)
b=[]
for i in a:
    b.append(i**2)
print(b)
def join_str (ar):
    arr=[]
    n = len(ar)
    #print(n)
    
    if(n%2==0):
        for i in range(0,n,2):
            arr.append(ar[i]+ar[i+1])
    else:
        for i in range(0,n-1,2):
            arr.append(ar[i]+ar[i+1])
        arr.append(ar[n-1])
        
        
    return arr
    
join_str(   ['abc','def', 'ghi','jkl', 'mno'] )