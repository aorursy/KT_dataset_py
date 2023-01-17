print("practical-1")

n=52
t=2
p=1
k=4
d=int(input("enter the number of dack : "))

print(d)
n=n*d
k=k*d
for i in range(t):
    p=p*float(k/n)
    k=k-1
    n=n-1
print(p)
    
    
    
