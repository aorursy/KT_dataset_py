# Menentukan bilangan faktorial

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
n=int(input("Bilangan : "))
print("Jumlah bilangan faktorial yaitu",factorial(n))
# Menghitung deret fibonacci

n=int(input("banyak deret bilangan : ")) #masukan banyak bilangan yang akan dicari

# Bilangan fibonacci ke-1 dan ke-2
n1 = 0
n2 = 1
count=0

#mengeksekusi
if n <= 0 :
    print("masukan bilangan bulat positif")
elif n == 1 :
    print(n1)
else:
    print("deret fibonacci:")
    while count<n:
        n3 = n1 + n2
        print(n1)
        
        n1=n2
        n2=n3
        count +=1
        
