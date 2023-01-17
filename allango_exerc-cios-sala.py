n = 4

v = [None]*n



for i in range(n):

    v[i] = float(input())



for i in range(0,n-1):

    for j in range(i+1,n):

        if (v[i] > v[j]):

            aux = v[i]

            v[i] = v[j]

            v[j] = aux

print(v)

    

n = 10

v = [None]*n



for i in range(n):

    v[i] = float(input())



m = (n // 2)

soma = v[0] + v[n-1]

ini = 0

fim  = n-1

for i in range(0, m):

    if  (v[i] + v[n-i-1] > soma):

        soma = v[i] + v[n-i-1]

        ini = i

        fim = n-i-1



print(v)

print ("inicio: ",ini, "fim: ", fim, "Soma: ",soma)