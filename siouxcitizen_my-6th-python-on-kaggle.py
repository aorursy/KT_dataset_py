#関数その１

def add(x, y):

    ans = x + y

    return ans



#関数その２

def addNTimes(x, y, n = 5):

    for num in range(n):

        x = x + y



    return x



#関数その１呼び出し

n = add(3, 5)

print(n)



#関数その２呼び出し①

n = addNTimes(3, 5)

print(n)   



#関数その２呼び出し②

n = addNTimes(3, 5, 3)

print(n)   

   