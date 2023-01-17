import math



def d1(x):

    return math.sqrt((x+1)/2) # sqrt: karekök



def d2(x, k):

    return (-math.sqrt(x**2 * k**4 - k**4 + 4 * k**2 - 4 * x**2 * k**2)- x * k**2 + 2*x)/2

    

donus = 15 # maks. 15 yapılmalı çünkü python noktadan sonra 16 hane alıyor.

x = 0      # noktadan sonra daha fazla hane alınabilse pi'ye daha çok yaklaşılır.



for i in range(donus):

    x = d1(x)



y = math.sqrt(1-x**2)



k = math.sqrt((x-1)**2+y**2)



pi = k*(2**(donus+1))

print("bulduğum pi:", pi)

print("gerçek pi:", math.pi)



def cos(derece):

    if derece < 180:

        if derece > 0:

            kn = round((derece * 2**donus)/90)

            x1 = 1

            for j in range(kn):

                x1 = d2(x1, k)

            return x1

    else:

        return None

    

def sin(derece):

    if derece < 180:

        if derece > 0:

            return math.sqrt(1-cos(derece)**2)

    else:

        return None

    

def tan(derece):

    if derece < 180:

        if derece > 0:

            return sin(derece)/cos(derece)

    else:

        return None



def cot(derece):

    if derece < 180:

        if derece > 0:

            return 1/tan(derece)

    else:

        return None

    

def csc(derece):

    if derece < 180:

        if derece > 0:

            return 1/sin(derece)

    else:

        return None

    

def sec(derece):

    if derece < 180:

        if derece > 0:

            return 1/cos(derece)

    else:

        return None 

    

print("sin53:", sin(53))

print("sin143:", sin(143))



print("cos30:", cos(30))

print("cos154:", cos(154))



print("tan125", tan(125))

print("tan40:", tan(40))