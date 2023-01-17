

x = [1,2,4,3,5]

y = [1,3,3,2,5]

w0 = 0

w1 = 0



le = len(x)

alpha = 0.01

jw0w1 = 0

dd = 0

u=0

tt = []

while(True):

    hw = []

    hwt = []

    hwt2 = []

    total = 0

    total2 = 0

 

    for i in x:

        dd= dd+i

        tm = w0+w1*i

        hw.append(tm)    

    print(f'Predicted Output: {hw}')    

    for i in range(len(y)):

        hwt.append(hw[i]-y[i])    

    print(f'Error: {hwt}')

    

    for i in hwt:

        total = total + i

    print(f'Error total: {total}')



    for i in range(len(y)):

        hwt2.append((hw[i]-y[i])**2)    

    print(f'Error Square: {hwt2}')



    for i in hwt2:

        total2 = total2 + i

    print(f'Error Square total: {total2}')   



    jw0w1 = (1/(2*le))*total2

    print(f'Cost: {jw0w1}')  



    w0 = w0 - alpha*(1/le)*total

    w1 = w1 - alpha*(1/le)*total*dd

    print(f'w0: {w0}')

    print(f'w1: {w1}')

    

    tt.append(float(str(jw0w1)[:4]))

    

    if(u>=1):

        if(tt[u-1] == tt[u]):

            break

        

    u = u+1

xx = 6

hwx = w0+w1*xx



print(f'Prediction: {hwx}')