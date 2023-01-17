import numpy as np

import time

start_time = time.time()

#n = int(input())

n = 1000

# x=[]

# y=[]

# for ii in range(n):

#     a, b = map(int, input().split())

#     x.append(a)

#     y.append(b)

x = np.random.randint(1,100,n).tolist()

y = np.random.randint(1,100,n).tolist()

# c = list(map(int, input().split()))

c = [30, 40, 50, 5]

if len(c) == 4:

        cg = c[0:2]

        ca = c[2:4]

if len(c) == 2:

        cg=c[0:2]

        c = list(map(int, input().split()))

        ca = c[0:2]



#ncase = int(input())

ncase = n

rg = np.random.randint(10,30,n).tolist()

ra = np.random.randint(10,30,n).tolist()

for ii in range(ncase):

    # rg, ra = map(int,input().split())

    rg2 = rg[ii]*rg[ii]

    ra2 = ra[ii]*ra[ii]

    outwifi = 0

    for jj in range(n):

        dx = x[jj]-cg[0]

        dy = y[jj]-cg[1]

        tmp1 = dx*dx+dy*dy <= rg2

        dx = x[jj]-ca[0]

        dy = y[jj]-ca[1]

        tmp2 = dx*dx+dy*dy <= ra2

        if tmp1 == tmp2:       

          outwifi += 1

    # print(outwifi) 

    

end_time = time.time()

print(end_time-start_time)
