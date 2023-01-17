import numpy as np

# n = int(input())

# Simulate input: Set the N number of mountain ranges to 10

n = 50
for kk in range(n):

#    space= input()

#     L = int(input())

# Simulate input: maximum L is set to 50

    L = np.random.randint(1,50,1)[0]

#uncomment to see length L

#     print(L)

#     hight = list(map(int,input().split()))

# Simulate input: maximum hight is set to 10

    hight = np.random.randint(1,10,L).tolist()

    hight = np.array(hight)

#uncomment to see the height (H)

#     print(hight)

    ind_one = np.where(hight==1)

    #print(ind_one[0])

    if len(ind_one[0]) == 0:

        print('Case #'+str(kk+1)+':',-1,-1)

    else:

        maxh = 1

        h = 1

        oind = L-1

        ind = ind_one[0][0]

        # print(hight)

        # print(ind_one[0])

        for ii in range(len(ind_one[0])):

            #chk left             

            if ind_one[0][ii] > 0:

                for jj in range(ind_one[0][ii]):

                    if hight[ind_one[0][ii]-jj-1] == jj+2:

                        h = jj+2

                        ind = ind_one[0][ii]-jj-1

                    else:

                        break

                        #print(h)

                if h > maxh:

                    maxh = h

                    oind = ind

                if h == maxh and ind < oind:

                    oind = ind

            #chk right

            if ind_one[0][ii] < L:

                for jj in range(L-1-ind_one[0][ii]):

                    if hight[ind_one[0][ii]+jj+1] == jj+2:

                        h = jj+2

                        ind = ind_one[0][ii]+jj+1

                    else:

                        break

                if h > maxh:

                    maxh = h  

                    oind = ind  

                if h == maxh and ind < oind:

                    oind = ind

        print('Case #'+str(kk+1)+':',maxh,oind)