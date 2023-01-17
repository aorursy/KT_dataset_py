import random
A = [9, 21, 1, 75, 88, 12, 4, 53];

B = [9, 21, 1, 75, 88, 12, 4, 53];

def find_in(k, B):

    for i in range(0,len(B)):

        if (B[i]==k):

            return i;
def total_sc(pl):

    sc = 0;

    for x in pl:

        sc+=x;

    return int(sc);
"""pl1 = [];

pl2 = [];

rd = [];

while (len(A)!=0):

    m = max(A);

    t = int(find_in(m, A));

    l = len(A);

    if len(A)== 2:

        sl = max(A[0],A[l-1]);

        pl1.append(sl);

        z = find_in(sl, A);

        A.remove(A[z]);

    elif len(A) > 4:

        if (A[t]>(A[t-1]+A[t+1])):

            

            if ((t+1)<(l-1)) and (t>1):

                sl = max(A[0],A[l-1]);

                pl1.append(int(sl));

                z = find_in(sl, A);

                A.remove(A[z]);

            else:

                if (t==1):

                    pl1.append(int(A[l-1]));

                    A.remove(A[l-1]);

                elif ((t+1)==l):

                    pl1.append(int(A[0]));

                    A.remove(B[0]);

        else:



            sl = max(A[0],B[l-1]);

            pl1.append(int(sl));

            z = find_in(sl, A)

            A.remove(A[z]);

    l = len(A);

    rd.append(int(A[0]));

    rd.append(int(A[l-1]));

    sl = random.choice(rd);

    pl2.append(int(sl));

    x = find_in(sl, A)

    A.remove(A[x]);



print(total_sc(pl1),total_sc(pl2));

"""
pl1 = [];

pl2 = [];

while (len(B)!=0):

    m = max(B);

    t = int(find_in(m, B));

    l = len(B);

    if len(B)== 2:

        sl = max(B[0],B[l-1]);

        pl1.append(sl);

        z = find_in(sl, B);

        B.remove(B[z]);

    elif len(B) > 4:

        if (B[t]>(B[t-1]+B[t+1])):

            

            if ((t+1)<(l-1)) and (t>1):

                sl = max(B[0],B[l-1]);

                pl1.append(int(sl));

                z = find_in(sl, B);

                B.remove(B[z]);

            else:

                if (t==1):

                    pl1.append(int(B[l-1]));

                    B.remove(B[l-1]);

                elif ((t+1)==l):

                    pl1.append(int(B[0]));

                    B.remove(B[0]);

        else:



            sl = max(B[0],B[l-1]);

            pl1.append(int(sl));

            z = find_in(sl, B)

            B.remove(B[z]);

    l = len(B);

    sl = max(B[0],B[l-1]);

    pl2.append(int(sl));

    x = find_in(sl, B)

    B.remove(B[x]);



print(total_sc(pl1),total_sc(pl2));