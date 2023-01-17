points = [2,4,10,12,3,20,30,11,25]
m1,m2,m3 = 2,4,6

D1,D2,D3 = [],[],[]
for i in range(9):

        D1.append(abs(m1-points[i]))

        D2.append(abs(m2-points[i]))

        D3.append(abs(m3-points[i]))
C1,C2,C3 = [],[],[]
for i in range(9):

    minimum = min(D1[i],D2[i],D3[i])

    if D1[i] == minimum:

        C1.append(points[i])

    elif D2[i] == minimum:

        C2.append(points[i])

    else:

        C3.append(points[i])

C1,C2,C3

m1 = sum(C1)/len(C1)

m2 = sum(C2)/len(C2)

m3 = sum(C3)/len(C3)

D1,D2,D3 = [],[],[]

C1,C2,C3 = [],[],[]

for i in range(9):

        D1.append(abs(m1-points[i]))

        D2.append(abs(m2-points[i]))

        D3.append(abs(m3-points[i]))

        

for i in range(9):

    minimum = min(D1[i],D2[i],D3[i])

    if D1[i] == minimum:

        C1.append(points[i])

    elif D2[i] == minimum:

        C2.append(points[i])

    else:

        C3.append(points[i])

C1,C2,C3

m1 = sum(C1)/len(C1)

m2 = sum(C2)/len(C2)

m3 = sum(C3)/len(C3)
C3