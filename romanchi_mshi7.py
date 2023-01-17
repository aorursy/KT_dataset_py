import numpy as np

import math

from time import time

def F_Matias(x):   

    return 0.26* (x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]  

def F_Levi13(x):  

    return pow(math.sin(3*math.pi*x[0]),2) + pow(x[0]-1,2)*(1+pow(math.sin(3*math.pi*x[1]),2)) + pow(x[1]-1,2)*(1+ pow(2*math.pi*x[1],2))  

def Shaffer4(x):  

    nominator = pow(math.cos(math.sin(math.fabs(x[0]**2 - x[1]**2))),2) - 0.5  

    dominator = pow(1 + 0.001*(x[0]**2 + x[1]**2), 2)  

    return 0.5 + float(nominator)/float(dominator)  



def sign_(x):

    if x > 0:

        return 1

    return -1



def main(function):



    #Rest = [[-100,100],[-100,100]]

    #Rest = [[-1.5,4],[-3,4]]

    restrictions = [[-1.5,4],[-3,4]]

    Rest = [[-10,10],[-10,10]]

    Points = list([tuple([(np.random.uniform(Rest[j][0] + ((Rest[j][1] - Rest[j][0]) * i) / 10., Rest[j][0] + abs((Rest[j][1] - Rest[j][0]) * (i + 1)) /10.)) for j in range(len(Rest))]) for i in range(5)])

    Params = [[100,5],15,np.pi,np.pi,1,500,10]

    def S(Ps, k):

        for i in Ps:

            if k < i[1] and k > i[0]:

                return Ps.index(i)

    def Lim(individ):

        desc_x = individ[0]

        desc_y = individ[1]

        if not ((restrictions[0][0] <= desc[0] <= restrictions[0][1]) and (

            restrictions[1][0] <= desc[1] <= restrictions[1][1])):

            while (desc_x > restrictions[0][1]) or (desc_x < restrictions[0][0]):

                desc_x = desc_x + (restrictions[0][0] - restrictions[0][1]) * sign_(desc_x - restrictions[0][1])

            while (desc_y > restrictions[1][1]) or (desc_y < restrictions[1][0]):

                desc_y = desc_y + (restrictions[1][0] - restrictions[1][1]) * sign_(desc_y - restrictions[1][1])

        return desc_x, desc_y

    Conc = Params[0]

    Steps = Params[5]

    Period = Params[6]

    Temp = []

    l = dict()

    drop = []

    t = Conc[1]

    Ps = [0] + [(0.5)**(i + 1) for i in range(t)]

    Ps[0] = (Ps[0], Ps[0] + Ps[1])

    for i in range(1, len(Ps) - 1):

        Ps[i] = (Ps[i - 1][1], Ps[i - 1][1] + Ps[i + 1])

    Ps[-2] = (Ps[i - 1][1], 1)

    Ps = Ps[:-1]

    Probs = Ps

     

    for cycle in range(Steps):

        Nums = dict()

        PointsA = []

        PointsB = []

        PointsC = []

        for ind in Points:

            Nums[ind] = function(ind)

            if ind in l.keys():

                l[ind] = l[ind] + 1

            else:

                l[ind] = 1

            if (l[ind] > Period):

                l.pop(ind, None)

                drop.append(ind)

        for i in range(Conc[0]):

            Descs = []

            distance = np.random.uniform(-Params[1], Params[1])

            angle_1 = np.random.uniform(0, Params[2])

            angle_2 = np.random.uniform(0, Params[3])

            h = np.random.uniform(0, Params[4])

            if (len(set(list(map(lambda n: tuple(map(lambda el: round(el, 8),n)), Points)))) < 2):

                if (Temp != []):

                    Temp[0] = Points[0] if function(Points[0]) < function(Points[0]) else Points[0]

                else:

                    Temp.append(Points[0])

                l = dict()

                drop = []

                Points = list([tuple([np.random.uniform(Rest[j][0], Rest[j][1]) for j in range(len(Rest))]) for i in range(Conc[1])])

            A = -1

            B = -1

            while (A == B):

                Stohastic = np.random.uniform(0, 1, 2)

                A = S(Probs, Stohastic[0])

                B = S(Probs, Stohastic[1])

            try:

                AncA, AncB = Points[A], Points[B]

            except IndexError:

                AncA, AncB = Points[0], Points[0]





            Descs.append((AncA[0] + distance * np.cos(angle_1), AncA[1] + distance * np.sin(angle_1)))

            Descs.append((AncB[0] + distance * np.cos(angle_1), AncB[1] + distance * np.sin(angle_1)))

            for desc in Descs:

                PointsA.append(Lim(desc))

            if (function(AncA) < function(AncB)):

                Descs.append(AncB)

                Descs.append((AncA[0] + (AncA[0] - AncB[0]) * np.cos(angle_2) - ( AncA[1] - AncB[1]) * np.sin(angle_2), AncA[1] + (AncA[0] - AncB[0]) * np.sin(angle_2) + (AncA[1] - AncB[1]) * np.cos(angle_2)))

            else:

                Descs.append(AncA)

                Descs.append((AncB[0] + (AncB[0] - AncA[0]) * np.cos(angle_2) - ( AncB[1] - AncA[1]) * np.sin(angle_2), AncB[1] + (AncB[0] - AncA[0]) * np.sin(angle_2) + (AncB[1] - AncA[1]) * np.cos(angle_2)))

            for desc in Descs:

                PointsB.append(Lim(desc))

            Descs = []

            if (function(AncA) < function(AncB)):

                Descs.append((h * (AncA[0] + AncB[0]), h * (AncA[1] + AncB[1])))

            for desc in Descs:

                PointsC.append(Lim(desc))

        Points += PointsC

        Points += PointsA

        Points += PointsB

        Points.sort(key=lambda x: function(x))

        Points = Points[:Conc[1]]

        for i in drop:

            drop.remove(i)

            if (i in Points):

                Points.remove(i)

                if (Temp == []):

                    Temp.append(i)

                else:

                    print(function(i), function(Temp[0]))

                    if function(i) < function(Temp[0]):

                        Temp[0] = i

        #print("Iter #" , cycle, ": ", str(list(map(lambda n: tuple(n), Points))))

    Points += Temp

    Points.sort(key=lambda n: function(n))

    return Points[0]



#res = main(F_Levi13)

start = time()

res = main(F_Levi13)

end = time()

print((end - start))

#res = main(F_Matias)

#print("BEST: {} at ({})".format(F_Levi13(res), res))

print("BEST: {} at ({};{})".format(F_Levi13(res), math.fabs(res[0]), math.fabs(res[1])))

#print("BEST: {} at ({})".format(F_Matias(res), res))
