import matplotlib.pyplot as plt

import random

import time 

def dotProduct (p1,p2): 

    return sum([p1[i]*p2[i] for i in range(3)])

def sign(y): 

    if y>0:

        return "positive"

    else: 

        return "negative"



p0 = [1, 0, 0]

y0 = -1

p1 = [1, 0, 1]

y1 = -1

p2 = [1, 1, 0]

y2 = -1

p3 = [1, 1, 1]

y3 = +1





w = [0, 1/2, 1]



while (dotProduct(w,p0)>=0 or dotProduct(w,p1)>=0 or dotProduct(w,p2)>=0 or dotProduct(w,p3) <=0):

    # plot the line given by w 

    (strp,p,y) = random.choice([("p0",p0,y0),("p1",p1,y1),("p2",p2,y2),("p3",p3,y3)])

    if ( dotProduct(w, p) *  y > 0 ): 

        continue

    else: 

        print("We have chosen", (strp,y), "because this point is misclassified")

        print("dotProduct of", strp, "and w is", dotProduct(p,w), "while y of", strp, "is", sign(y))

        w = [w[i] + y * p[i] for i in range(3)] 

        print("new w=", w)

        

        if (w[2]!=0): 

            plt.plot([0,1],[-w[0]/w[2],(-w[0]-w[1])/w[2]],'k')

        else: 

            if (w[1]!=0): 

                plt.plot([-w[0]/w[1],-w[0]/w[1]],[0,1],'k')

            else: 

                plt.plot([0,0],[0,1],'k')

        plt.scatter(0,0,marker='o',color='b')

        plt.scatter(0,1,marker='o',color='b')

        plt.scatter(1,0,marker='o',color='b')

        plt.scatter(1,1,marker='s',color='r')

        plt.annotate('p0',(0,0))

        plt.annotate('p1',(0,1))

        plt.annotate('p2',(1,0))

        plt.annotate('p3',(1,1))

        plt.show()

        print("Now", strp, "is well classified")

        print("dotProduct of p and w =", dotProduct(p,w), "while y is", sign(y))

        print()

        input("press to continue")

        