w1= -0.2

w2= 0.4

Learning_rate = 0.2

Thold = 0
exp = [0,1,1,1]
def step(num):

    if(num>Thold):

        num=1

    elif(num<=Thold):

        num=0

    return num
def errorcal(act,exp):

    e = exp-act

    return e
def weightupdate(wi,xi,error):

    newweight = wi+(Learning_rate*xi*error)

    return newweight
for epoch in range (20):

    print("")

    print("*****EPOCH # {}*****".format(epoch+1))

    ite = 0

    for x1 in range(2):

        for x2 in range(2):

            print(" ")

            print("for x1 = {} , for x2 = {}".format(x1,x2))

            print("w1 = {} ,  w2 = {}".format(w1,w2))

            y = (w1*x1)+(w2*x2)

            y = step(y)

            print("Our Actual Value is {}".format(y))

            print("Our Expected Value is {}".format(exp[ite]))

            error = errorcal(y,exp[ite])

            print("Our error Value is {}".format(error))

            if(error!=0):

                w1=weightupdate(w1,x1,error)

                print("Our Updated Weight w1 is {}".format(w1))

                w2=weightupdate(w2,x2,error)

                print("Our Updated Weight w2 is {}".format(w2))

            ite= ite+1

    