w1=0.1

w2=0.4

w3=0.8

w4=0.6

w5=0.3

w6=0.9

Learning_rate = 0.2
exp = [0,1,1,0]
import math as m

def sigmoid(num):

    num = (1/(1+m.exp(-num)))

    return num
def errorcal(act,val):

    error = val*((1-val)*(act-val))

    return error
def weightupdate(wi,xi,error):

    newweight = wi+(error*xi)

    return newweight


for epoch in range (20):

    print("")

    print("*****EPOCH # {}*****".format(epoch+1))

    ite = 0

    for a in range(2):

        for b in range(2):

            print("")

            print("for x1={} and x2={}".format(a,b))

            z1 = (w1*a)+(b*w3)

            z1 = sigmoid(z1)

            print("value of first neuron = {}".format(z1))

            z2 = (b*w4)+(a*w2)

            z2 = sigmoid(z2)

            print("Value of second neuron = {}".format(z2))

            f = (z1*w5)+(z2*w6)

            f = sigmoid(f)

            print("Output is {}".format(f))

            error = errorcal(exp[ite],f)

            print("error is {}".format(error))

            w5 = weightupdate(w5,z1,error)

            print("Updated First Weight for output is {}".format(w5))

            w6 = weightupdate(w6,z2,error)

            print("Updated Second Weight for output is {}".format(w6))

            hiddenerror1 = (error * w5) * ((1-f)*f)

            print("Error for Hidden layer = {}".format(hiddenerror1))

            hiddenerror2 = (error * w6) * ((1-f)*f)

            print("Error for hidden layer = {}".format(hiddenerror2))

            w1 = w1 + (hiddenerror1 * a)

            w3 = w3 + (hiddenerror2 * b)

            w2 = w2 + (hiddenerror1 * a)

            w4 = w4 + (hiddenerror2 * b)

            print("New Weights for w1 {}".format(w1))

            print("New Weights for w3 {}".format(w3))

            print("New Weights for w2 {}".format(w2))

            print("New Weights for w4 {}".format(w4))