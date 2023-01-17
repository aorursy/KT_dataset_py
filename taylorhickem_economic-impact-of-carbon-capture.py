import matplotlib.pyplot as plt
#define constants

R0 = 500   #initial revenue mil USD/yr

f = 0.15   #energy penalty with carbon capture

z = 0.5    #% carbon capture

pC = 40    #USD/ton CO2eq

pE = 0.015 #USD/kWh energy price

EF = 0.18  #kg CO2eq/kWh of energy

m0 = 0.25  #initial margin

xE = 0.8   #energy share of costs

cL = 0.1   #labor cost per $ of revenue

pL = 60    #labor cost kUSD per worker per year

beta = 1   #elasticity margin to volumes
# define functions

def margin(zcc,p=pC,x=xE,i=m0,pnt=f):

    delta = (zcc*pnt+(1-zcc)*10**-3*p/pE*EF)

    m = 1-(1+delta*x)*(1-i)

    return m

    

def vapw(m): #value added per worker

    return 1000*m*pL/cL



def revenue(k):

    return R0*(1+beta*k)



def W(m): #number of workers

    R = revenue(m/m0-1)

    return R*m/vapw(m)*10**6



    
W(0.25)
# simple example no carbon capture, carbon price $10/ton, 

# energy 20% of costs and 20% margins



margin(0,p=10,x=0.2,i=0.20)
#set of series at different carbon prices from 5, 20, 40, 80 and 120 USD/ton

# vary from 0-100% cc implementation

#prices = [5,20,40,80,120]

prices = [5,10,20,30,40]

zcc = [x/100 for x in range(101)]
margins = [[margin(x,p=y) for x in zcc] for y in prices]

for m in margins:

    plt.plot(zcc,m)

plt.legend(['$5/ton','$20/ton','$30/ton','$40/ton','$50/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('profit margin')
margins = [[margin(x,p=y,pnt=0.1) for x in zcc] for y in prices]

for m in margins:

    plt.plot(zcc,m)

plt.legend(['$5/ton','$20/ton','$30/ton','$40/ton','$50/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('profit margin')
prices = [5,20,40,80,120]

margins = [[margin(x,p=y,x=0.2,pnt=0.15) for x in zcc] for y in prices]

for m in margins:

    plt.plot(zcc,m)

plt.legend(['$5/ton','$20/ton','$40/ton','$80/ton','$120/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('profit margin')
margins = [[margin(x,p=y,x=0.2,pnt=0.10) for x in zcc] for y in prices]

for m in margins:

    plt.plot(zcc,m)

plt.legend(['$5/ton','$20/ton','$40/ton','$80/ton','$120/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('profit margin')
prices = [5,10,20,30,40]

va = [[vapw(margin(x,p=y,pnt=0.15)) for x in zcc] for y in prices]

for v in va:

    plt.plot(zcc,v)

plt.legend(['$5/ton','$20/ton','$30/ton','$40/ton','$50/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('value added per worker $USD')
workers = [[W(margin(x,p=y,pnt=0.15)) for x in zcc] for y in prices]

for w in workers:

    plt.plot(zcc,w)

plt.legend(['$5/ton','$20/ton','$30/ton','$40/ton','$50/ton'])

plt.xlabel('% carbon captured')

plt.ylabel('total employed')
#simple example to refresh my memory

x = [x for x in range(10)]

y1 = [2*i for i in x]

y2 = [3*i for i in x]

plt.plot(x,y1)

plt.plot(x,y2)

plt.legend(['y1','y2'])