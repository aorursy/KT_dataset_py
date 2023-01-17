print('hello')
import numpy as np
import math,cmath
import random
import matplotlib.pyplot as plt

"""definition about the problem"""
M=100
N=20
global totalN
totalN=0
"""definition of gates,states & GD related functions"""
FredkinGate=np.array(
               [[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1]])
HadamardGate=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
IdentityGate=np.array([[1,0],[0,1]])
NotGate=np.array([[0,1],[1,0]])#also pauli-x
zero=np.array([1,0]).reshape(2,1)
"""for convenience, some multiple-qubit gate"""
HII=np.kron(np.kron(HadamardGate,IdentityGate),IdentityGate)
NII=np.kron(np.kron(NotGate,IdentityGate),IdentityGate)
E0II=np.kron(np.array([[1,0],[0,0]]),np.kron(IdentityGate, IdentityGate))

#magnitude of a complex number
def Mag(x):
    x=np.array(x)
    return np.sqrt(x.dot(x))
#rotation operators
def Rx(theta):
    return np.array([[math.cos(theta/2),-1j*math.sin(theta/2)],
                     [-1j*math.sin(theta/2),math.cos(theta/2)]])
def Ry(theta):
    return np.array([[math.cos(theta/2),-math.sin(theta/2)],
                     [math.sin(theta/2),math.cos(theta/2)]])
def Rz(theta):
    return np.array([[cmath.exp(-1j*theta/2),0],
                     [0,cmath.exp(1j*theta/2)]])

def RandomUnitary():
    """slight different from the parameterization used in approximation"""
    Utheta=np.random.rand()*math.pi
    Uphi=np.random.rand()*math.pi*2
    Ulambd=np.random.random()*math.pi*2
    U= np.array([[math.cos(Utheta/2),-cmath.exp(1j*Ulambd)*math.sin(Utheta/2)],
             [cmath.exp(1j*Uphi)*math.sin(Utheta/2), cmath.exp(1j*(Ulambd+Uphi))*math.cos(Utheta/2)]])
    return U

def UPara(theta):
    """U=R_z(theta0)*R_y(theta1)*R_z(theta2)"""
    """TODO:A more generalized form"""
    return np.dot(np.dot(Rz(theta[0]),Ry(theta[1])),Rz(theta[2]))

def RandomState():
    t=np.random.rand(4)
    t/=math.sqrt(np.sum(t*t))
    return np.array([t[0]+1j*t[1],t[2]+1j*t[3]]).reshape(2,1)


def CSWAP(s0,s1,s2):
    afterCSWAP=np.dot(HII, np.dot(FredkinGate,np.dot(HII,np.kron(s0,np.kron(s1,s2)))))
    return afterCSWAP
"""Prob0 in CSWAP gate"""
"""update to N measurement!"""
def Prob0(s0, s1, s2,isIdeal):
    afterCSWAP=CSWAP(s0,s1,s2)
    #return np.absolute(np.sum(np.power(afterCSWAP[:4].real,2)+np.power(afterCSWAP[:4].imag,2)))
    Prob=np.absolute(np.sum(np.power(afterCSWAP[:4].real,2)+np.power(afterCSWAP[:4].imag,2)))
    if(isIdeal):
        return Prob
    global totalN
    totalN+=N
    return np.sum(np.array([np.random.rand()<Prob for x in range(N)]))/N
    
"""measure first qubit in 0 basis"""
def trace0(state, isIdeal):
    statedag=state.conjugate().reshape(1,8)
    E0IIrho3=np.dot(E0II,np.dot(state,statedag))
    l=np.array([E0IIrho3[x][x] for x in range(8)])
    F=np.sum(np.sqrt(l.imag**2+l.real**2))
    if(isIdeal):
        return F
    
    global totalN
    totalN+=N
    #print(np.sum(np.array([np.random.rand()<F for x in range(N)]))/N)
    return np.sum(np.array([np.random.rand()<F for x in range(N)]))/N
"""Gradient Descent"""
def cost_func(exact_output, test_output, isIdeal):
    """fidelity: CSWAP gate"""
    C=0.0
    for i in range(len(test_output)):
        state=CSWAP(zero,exact_output[i],test_output[i])
        F=trace0(state,isIdeal)
        #print(F)
        C+=F
    C=1-C/len(test_output)
    return C


def gradient(theta, j, test_input,exact_output, isIdeal):
    Utheta0=UPara([theta[x]+(math.pi/2)*(j==x) for x in range(len(theta))])
    Utheta1=UPara([theta[x]-(math.pi/2)*(j==x) for x in range(len(theta))])
    Utheta=UPara(theta)
    Utheta_tmp=UPara([theta[x]+0.001*(j==x) for x in range(len(theta))])
    G=0
    for i in range(len(test_input)):
        state=CSWAP(zero, exact_output[i], np.dot(Utheta,test_input[i]))
        state_tmp=CSWAP(zero, exact_output[i], np.dot(Utheta_tmp,test_input[i]))
        """HII*Fredkin*HII"""
        Pr00=Prob0(zero, exact_output[i], np.dot(Utheta0,test_input[i]),isIdeal)
        Pr01=Prob0(zero, exact_output[i], np.dot(Utheta1,test_input[i]),isIdeal)
        F=trace0(state,isIdeal)
        #F_tmp=trace0(state_tmp,isIdeal)
        if (np.abs(F)<1e-05):
           F+=0.00001 
        #print((Pr00-Pr01)/F, (F_tmp-F)*100000)
        G+=(Pr00-Pr01)/F
    G=G*(-1/len(test_input))
    return G

def modify(x):
    ans=x
    if(ans>0.5*math.pi):
        ans-=int((ans-0.5*math.pi)/(2*math.pi))*2*math.pi
    if (ans<=-0.5*math.pi):
        ans+=int((-0.5*math.pi-ans)/(2*math.pi))*2*math.pi
    return ans

thetaUnknown=[(np.random.rand()-0.5)*math.pi/2 for x in range(3)]
U=UPara(thetaUnknown)*np.exp(1j*np.random.rand())
print("U=\n",U)
Psi=[RandomState() for x in range(M)]
UPsi=[np.dot(U, psi) for psi in Psi]
Mtest=int(M/5)
if(Mtest<5):
    Mtest=50
Psi_test=[RandomState() for x in range(Mtest)]
UPsi_test=[np.dot(U, psi) for psi in Psi_test]
print("Psi0=\n",Psi[0])
print("UPsi0=\n",UPsi[0])

"""Initialization"""
#theta=[(np.random.rand()-0.5)*math.pi for x in range(3)]
def process(M, N):
    global totalN
    totalN=0
    theta=[1,1,1]
    Utheta=UPara(theta)
    Psi=[RandomState() for x in range(M)]
    UPsi=[np.dot(U, psi) for psi in Psi]
    C=cost_func(UPsi[:M], [np.dot(Utheta,x) for x in Psi[:M]],False)
    #print(theta)
    #print(thetaUnknown)
    #print(UPara(theta))
    #print(UPara(thetaUnknown))
    """Psi, UPsi"""
    step=0
    alpha=0.3
    thetalist=[]
    thetalist.append(theta)
    testcostlist=[]
    testcostlist.append(cost_func(UPsi_test, [np.dot(Utheta,x) for x in Psi_test],True))
    costlist=[]
    costlist.append(C)
    while(step<200):
        #print(step)
        gradientC=np.array([gradient(theta, j, Psi[:M],UPsi[:M],False) for j in range(len(theta))])
        #print('gradient')
        theta=list(np.array(theta)-alpha*gradientC)
        for i in range(len(theta)):
            theta[i]=modify(theta[i])
        #print('theta')
        Utheta=UPara(theta)
        #print('Utheta')
        C=cost_func(UPsi[:M], [np.dot(Utheta,x) for x in Psi[:M]], False)
        #print('costfunc')
        #print(theta)
        thetalist.append(theta)
        testcostlist.append(cost_func(UPsi_test, [np.dot(Utheta,x) for x in Psi_test], True))
        costlist.append(C)
        #print(gradientC)
        #print(C)
        Psi=[RandomState() for x in range(M)]
        UPsi=[np.dot(U, psi) for psi in Psi]
        step+=1

    print('total measurements:', totalN)
    #print('process end')
    return theta, costlist, testcostlist
def drawfig(num, title, costlist, testcostlist):
#    plt.figure(0)
#    plt.plot(list(range(len(thetalist))),thetalist,'-')
#    plt.plot(0,thetaUnknown[0],'b.')
#    plt.plot(0,thetaUnknown[1],'b*')
#    plt.plot(0,thetaUnknown[2],'bo')
#    plt.show()
    plt.subplot(5,4,num+1).plot(list(range(len(costlist))),costlist,'r')
    plt.subplot(5,4,num+1).plot(list(range(len(testcostlist))),testcostlist,'g')
    plt.subplot(5,4,num+1).set_title(title)
#    plt.show()
"""red:cost function of training set green:ideal cost function of test set"""

MM=[20,10,5,2]
NN=[20,10,5,2]
MN=[(20,20),
    (20,10),
    (20,5),
    (20,2),
    (10,20),
   (10,10),
   (10,5),
   (10,2),
   (5,20),
   (5,10),
   (5,5),
   (5,2),
   (2,20),
   (2,10),
   (2,5),
   (2,2),
   (1000,1),
   (500,1),
   (200,1),
   (100,1)]
totTcl=[]
cnt=0
print(UPara(thetaUnknown))

fig=plt.figure(0)
fig.set_size_inches(40, 20)
for mn in MN:
    M,N = mn
    theta,cl,tcl=process(M,N)
    drawfig(cnt,'M=%d N=%d' % (M,N), cl, tcl)
    totTcl.append((tcl, cnt))
    cnt+=1
    #print('M=%d N=%d' % (M,N))
    #print(UPara(theta))

plt.show()    
fig.savefig('costfunctionMN.png', dpi=720)        
        

fig1=plt.figure(100)
fig.set_size_inches(18.5, 10.5)
for i in totTcl:
    tcl, clr=i
    if(clr%5==0):
        color=((clr+1)/cnt,(clr+1)/cnt,(clr+1)/cnt)
    elif (clr%5==1):
        color=((clr+1)/cnt,0,0)
    elif (clr%5==2):
        color=(0,(clr+1)/cnt,0)
    elif (clr%5==3):
        color=(0,0,(clr+1)/cnt)
    else:
        color=((clr+1)/cnt,(clr+1)/cnt,0)
    plt.plot(list(range(len(tcl[:50]))),tcl[:50],color=color)
    
plt.show()
fig.savefig('costfunctiontrend.png', dpi=100)
