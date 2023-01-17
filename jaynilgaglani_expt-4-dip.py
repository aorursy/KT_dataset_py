import numpy as np
import math
def findNumber(N):
    a = np.zeros(N,dtype=float)
    for i in range(N):
        if(i==0):
            a[i] = math.sqrt(1/N)
        else:
            a[i] = math.sqrt(2/N)
    return a
def generateDCT(N):
    C = np.zeros([N,N], dtype=float)
    a = findNumber(N)
    for i in range(N):
        for j in range(N):
            k = ((2*j+1)*i)/N
            k = (math.pi/2)*k
            C[i][j] = round(a[i]*math.cos(k),3)
    return C
def generateHadamardMatrix(N):
    H = np.ones([N,N])
    i=1
    while i<N:
        for j in range(i):
            for k in range(i):
                H[j+i][k] = H[j][k]
                H[j][k+i] = H[j][k]
                H[j+i][k+i] = -1*H[j][k]
        i += i
    return H

def calculateWalshTransform(N):
    H = generateHadamardMatrix(N)
    W = np.zeros([N,N],dtype=int)
    temp = []
    for i in range(N):
        count = 0
        for j in range(N-1):
            if H[i][j]!=H[i][j+1]:
                count = count + 1
        temp.append(count)
        
    for i in range(N):
        W[temp[i]] = H[i]
    return W
def DCT(F):
    C = generateDCT(F.shape[0])
    DCT = np.dot(C,F)
    DCT = np.dot(DCT,C.T)
    print("----------------Discrete Cosine Transform-----------------\n")
    print(DCT)
def HadamardMatrix(F):
    H = generateHadamardMatrix(F.shape[0])
    HCT = np.dot(H,F)
    HCT = np.dot(HCT,H.T)
    print("\n-----------------Hadamard Matrix-----------------\n")
    print(HCT)
def WalshTransform(F):
    W = calculateWalshTransform(F.shape[0])
    WCT = np.dot(W,F)
    WCT = np.dot(WCT,W.T)
    print("\n-----------------Walsh Transform-----------------\n")
    print(WCT)
F = np.array([
 [2,4,4,2],
 [4,6,8,3],
 [2,8,10,4],
 [3,8,6,2]
 ])
DCT(F)
HadamardMatrix(F)
WalshTransform(F)
print("\n")
