# Import the dependencies

import numpy as np

from scipy.linalg import toeplitz, cholesky, sqrtm

from scipy.linalg import inv

from scipy import signal

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")

print("Done")
# Setting up the time data:

dt = 0.005; # integration step, average neuron resets 200 times per second

T = 5+dt; # maximum time considered

N = np.int(np.round(T/dt)) #Amount of data points

t = np.arange(0,T,dt)

print ('Amount of data points: ', N)

print ('Starting with', t[0:5])

print ('Ending with', t[N-5:N])

print ('Data elements', np.size(t))
# Example to draw random samples using the python numpy.random.normal function  

# Generates a normal (Gaussian) distribution with a certain mean (0) and standard deviation (3)

w_first = np.random.normal(0,3,N)

print('Example 1')

print('Generate random single zero-mean sequences with standard deviation 3')

print('First 10 values of a zero-mean random sequence for visual inspection')

print(w_first[0:10])

print('')



print ('Example 2')

print('Generate random zero-mean sequences from a covaraince matrix')

# Example to generate random samples series with desired covariance matrix 

# note: on the diagonal you will find the variances, is is a symmetric matrix

# This method is capable to generate noise with covariances but in case of active inference we generate independent noise, 

# hence 0 on the non-diagonals

np.random.seed(123456)

Sw = np.matrix('3 0 0;0 9 0; 0 0 16')

n = Sw.shape[1] # dimension of noise = amount of sequences to be generated 

L =cholesky(Sw, lower=True)  #Cholesky method

w = np.dot(L,np.random.randn(n,N))



# Plot the first white noise sequence:

plt.plot(t[0:200],w.T[0:200,1],label='white noise'); 

plt.title('Noise sequence 1, first second')

plt.legend(loc='upper right')

plt.show;

# some plt versions expect data in same dimension, hence the w.T to align with w



# Calculate the variance/covariance of the generated data sets



print ("Covariance matrix")

print(Sw)

print ("Estimated variance/covariance of generated random sequence")

print(np.cov(w))

print ("The mean of sequence 1: ", np.mean(w[1,:]))

# Example of a univariate case (vector of 1 number)

# In this example an embedding order of 3 (and thus has 4 entries)

D= np.matrix('0 1 0 0 ; 0 0 1 0 ; 0 0 0 1 ; 0 0 0 0')

print ('Derivative operator')

print (D)

print ('vector with 1 data point in generalized coordinates of motion with embedding order 3')

x = np.matrix('1; 2; 3; 4')

print (x)

print ('Result')

dx = np.dot(D,x)

print (dx)

print ('vector with 2 data points in generalized coordinates of motion with embedding order 3')

y = np.matrix('1 4; 2 3; 3 2 ; 4 1')

print (y)

print ('Result')

dy = np.dot(D,y)

print (dy)
def makeNoise(C,s2,t):

    """

    Generate coloured noise 

    Code by Sherin Grimbergen (V1 2019) and Charel van Hoof (V2 2020)

    

    INPUTS:

        C       - variance of the required coloured noise expressed as desired covariance matrix

        s2      - temporal smoothness of the required coloured noise, expressed as variance of the filter

        t       - timeline 

        

    OUTPUT:

        ws      - coloured noise, noise sequence with temporal smoothness

    """

    

    if np.size(C)== 1:

        n = 1

    else:

        n = C.shape[1]  # dimension of noise

        

    # Create the white noise with correct covariance

    N = np.size(t)      # number of elements

    L =cholesky(C, lower=True)  #Cholesky method

    w = np.dot(L,np.random.randn(n,N))

    

    if s2 < 1e-5: # return white noise

        return w

    else: 

        # Create the noise with temporal smoothness

        P = toeplitz(np.exp(-t**2/(2*s2)))

        F = np.diag(1./np.sqrt(np.diag(np.dot(P.T,P))))

        K = np.dot(P,F)

        ws = np.dot(w,K)

        return ws
# Example to generate coloured noise with desired covariance matrix 



np.random.seed(123456) # same random seed so same random white noise generated as previous example

ws_64 = makeNoise(Sw,1/64,t)

np.random.seed(123456) # same random seed so same random white noise generated as previous example

ws_4 = makeNoise(Sw,1/4,t)

np.random.seed(123456) # same random seed so same random white noise generated as previous example

ws_512 = makeNoise(Sw,1/512,t)





# Plot the noise with temporal smoothness sequence (first second of sequence 1):

plt.plot(t,w.T[:,1],label='white noise')

plt.plot(t,ws_64.T[:,1],label='coloured noise 1/64')

plt.plot(t,ws_4.T[:,1],label='coloured noise 1/4') 

plt.plot(t,ws_512.T[:,1],label='coloured noise 1/512') 

plt.title('Coloured noise ')

plt.legend(loc='upper right')

plt.show;



print ("Covariance matrix")

print(Sw)

print ("Estimated variance/covariance of generated random sequence with temporal smoothness")

print(np.cov(ws_64))

def temporalC(p,s2):

    """

    Construct the temporal covariance matrix S for noise with embedding order p and smoothness parameter s

    

    Code by Sherin Grimbergen (V1 2019) and Charel van Hoof (V2 2020)

    

    INPUTS:

        p       - embedding order (>0)

        s2      - smoothness parameter (>0), variance of the filter (sigma^2)

        

    OUTPUT:

        S       - temporal covariance matrix ((p+1) x (p+1))

    """ 



    q = np.arange(p+1)

    

    r = np.zeros(1+2*(p))

    r[2*q] = np.cumprod(1-2*q)/(2*s2)**(q)    

    

    S = np.empty([0,p+1])



    for i in range(p+1):

        S = np.vstack([S,r[q+i]])

        r = -r

           

    return S 
# Example temporal covariance matrix

p=5 # embedding order 5 of generative model, is the number of derivatives. p=5  means we have μ(0)  until  μ(6) , which means the vector has 6 entries

variance = 1 # selected variance of 1 so you can easily compare with the printed example above

Cv= temporalC(p,variance)

print(Cv)
# Example temporal covariance matrix

p=5 # embedding order 5 of generative model, is the number of derivatives. p=5  means we have μ(0)  until  μ(6) , which means the vector has 6 entries

variance = 0.5 # selected variance of 0.5 so you can easily see the variances on the diagonal grow quickly and thus less influence of higher order motions

Cv= temporalC(p,variance)

print(Cv)
# Same sample code to highlight the power of matrix calculations



v = np.matrix('1; 2; 3')

A= np.matrix(' 1 2 3 ; 4 5 6 ; 7 8 9 ')

B= np.matrix(' 3 0 0 ; 0 6 0 ; 0 0 9 ')



print ('Example matrix general A')

print(A)

print ('Example covariance matrix B')

print(B)

print ('Example vector v')

print(v)

print ('Sum of matrix A and B')

print(A+B)

print ('Transpose of matrix A')

print(A.T)

print ('Transpose of vector v')

print(v.T)

print ('Matrix B multiplied with vector v')

print(np.dot(B,v))

print ('Matrix B multiplied with matrix A')

print(np.dot(B,A))

print ('Matrix A multiplied with matrix B (notice A*B is not B*A)')

print(np.dot(A,B))

print ('Kronecker product Matrix A with matrix B')

print(np.kron(A,B))



# Same sample code to highlight the simplified math of 



v = np.matrix('-1; 2; 3')

A= np.matrix(' 1 2 3 ; 4 5 6 ; 7 8 9 ')

B= np.matrix(' 3 0 0 ; 0 6 0 ; 0 0 9 ')



print ('Example matrix general A')

print(A)

print ('Example covariance matrix B')

print(B)

print ('Transpose of matrix A')

print(A.T)

print ('Transpose of matrix B')

print(B.T)

print ('Inverse of Matrx A ')

print(np.linalg.inv(A))

print ('Inverse of Matrx B (notice non-zero values are replaced with its reciprocal in case of an inverse of a covariance matrix) ')

print(np.linalg.inv(B))

print ('Determinant of Matrix A')

print(np.linalg.det(A))

print ('Determinant of Matrix B (notice in case of a covariance matrix the determinant is the multiplication of the variances on the diagonal)')

print(round(np.linalg.det(B)))

print ('Show example covariance matrix B is semi-definite')

print(np.dot(v.T,np.dot(B,v)))

print ('Show example precision matrix B is semi-definite')

print(np.dot(v.T,np.dot(np.linalg.inv(B),v)))