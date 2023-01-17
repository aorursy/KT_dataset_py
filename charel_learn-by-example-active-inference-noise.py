# Import the dependencies

import numpy as np

from scipy.linalg import toeplitz, cholesky, sqrtm

from scipy.linalg import inv

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")

print("Import completed")
# Setting up the time data:

dt = 0.05

T = 5 + dt

N = int(round(T/dt))

t = np.arange(0,T,dt)

print(N, "data elements")

print ('Starting with', t[0:5])

print ('Ending with', t[N-5:N])

print ('Data elements', np.size(t))
variance = 4 #input



# randn generates an array of shape (in this case N), filled with random floats 

# sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1 

# so 68.3% of the samples are between -1 and + 1; 

# 95.5% of the samples between -2 and +2; 

# 98.7% of the samples between -3 and +3

# Multiple with the standard deviation to get the right variance

w = np.sqrt(variance) * np.random.randn(N)



# or simply use the python numpy.random.normal function to draw random samples from a normal (Gaussian) distribution

# w = np.random.normal(0,np.sqrt(variance),N)



#Show the first 80 values for visual inspection

print(w[0:80])



# Plot Gaussian white noise sequence to get the rough ragged noise due to the Identically distributed and statistically independent:

plt.plot(t,w,label ='White noise')

plt.title('White noise')

plt.show
# Print the distribution plot to showcase the bell-curve 

# (The more values the better the curve becomes visible)

Min_graph = min(w)

Max_graph = max(w)

x = np.linspace(Min_graph, Max_graph, N) # to plot the data

sns.distplot(w, bins=20, kde=False)

print ("The mean is: ", np.mean(w))
#Show with the same random seed we get the same random sequence

random_seed=1234



np.random.seed(random_seed)

w1 = np.random.randn(N)

plt.plot(t[0:80],w1[0:80],label ='White noise 1')



np.random.seed(random_seed)

w2 = np.random.randn(N)

plt.plot(t[0:80],w2[0:80],linestyle='dashed',label ='White noise 2')



plt.title('White noise comparison')

plt.show



#Let's compare both methods with a low and high correlation covariance matrix

Sw = np.matrix('9 2;2 16') # input covariance matrix

Sw_high = np.matrix('9 9;9 16') # input, high correlation

# note: this matrix must be symmetric since it is a covariance matrix

# note: on the diagonal you will find the variances



# Generate white Gaussian noise sequences:

n = Sw.shape[1] # dimension of noise

np.random.seed(random_seed)



L =sqrtm(Sw)  #Sqrtm method

L_cho =cholesky(Sw, lower=True)  #Cholesky method



x=np.random.randn(n,N)

w = np.dot(L,x)

w_cho=np.dot(L_cho,x)



np.random.seed(random_seed) #same random seed to generate the same sequences

w_high = np.dot(cholesky(Sw_high).T,np.random.randn(n,N))

w_cho_high = np.dot(sqrtm(Sw_high).T,np.random.randn(n,N))



# Plot the white noise sequence:

fig, axes = plt.subplots(2,2)

plt.subplots_adjust(wspace=0.35, hspace=0.35)

fig.suptitle('Two dimensional correlated white noise')

axes[0,0].plot(t,w[0,:],label='Gaussian white noise 1 low correlation')

axes[0,0].set_xlabel('time [s]')

axes[0,0].set_ylabel('Noise 1 L')

axes[0,0].grid(1)

axes[0,1].plot(t,w[1,:],label='Gaussian white noise 2 low correlation')

axes[0,1].set_xlabel('time [s]')

axes[0,1].set_ylabel('Noise 2 L')

axes[0,1].grid(1)

axes[1,0].plot(t,w_high[0,:],label='Gaussian white noise 1 high correlation')

axes[1,0].set_xlabel('time [s]')

axes[1,0].set_ylabel('Noise 1 H')

axes[1,0].grid(1)

axes[1,1].plot(t,w_high[1,:],label='Gaussian white noise 2 high correlation')

axes[1,1].set_xlabel('time [s]')

axes[1,1].set_ylabel('Noise 2 H')

axes[1,1].grid(1)



# Calculate the variance/covariance of the generated data sets

print ("Covariance matrix with low covariance as input for first sequence")

print(Sw)

print ("Covariance matrix with high covariance as input for second sequence")

print(Sw_high)

print ("Estimated variance/covariance of first sequence (sqrtm method)")

print(np.cov(w))

print ("Estimated variance/covariance of second sequence (sqrtm method)")

print(np.cov(w_high))

print ("Estimated variance/covariance of first sequence (cholesky method)")

print(np.cov(w_cho))

print ("Estimated variance/covariance of second sequence (cholesky method)")

print(np.cov(w_cho_high))

print ("The mean of the first sequence (sqrtm method): ", np.mean(w[0,:]))

print ("The mean of the second sequence (sqrtm method): ", np.mean(w_high[0,:]))

print ("The mean of the first sequence (cholesky method): ", np.mean(w_cho[0,:]))

print ("The mean of the second sequence (cholesky method): ", np.mean(w_cho_high[0,:]))
print('Square root matrix:')

print(L)

print('Cholesky decomposition:')

print(L_cho)

print('quick check both cholesky and sqrtm have the property L * L^T = Sw')

print(np.dot(L,L.T))

print(np.dot(L_cho,L_cho.T))
sigma = 0.158 # in FEP sigma is usually 0.158 and << 1



#Showcase the effect of sigma

m=np.arange(-T,T,dt)



sigma_1508=np.exp(-m**2/(2*15.8**2))    #gamma=0.008

sigma_5=np.exp(-m**2/(2*5**2))          #gamma=0.08

sigma_1058=np.exp(-m**2/(2*1.58**2))    #gamma=0.8

sigma_05=np.exp(-m**2/(2*0.5**2))       #gamma=8

sigma_0158=np.exp(-m**2/(2*0.158**2))   #gamma=80

sigma_005=np.exp(-m**2/(2*0.05**2))    #gamma=800



#rho_normalized=1./(np.sqrt(2*np.pi/gamma))*np.exp(-gamma/4*tau**2)



plt.plot(m,sigma_1508,label ='$\sigma=$ 15.8')

plt.plot(m,sigma_5,label ='$\sigma=$ 5')

plt.plot(m,sigma_1058,label ='$\sigma=$ 1.58')

plt.plot(m,sigma_05,label ='$\sigma=$ 0.5')

plt.plot(m,sigma_0158,label ='$\sigma=$ 0.158')

plt.plot(m,sigma_005,label ='$\sigma=$ 0.05')



plt.legend(loc='upper left')

# Let's calculate the convoluted noise in an old-fashioned for-next embedded loop 

# for visual inspection/understanding what is happening

# intuition, the top of the filter ho (h(0)) in the inner loop is centered on position p

wn=w[1,:] #1 dimensional white noise example

wc_trad = np.zeros(N) #initialize convoluted noise on zeros

for p in range(0, N):

    sum=0

    average=0

    for h in range(-p, N-p):

        delta = wn[p+h]*np.exp(-(h*dt)**2/(2*sigma**2)) 

        sum = sum + delta

        average = average + wn[p+h]

        # Below some code for visual inspection of what is happening in the first 5 iterations

        # Show the significant deltas

        if abs(delta) > 0.05 and p<5:

            print ('top=',p,'h=',h, 'delta=','{0:.5f}'.format(delta) ,\

                   '(','{0:.5f}'.format(wn[p+h]), '*' , \

                   '{0:.5f}'.format(np.exp(-(h*dt)**2/(2*sigma**2))), ')', \

                   'sum=','{0:.6f}'.format(sum), 'plain sum=', '{0:.5f}'.format(average) )

    next

    wc_trad[p] = sum

    if p<5:

            print ('top=',p, 'total sum =', sum ) #visual inspection

            print ('top=',p, 'plain sum =', average, 'Average=', average/N ) #visual inspection 

next

plt.plot(t,wn,label='white noise')

plt.plot(t,wc_trad,label='Convoluted noise')

plt.axhline(y=0.0, color='red')

plt.legend(loc='upper left')

plt.title('$\sigma=$ ' + str(sigma))

plt.show
# Scypy has a toeplitz function that we can use to calculate the convolution very handy

# Below example shows what toeplitz calculates

# See the kernel in the the kolomns shifting, by observing the top of the Gaussian (in this case 8)



print(toeplitz([8,4,2,1,1,0]))
# Now we can calculate the convoluted noise with 2 lines of code (for all dimensions of w)

# The noise graphs are identical

P = toeplitz(np.exp(-t**2/(2*sigma**2)))

wc=np.dot(w,P) #Calculate the convoluted noise



# Plot results

plt.plot(t,w[1,:],label='white noise') 

plt.plot(t,wc[1,:],label='Convoluted noise') #Show one dimension

plt.axhline(y=0.0, color='red')

plt.title('$\sigma=$ ' + str(sigma))

plt.legend(loc='upper left')

plt.show
wc_1508=np.dot(w,toeplitz(np.exp(-t**2/(2*15.8**2)))) #gamma=0.008

wc_5=np.dot(w,toeplitz(np.exp(-t**2/(2*5**2)))) #gamma=0.08

wc_1058=np.dot(w,toeplitz(np.exp(-t**2/(2*1.58**2)))) #gamma=0.8            

wc_05=np.dot(w,toeplitz(np.exp(-t**2/(2*0.5**2)))) #gamma=8

wc_0158=np.dot(w,toeplitz(np.exp(-t**2/(2*0.158**2)))) #gamma=80

wc_005=np.dot(w,toeplitz(np.exp(-t**2/(2*0.05**2)))) #gamma=800

wc_00158=np.dot(w,toeplitz(np.exp(-t**2/(2*0.0158**2)))) #gamma=8000              

               

#Example with a forced exact zero mean

wn=w[1,:].copy() #1 dimensional white noise example

print ('White noise mean',np.mean(wn))

print ('White noise sum',np.sum(wn))

wn-=np.mean(wn) # normalize to zero mean by substracting the mean

print ('White noise with forced 0 mean',np.mean(wn))

print ('White noise with forced 0 sum',np.sum(wn))

wc_1508_zm=np.dot(wn,toeplitz(np.exp(-t**2/(2*15.8**2))))



# Plot results

plt.plot(t,w[1,:],label='white') 

plt.plot(t,wc_1508[1,:],label='$\sigma=$ 15.8') 

plt.plot(t,wc_5[1,:],label='$\sigma=$ 5') 

plt.plot(t,wc_1058[1,:],label='$\sigma=$ 1.58') 

plt.plot(t,wc_05[1,:],label='$\sigma=$ 0.5')

plt.plot(t,wc_0158[1,:],label='$\sigma=$ 0.158')

#plt.plot(t,wc_005[1,:],label='$\sigma=$ 0.05') 

plt.plot(t,wc_00158[1,:],label='$\sigma=$ 0.0158',linestyle='dashed') #Dashed to show it overlaps with white noise

plt.plot(t,wc_1508_zm,label='$\sigma=$ 15.8 forced') #Show random numbers with forced zero mean



plt.title('gamma pallette')

plt.legend(loc='best')

plt.show
#Show how the cumulative sum of all random numbers develops



w_sum=w.copy()

print('Sum', np.sum(w[1,:]))

for j in range(1,N):

    w_sum[:,j]=w_sum[:,j-1]+w[:,j]

plt.plot(t,w[1,:],label='white') 

plt.plot(t,w_sum[1,:],label='sum') 
#And simular example some a larger series of 5M random numbers

np.random.seed(random_seed)

w_lots =  np.random.randn(5000000)

print(np.mean(w_lots))

print(np.sum(w_lots))

w_sum=w_lots.copy()

for j in range(1,5000000):

    w_sum[j]=w_sum[j-1]+w_lots[j]

plt.plot(w_lots,label='white') 

plt.plot(w_sum,label='sum') 

    
# Calculate the variance/covariance of the generated data sets

print ("Covariance matrix as input for the white noise")

print(Sw)

print ("Estimated variance/covariance of the white noise")

print(np.cov(w))

print ("Estimated variance/covariance of the convoluted noise")

print(np.cov(wc))

print ("The mean of the white first sequence: ", np.mean(w[0,:]))

print ("The mean of the convoluted first sequence: ", np.mean(wc[0,:]))

print ("The mean of the white second sequence: ", np.mean(w[1,:]))

print ("The mean of the convoloted second sequence: ", np.mean(wc[1,:]))
#Show example

p=toeplitz([1,0.7,0.3,0.2,0.1,0])

#p=toeplitz([9,4,2,1,0])

print('toeplitz P: ')

print(p)

p2=np.dot(p,p.T)

print('squared: ')

print(p2)

p3=np.diag(p2)

print('take diagonal: ')

print(p3)

p4=np.sqrt(p3)

print('take square root: ')

print(p4)

p5=1./p4

print('inverse ')

print(p5)

f=np.diag(p5) #F

print('create matrix F: ')

print(f)

print(f.T)

print('Multiplying matrix P with F ')

p7=np.dot(p, f)

print(p7)

print('Verify that F.T dot P.T dot P dot F = I for the diagonal (identity matrix ) ')

p8=np.dot(f.T,p.T)

p9=np.dot(p8,p7)

print(p9)

# Make the smoothened noise:

F = np.diag(1./np.sqrt(np.diag(np.dot(P.T,P))))

ws= np.dot(wc,F)

plt.plot(t,w[1,:],label='white noise') 

plt.plot(t,wc[1,:],label='Convoluted noise')

plt.plot(t,ws[1,:],label='noise with temporal smoothness')

plt.title('$\sigma=$ ' + str(sigma))

plt.legend(loc='upper left')

plt.show



# Calculate the variance/covariance of the generated data sets

print ("Covariance matrix as input for the white noise")

print(Sw)

print ("Estimated variance/covariance of the white noise")

print(np.cov(w))

print ("Estimated variance/covariance of the normalized convoluted noise")

print(np.cov(ws))

print ("The mean of the white first sequence: ", np.mean(w[0,:]))

print ("The mean of the normalized convoluted first sequence: ", np.mean(ws[0,:]))

print ("The mean of the white second sequence: ", np.mean(w[1,:]))

print ("The mean of the normalized convoloted second sequence: ", np.mean(ws[1,:]))

print ("Covariance matrix as input for the white noise")

print(Sw)

print ("Estimated covariance matrix of the white noise")

print(np.cov(w))



#wc_1508=np.dot(w,toeplitz(np.exp(-t**2/(2*15.8**2)))) #gamma=0.008

#wc_5=np.dot(w,toeplitz(np.exp(-t**2/(2*5**2)))) #gamma=0.08

#wc_1058=np.dot(w,toeplitz(np.exp(-t**2/(2*1.58**2)))) #gamma=0.8            

#wc_05=np.dot(w,toeplitz(np.exp(-t**2/(2*0.5**2)))) #gamma=8

#wc_0158=np.dot(w,toeplitz(np.exp(-t**2/(2*0.158**2)))) #gamma=80

#wc_005=np.dot(w,toeplitz(np.exp(-t**2/(2*0.05**2)))) #gamma=800

#wc_00158=np.dot(w,toeplitz(np.exp(-t**2/(2*0.0158**2)))) #gamma=8000              

               



P_1508 = toeplitz(np.exp(-t**2/(2*15.8**2)))

K_1508=np.dot(P_1508,np.diag(1./np.sqrt(np.diag(np.dot(P_1508.T,P_1508)))))

ws_1508= np.dot(w,K_1508)

print ('Sigma 15.8')

print('Mean:' ,np.mean(ws_1508[1,:]))

print(np.cov(ws_1508))



P_5 = toeplitz(np.exp(-t**2/(2*5**2)))

K_5=np.dot(P_5,np.diag(1./np.sqrt(np.diag(np.dot(P_5.T,P_5)))))

ws_5= np.dot(w,K_5)

print ('Sigma 5')

print('Mean:' ,np.mean(ws_5[1,:]))

print(np.cov(ws_5))



P_1058 = toeplitz(np.exp(-t**2/(2*1.58**2)))

K_1058=np.dot(P_1058,np.diag(1./np.sqrt(np.diag(np.dot(P_1058.T,P_1058)))))

ws_1058= np.dot(w,K_1058)

print ('Sigma 1.58')

print('Mean:' ,np.mean(ws_1058[1,:]))

print(np.cov(ws_1058))



P_05 = toeplitz(np.exp(-t**2/(2*0.5**2)))

K_05=np.dot(P_05,np.diag(1./np.sqrt(np.diag(np.dot(P_05.T,P_05)))))

ws_05= np.dot(w,K_05)

print ('Sigma 0.5')

print('Mean:' ,np.mean(ws_05[1,:]))

print(np.cov(ws_05))



P_0158 = toeplitz(np.exp(-t**2/(2*0.158**2)))

K_0158=np.dot(P_0158,np.diag(1./np.sqrt(np.diag(np.dot(P_0158.T,P_0158)))))

ws_0158= np.dot(w,K_0158)

print ('Sigma 0.158')

print('Mean:' ,np.mean(ws_0158[1,:]))

print(np.cov(ws_0158))



P_005 = toeplitz(np.exp(-t**2/(2*0.05**2)))

K_005=np.dot(P_005,np.diag(1./np.sqrt(np.diag(np.dot(P_005.T,P_005)))))

ws_005= np.dot(w,K_005)

print ('Sigma 0.05')

print('Mean:' ,np.mean(ws_005[1,:]))

print(np.cov(ws_005))



P_00158 = toeplitz(np.exp(-t**2/(2*0.0158**2)))

K_00158=np.dot(P_00158,np.diag(1./np.sqrt(np.diag(np.dot(P_00158.T,P_00158)))))

ws_00158= np.dot(w,K_00158)

print ('Sigma 0.0158')

print('Mean:' ,np.mean(ws_00158[1,:]))

print(np.cov(ws_00158))



# Plot results

plt.plot(t,w[1,:],label='white') 

plt.plot(t,ws_1508[1,:],label='S15.8') 

plt.plot(t,ws_5[1,:],label='S5') 

plt.plot(t,ws_1058[1,:],label='S1.58') 

plt.plot(t,ws_05[1,:],label='S0.5')

plt.plot(t,ws_0158[1,:],label='S0.158')

plt.plot(t,ws_005[1,:],label='S0.05') 

plt.plot(t,ws_00158[1,:],label='S0.0158',linestyle='dashed') #Dashed to show it overlaps with white noise



plt.title('gamma pallette')

plt.legend(loc='best')

plt.show
plt.acorr(ws[1,:], maxlags=50, label='smooth', color='red')

plt.acorr(w[1,:], maxlags=50, label='white', color='blue')

plt.title('autocorrelate')

plt.legend(loc='best')

plt.show
np.random.seed(random_seed)

# Setting up the time data:

dt = 0.05

T = 5+dt

N = int(round(T/dt))

t = np.arange(0,T,dt)

# Desired covariance matrix (noise in Rˆ2):

# note: this matrix must be symmetric

Sw = np.matrix('9 2;2 16')

# Generate white Gaussian noise sequences:

n = Sw.shape[1] # dimension of noise

L =sqrtm(Sw)  #Sqrtm method

#L=cholesky(Sw, lower=True)  #Cholesky method

w = np.dot(L,np.random.randn(n,N))

# Plot the first white noise sequence:

plt.plot(t,w.T[:,1],label='test')

# Set up convolution matrix:

sigma = 0.158

P = toeplitz(np.exp(-t**2/(2*sigma**2)))

F = np.diag(1./np.sqrt(np.diag(np.dot(P.T,P))))

# Make the smoothened noise:

K = np.dot(P,F)

ws = np.dot(w,K)

plt.plot(t,ws.T[:,1]) # some plot versions plot expect data in same dimension, hence the ws.T to align with w

plt.title('$\sigma=$ ' + str(sigma))

plt.show
#Try traditional normalisation as a way to get correct zero mean and correct variance 

# Setting up the data:

dt = 0.05

T = 5 + dt

N = int(round(T/dt))

t = np.arange(0,T,dt)

np.random.seed(random_seed)

Sw = np.matrix('9 2;2 16') 

#Sw = np.matrix('9 2 4;2 8 5; 4 5 24')

n = Sw.shape[1] # dimension of noise

#white noise

L =sqrtm(Sw)  #Sqrtm method

w = np.dot(L,np.random.randn(n,N))

#convoluted noise

sigma = 0.158

P = toeplitz(np.exp(-t**2/(2*sigma**2)))

wc=np.dot(w,P) 

#Smoothened noise

F = np.diag(1./np.sqrt(np.diag(np.dot(P.T,P))))

ws = np.dot(wc,F)



#Alternative smoothened noise

ws_alt = wc.copy()

for j in range(0,n):

    ws_alt[j,:] -= np.mean(ws_alt[j,:]) # normalize mean to 0

    ws_alt[j,:] /= np.std(ws_alt[j,:]) / np.sqrt(Sw[j,j]) #normalize variance to original covariance matrix



#Plot the result

plt.plot(t,w[1,:],label='white noise') 

#plt.plot(t,wc[1,:],label='Convoluted noise')

plt.plot(t,ws[1,:],label='noise with temporal smoothness')

plt.plot(t,ws_alt[1,:],label='noise with alternative temporal smoothness')

plt.title('$\sigma=$ ' + str(sigma))

plt.legend(loc='upper left')

plt.show



# Calculate the variance/covariance of the generated data sets

print ("Covariance matrix as input for the white noise")

print(Sw)

print ("Estimated variance/covariance of the white noise")

print(np.cov(w))

print ("Estimated variance/covariance of the normalized convoluted noise")

print(np.cov(ws))

print ("Estimated variance/covariance of the alternative normalized convoluted noise")

print(np.cov(ws_alt))

print ("The mean of the white first sequence: ", np.mean(w[0,:]))

print ("The mean of the normalized convoluted first sequence: ", np.mean(ws[0,:]))

print ("The mean of the alternative normalized convoluted first sequence: ", np.mean(ws_alt[0,:]))

print ("The mean of the white second sequence: ", np.sum(w[1,:]))

print ("The mean of the normalized convoloted second sequence: ", np.sum(ws[1,:]))

print ("The mean of the alternative convoloted second sequence: ", np.sum(ws_alt[1,:]))
