# Import the dependencies

import numpy as np

from scipy.linalg import toeplitz, cholesky, sqrtm, inv

# import scipy.linalg as la

from scipy import signal

from scipy.integrate import odeint

import time

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")

print("Imports done")
def g_gp(x,v):

    """

    Generative process, equation of sensory mapping: g(x) at point x    

   

    INPUTS:

        x       - Hidden state, depth position in centimetres

        v       - Hidden causal state, in this example not used

        

    OUTPUT:

        y       - Temperature in degrees celsius

        

    """

    #t0= 20

    #y=(t0-8)/(0.2*x**2+1)+8

    t0=25

    return t0 -16 / (1 + np.exp(5-x/5))



def dg_gp(x):

    """

    Partial derivative of generative process towards x, equation of sensory mapping: g'(x) at point x    

   

    INPUTS:

        x       - Position in centimetres    

        

    OUTPUT:

        y       - Temperature in degrees celsius

        

    """

    #t0= 20

    #y=-2*0.2*x*(t0-8)/(0.2*x**2+1)**2

    

    return -16/5* np.exp(5-x/5) / (np.exp(5-x/5)+1)**2



# Show the temperature curve

x_show = np.arange (-0,50,0.01)

y_show = g_gp(x_show,0)

dy_show = dg_gp(x_show)

plt.plot(y_show, x_show)

#plt.plot(dy_show, x_show)

plt.ylabel('Depth (centimeters)')

plt.xlabel('Temperature (° C)')

plt.gca().invert_yaxis()

plt.vlines(17, 50, 25, colors='r', linestyles='dashed')

plt.hlines(25, 10,17, colors='r', linestyles='dashed')

plt.text(17.3,27,"Optimal temparature 17° C")

plt.show;



print('Temperature at 25 centimetres is: ', g_gp(25,0), ' degrees celsius')
# Setting up the time data:

dt = 0.005; # integration step, average neuron resets 200 times per second

T = 15+dt; # maximum time considered

t = np.arange(0,T,dt)

N= t.size #Amount of data points

print ('Amount of data points: ', N)

print ('Starting with', t[0:5])

print ('Ending with', t[N-5:N])

print ('Data elements', np.size(t))
class ai_capsule():

    """

        Class that constructs a group of neurons that perform Active Inference for one hidden state, one sensory input, one prior

        In neurology it could eg represent a (micro) column

        

        Version 0.3 Predictive coding solution

    """

    def __init__(self,dt, mu_v, Sigma_w, Sigma_z, a_mu):   

        self.dt = dt    # integration step

        self.mu_x = mu_v   # initializing the best guess of hidden state by the hierarchical prior

        self.F = 0      # Free Energy

        self.eps_x = 0  # delta epsilon_x, prediction error on hidden state

        self.eps_y = 0  # delta epsilon_y, prediction error on sensory measurement

        self.Sigma_w = Sigma_w #Estimated variance of the hidden state 

        self.Sigma_z = Sigma_z # Estimated variance of the sensory observation 

        self.alpha_mu = a_mu # Learning rate of the gradient descent mu (hidden state)

    

    def g(self,x,v):

        """

            equation of sensory mapping of the generative model: g(x) at point x 

            Given as input for this example equal to the true generative process g_gp(x)

        """

        return g_gp(x,v)

    

    def dg(self, x):

        """

            Partial derivative of the equation of sensory mapping of the generative model towards x: g'(x) at point x 

            Given as input for this example equal to the true derivative of generative process dg_gp(x)

        """

        return dg_gp(x)

    

    def f(self,x,v):

        """

            equation of motion of the generative model: f(x) at point x 

            Given as input for this example equal to the prior belief v

        """

        return v

    

    # def df(self,x): Derivative of the equation of motion of the generative model: f'(x) at point x

    # not needed in this example 



    

    def inference_step (self, i, mu_v, y):

        """

        Perceptual inference    



        INPUTS:

            i       - tic, timestamp

            mu_v    - Hierarchical prior input signal (mean) at timestamp

            y       - sensory input signal at timestamp



        INTERNAL:

            mu      - Belief or hidden state estimation



        """

       

        # Calculate the prediction errors

        e_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state

        e_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation

        # motion of prediction error hidden state

        self.eps_x = self.eps_x + dt * self.alpha_mu * (e_x - self.Sigma_w * self.eps_x)

        # motion of prediction error sensory observation

        self.eps_y = self.eps_y + dt * self.alpha_mu * (e_y - self.Sigma_z * self.eps_y)

        # motion of hidden state mu_x 

        self.mu_x = self.mu_x + dt * - self.alpha_mu * (self.eps_x - self.dg(self.mu_x) * self.eps_y)

        

        # Calculate Free Energy to report out

        # Recalculate the prediction errors because hidden state has been updated, could leave it out for performance reasons

        e_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state

        e_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation

        # Calculate Free Energy

        self.F = 0.5 * (e_x**2 / self.Sigma_w + e_y**2 / self.Sigma_z + np.log(self.Sigma_w * self.Sigma_z))

        

        return self.F, self.mu_x , self.g(self.mu_x,0)

class ai_capsule_v1():

    """

        Class that constructs a group of neurons that perform Active Inference for one hidden state, one sensory input, one prior

        In neurology it could eg represent a (micro) column

        

        Version 0.1, Engineering solution, same as the first experiment for base reference 

    """

    def __init__(self,dt, mu_v, Sigma_w, Sigma_z, a_mu):   

        self.dt = dt    # integration step

        self.mu_x = mu_v   # initializing the best guess of hidden state by the hierarchical prior

        self.F = 0      # Free Energy

        self.eps_x = 0  # epsilon_x, prediction error on hidden state

        self.eps_y = 0  # epsilon_y, prediction error on sensory measurement

        self.Sigma_w = Sigma_w #Estimated variance of the hidden state 

        self.Sigma_z = Sigma_z # Estimated variance of the sensory observation 

        self.alpha_mu = a_mu # Learning rate of the gradient descent mu (hidden state)

    

    def g(self,x,v):

        """

            equation of sensory mapping of the generative model: g(x,v) at point x 

            Given as input for this example equal to the true generative process g_gp(x)

        """

        return g_gp(x,v)

    

    def dg(self, x):

        """

            Partial derivative of the equation of sensory mapping of the generative model towards x: g'(x) at point x 

            Given as input for this example equal to the true derivative of generative process dg_gp(x)

        """

        return dg_gp(x)

    

    def f(self,x,v):

        """

            equation of motion of the generative model: f(x,v) at point x 

            Given as input for this example equal to the prior belief v

        """

        return v

    

    # def df(self,x): Derivative of the equation of motion of the generative model: f'(x) at point x

    # not needed in this example 

  

    def inference_step (self, i, mu_v, y):

        """

        Perceptual inference    



        INPUTS:

            i       - tic, timestamp

            mu_v    - Hierarchical prior input signal (mean) at timestamp

            y       - sensory input signal at timestamp



        INTERNAL:

            mu      - Belief or hidden state estimation



        """



        # Calculate prediction errors

        self.eps_x = self.mu_x - self.f(self.mu_x, mu_v)  # prediction error hidden state

        self.eps_y = y - self.g(self.mu_x, mu_v) #prediction error sensory observation

        # Free energy gradient

        dFdmu_x = self.eps_x/self.Sigma_w - self.dg(self.mu_x) * self.eps_y/self.Sigma_z

        # Perception dynamics

        dmu_x   = 0 - self.alpha_mu*dFdmu_x  # Note that this is an example without generalised coordinates of motion hence u'=0

        # motion of mu_x 

        self.mu_x = self.mu_x + self.dt * dmu_x

        

        # Calculate Free Energy to report out

        self.F = 0.5 * (self.eps_x**2 / self.Sigma_w + self.eps_y**2 / self.Sigma_z + np.log(self.Sigma_w * self.Sigma_z))

        

        return self.F, self.mu_x , self.g(self.mu_x,0)
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
def simulation (v, mu_v, Sigma_w, Sigma_z, noise, a_mu, version):

    """

    Basic simplist example perceptual inference    

   

    INPUTS:

        v        - Hydars actual depth, used in generative model, since it is a stationary example hidden state x = v + random fluctuation

        mu_v     - Hydar prior belief/hypotheses of the hidden state

        Sigma_w  - Estimated variance of the hidden state 

        Sigma_z  - Estimated variance of the sensory observation  

        noise    - white, smooth or none

        a_mu     - Learning rate for mu

        version  - 1=engineering solution 1; 3=predictive coding solution



    """





    

    # Init tracking

    mu_x = np.zeros(N) # Belief or estimation of hidden state 

    F = np.zeros(N) # Free Energy of AI neuron

    mu_y = np.zeros(N) # Belief or prediction of sensory signal 

    x = np.zeros(N) # True hidden state

    y = np.zeros(N) # Sensory signal as input to AI neuron



    # Create active inference neuron

    if version==1:

        capsule = ai_capsule_v1(dt, mu_v, Sigma_w, Sigma_z, a_mu) 

    elif version == 3:

        capsule = ai_capsule(dt, mu_v, Sigma_w, Sigma_z, a_mu)



    # Construct noise signals with emporal smoothness:

    np.random.seed(1234)

    sigma = 1/64 # smoothness of the noise parameter, variance of the filter

    w = makeNoise(Sigma_w,sigma,t)

    z = makeNoise(Sigma_z,sigma,t)



    ssim = time.time() # start sim

    

    # Simulation

    for i in np.arange(1,N):

        # Generative process

        if noise == 'white':

            x[i] = v + np.random.randn(1)* Sigma_w

            y[i] = g_gp(x[i],v) + np.random.randn(1)* Sigma_z

        elif noise == 'smooth':

            x[i]= v + w[0,i]

            y[i] = g_gp(x[i],v) + z[0,i]

        else: #no noise

            x[i]= v 

            y[i] = g_gp(x[i],v)

        #Active inference

        F[i], mu_x[i], mu_y[i] =capsule.inference_step(i,mu_v,y[i])



    # Print the results

    tsim = time.time() - ssim

    #print('Simulation time: ' + "%.2f" % tsim + ' sec' )



    return F, mu_x, mu_y, x, y



# Test case



v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'no noise',1,3) # predictive coding

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'no noise',1,1) # engineering version (1.01)



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='Belief predictive coding');

axes[0].plot(t[1:],mu_x2[1:],label='Belief engineering coding');

axes[0].plot(t[1:],x1[1:],label='Generative process');

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[0].grid(1);

axes[1].plot(t[1:],mu_y1[1:],label='Belief predictive coding');

axes[1].plot(t[1:],mu_y2[1:],label='Belief engineering coding');

axes[1].plot(t[1:],y1[1:],label='Generative process');

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='Belief predictive coding');

axes[2].semilogy(t[1:],F2[1:],label='Belief engineering coding');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 25 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'no noise',1,3) # prior and observation balanced





# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='Belief predicting coding');

axes[0].plot(t[1:],x1[1:],label='Generative process');

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[0].grid(1);

axes[1].plot(t[1:],mu_y1[1:],label='Belief predictive coding');

axes[1].plot(t[1:],y1[1:],label='Generative process');

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].plot(t[1:],F1[1:],label='Belief predictive coding');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'no noise',1,3) # prior and observation balanced

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,0.1,'no noise',1,3) # underctain about prior

F3, mu_x3, mu_y3, x3, y3 = simulation(v,mu_v,0.1,1,'no noise',1,3) # Trust generative model, belief high variance in sensor



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='balanced');

axes[0].plot(t[1:],mu_x2[1:],label='Trust sensor');

axes[0].plot(t[1:],mu_x3[1:],label='Trust Genmodel');

axes[0].plot(t[1:],x1[1:],label='Generative process');

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='balanced');

axes[1].plot(t[1:],mu_y2[1:],label='Trust sensor');

axes[1].plot(t[1:],mu_y3[1:],label='Trust Genmodel');

axes[1].plot(t[1:],y1[1:],label='Generative process');

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

#axes[1].legend(loc='upper right');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='balanced');

axes[2].semilogy(t[1:],F2[1:],label='Trust sensor');

axes[2].semilogy(t[1:],F3[1:],label='Trust Genmodel');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel;

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,0.1,0.1,'white',1,3) # predictive coding

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,0.1,0.1,'white',1,1) # engineering version (1.01)



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='Belief predictive coding');

axes[0].plot(t[1:],mu_x2[1:],label='Belief engineering coding');

axes[0].plot(t[1:],x1[1:],label='Generative process');

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[0].grid(1);

axes[1].plot(t[1:],mu_y1[1:],label='Belief predictive coding');

axes[1].plot(t[1:],mu_y2[1:],label='Belief engineering coding');

axes[1].plot(t[1:],y1[1:],label='Generative process');

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='Belief predictive coding');

axes[2].semilogy(t[1:],F2[1:],label='Belief engineering coding');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
# showcase the coloured noise of the experiment, use same random seed to reproduce exact same noise

np.random.seed(42)

z = makeNoise(1,1/64,t)

plt.plot(t,z[0,:]);


v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'smooth',1,3) # predictive coding

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'smooth',1,1) # engineering version (1.01)



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='Belief predictive coding');

axes[0].plot(t[1:],mu_x2[1:],label='Belief engineering coding');

axes[0].plot(t[1:],x1[1:],label='Generative process');

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[0].grid(1);

axes[1].plot(t[1:],mu_y1[1:],label='Belief predictive coding');

axes[1].plot(t[1:],mu_y2[1:],label='Belief engineering coding');

axes[1].plot(t[1:],y1[1:],label='Generative process');

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='Belief predictive coding');

axes[2].semilogy(t[1:],F2[1:],label='Belief engineering coding');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,10,10,'no noise',1,3) # high variance = high uncertainty =low precision

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'no noise',1,3) 

F3, mu_x3, mu_y3, x3, y3 = simulation(v,mu_v,0.1,0.1,'no noise',1,3) # low variance =  low uncertainty = high precision



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='$\sigma^2$=10 low precision');

axes[0].plot(t[1:],mu_x2[1:],label='$\sigma^2$=1');

axes[0].plot(t[1:],mu_x3[1:],label='$\sigma^2$=0.1 high precision');

axes[0].plot(t[1:],x1[1:],label='Generative process'); #note x1=x2=x3 because no noise

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='$\sigma^2$=10 low precision');

axes[1].plot(t[1:],mu_y2[1:],label='$\sigma^2$=1');

axes[1].plot(t[1:],mu_y3[1:],label='\sigma^2$=0.1 high precision');

axes[1].plot(t[1:],y1[1:],label='Generative process'); #note y1=y2=y3 because no noise

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='$\sigma^2$=10 low precision');

axes[2].semilogy(t[1:],F2[1:],label='$\sigma^2$=1');

axes[2].semilogy(t[1:],F3[1:],label='\sigma^2$=0.1 high precision');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);

v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,10,10,'smooth',1,3) # high variance = high uncertainty =low precision

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'smooth',1,3) 

F3, mu_x3, mu_y3, x3, y3 = simulation(v,mu_v,0.1,0.1,'smooth',1,3) # low variance =  low uncertainty = high precision



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Predictive Coding Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='$\sigma^2$=10 low precision');

axes[0].plot(t[1:],mu_x2[1:],label='$\sigma^2$=1');

axes[0].plot(t[1:],mu_x3[1:],label='$\sigma^2$=0.1 high precision');

axes[0].plot(t[1:],x1[1:],label='Generative process (low precision)'); 

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='$\sigma^2$=10 low precision');

axes[1].plot(t[1:],mu_y2[1:],label='$\sigma^2$=1');

axes[1].plot(t[1:],mu_y3[1:],label='\sigma^2$=0.1 high precision');

axes[1].plot(t[1:],y1[1:],label='Generative process (low precision)'); 

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='$\sigma^2$=10 low precision');

axes[2].semilogy(t[1:],F2[1:],label='$\sigma^2$=1');

axes[2].semilogy(t[1:],F3[1:],label='\sigma^2$=0.1 high precision');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'no noise',1,3) # 

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'no noise',2,3) 

F3, mu_x3, mu_y3, x3, y3 = simulation(v,mu_v,1,1,'no noise',4,3) # 



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Basic Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='LR=1');

axes[0].plot(t[1:],mu_x2[1:],label='LR=2');

axes[0].plot(t[1:],mu_x3[1:],label='LR=4');

axes[0].plot(t[1:],x1[1:],label='Generative process'); 

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='LR=1');

axes[1].plot(t[1:],mu_y2[1:],label='LR=2');

axes[1].plot(t[1:],mu_y3[1:],label='LR=4');

axes[1].plot(t[1:],y1[1:],label='Generative process'); 

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='LR=1');

axes[2].semilogy(t[1:],F2[1:],label='LR=2');

axes[2].semilogy(t[1:],F3[1:],label='LR=4');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'smooth',1,3) # 

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'smooth',2,3) 

F3, mu_x3, mu_y3, x3, y3 = simulation(v,mu_v,1,1,'smooth',4,3) # 



# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Basic Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='LR=1');

axes[0].plot(t[1:],mu_x2[1:],label='LR=2');

axes[0].plot(t[1:],mu_x3[1:],label='LR=4');

axes[0].plot(t[1:],x1[1:],label='Generative process'); 

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='LR=1');

axes[1].plot(t[1:],mu_y2[1:],label='LR=2');

axes[1].plot(t[1:],mu_y3[1:],label='LR=4');

axes[1].plot(t[1:],y1[1:],label='Generative process'); 

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='LR=1');

axes[2].semilogy(t[1:],F2[1:],label='LR=2');

axes[2].semilogy(t[1:],F3[1:],label='LR=4');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);
v = 30 # actual depth Hydar

mu_v = 25 # Hydars belief of the depth

F1, mu_x1, mu_y1, x1, y1 = simulation(v,mu_v,1,1,'smooth',10,1) # Engineering solution

F2, mu_x2, mu_y2, x2, y2 = simulation(v,mu_v,1,1,'smooth',10,3) # Predictive coding





# Plot results:

fig, axes = plt.subplots(3, 1, sharex='col');

fig.suptitle('Basic Active Inference');

axes[0].plot(t[1:],mu_x1[1:],label='Engineering solution');

axes[0].plot(t[1:],mu_x2[1:],label='Predictive coding');

axes[0].plot(t[1:],x1[1:],label='Generative process'); 

axes[0].hlines(mu_v, 0,T, label='Prior belief')

axes[0].set_ylabel('position');

axes[0].grid(1);

fig.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes[1].plot(t[1:],mu_y1[1:],label='Engineering solution');

axes[1].plot(t[1:],mu_y2[1:],label='Predictive coding');

axes[1].plot(t[1:],y1[1:],label='Generative process'); 

axes[1].hlines(g_gp(mu_v,0), 0,T, label='Prior belief')

axes[1].set_ylabel('Temperature');

axes[1].grid(1);

axes[2].semilogy(t[1:],F1[1:],label='LR=1');

axes[2].semilogy(t[1:],F2[1:],label='LR=2');

axes[2].set_xlabel('time [s]');

axes[2].set_ylabel('Free energy');

axes[2].grid(1);