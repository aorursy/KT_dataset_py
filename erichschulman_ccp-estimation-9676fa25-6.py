import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d #pre written interpolation function
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats #for kernel function
#constants/data set up
BETA = .9999
GAMMA = .5772 #euler's constant

#load data into memory
data = pd.read_csv('../input/bus1234.csv')

#divide by 1e6 (use the same scale are Rust and AM)
data['miles'] = (data['miles'])/1e6

#switch to date time for ease 
data['date'] = pd.to_datetime(data[['year', 'month']].assign(Day=1))
data = data[['id','group','date','replace','miles']]

#lag date
date_lag = data.copy()
date_lag['date'] = date_lag['date'] - pd.DateOffset(months=1)
data = data.merge(date_lag, how='left', on=['id','group','date'] , suffixes=('','_next'))
data = data.dropna()
#size of step in discretization
STEP = .002

#make states global variables
STATES = np.arange(data['miles'].min(),data['miles'].max(), STEP)
def miles_pdf(i_obs, x_obs, x_next):
    """estimation of mileage pdf following AM using the
    kernel function
    
    this corresponds to pdfdx in AM's code"""
    
    #figure out max number of steps
    dx = (1-i_obs)*(x_next - x_obs) + i_obs*x_next
    
    #number of 'transition' states
    dx_states = np.arange(dx.min(),dx.max() , STEP)
    
    #use kernel groups to make pdf
    kernel1 = stats.gaussian_kde(dx, bw_method='silverman')
    pdfdx = kernel1(dx_states)
    
    return np.array([pdfdx/pdfdx.sum()]).transpose()


MILES_PDF = miles_pdf(data['replace'], data['miles'], data['miles_next'])
#set up plot of both value functions
fig = plt.figure()

#make a plot of both value functions
plt.ylabel('Probability Density')
plt.xlabel('Additional Mileage (divided by 1e6)')
plt.plot(STATES[0:len(MILES_PDF)],MILES_PDF)

plt.show()
def transition_1(i_obs, x_obs , x_next):
    """calculate transitions probabilities,
    non-parametrically
    
    this corresponds to fmat2 in AM's code"""
    
    #transitions when i=1
    pdfdx = miles_pdf(i_obs, x_obs, x_next).transpose()
    
    #zero probability of transitioning to large states
    zeros = np.zeros( (len(STATES),len(STATES)-pdfdx.shape[1]) )
    
    #transitioning to first state and 'jumping' dx states
    fmat1 = np.tile(pdfdx,(len(STATES),1))
    fmat1 = np.concatenate( (fmat1, zeros), axis=1 )

    return fmat1

FMAT1 = transition_1(data['replace'], data['miles'],data['miles_next'])
def transition_0(i_obs, x_obs, x_next):
    """calculate transitions probabilities,
    non-parametrically
    
    this corresponds to fmat1 in AM's code"""
    
    pdfdx = miles_pdf(i_obs, x_obs, x_next).transpose()
    
    #initialize fmat array, transitions when i=0
    end_zeros = np.zeros((1, len(STATES) - pdfdx.shape[1]))
    fmat0 = np.concatenate( (pdfdx, end_zeros), axis=1 )

    for row in range(1, len(STATES)):
        
        #this corresponds to colz i think
        cutoff = ( len(STATES) - row - pdfdx.shape[1] )
        
        #case 1 far enough from the 'end' of the matrix
        if cutoff >= 0:
            start_zeros = np.zeros((1,row))
            end_zeros = np.zeros((1, len(STATES) - pdfdx.shape[1] - row))
            fmat_new = np.concatenate( (start_zeros, pdfdx, end_zeros), axis=1 )
            fmat0 = np.concatenate((fmat0, fmat_new))
       
        #case 2, too far from the end and need to adjust probs
        else:
            pdf_adj = pdfdx[:,0:cutoff]
            pdf_adj = pdf_adj/pdf_adj.sum(axis=1)
            
            start_zeros = np.zeros((1,row))
            fmat_new = np.concatenate( (start_zeros, pdf_adj), axis=1 )
            fmat0 = np.concatenate((fmat0, fmat_new))
            
    return fmat0

FMAT0 = transition_0(data['replace'],data['miles'],data['miles_next'])

PR_TRANS = FMAT0, FMAT1
def initial_pr(i_obs, x_obs, d=0):
    """initial the probability of view a given state following AM.
    just involves logit to predict
    
    Third arguement involves display"""
    
    X = np.array([x_obs, x_obs**2, x_obs**3]).transpose()
    X = sm.add_constant(X)
    
    model = sm.Logit(i_obs,X)
    fit = model.fit(disp=d)
    if d: print(fit.summary())
    
    x_states = np.array([STATES, STATES**2, STATES**3]).transpose()
    x_states = sm.add_constant(x_states)
    
    return fit.predict(x_states)

PR_OBS = initial_pr(data['replace'], data['miles'], d=1)
def hm_value(params, cost, pr_obs, pr_trans):
    """calculate value function using hotz miller approach"""
    
    #set up matrices, transition is deterministic
    trans0, trans1 = pr_trans
    
    #calculate value function for all state
    pr_tile = np.tile( pr_obs.reshape( len(STATES) ,1), (1, len(STATES) ))
    
    denom = (np.identity( len(STATES) ) - BETA*(1-pr_tile)*trans0 - BETA*pr_tile*trans1)
    
    numer = ( (1-pr_obs)*(cost(params, STATES, 0) + GAMMA - np.log(1-pr_obs)) + 
                 pr_obs*(cost(params, STATES, 1) + GAMMA - np.log(pr_obs) ) )
    
    value = np.linalg.inv(denom).dot(numer)
    return value
def hm_prob(params, cost, pr_obs, pr_trans):
    """calculate psi (i.e. CCP likelihood) using 
    the value function from the hotz miller appraoch"""
    
    value = hm_value(params, cost, pr_obs, pr_trans)
    value = value - value.min() #subtract out smallest value
    trans0, trans1 = pr_trans
    
    delta1 = np.exp( cost(params, STATES, 1) + BETA*trans1.dot(value))
    delta0 = np.exp( cost(params, STATES, 0) + BETA*trans0.dot(value) )
    
    return delta1/(delta1+delta0)
class CCP(GenericLikelihoodModel):
    """class for estimating the values of R and theta
    using the CCP routine and the helper functions
    above"""
    
    def __init__(self, i, x, x_next, params, cost, **kwds):
        """initialize the class
        
        i - replacement decisions
        x - miles
        x_next - next periods miles
        params - names for cost function parameters
        cost - cost function specification, takes agruements (params, x, i) """
        
        super(CCP, self).__init__(i, x, **kwds)
        
        #data
        self.endog = i #these names don't work exactly
        self.exog = x #the idea is that x is mean indep of epsilon
        self.x_next = x_next
        
        #transitions
        self.pr_obs = initial_pr(i, x)
        self.trans =  transition_0(i,x,x_next), transition_1(i,x,x_next)
        
        #initial model fit
        self.cost = cost
        self.num_params = len(params)
        self.data.xnames =  params
        self.results = self.fit( start_params=np.ones(self.num_params) )
        
        
    def nloglikeobs(self, params, v=False):
        """psuedo log likelihood function for the CCP estimator"""
        
        # Input our data into the model
        i = self.endog
        x = (self.exog/STEP).astype(int)*STEP #discretized x
           
        #set up hm state pr
        prob = hm_prob(params, self.cost, self.pr_obs, self.trans).transpose()
        prob = interp1d(STATES, prob)
        prob = prob(x)
        
        log_likelihood = (1-i)*np.log(1-prob) + i*np.log(prob)
        
        return -log_likelihood.sum()
    
    
    def iterate(self, numiter):
        """iterate the Hotz Miller estimation procedure 'numiter' times"""
        i = 0
        while(i < numiter):
            #update pr_obs based on parameters
            self.pr_obs = hm_prob(self.results.params, self.cost, self.pr_obs, self.trans)
            
            #refit the model
            self.results = self.fit(start_params=np.ones(self.num_params))
            i = i +1
#define cost functon using lambda expression
linear_cost = lambda params, x, i: (1-i)*x*params[i] + i*params[i]

linear_model = CCP(data['replace'], data['miles'], data['miles_next'], ['theta1','RC'], linear_cost)
print(linear_model.results.summary())
#save params from 1 iteration for later
linear_params0, linear_pr0 = linear_model.results.params, linear_model.pr_obs

linear_model.iterate(2)
print(linear_model.results.summary())
#save first iteration params for later
p_linear0 = hm_prob(linear_params0, linear_model.cost, linear_pr0, linear_model.trans)
p_linear = hm_prob(linear_model.results.params, linear_model.cost, linear_model.pr_obs, linear_model.trans)

#make a plot of both value functions
plt.ylabel('Engine Replacment Probability')
plt.xlabel('Mileage (divided by 10e6)')
plt.plot(STATES, PR_OBS, label='Logit')
plt.plot(STATES,p_linear0,label='Linear (0 iterations)')
plt.plot(STATES, p_linear, label='Linear (2 iterations)')


plt.legend()
plt.show()
quad_cost = lambda params, x, i: (1-i)*(params[0]*x + params[1]*x**2) + i*params[2]
quad_model = CCP(data['replace'], data['miles'], data['miles_next'], ['theta1','theta2', 'R'], quad_cost)
quad_model.iterate(2)
print(quad_model.results.summary())
#set up plot of both value functions
fig = plt.figure()
p_quad = hm_prob(quad_model.results.params, quad_model.cost, quad_model.pr_obs, quad_model.trans)

#make a plot of both value functions
plt.ylabel('Engine Replacment Probability')
plt.xlabel('Mileage (divided by 10e6)')
plt.plot(STATES, PR_OBS, label='Logit')
plt.plot(STATES,p_linear,label='Linear (2 Iterations)')
plt.plot(STATES, p_quad, label='Quadratic (2 Iterations)')

plt.legend()
plt.show()