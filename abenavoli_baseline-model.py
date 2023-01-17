import pymc3 as pm

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline



#define  basis functions

def make_basis(x):

    # the components are 1,x, cos(2 \pi x), sin(2 \pi x)...

    periodic = np.hstack([np.cos(2*np.pi*x),np.sin(2*np.pi*x),np.cos(4*np.pi*x),np.sin(4*np.pi*x),np.cos(6*np.pi*x),np.sin(6*np.pi*x),np.cos(8*np.pi*x),np.sin(8*np.pi*x),np.cos(10*np.pi*x),np.sin(10*np.pi*x),np.cos(12*np.pi*x),np.sin(12*np.pi*x)])

    return np.hstack([np.ones((x.shape[0],1)),x, periodic ]) 



#  periodic basis starts at column:

col_per = 2



#baseline model

def run_model(x,y,plot=False):    

    H = make_basis(x.reshape(-1,1))

    #normalize the data

    yn = (y-np.mean(y))/np.std(y)

    

    #select non periodic components of the basis

    H_np = H[:,0:col_per]



    with pm.Model() as model:

        #prior

        w = pm.Normal('weights', mu=0, sd=50, shape=(H.shape[1],))

        sigma = pm.HalfCauchy('sigma', 5)



        #linear model

        mu = pm.Deterministic('mu', pm.math.matrix_dot(H,w).T)



        #likelihood

        y_obs = pm.Normal('y', mu=mu, sd=sigma, observed=yn)



        #we can do  an approximated inference

    with model:

        inference = pm.ADVI()

        approx = pm.fit(60000, method=inference)

        

    posterior = approx.sample(draws=500)

    

    all_prediction = np.dot(H,posterior['weights'].T).T

    non_periodic_prediction = np.dot(H_np,posterior['weights'][:,0:col_per].T).T

    if plot==True:

        plt.figure()

        plt.plot(x,np.mean(all_prediction,axis=0),'r', label='Overall Mean')

        plt.plot(x,np.mean(non_periodic_prediction,axis=0),'b', label='Mean of the non-periodic comp.')

        plt.legend()

        plt.scatter(x,yn)

    Gradients = []

    for i in range(non_periodic_prediction.shape[0]):

        Gradients.append(np.min(np.gradient(non_periodic_prediction[i,:], x)))

        

    posterior_probability_deriviative_is_positive = len(np.where(np.array(Gradients)>0)[0])/len(Gradients)

    print("probability that the function is increasing=", posterior_probability_deriviative_is_positive)

    if posterior_probability_deriviative_is_positive>0.95:

        return 1

    else:

        return 0



    

#this is the inpu

x = np.linspace(0,1,100)    


y =  x + np.cos(4*np.pi*x) + np.random.randn(len(x))*0.2

plt.plot(x,y)
run_model(x,y,plot=True)
#toy example

y =  x/2-2*np.exp(-(x-0.5)**2) + 2 + np.random.randn(len(x))*0.05

plt.plot(x,y)
run_model(x,y, plot=True)
x = np.linspace(0,1,100)

test_df = pd.read_csv("Dataset/test.csv")





Decision = pd.DataFrame(columns=['Id','Category'])

for r in range(train_df.shape[0]):

    id_row = test_df.iloc[r,0]

    y = test_df.iloc[r,1:].values

    decision = run_model(x,y)

    Decision = Decision.append({'Id': int(id_row), 'Category': int(decision)}, ignore_index=True) 

    print(Decision)

    Decision.to_csv("Decision_baseline.csv",  index=False)