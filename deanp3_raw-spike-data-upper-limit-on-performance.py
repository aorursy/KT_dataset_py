import xarray as xr #importing xarray
import pandas as pd # importing pandas to make comparison to processed data
import numpy as np
import os 
import matplotlib.pyplot as plt 
%matplotlib inline

#load the processed data you originally got
df = pd.read_csv('../input/train.csv')
df.index = df.Id#turn the first col to index
df = df.iloc[:,1:]#get rid of first col that is now index
stim = np.load('../input/stim.npy')

da = xr.open_dataset('../input/spike_data.nc')['resp']#read in the spike data (1gb)
print(da)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.imshow(stim[50]);plt.xticks([]);plt.yticks([]);
plt.subplot(122)
#isel selects based on the order of the data (0-n) and sel based on the labels of the data.
#nans fill up where data was not collected so for this cell 5 trials were collected so the other 
#trials are filled with nans so I drop them along the trial axis.
da.isel(unit=4).sel(stim=50).dropna('trial').plot();#the 4th unit, the 50th stimulis
da_p = da.sel(t=slice(.05, .35))#grab only the times from 50ms to 350 ms post stimulus
da_p = da_p.transpose('unit', 'stim', 'trial', 't')#transpose for ease

#the overall array is 'ragged' units can have different number of stim, 
#and each stim can have different number of trials so we make a list of non-ragged arrays
units = [unit.dropna('stim',how='all').dropna('trial',how='any') for unit in da_p]
#get number of spikes and take sqrt
units = [unit.sum('t')**0.5 for unit in units]
#get number of trials for each unit
ns = [len(unit.coords['trial']) for unit in units]
#average number of spikes
m_units = [unit.mean('trial') for unit in units]
#simulation
sim_results = pd.DataFrame(np.zeros((len(ns), 2)), columns=['mean', 'sd'])
nsims = 500
for i, n, m_unit in zip(range(len(ns)), ns, m_units): 
    perfect_model = m_unit.values#perfect model is the mean to be used in the simulation
    sims = np.random.normal(loc=perfect_model,#use sample means as true means in sim
                            scale=(4*n)**-0.5,#variance is scaled by number of trials
                            size=(nsims,len(perfect_model)))#make n simulations
    #get simulated fraction variance explained by fitting the true means to noisy simulations.
    sim_r = [np.corrcoef(perfect_model, sim)[0,1]**2 for sim in sims]
    #store results
    sim_results['mean'][i] = np.mean(sim_r)
    sim_results['sd'][i] = np.std(sim_r)
sim_results
plt.errorbar(x=range(len(ns)), y=sim_results['mean'], yerr=2*sim_results['sd']);
plt.grid();plt.xlabel('Unit');plt.ylabel(r'$R^2$ upper limit');plt.ylim(0,1);
plt.title('Theoretical optimal model performance on neural data');