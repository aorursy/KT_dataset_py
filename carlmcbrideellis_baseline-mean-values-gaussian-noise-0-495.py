import numpy  as np

import pandas as pd

import json
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

sample = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
reactivity_data   = np.mean(train['reactivity'].tolist(), axis=1)

reactivity_mu     = np.mean(reactivity_data)

reactivity_sigma  = np.std(reactivity_data)



deg_Mg_pH10_data  = np.mean(train['deg_Mg_pH10'].tolist(), axis=1)

deg_Mg_pH10_mu    = np.mean(deg_Mg_pH10_data)

deg_Mg_pH10_sigma = np.std(deg_Mg_pH10_data)



deg_pH10_data     = np.mean(train['deg_pH10'].tolist(), axis=1)

deg_pH10_mu       = np.mean(deg_pH10_data)

deg_pH10_sigma    = np.std(deg_pH10_data)



deg_Mg_50C_data   = np.mean(train['deg_Mg_50C'].tolist(), axis=1)

deg_Mg_50C_mu     = np.mean(deg_Mg_50C_data)

deg_Mg_50C_sigma  = np.std(deg_Mg_50C_data)



deg_50C_data      = np.mean(train['deg_50C'].tolist(), axis=1)

deg_50C_mu        = np.mean(deg_50C_data)

deg_50C_sigma     = np.std(deg_50C_data)
print("reactivity:  μ=%.4f" %reactivity_mu  +"  σ=%.3f" %reactivity_sigma)

print("deg_Mg_pH10: μ=%.4f" %deg_Mg_pH10_mu +"  σ=%.3f" %deg_Mg_pH10_sigma)

print("deg_pH10:    μ=%.4f" %deg_pH10_mu    +"  σ=%.3f" %deg_pH10_sigma)

print("deg_Mg_50C:  μ=%.4f" %deg_Mg_50C_mu  +"  σ=%.3f" %deg_Mg_50C_sigma)

print("deg_50C:     μ=%.4f" %deg_50C_mu     +"  σ=%.3f" %deg_50C_sigma)
n_values = sample.shape[0]



sample.loc[:,'reactivity']  = np.random.normal(reactivity_mu,  reactivity_sigma,  n_values)

sample.loc[:,'deg_Mg_pH10'] = np.random.normal(deg_Mg_pH10_mu, deg_Mg_pH10_sigma, n_values)

sample.loc[:,'deg_pH10']    = np.random.normal(deg_pH10_mu,    deg_pH10_sigma,    n_values)

sample.loc[:,'deg_Mg_50C']  = np.random.normal(deg_Mg_50C_mu,  deg_Mg_50C_sigma,  n_values)

sample.loc[:,'deg_50C']     = np.random.normal(deg_50C_mu,     deg_50C_sigma,     n_values)



sample.to_csv('submission.csv',index=False)